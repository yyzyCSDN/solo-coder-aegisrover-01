"""Versioned record repository with optimistic concurrency and cursor pagination.

The repository is the platform's single source of truth for mission records, map
revisions, session state and operational settings. Every write bumps a per-record
version and records an immutable history row, so a caller that read version *n* can
detect that somebody else already wrote version *n + 1* instead of silently
overwriting it.

Multi-record changes run inside :meth:`Repository.transaction` (or the convenience
wrapper :meth:`Repository.batch_mutate`): every write in the block shares one SQLite
transaction, so an error halfway through a batch rolls *all* of it back instead of
leaving half-applied interrelated records. History itself is reclaimed with
:meth:`Repository.prune_history` under an explicit retention policy; hash-chained and
append-only namespaces (audit trail, event log) are protected by the policy defaults.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Sequence

__all__ = ('Record', 'Page', 'PutOp', 'DeleteOp', 'RetentionPolicy', 'PruneReport',
           'VersionConflict', 'NotFound', 'Repository', 'PROTECTED_NAMESPACES',
           'DEFAULT_RETENTION', 'canonical_json')

#: Append-only / hash-chained namespaces whose rows must never be pruned silently.
PROTECTED_NAMESPACES: tuple[str, ...] = ('audit', 'events')


def canonical_json(value: Any) -> str:
    """Deterministic JSON used for digests and stored payloads."""
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


def _digest(ns: str, key: str, version: int, payload: Any, updated_at: float) -> str:
    material = '\x1f'.join((ns, key, str(version), canonical_json(payload), f'{updated_at:.6f}'))
    return hashlib.sha256(material.encode()).hexdigest()


class VersionConflict(RuntimeError):
    """Raised when a write is based on a stale version."""

    def __init__(self, ns: str, key: str, expected: int, actual: int | None):
        super().__init__(f'{ns}/{key}: expected version {expected}, found {actual}')
        self.ns = ns
        self.key = key
        self.expected = expected
        self.actual = actual


class NotFound(KeyError):
    """Raised when a record does not exist."""

    def __init__(self, ns: str, key: str):
        super().__init__(f'{ns}/{key}')
        self.ns = ns
        self.key = key


@dataclass(frozen=True)
class Record:
    ns: str
    key: str
    version: int
    payload: Any
    etag: str
    updated_at: float
    deleted: bool = False

    def to_dict(self) -> dict:
        return {
            'namespace': self.ns,
            'key': self.key,
            'version': self.version,
            'payload': self.payload,
            'etag': self.etag,
            'updated_at': self.updated_at,
            'deleted': self.deleted,
        }


@dataclass(frozen=True)
class Page:
    items: tuple[Record, ...]
    cursor: str | None
    has_more: bool

    def to_dict(self) -> dict:
        return {'items': [i.to_dict() for i in self.items], 'cursor': self.cursor, 'has_more': self.has_more}


@dataclass(frozen=True)
class PutOp:
    """One upsert inside a batch."""
    ns: str
    key: str
    payload: Any
    expected: int | None = None


@dataclass(frozen=True)
class DeleteOp:
    """One tombstone inside a batch."""
    ns: str
    key: str
    expected: int | None = None


@dataclass(frozen=True)
class RetentionPolicy:
    """Convergence rule for ``record_history`` rows.

    A history version survives when it is the record's current/live version
    (``keep_live``), or one of the most recent ``keep_last`` versions, or newer than
    ``max_age_seconds``. Everything else is pruned. Namespaces in
    :data:`PROTECTED_NAMESPACES` are skipped unless explicitly listed in
    ``include_namespaces``; pruning is restricted to ``namespaces`` when it is set.
    """
    keep_last: int = 50
    max_age_seconds: float | None = None
    keep_live: bool = True
    namespaces: tuple[str, ...] | None = None
    include_namespaces: tuple[str, ...] = ()

    def applies_to(self, ns: str) -> bool:
        if self.namespaces is not None:
            return ns in self.namespaces
        if ns in PROTECTED_NAMESPACES:
            return ns in self.include_namespaces
        return True


@dataclass(frozen=True)
class PruneReport:
    deleted_history_rows: int = 0
    affected_records: int = 0

    def __bool__(self) -> bool:
        return self.deleted_history_rows > 0


DEFAULT_RETENTION = RetentionPolicy(keep_last=50)


SCHEMA = """
CREATE TABLE IF NOT EXISTS records (
    ns TEXT NOT NULL,
    key TEXT NOT NULL,
    version INTEGER NOT NULL,
    payload TEXT NOT NULL,
    etag TEXT NOT NULL,
    updated_at REAL NOT NULL,
    deleted INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (ns, key)
);
CREATE TABLE IF NOT EXISTS record_history (
    ns TEXT NOT NULL,
    key TEXT NOT NULL,
    version INTEGER NOT NULL,
    payload TEXT NOT NULL,
    etag TEXT NOT NULL,
    updated_at REAL NOT NULL,
    deleted INTEGER NOT NULL DEFAULT 0,
    recorded_at REAL NOT NULL,
    PRIMARY KEY (ns, key, version)
);
CREATE INDEX IF NOT EXISTS records_ns_key ON records (ns, key);
"""


class Repository:
    """SQLite-backed versioned repository.

    Parameters
    ----------
    path:
        Filesystem path for the database, or ``':memory:'`` for tests.
    clock:
        Callable returning the current time; injectable so tests are deterministic.
    """

    def __init__(self, path: str = ':memory:', clock=time.time):
        self.path = path
        self._clock = clock
        self._conn = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute('PRAGMA journal_mode=WAL')
        self._conn.execute('PRAGMA foreign_keys=ON')
        self._conn.executescript(SCHEMA)
        self._savepoint_depth = 0

    # -- transactions ----------------------------------------------------------
    @contextmanager
    def transaction(self):
        """One atomic block for several interrelated writes.

        The outermost block opens ``BEGIN IMMEDIATE``; nested blocks become
        SAVEPOINTs. Raising any exception rolls back exactly that block (and every
        write done through this repository inside it, including audit entries);
        completing normally commits everything together.
        """
        if self._savepoint_depth == 0:
            self._conn.execute('BEGIN IMMEDIATE')
            self._savepoint_depth = 1
            try:
                yield self
            except Exception:
                self._conn.execute('ROLLBACK')
                self._savepoint_depth = 0
                raise
            else:
                self._conn.execute('COMMIT')
                self._savepoint_depth = 0
        else:
            name = f'sp_{self._savepoint_depth}'
            self._conn.execute(f'SAVEPOINT {name}')
            self._savepoint_depth += 1
            depth = self._savepoint_depth
            try:
                yield self
            except Exception:
                self._conn.execute(f'ROLLBACK TO SAVEPOINT {name}')
                self._conn.execute(f'RELEASE SAVEPOINT {name}')
                self._savepoint_depth = depth - 1
                raise
            else:
                self._conn.execute(f'RELEASE SAVEPOINT {name}')
                self._savepoint_depth = depth - 1

    def batch_mutate(self, ops: Sequence[PutOp | DeleteOp]) -> tuple[Record, ...]:
        """Apply many upserts/tombstones atomically; all succeed or none do."""
        with self.transaction():
            return tuple(self.apply(op) for op in ops)

    def apply(self, op: PutOp | DeleteOp) -> Record:
        """Apply a single :class:`PutOp` / :class:`DeleteOp` inside open txn."""
        if isinstance(op, PutOp):
            return self.put(op.ns, op.key, op.payload, expected=op.expected)
        if isinstance(op, DeleteOp):
            return self.delete(op.ns, op.key, expected=op.expected)
        raise TypeError(f'unsupported batch op: {op!r}')

    # -- reads -----------------------------------------------------------------
    def get(self, ns: str, key: str, *, include_deleted: bool = False) -> Record:
        row = self._conn.execute(
            'SELECT ns, key, version, payload, etag, updated_at, deleted FROM records WHERE ns=? AND key=?',
            (ns, key),
        ).fetchone()
        if row is None or (row['deleted'] and not include_deleted):
            raise NotFound(ns, key)
        return self._record(row)

    def maybe_get(self, ns: str, key: str) -> Record | None:
        try:
            return self.get(ns, key)
        except NotFound:
            return None

    def history(self, ns: str, key: str) -> tuple[Record, ...]:
        rows = self._conn.execute(
            'SELECT ns, key, version, payload, etag, updated_at, deleted FROM record_history'
            ' WHERE ns=? AND key=? ORDER BY version',
            (ns, key),
        ).fetchall()
        return tuple(self._record(r) for r in rows)

    def namespaces(self) -> tuple[str, ...]:
        rows = self._conn.execute('SELECT DISTINCT ns FROM records ORDER BY ns').fetchall()
        return tuple(r['ns'] for r in rows)

    def keys_with_prefix(self, ns: str, prefix: str) -> tuple[str, ...]:
        """Distinct stored keys in ``ns`` starting with ``prefix`` (any table)."""
        rows = self._conn.execute(
            'SELECT DISTINCT key FROM record_history WHERE ns=? AND key LIKE ?',
            (ns, f'{prefix}%')).fetchall()
        return tuple(r['key'] for r in rows)

    # -- writes ----------------------------------------------------------------
    def put(self, ns: str, key: str, payload: Any, *, expected: int | None = None) -> Record:
        """Insert or update ``payload``.

        ``expected=None`` means "create or overwrite unconditionally"; pass the
        version you read to get optimistic concurrency instead.
        """
        with self.transaction():
            row = self._conn.execute(
                'SELECT version, deleted FROM records WHERE ns=? AND key=?', (ns, key)
            ).fetchone()
            current = None if row is None else int(row['version'])
            if expected is not None:
                if current is None or current != expected:
                    raise VersionConflict(ns, key, expected, current)
            version = 1 if current is None else current + 1
            updated_at = self._clock()
            etag = _digest(ns, key, version, payload, updated_at)
            body = canonical_json(payload)
            self._conn.execute(
                'INSERT INTO records (ns, key, version, payload, etag, updated_at, deleted)'
                ' VALUES (?,?,?,?,?,?,0)'
                ' ON CONFLICT(ns, key) DO UPDATE SET version=excluded.version, payload=excluded.payload,'
                ' etag=excluded.etag, updated_at=excluded.updated_at, deleted=0',
                (ns, key, version, body, etag, updated_at),
            )
            self._conn.execute(
                'INSERT INTO record_history (ns, key, version, payload, etag, updated_at, deleted, recorded_at)'
                ' VALUES (?,?,?,?,?,?,0,?)',
                (ns, key, version, body, etag, updated_at, self._clock()),
            )
        return Record(ns, key, version, payload, etag, updated_at)

    def delete(self, ns: str, key: str, *, expected: int | None = None) -> Record:
        """Tombstone a record. History is preserved (subject to retention pruning)."""
        with self.transaction():
            row = self._conn.execute(
                'SELECT version, payload FROM records WHERE ns=? AND key=? AND deleted=0', (ns, key)
            ).fetchone()
            if row is None:
                raise NotFound(ns, key)
            current = int(row['version'])
            if expected is not None and expected != current:
                raise VersionConflict(ns, key, expected, current)
            version = current + 1
            updated_at = self._clock()
            etag = _digest(ns, key, version, None, updated_at)
            self._conn.execute(
                'UPDATE records SET version=?, deleted=1, etag=?, updated_at=? WHERE ns=? AND key=?',
                (version, etag, updated_at, ns, key),
            )
            self._conn.execute(
                'INSERT INTO record_history (ns, key, version, payload, etag, updated_at, deleted, recorded_at)'
                ' VALUES (?,?,?,?,?,?,1,?)',
                (ns, key, version, row['payload'], etag, updated_at, self._clock()),
            )
        return Record(ns, key, version, None, etag, updated_at, deleted=True)

    # -- retention / convergence -----------------------------------------------
    def prune_history(self, policy: RetentionPolicy | None = None) -> PruneReport:
        """Delete ``record_history`` rows that fall outside ``policy``.

        Always preserves the live row (current version) and the tombstone row per
        record unless ``keep_live`` is disabled, so optimistic concurrency and
        rollback-to-last stay intact. Protected namespaces (audit hash chain,
        ordered event log) are skipped unless the policy opts in explicitly.
        """
        policy = policy or DEFAULT_RETENTION
        # Parameters must be appended in the same order the placeholders appear.
        conditions: list[str] = []
        params: list[Any] = []
        if policy.keep_live:
            conditions.append(
                'h.version < COALESCE((SELECT r.version FROM records r'
                ' WHERE r.ns = h.ns AND r.key = h.key), 0)')
        if policy.keep_last > 0:
            conditions.append(
                'h.version <= (SELECT MAX(version) - ? FROM record_history'
                ' WHERE ns = h.ns AND key = h.key)')
            params.append(policy.keep_last)
        if policy.max_age_seconds is not None:
            cutoff = float(self._clock()) - float(policy.max_age_seconds)
            conditions.append('h.updated_at < ?')
            params.append(cutoff)
        where = ' AND '.join(conditions)
        applicable = [ns for (ns,) in self._conn.execute(
            'SELECT DISTINCT ns FROM record_history').fetchall() if policy.applies_to(ns)]
        if not applicable:
            return PruneReport()
        ns_placeholders = ','.join('?' for _ in applicable)
        candidate_sql = (
            f'SELECT h.ns, h.key, h.version FROM record_history h'
            f' WHERE h.ns IN ({ns_placeholders}) AND ({where})')
        with self.transaction():
            candidates = self._conn.execute(
                candidate_sql, applicable + params).fetchall()
            removed = len(candidates)
            affected = len({(r['ns'], r['key']) for r in candidates})
            self._conn.execute(
                f'DELETE FROM record_history WHERE rowid IN'
                f' (SELECT h.rowid FROM record_history h'
                f'  WHERE h.ns IN ({ns_placeholders}) AND ({where}))',
                applicable + params)
        return PruneReport(deleted_history_rows=removed, affected_records=affected)

    def purge_keys(self, ns: str, keys: Iterable[str]) -> int:
        """Hard-delete records *and* their history (used by retention on map logs).

        Unlike :meth:`delete` this leaves no tombstone; callers are expected to run
        it only under an explicit retention policy. Returns the number of records
        removed.
        """
        keys = tuple(keys)
        if not keys:
            return 0
        placeholders = ','.join('?' for _ in keys)
        with self.transaction():
            cur = self._conn.execute(
                f'DELETE FROM records WHERE ns=? AND key IN ({placeholders})', (ns, *keys))
            removed = cur.rowcount
            self._conn.execute(
                f'DELETE FROM record_history WHERE ns=? AND key IN ({placeholders})', (ns, *keys))
        return removed

    # -- paging ----------------------------------------------------------------
    def page(self, ns: str, *, limit: int = 50, cursor: str | None = None, prefix: str = '') -> Page:
        """Stable keyset pagination ordered by key.

        ``cursor`` is an opaque encoding of the last key returned. Callers must treat
        it as opaque; the implementation is free to change it.
        """
        if limit <= 0:
            raise ValueError('limit must be positive')
        after = _decode_cursor(cursor) if cursor else ''
        rows = self._conn.execute(
            'SELECT ns, key, version, payload, etag, updated_at, deleted FROM records'
            ' WHERE ns=? AND deleted=0 AND key>? AND key LIKE ? ORDER BY key LIMIT ?',
            (ns, after, f'{prefix}%', limit + 1),
        ).fetchall()
        has_more = len(rows) > limit
        selected = rows[:limit]
        items = tuple(self._record(r) for r in selected)
        next_cursor = _encode_cursor(selected[-1]['key']) if selected else None
        return Page(items, next_cursor if has_more else None, has_more)

    def scan(self, ns: str, prefix: str = '') -> Iterator[Record]:
        cursor = None
        while True:
            page = self.page(ns, limit=200, cursor=cursor, prefix=prefix)
            yield from page.items
            if not page.has_more:
                return
            cursor = page.cursor

    def count(self, ns: str, prefix: str = '') -> int:
        row = self._conn.execute(
            'SELECT COUNT(*) AS n FROM records WHERE ns=? AND deleted=0 AND key LIKE ?',
            (ns, f'{prefix}%'),
        ).fetchone()
        return int(row['n'])

    def last_key(self, ns: str) -> str | None:
        """Highest key in ``ns``; append-only logs use this to find their head."""
        row = self._conn.execute(
            'SELECT key FROM records WHERE ns=? AND deleted=0 ORDER BY key DESC LIMIT 1', (ns,)
        ).fetchone()
        return None if row is None else row['key']

    def close(self) -> None:
        self._conn.close()

    # -- helpers ---------------------------------------------------------------
    @staticmethod
    def _record(row: sqlite3.Row) -> Record:
        return Record(
            ns=row['ns'],
            key=row['key'],
            version=int(row['version']),
            payload=json.loads(row['payload']) if row['payload'] is not None else None,
            etag=row['etag'],
            updated_at=float(row['updated_at']),
            deleted=bool(row['deleted']),
        )


def _encode_cursor(key: str) -> str:
    import base64

    return base64.urlsafe_b64encode(key.encode()).decode().rstrip('=')


def _decode_cursor(cursor: str) -> str:
    import base64

    padding = '=' * (-len(cursor) % 4)
    return base64.urlsafe_b64decode(cursor + padding).decode()
