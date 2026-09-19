"""Versioned record repository with optimistic concurrency and cursor pagination.

The repository is the platform's single source of truth for mission records, map
revisions, session state and operational settings. Every write bumps a per-record
version and records a history row, so a caller that read version *n* can detect
that somebody else already wrote version *n + 1* instead of silently overwriting it.

Two guards keep the store usable over months of writes. :meth:`Repository.transaction`
groups any number of writes into one atomic unit, so a batch of related records
commits all-or-nothing instead of leaving a half-applied set behind when something
fails midway. And :meth:`Repository.prune_history` converges the history table
according to a :class:`RetentionPolicy` — the newest versions of each record always
survive, older ones are physically removed — so months of edits do not fill the disk.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import groupby
from typing import Any, Iterable, Iterator

__all__ = ('Record', 'Page', 'VersionConflict', 'NotFound', 'TransactionError',
           'RetentionPolicy', 'Repository', 'canonical_json')


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


class TransactionError(RuntimeError):
    """Raised when a transaction cannot commit because a nested unit failed."""


@dataclass(frozen=True)
class RetentionPolicy:
    """How much history to keep per record when pruning.

    A history row survives pruning when it is among the newest ``keep_versions``
    rows of its record *or* younger than ``keep_seconds``; only a row that fails
    both tests is removed, so the policy can be driven by count, by age, or both.
    ``keep_versions`` must be at least 1 so the current version always keeps its
    history row.
    """

    keep_versions: int = 10
    keep_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.keep_versions < 1:
            raise ValueError('keep_versions must be at least 1')
        if self.keep_seconds is not None and self.keep_seconds <= 0:
            raise ValueError('keep_seconds must be positive')


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
        self._tx_depth = 0
        self._tx_doomed = False

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

    # -- transactions ----------------------------------------------------------
    @contextmanager
    def transaction(self) -> Iterator['Repository']:
        """Group any number of writes into one atomic unit.

        Every ``put``/``delete`` (and every service method built on them) issued
        inside the block commits together when the block exits cleanly; if the
        body raises, everything is rolled back, so a batch of related records is
        never left half applied. Blocks may nest: an inner block joins the outer
        transaction, and if an inner failure is swallowed the outermost commit
        raises :class:`TransactionError` rather than committing a partial set.

        A single failed ``put``/``delete`` inside the block (for example a
        :class:`VersionConflict` the caller catches and handles) does not doom
        the batch — the failed statement simply never happened. Letting the
        exception propagate out of the block always rolls everything back.
        """
        self._begin()
        try:
            yield self
        except BaseException as exc:
            self._finish(exc, doom=True)
            raise
        if self._tx_doomed:
            self._finish(None, doom=False)
            raise TransactionError(
                'a nested operation failed; rolled back instead of committing a partial set')
        self._finish(None, doom=False)

    def _begin(self) -> None:
        if self._tx_depth == 0:
            self._conn.execute('BEGIN IMMEDIATE')
            self._tx_doomed = False
        self._tx_depth += 1

    def _finish(self, error: BaseException | None, *, doom: bool) -> None:
        if doom:
            self._tx_doomed = True
        self._tx_depth -= 1
        if self._tx_depth == 0:
            self._conn.execute('ROLLBACK' if error is not None or self._tx_doomed else 'COMMIT')
            self._tx_doomed = False

    # -- writes ----------------------------------------------------------------
    def put(self, ns: str, key: str, payload: Any, *, expected: int | None = None) -> Record:
        """Insert or update ``payload``.

        ``expected=None`` means "create or overwrite unconditionally"; pass the
        version you read to get optimistic concurrency instead. Inside
        :meth:`transaction` the write joins the surrounding atomic unit.
        """
        self._begin()
        try:
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
        except Exception as exc:
            self._finish(exc, doom=False)
            raise
        self._finish(None, doom=False)
        return Record(ns, key, version, payload, etag, updated_at)

    def delete(self, ns: str, key: str, *, expected: int | None = None) -> Record:
        """Tombstone a record. History is preserved."""
        self._begin()
        try:
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
        except Exception as exc:
            self._finish(exc, doom=False)
            raise
        self._finish(None, doom=False)
        return Record(ns, key, version, None, etag, updated_at, deleted=True)

    # -- retention -------------------------------------------------------------
    def prune_history(self, policy: RetentionPolicy, *, ns: str | None = None,
                      now: float | None = None) -> dict[str, int]:
        """Physically remove history rows that fall outside ``policy``.

        For every record the newest ``policy.keep_versions`` history rows always
        survive, and so does any row recorded within ``policy.keep_seconds`` of
        ``now``; only rows that fail both tests are removed. The live record is
        never touched. Returns the number of rows removed per namespace
        (namespaces where nothing was pruned are omitted).
        """
        cutoff = None
        if policy.keep_seconds is not None:
            cutoff = (self._clock() if now is None else now) - policy.keep_seconds
        removed: dict[str, int] = {}
        with self.transaction():
            query = 'SELECT ns, key, version, recorded_at FROM record_history'
            args: tuple = ()
            if ns is not None:
                query += ' WHERE ns=?'
                args = (ns,)
            rows = self._conn.execute(query + ' ORDER BY ns, key, version', args).fetchall()
            for (row_ns, key), group in groupby(rows, key=lambda r: (r['ns'], r['key'])):
                stale = list(group)[:-policy.keep_versions]
                for row in stale:
                    if cutoff is not None and float(row['recorded_at']) >= cutoff:
                        continue
                    self._conn.execute(
                        'DELETE FROM record_history WHERE ns=? AND key=? AND version=?',
                        (row_ns, key, int(row['version'])),
                    )
                    removed[row_ns] = removed.get(row_ns, 0) + 1
        return removed

    def purge(self, ns: str, key: str) -> bool:
        """Physically remove a record and all of its history.

        This is the retention counterpart of :meth:`delete`: where ``delete``
        tombstones and preserves history, ``purge`` reclaims the space. It is
        meant for retention jobs — purging a namespace whose readers rely on an
        unbroken sequence (such as the audit chain) breaks their guarantees.
        Returns True if a live record was removed.
        """
        with self.transaction():
            gone = self._conn.execute(
                'DELETE FROM records WHERE ns=? AND key=?', (ns, key)
            ).rowcount
            self._conn.execute('DELETE FROM record_history WHERE ns=? AND key=?', (ns, key))
        return bool(gone)

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
