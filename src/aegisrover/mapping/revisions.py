"""Versioned map revisions with three-way patch merge.

Two robots edit different parts of the same map. Storing every revision and merging
three-way (base / ours / theirs) means non-overlapping edits combine automatically,
while a cell that both sides changed to *different* values is reported as a conflict
instead of one side silently winning. The interchange format carries an explicit
version so a legacy v1 payload can be migrated instead of misread.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field, replace
from typing import Iterable

from aegisrover.storage.audit import AuditLog
from aegisrover.storage.repository import Repository, canonical_json

__all__ = ('MapRevision', 'MergeResult', 'MapRepository', 'MapFormatError',
           'MapPruneReport')

NAMESPACE = 'maps'
FORMAT_VERSION = 2


class MapFormatError(ValueError):
    pass


@dataclass(frozen=True)
class MapPruneReport:
    removed_revisions: int = 0
    affected_maps: tuple[str, ...] = ()


def _digest(payload: dict) -> str:
    body = canonical_json({k: v for k, v in payload.items() if k != 'digest'})
    return hashlib.sha256(body.encode()).hexdigest()


def _normalise_cells(cells: dict) -> dict[str, int]:
    out: dict[str, int] = {}
    for key, value in cells.items():
        if isinstance(key, tuple):
            key = f'{key[0]},{key[1]}'
        out[str(key)] = int(value)
    return dict(sorted(out.items(), key=lambda item: tuple(int(p) for p in item[0].split(','))))


@dataclass(frozen=True)
class MapRevision:
    map_id: str
    revision: int
    parent: int | None
    author: str
    note: str
    cells: dict
    digest: str
    created_at: float

    def to_dict(self) -> dict:
        return {'map_id': self.map_id, 'revision': self.revision, 'parent': self.parent,
                'author': self.author, 'note': self.note, 'cells': self.cells,
                'digest': self.digest, 'created_at': self.created_at}

    @staticmethod
    def from_dict(payload: dict) -> 'MapRevision':
        return MapRevision(payload['map_id'], int(payload['revision']), payload.get('parent'),
                           payload['author'], payload.get('note', ''), dict(payload['cells']),
                           payload['digest'], float(payload['created_at']))

    def cell(self, x: int, y: int) -> int | None:
        return self.cells.get(f'{x},{y}')


@dataclass(frozen=True)
class MergeResult:
    cells: dict
    conflicts: tuple[str, ...] = ()
    applied_from_ours: tuple[str, ...] = ()
    applied_from_theirs: tuple[str, ...] = ()

    @property
    def clean(self) -> bool:
        return not self.conflicts


class MapRepository:
    def __init__(self, repository: Repository, audit: AuditLog | None = None, clock=time.time):
        self._repo = repository
        self._audit = audit if audit is not None else AuditLog(repository, clock)
        self._clock = clock

    # -- writes ----------------------------------------------------------------
    def save(self, map_id: str, cells: dict, *, expected_revision: int | None = None,
             actor: str = 'operator', note: str = '') -> MapRevision:
        with self._repo.transaction():
            return self._save_locked(map_id, cells, expected_revision, actor, note)

    def save_many(self, changes: Iterable[tuple], *, actor: str = 'operator',
                  note: str = '') -> tuple[MapRevision, ...]:
        """Apply many (possibly interrelated) map changes atomically.

        Each change is ``(map_id, cells)`` or ``(map_id, cells,
        expected_revision)``. All revisions and their audit entries commit together;
        a stale revision, bad payload or any other failure rolls the whole group
        back so a caller never sees half a batch.
        """
        normalised = []
        for change in changes:
            if len(change) == 2:
                map_id, cells = change
                expected = None
            elif len(change) == 3:
                map_id, cells, expected = change
            else:
                raise MapFormatError('change must be (map_id, cells) or'
                                     ' (map_id, cells, expected_revision)')
            normalised.append((map_id, cells, expected))
        with self._repo.transaction():
            saved = tuple(self._save_locked(map_id, cells, expected, actor,
                                            note or 'batch save')
                          for map_id, cells, expected in normalised)
            self._audit.append(actor, 'map.save_many',
                               ','.join(m.map_id for m in saved),
                               {'revisions': [m.revision for m in saved],
                                'maps': len(saved)})
        return saved

    def _save_locked(self, map_id: str, cells: dict,
                     expected_revision: int | None, actor: str, note: str) -> MapRevision:
        """Build and persist one revision; caller owns the transaction."""
        latest = self.latest(map_id)
        current = 0 if latest is None else latest.revision
        if expected_revision is not None and expected_revision != current:
            from aegisrover.storage.repository import VersionConflict

            raise VersionConflict(NAMESPACE, map_id, expected_revision, current)
        normalised_cells = _normalise_cells(cells)
        staged = {'map_id': map_id, 'revision': current + 1,
                  'parent': None if latest is None else latest.revision,
                  'author': actor, 'note': note, 'cells': normalised_cells,
                  'created_at': self._clock()}
        revision = MapRevision(**staged, digest=_digest(staged))
        self._repo.put(NAMESPACE, f'{map_id}@{revision.revision:05d}', revision.to_dict())
        self._audit.append(actor, 'map.save', map_id,
                           {'revision': revision.revision, 'cells': len(normalised_cells),
                            'digest': revision.digest})
        return revision

    def prune_revisions(self, *, keep_last: int = 50,
                        max_age_seconds: float | None = None,
                        actor: str = 'retention') -> MapPruneReport:
        """Converge stored map revisions under a retention policy.

        Every map keeps its latest revision plus the ``keep_last`` newest ones (and
        anything younger than ``max_age_seconds``). Older revision *records* (not
        just history rows) are hard-deleted, so the maps namespace stops growing
        without bound. Returns which revisions and maps were affected.
        """
        cutoff = None if max_age_seconds is None else self._clock() - float(max_age_seconds)
        removed = 0
        affected: list[str] = []
        with self._repo.transaction():
            all_keys = self._repo.keys_with_prefix(NAMESPACE, '')
            map_ids = sorted({key.rsplit('@', 1)[0] for key in all_keys
                              if '@' in key})
            for map_id in map_ids:
                revisions = self.history(map_id)
                latest_rev = revisions[-1].revision
                keep_floor = {m.revision for m in revisions[-keep_last:]} if keep_last > 0 else set()
                stale = []
                for revision in revisions:
                    # The live head and the newest ``keep_last`` are never pruned;
                    # everything older than the age window (or everything, when no
                    # window is configured) is eligible.
                    if revision.revision == latest_rev or revision.revision in keep_floor:
                        continue
                    if cutoff is not None and revision.created_at >= cutoff:
                        continue
                    stale.append(f'{map_id}@{revision.revision:05d}')
                if stale:
                    removed += self._repo.purge_keys(NAMESPACE, stale)
                    affected.append(map_id)
            if removed:
                self._audit.append(actor, 'map.prune', NAMESPACE,
                                   {'removed_revisions': removed,
                                    'maps': affected, 'keep_last': keep_last})
        return MapPruneReport(removed, tuple(affected))

    def rollback(self, map_id: str, revision: int, *, actor: str = 'operator') -> MapRevision:
        target = self.get(map_id, revision)
        return self.save(map_id, target.cells, actor=actor, note=f'rollback to r{revision}')

    # -- reads -----------------------------------------------------------------
    def latest(self, map_id: str) -> MapRevision | None:
        items = self.history(map_id)
        return None if not items else items[-1]

    def get(self, map_id: str, revision: int | None = None) -> MapRevision:
        if revision is None:
            latest = self.latest(map_id)
            if latest is None:
                raise KeyError(map_id)
            return latest
        record = self._repo.maybe_get(NAMESPACE, f'{map_id}@{revision:05d}')
        if record is None:
            raise KeyError(f'{map_id}@{revision}')
        return MapRevision.from_dict(record.payload)

    def history(self, map_id: str) -> list[MapRevision]:
        items = [MapRevision.from_dict(r.payload) for r in self._repo.scan(NAMESPACE)
                 if r.key.startswith(f'{map_id}@')]
        return sorted(items, key=lambda m: m.revision)

    def verify(self, revision: MapRevision) -> bool:
        staged = {'map_id': revision.map_id, 'revision': revision.revision, 'parent': revision.parent,
                  'author': revision.author, 'note': revision.note, 'cells': revision.cells,
                  'created_at': revision.created_at}
        return revision.digest == _digest(staged)

    def diff(self, left: MapRevision, right: MapRevision) -> dict:
        keys = set(left.cells) | set(right.cells)
        changed, added, removed = {}, {}, {}
        for key in keys:
            a, b = left.cells.get(key), right.cells.get(key)
            if a == b:
                continue
            if a is None:
                added[key] = b
            elif b is None:
                removed[key] = a
            else:
                changed[key] = {'from': a, 'to': b}
        return {'from': left.revision, 'to': right.revision, 'changed': changed,
                'added': added, 'removed': removed}

    # -- merge -----------------------------------------------------------------
    @staticmethod
    def three_way_merge(base: MapRevision, ours: MapRevision, theirs: MapRevision) -> MergeResult:
        keys = set(base.cells) | set(ours.cells) | set(theirs.cells)
        merged: dict[str, int] = {}
        conflicts: list[str] = []
        from_ours: list[str] = []
        from_theirs: list[str] = []
        for key in sorted(keys, key=lambda k: tuple(int(p) for p in k.split(','))):
            b, o, t = base.cells.get(key), ours.cells.get(key), theirs.cells.get(key)
            if o == t:
                if o is not None:
                    merged[key] = o
                continue
            if o == b:
                if t is not None:
                    merged[key] = t
                from_theirs.append(key)
                continue
            if t == b:
                if o is not None:
                    merged[key] = o
                from_ours.append(key)
                continue
            conflicts.append(key)
            merged[key] = o  # deterministic provisional value; caller must resolve
        return MergeResult(merged, tuple(conflicts), tuple(from_ours), tuple(from_theirs))

    def merge_and_save(self, map_id: str, ours_revision: int, theirs_revision: int, *,
                       actor: str = 'merger') -> tuple[MapRevision, MergeResult]:
        ours = self.get(map_id, ours_revision)
        theirs = self.get(map_id, theirs_revision)
        base_revision = ours.parent if ours.parent is not None else theirs.parent
        if base_revision is None:
            raise MapFormatError('cannot merge revisions without a common ancestor')
        base = self.get(map_id, base_revision)
        result = self.three_way_merge(base, ours, theirs)
        if result.conflicts:
            return ours, result
        saved = self.save(map_id, result.cells, actor=actor,
                          note=f'merge r{ours.revision}+r{theirs.revision}')
        return saved, result

    # -- interchange format ----------------------------------------------------
    @staticmethod
    def encode(payload: dict) -> str:
        body = {'format': FORMAT_VERSION, 'width': int(payload['width']), 'height': int(payload['height']),
                'resolution': float(payload.get('resolution', 1.0)),
                'origin': [float(payload.get('origin', [0.0, 0.0])[0]),
                           float(payload.get('origin', [0.0, 0.0])[1])],
                'cells': _normalise_cells(payload['cells'])}
        body['digest'] = _digest(body)
        return json.dumps(body, sort_keys=True, separators=(',', ':'))

    @staticmethod
    def decode(text: str) -> dict:
        payload = json.loads(text)
        version = int(payload.get('format', 0))
        if version != FORMAT_VERSION:
            raise MapFormatError(f'unsupported map format version {version}')
        body = {k: v for k, v in payload.items() if k != 'digest'}
        if _digest(body) != payload['digest']:
            raise MapFormatError('digest mismatch')
        return body

    @staticmethod
    def migrate(text: str) -> str:
        """Migrate a legacy v1 payload (row-major JSON grid) to the v2 format."""
        payload = json.loads(text)
        if 'format' in payload:
            return json.dumps(payload, sort_keys=True, separators=(',', ':'))
        width = int(payload['width'])
        height = int(payload['height'])
        flat = list(payload['cells'])
        if len(flat) != width * height:
            raise MapFormatError('legacy cell count mismatch')
        cells = {f'{x},{y}': int(flat[y * width + x]) for y in range(height) for x in range(width)}
        return MapRepository.encode({'width': width, 'height': height, 'cells': cells})
