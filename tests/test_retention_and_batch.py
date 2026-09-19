"""History retention (convergence) and atomic batch mutation tests.

Covers the two operational guarantees added on top of the versioned repository:

* history rows grow under a configurable retention policy instead of forever,
  while the audit hash chain and ordered event log stay protected;
* groups of interrelated writes (config rows, map revisions plus their audit
  entries) commit as one SQLite transaction - any failure rolls the whole group
  back, so callers never observe half a batch.
"""
import pytest

from aegisrover.mapping.revisions import MapRepository
from aegisrover.service.platform import PlatformService, ServiceError
from aegisrover.storage.audit import AuditLog
from aegisrover.storage.repository import (
    DeleteOp,
    PutOp,
    Repository,
    RetentionPolicy,
    VersionConflict,
)


class FakeClock:
    def __init__(self, now=1000.0):
        self.now = now

    def __call__(self):
        self.now += 0.5
        return self.now


@pytest.fixture()
def repo():
    r = Repository(':memory:', clock=FakeClock())
    yield r
    r.close()


# -- atomic batches -----------------------------------------------------------
def test_batch_mutate_is_all_or_nothing(repo):
    for i in range(5):
        repo.put('settings', f'k{i}', {'n': 0})
    ops = [
        PutOp('settings', 'k0', {'n': 1}, expected=1),
        PutOp('settings', 'k1', {'n': 1}, expected=1),
        PutOp('settings', 'k2', {'n': 1}, expected=99),  # stale -> fails
        PutOp('settings', 'k3', {'n': 1}, expected=1),
    ]
    with pytest.raises(VersionConflict):
        repo.batch_mutate(ops)
    # None of the earlier, already-executed writes survived.
    for i in range(5):
        record = repo.get('settings', f'k{i}')
        assert record.payload == {'n': 0}
        assert record.version == 1


def test_batch_mutate_commits_mixed_puts_and_deletes(repo):
    for i in range(3):
        repo.put('settings', f'k{i}', {'n': 0})
    results = repo.batch_mutate([
        PutOp('settings', 'k0', {'n': 9}, expected=1),
        PutOp('settings', 'k1', {'n': 9}, expected=1),
        DeleteOp('settings', 'k2', expected=1),
    ])
    assert [r.version for r in results] == [2, 2, 2]
    assert repo.get('settings', 'k0').payload == {'n': 9}
    assert repo.get('settings', 'k2', include_deleted=True).deleted


def test_explicit_transaction_rolls_back_on_arbitrary_exception(repo):
    repo.put('settings', 'k0', {'n': 0})
    with pytest.raises(RuntimeError):
        with repo.transaction():
            repo.put('settings', 'k0', {'n': 5}, expected=1)
            repo.put('settings', 'brand_new', {'n': 5})
            raise RuntimeError('boom')
    assert repo.get('settings', 'k0').payload == {'n': 0}
    assert repo.maybe_get('settings', 'brand_new') is None


def test_nested_savepoint_rolls_back_only_the_inner_block(repo):
    with repo.transaction():
        repo.put('settings', 'outer1', {'n': 1})
        with pytest.raises(ValueError):
            with repo.transaction():
                repo.put('settings', 'inner', {'n': 1})
                raise ValueError('inner failure')
        repo.put('settings', 'outer2', {'n': 1})
    assert repo.get('settings', 'outer1').payload == {'n': 1}
    assert repo.get('settings', 'outer2').payload == {'n': 1}
    assert repo.maybe_get('settings', 'inner') is None


def test_batch_failure_does_not_poison_following_transactions(repo):
    with pytest.raises(VersionConflict):
        repo.batch_mutate([PutOp('settings', 'k0', {'n': 1}, expected=5)])
    # The connection is still usable and commits normally afterwards.
    assert repo.put('settings', 'k0', {'n': 1}).version == 1


# -- retention ----------------------------------------------------------------
def _versions(repo, ns, key):
    return [h.version for h in repo.history(ns, key)]


def test_prune_keeps_live_and_newest_versions(repo):
    repo.put('settings', 'grow', {'v': 1})
    for v in range(2, 12):
        repo.put('settings', 'grow', {'v': v}, expected=v - 1)
    assert _versions(repo, 'settings', 'grow') == list(range(1, 12))
    report = repo.prune_history(RetentionPolicy(keep_last=3))
    assert _versions(repo, 'settings', 'grow') == [9, 10, 11]
    assert report.deleted_history_rows == 8
    assert report.affected_records == 1
    # Live row untouched, optimistic concurrency still enforced.
    live = repo.get('settings', 'grow')
    assert live.version == 11 and live.payload == {'v': 11}
    repo.put('settings', 'grow', {'v': 12}, expected=11)
    with pytest.raises(VersionConflict):
        repo.put('settings', 'grow', {'v': 99}, expected=9)


def test_prune_keep_last_zero_retains_only_live(repo):
    repo.put('ns', 'k', 1)
    for v in range(2, 6):
        repo.put('ns', 'k', v, expected=v - 1)
    report = repo.prune_history(RetentionPolicy(keep_last=0))
    assert _versions(repo, 'ns', 'k') == [5]
    assert report.deleted_history_rows == 4


def test_prune_respects_age_window(repo):
    clock = FakeClock()
    r = Repository(':memory:', clock=clock)
    try:
        r.put('settings', 'aged', {'v': 1})
        for v in range(2, 8):
            r.put('settings', 'aged', {'v': v}, expected=v - 1)
        report = r.prune_history(RetentionPolicy(keep_last=0, max_age_seconds=2.0))
        versions = _versions(r, 'settings', 'aged')
        assert versions[-1] == 7 and len(versions) < 7
        assert report.deleted_history_rows > 0
        # A generous window prunes nothing.
        assert r.prune_history(RetentionPolicy(keep_last=1000)).deleted_history_rows == 0
    finally:
        r.close()


def test_prune_skips_protected_namespaces_by_default(repo):
    audit = AuditLog(repo, clock=FakeClock())
    for subject in ('a', 'b', 'c'):
        audit.append('operator', 'note', subject)
    # A legitimate rewrite gives the audit row a second history version.
    record = repo.get('audit', '000000000001')
    repo.put('audit', '000000000001', record.payload, expected=1)
    assert _versions(repo, 'audit', '000000000001') == [1, 2]
    assert repo.prune_history(RetentionPolicy(keep_last=0)).deleted_history_rows == 0
    assert _versions(repo, 'audit', '000000000001') == [1, 2]
    # Explicit opt-in prunes the old version but keeps the live head.
    report = repo.prune_history(
        RetentionPolicy(keep_last=1, include_namespaces=('audit',)))
    assert report.deleted_history_rows == 1
    assert _versions(repo, 'audit', '000000000001') == [2]


def test_prune_namespace_allowlist(repo):
    repo.put('nsA', 'x', 1)
    repo.put('nsA', 'x', 2, expected=1)
    repo.put('nsA', 'x', 3, expected=2)
    repo.put('nsB', 'x', 1)
    repo.put('nsB', 'x', 2, expected=1)
    repo.prune_history(RetentionPolicy(keep_last=1, namespaces=('nsA',)))
    assert _versions(repo, 'nsA', 'x') == [3]
    assert _versions(repo, 'nsB', 'x') == [1, 2]


def test_purge_keys_removes_record_and_history(repo):
    repo.put('nsB', 'x', 1)
    repo.put('nsB', 'x', 2, expected=1)
    assert repo.purge_keys('nsB', ['x']) == 1
    assert repo.maybe_get('nsB', 'x') is None
    assert repo.history('nsB', 'x') == ()
    assert repo.purge_keys('nsB', []) == 0


# -- map batch + map revision retention ---------------------------------------
@pytest.fixture()
def maps(repo):
    return MapRepository(repo, AuditLog(repo, clock=FakeClock()), clock=FakeClock())

def test_save_many_commits_interrelated_maps(maps):
    saved = maps.save_many([
        ('yard', {'1,1': 1}),
        ('dock', {'2,2': 2}),
        ('shed', {'3,3': 3, (4, 4): 4}),
    ], actor='planner', note='initial layout')
    assert [(m.map_id, m.revision) for m in saved] == [
        ('yard', 1), ('dock', 1), ('shed', 1)]
    assert maps.latest('shed').cells == {'3,3': 3, '4,4': 4}
    assert AuditLog(maps._repo).verify() == ()


def test_save_many_stale_revision_rolls_back_whole_group(maps):
    maps.save_many([('yard', {'1,1': 1}), ('dock', {'2,2': 2}), ('shed', {'3,3': 3})])
    maps.save('yard', {'1,1': 2})
    with pytest.raises(VersionConflict):
        maps.save_many([
            ('yard', {'1,1': 3}, 2),
            ('dock', {'2,2': 9}, 1),
            ('shed', {'3,3': 8}, 1),
            ('yard', {'1,1': 4}, 1),  # stale: yard already at 2
        ])
    assert maps.latest('yard').cells == {'1,1': 2}
    assert maps.latest('dock').cells == {'2,2': 2}
    assert maps.latest('shed').cells == {'3,3': 3}


def test_save_many_payload_error_rolls_back(maps):
    maps.save('dock', {'2,2': 2})
    with pytest.raises(Exception):
        maps.save_many([('dock', {'9,9': 9}), ('nope', {'bad': 'not-an-int'})])
    assert maps.latest('dock').cells == {'2,2': 2}
    assert maps.latest('nope') is None


def test_prune_map_revisions_keeps_newest_and_live_head(maps):
    maps.save('yard', {'0,0': 0})
    for i in range(11):
        maps.save('yard', {f'{i},0': i})
    assert [m.revision for m in maps.history('yard')] == list(range(1, 13))
    report = maps.prune_revisions(keep_last=3)
    remaining = [m.revision for m in maps.history('yard')]
    assert remaining == [10, 11, 12]
    assert report.removed_revisions == 9
    assert 'yard' in report.affected_maps
    # Live map readable; pruned revision is not.
    assert maps.latest('yard').revision == 12
    assert maps.get('yard').cells == {'10,0': 10}
    with pytest.raises(KeyError):
        maps.get('yard', 5)
    # Idempotent: pruning again removes nothing.
    assert maps.prune_revisions(keep_last=3).removed_revisions == 0


def test_prune_map_revisions_keeps_head_even_with_keep_last_zero(maps):
    maps.save('dock', {'0,0': 0})
    for i in range(3):
        maps.save('dock', {f'{i},5': i})
    report = maps.prune_revisions(keep_last=0)
    assert report.removed_revisions == 3
    assert [m.revision for m in maps.history('dock')] == [4]
    assert maps.latest('dock').revision == 4
    assert maps.prune_revisions(keep_last=0).removed_revisions == 0


def test_prune_map_revisions_age_window(maps):
    maps.save('shed', {'0,0': 0})
    for i in range(5):
        maps.save('shed', {f'{i},1': i})
    old = [m.revision for m in maps.history('shed')]
    # keep_last=1 so the age window decides among everything but the newest two.
    report = maps.prune_revisions(keep_last=1, max_age_seconds=0.001)
    after = [m.revision for m in maps.history('shed')]
    assert report.removed_revisions == len(old) - len(after)
    assert after[-1] == old[-1]
    assert 1 <= len(after) < len(old)
    # A generous age window removes nothing.
    assert maps.prune_revisions(keep_last=50, max_age_seconds=10_000.0).removed_revisions == 0


def test_rollback_after_prune_uses_retained_revision(maps):
    maps.save('yard', {'0,0': 0})
    for i in range(1, 6):
        maps.save('yard', {f'{i},0': i})
    maps.prune_revisions(keep_last=2)
    restored = maps.rollback('yard', 5)
    assert restored.cells == {'4,0': 4}
    assert restored.parent == 6


# -- service facade ------------------------------------------------------------
def test_service_batch_and_retention_endpoints(repo):
    service = PlatformService(repo, clock=FakeClock())
    result = service.save_maps([('a', {'1,1': 1}), ('b', {'2,2': 2})], actor='op')
    assert [s['revision'] for s in result['saved']] == [1, 1]
    with pytest.raises(ServiceError) as exc:
        service.save_maps([('a', {'1,1': 9}, 1), ('b', {'2,2': 9}, 99)])
    assert exc.value.code == 'revision_conflict' and exc.value.status == 409
    # Whole group rejected.
    assert service.get_map('a')['revision'] == 1
    assert service.get_map('b')['revision'] == 1

    for i in range(10):
        service.save_map('a', {f'{i},0': i})
    report = service.prune_maps(keep_last=2)
    assert report['removed_revisions'] == 9
    assert service.get_map('a')['revision'] == 11
    assert service.health()['checks']['audit_chain']['ok'] is True
