"""Storage, audit and mission-domain acceptance tests for the v1.1 platform."""
import pytest

from aegisrover.mission.allocation import Conflict, Deadlock, Reservation, ReservationBook
from aegisrover.mission.lifecycle import InvalidTransition, MissionService
from aegisrover.service.platform import PlatformService, ServiceError
from aegisrover.storage.audit import AuditLog
from aegisrover.storage.repository import (
    NotFound, Repository, RetentionPolicy, TransactionError, VersionConflict,
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


def test_repository_versions_and_history(repo):
    first = repo.put('maps', 'yard', {'rev': 1})
    assert first.version == 1
    second = repo.put('maps', 'yard', {'rev': 2}, expected=1)
    assert (second.version, second.etag) != (first.version, first.etag)
    with pytest.raises(VersionConflict):
        repo.put('maps', 'yard', {'rev': 3}, expected=1)
    assert [r.version for r in repo.history('maps', 'yard')] == [1, 2]
    assert repo.get('maps', 'yard').payload == {'rev': 2}


def test_repository_paging_and_tombstones(repo):
    for i in range(7):
        repo.put('missions', f'm{i:02d}', {'i': i})
    page = repo.page('missions', limit=3)
    assert [r.key for r in page.items] == ['m00', 'm01', 'm02']
    assert page.has_more and page.cursor
    page2 = repo.page('missions', limit=3, cursor=page.cursor)
    assert [r.key for r in page2.items] == ['m03', 'm04', 'm05']
    repo.delete('missions', 'm04', expected=1)
    with pytest.raises(NotFound):
        repo.get('missions', 'm04')
    assert repo.get('missions', 'm04', include_deleted=True).deleted
    assert repo.count('missions') == 6


def test_transaction_commits_related_writes_atomically(repo):
    repo.put('missions', 'm0', {'i': 0})
    with repo.transaction():
        repo.put('configs', 'robot-1', {'speed': 1.0})
        repo.put('configs', 'robot-2', {'speed': 2.0})
        repo.delete('missions', 'm0')
    assert repo.get('configs', 'robot-1').payload == {'speed': 1.0}
    assert repo.get('configs', 'robot-2').version == 1
    with pytest.raises(NotFound):
        repo.get('missions', 'm0')
    assert [r.version for r in repo.history('configs', 'robot-1')] == [1]


def test_transaction_rolls_back_everything_on_error(repo):
    repo.put('configs', 'robot-1', {'speed': 1.0})
    with pytest.raises(VersionConflict):
        with repo.transaction():
            repo.put('configs', 'robot-1', {'speed': 2.0}, expected=1)
            repo.put('configs', 'robot-2', {'speed': 3.0})
            repo.put('configs', 'robot-1', {'speed': 9.0}, expected=99)
    assert repo.get('configs', 'robot-1').version == 1
    assert repo.get('configs', 'robot-1').payload == {'speed': 1.0}
    with pytest.raises(NotFound):
        repo.get('configs', 'robot-2')
    assert [r.version for r in repo.history('configs', 'robot-1')] == [1]


def test_transaction_statement_failure_does_not_doom_batch(repo):
    # A conflict the caller catches and handles must not sink the whole batch.
    with repo.transaction():
        repo.put('configs', 'a', {'v': 1})
        with pytest.raises(VersionConflict):
            repo.put('configs', 'a', {'v': 2}, expected=99)
        repo.put('configs', 'b', {'v': 3})
    assert repo.get('configs', 'a').payload == {'v': 1}
    assert repo.get('configs', 'b').payload == {'v': 3}


def test_transaction_nested_failure_dooms_outer_commit(repo):
    with pytest.raises(TransactionError):
        with repo.transaction():
            repo.put('configs', 'a', {'v': 1})
            with pytest.raises(RuntimeError):
                with repo.transaction():
                    repo.put('configs', 'b', {'v': 2})
                    raise RuntimeError('boom')
    with pytest.raises(NotFound):
        repo.get('configs', 'a')
    with pytest.raises(NotFound):
        repo.get('configs', 'b')


def test_prune_history_keeps_newest_versions(repo):
    for i in range(5):
        repo.put('configs', 'robot-1', {'rev': i})
    removed = repo.prune_history(RetentionPolicy(keep_versions=2), ns='configs')
    assert removed == {'configs': 3}
    assert [r.version for r in repo.history('configs', 'robot-1')] == [4, 5]
    assert repo.get('configs', 'robot-1').payload == {'rev': 4}


def test_prune_history_respects_age_cutoff(repo):
    # FakeClock advances 0.5 per call and each put calls it twice, so the six
    # history rows are recorded at 1001.0 .. 1006.0.
    for i in range(6):
        repo.put('configs', 'robot-1', {'rev': i})
    removed = repo.prune_history(RetentionPolicy(keep_versions=1, keep_seconds=2.0),
                                 ns='configs', now=1006.0)
    assert removed == {'configs': 3}
    assert [r.version for r in repo.history('configs', 'robot-1')] == [4, 5, 6]


def test_retention_policy_requires_keeping_current():
    with pytest.raises(ValueError):
        RetentionPolicy(keep_versions=0)
    with pytest.raises(ValueError):
        RetentionPolicy(keep_seconds=0)


def test_audit_chain_detects_tampering(repo):
    audit = AuditLog(repo, clock=FakeClock())
    audit.append('operator', 'mission.create', 'm1', {'priority': 1})
    audit.append('operator', 'mission.queue', 'm1')
    audit.append('robot-7', 'mission.start', 'm1')
    assert audit.verify() == ()
    assert [e.seq for e in audit.entries()] == [1, 2, 3]
    audit.tamper(2, action='mission.cancel')
    breaks = audit.verify()
    assert breaks and breaks[0].seq == 2
    assert any(b.reason == 'entry digest mismatch' for b in breaks)


def test_audit_chain_detects_reordering(repo):
    audit = AuditLog(repo, clock=FakeClock())
    for name in ('a', 'b', 'c'):
        audit.append('operator', 'note', name)
    record = repo.get('audit', '000000000002')
    payload = dict(record.payload)
    payload['previous_digest'] = '0' * 64
    repo.put('audit', '000000000002', payload)
    reasons = {b.reason for b in audit.verify()}
    assert 'previous digest mismatch' in reasons


def test_mission_lifecycle_is_guarded_and_idempotent(repo):
    service = MissionService(repo, clock=FakeClock())
    mission = service.create('m1', [(0, 0), (5, 0)], priority=2, actor='planner')
    assert mission.state == 'draft'
    assert service.command('m1', 'queue')['applied'] is True
    assert service.command('m1', 'queue')['applied'] is False
    with pytest.raises(InvalidTransition):
        service.command('m1', 'complete')
    service.command('m1', 'assign', assignee='robot-1')
    service.command('m1', 'start')
    service.command('m1', 'complete')
    with pytest.raises(InvalidTransition):
        service.command('m1', 'start')
    assert service.get('m1').state == 'completed'


def test_mission_conflicting_commands_are_serialised(repo):
    service = MissionService(repo, clock=FakeClock())
    service.create('m1', [(0, 0)], actor='planner')
    service.command('m1', 'queue')
    with pytest.raises(VersionConflict):
        service.command('m1', 'assign', assignee='robot-1', expected_revision=99)
    history = service.get('m1').history
    assert [h['command'] for h in history] == ['create', 'queue']


def test_mission_dispatch_respects_priority_and_capabilities(repo):
    service = MissionService(repo, clock=FakeClock())
    service.create('low', [(0, 0)], priority=0, required_capabilities=['lidar'])
    service.create('high', [(1, 1)], priority=9, required_capabilities=['lidar', 'arm'])
    for name in ('low', 'high'):
        service.command(name, 'queue')
    # A robot that cannot satisfy the highest-priority mission must not silently
    # take it; it gets the best mission it is actually able to run.
    first = service.claim_next('robot-1', ['lidar'])
    assert first.mission_id == 'low' and first.state == 'assigned'
    second = service.claim_next('robot-2', ['lidar', 'arm'])
    assert second.mission_id == 'high'
    assert service.claim_next('robot-3', ['lidar', 'arm']) is None


def test_service_transaction_applies_batch_all_or_nothing(repo):
    service = PlatformService(repo, clock=FakeClock())
    with service.transaction():
        service.create_mission('m1', [(0, 0)], actor='ops')
        service.save_map('yard', {'0,0': 1}, actor='ops')
    assert service.get_mission('m1')['state'] == 'draft'
    assert service.get_map('yard')['revision'] == 1

    with pytest.raises(ServiceError):
        with service.transaction():
            service.save_map('depot', {'0,0': 1})
            service.create_mission('m2', [(2, 2)])
            service.save_map('depot', {'0,0': 2}, if_match=99)
    with pytest.raises(ServiceError):
        service.get_map('depot')
    with pytest.raises(ServiceError):
        service.get_mission('m2')


def test_service_prune_history_is_audited(repo):
    service = PlatformService(repo, clock=FakeClock())
    service.create_mission('m1', [(0, 0)])
    service.command_mission('m1', 'queue')
    service.command_mission('m1', 'assign', assignee='robot-1')
    service.command_mission('m1', 'start')
    report = service.prune_history(RetentionPolicy(keep_versions=1), namespaces=['missions'])
    assert report['removed'] == {'missions': 3}
    assert [r.version for r in repo.history('missions', 'm1')] == [4]
    assert service.get_mission('m1')['state'] == 'running'
    assert 'maintenance.prune_history' in {e.action for e in service.audit.entries()}
    assert service.audit.verify() == ()


def test_reservations_are_half_open():
    book = ReservationBook()
    book.reserve(Reservation('door-a', 'robot-1', 10.0, 12.0))
    book.reserve(Reservation('door-a', 'robot-2', 12.0, 14.0))
    with pytest.raises(Conflict):
        book.reserve(Reservation('door-a', 'robot-3', 11.5, 12.5))
    assert book.utilization('door-a', 10.0, 14.0) == pytest.approx(1.0)
    assert book.free_windows('door-a', 9.0, 15.0) == [(9.0, 10.0), (14.0, 15.0)]


def test_reservation_preemption_only_when_lower_priority_has_not_started():
    book = ReservationBook()
    book.reserve(Reservation('dock', 'robot-1', 10.0, 12.0, priority=1))
    book.reserve(Reservation('dock', 'robot-2', 10.5, 12.5, priority=5), preempt=True, now=9.0)
    assert {r.holder for r in book.usage('dock')} == {'robot-2'}
    with pytest.raises(Conflict):
        # robot-2 already started at 10.5, so it may not be evicted implicitly.
        book.reserve(Reservation('dock', 'robot-3', 11.0, 13.0, priority=9), preempt=True, now=10.6)
    assert {r.holder for r in book.usage('dock')} == {'robot-2'}
    book.release('robot-2')
    book.reserve(Reservation('dock', 'robot-3', 11.0, 13.0, priority=9))
    assert {r.holder for r in book.usage('dock')} == {'robot-3'}


def test_wait_for_graph_reports_circular_wait():
    book = ReservationBook()
    book.reserve(Reservation('door-a', 'robot-1', 0.0, 5.0))
    book.reserve(Reservation('door-b', 'robot-2', 0.0, 5.0))
    book.wait_for('robot-1', 'robot-2')
    book.wait_for('robot-2', 'robot-1')
    cycle = book.detect_deadlock()
    assert cycle is not None and cycle[0] == cycle[-1] and set(cycle) == {'robot-1', 'robot-2'}
    with pytest.raises(Deadlock):
        book.require_no_deadlock()
    book.clear_wait('robot-2')
    assert book.detect_deadlock() is None
