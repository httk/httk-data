"""P3 permanentization coverage: roles, lease/counters, and residue cleanup."""

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, ClassVar

import pytest
import sqlalchemy
from httk.core.register import register_entry_family, register_entry_record
from httk.core.storage import IdentitySkip, StorageInfo
from sqlalchemy import event

from httk.store.backend.sql import Backend, EntryMetadataConflictError, SqlStore, StorageLayoutUpgradeRequiredError
from httk.store.backend.sql.mapping import entry_dispatch_table_name
from httk.store.backend.sql.store import _DegradedWriteCrash


@dataclass(frozen=True)
class RoleLeaf:
    value: str


@dataclass(frozen=True)
class RoleRoot:
    leaf: RoleLeaf


@dataclass(frozen=True)
class ValueLeaf:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(dedup="by_value")
    value: str


@dataclass(frozen=True)
class ValueRoot:
    leaf: ValueLeaf


@dataclass(frozen=True)
class CrashLeaf:
    value: str
    atoms: list[str]


@dataclass(frozen=True)
class CrashRoot:
    leaf: CrashLeaf
    values: list[str]


@dataclass(frozen=True)
class CrashSolo:
    values: list[str]


@dataclass(frozen=True)
class StatementRecord:
    value: str


@dataclass(frozen=True)
class MetadataLeaf:
    value: str
    note: Annotated[str, IdentitySkip()]


@dataclass(frozen=True)
class MetadataRoot:
    leaf: MetadataLeaf


class CrashDispatchFamily:
    """A private two-backing family used to expose the dispatch durable step."""


@dataclass(frozen=True)
class CrashDispatchA:
    value: str


@dataclass(frozen=True)
class CrashDispatchB:
    value: int


register_entry_family(name="test-permanentization-crash-dispatch", family=f"{__name__}:CrashDispatchFamily")
register_entry_record(
    name="test-permanentization-crash-dispatch-a",
    family="test-permanentization-crash-dispatch",
    record=f"{__name__}:CrashDispatchA",
)
register_entry_record(
    name="test-permanentization-crash-dispatch-b",
    family="test-permanentization-crash-dispatch",
    record=f"{__name__}:CrashDispatchB",
)


_CRASH_POINTS = (
    "dirty-upsert:crash_leaf",
    "counter-table-create:crash_leaf",
    "counter-init:crash_leaf",
    "counter-allocation:crash_leaf",
    "dirty-upsert:crash_leaf_atoms",
    "child-row-write:crash_leaf_atoms",
    "parent-row-write:crash_leaf",
    "dirty-upsert:crash_root",
    "counter-table-create:crash_root",
    "counter-init:crash_root",
    "counter-allocation:crash_root",
    "dirty-upsert:crash_root_values",
    "child-row-write:crash_root_values",
    "parent-row-write:crash_root",
    "dirty-delete:crash_leaf",
    "dirty-delete:crash_leaf_atoms",
    "dirty-delete:crash_root",
    "dirty-delete:crash_root_values",
)

_ROOT_COUNTER_POINTS = (
    "dirty-upsert:crash_solo",
    "counter-table-create:crash_solo",
    "counter-init:crash_solo",
    "counter-allocation:crash_solo",
    "dirty-upsert:crash_solo_values",
    "child-row-write:crash_solo_values",
    "parent-row-write:crash_solo",
    "dirty-delete:crash_solo",
    "dirty-delete:crash_solo_values",
)

_CRASH_SMOKE_POINTS = frozenset(
    {
        "dirty-upsert:crash_leaf",
        "parent-row-write:crash_root",
        "counter-allocation:crash_solo",
    }
)


def _tiered_crash_points(points: tuple[str, ...]) -> tuple[object, ...]:
    """Keep three representative crash windows in CI and reserve the whole battery for the full tier."""
    return tuple(
        pytest.param(point, marks=pytest.mark.extended) if point not in _CRASH_SMOKE_POINTS else point
        for point in points
    )


def _role(database: Backend, table: str, sid: int) -> int:
    with database.engine.connect() as connection:
        return int(
            connection.execute(
                sqlalchemy.text(f'SELECT _httk_role FROM "{table}" WHERE sid = :sid'), {"sid": sid}
            ).scalar_one()
        )


def test_top_level_dedup_promotes_dependency_for_content_and_by_value() -> None:
    with Backend.sqlite() as database:
        store = SqlStore(database, entry_records={})
        leaf = RoleLeaf("content")
        store.save(RoleRoot(leaf))
        leaf_sid = store.sid_of(leaf)
        assert leaf_sid is not None and _role(database, "role_leaf", leaf_sid) == 0
        assert store.save(leaf) == leaf_sid
        assert _role(database, "role_leaf", leaf_sid) == 1

        value = ValueLeaf("value")
        store.save(ValueRoot(value))
        with database.engine.connect() as connection:
            value_sid = int(connection.execute(sqlalchemy.text("SELECT sid FROM value_leaf")).scalar_one())
        assert _role(database, "value_leaf", value_sid) == 0
        assert store.save(value) == value_sid
        assert _role(database, "value_leaf", value_sid) == 1


def test_degraded_lease_counter_and_fsck(tmp_path: Path) -> None:
    path = tmp_path / "degraded.sqlite"
    first_database = Backend.sqlite(path, degraded=True)
    first = SqlStore(first_database, entry_records={})
    sid = first.save(RoleRoot(RoleLeaf("one")))
    assert sid == 1
    second_database = Backend.sqlite(path, degraded=True)
    second = SqlStore(second_database)
    with pytest.raises(RuntimeError, match="lease is held"):
        second.save(RoleLeaf("blocked"))
    second.steal_lease()
    assert second.save(RoleLeaf("after-steal")) == 2
    report = second.fsck(known_types=(RoleRoot,))
    assert not report.violations
    first_database.dispose()
    second_database.dispose()


def test_degraded_profile_requires_degraded_database(tmp_path: Path) -> None:
    path = tmp_path / "profile.sqlite"
    database = Backend.sqlite(path, degraded=True)
    SqlStore(database, entry_records={})
    database.dispose()
    with pytest.raises(StorageLayoutUpgradeRequiredError):
        SqlStore(Backend.sqlite(path))


def test_write_profile_validates_live_sqlite_connection_mode() -> None:
    autocommit_engine = sqlalchemy.create_engine("sqlite://", isolation_level="AUTOCOMMIT")
    try:
        with pytest.raises(StorageLayoutUpgradeRequiredError):
            SqlStore(Backend(autocommit_engine), entry_records={})
    finally:
        autocommit_engine.dispose()
    transactional_engine = sqlalchemy.create_engine("sqlite://")
    try:
        with pytest.raises(StorageLayoutUpgradeRequiredError):
            SqlStore(Backend(transactional_engine, degraded=True), entry_records={})
    finally:
        transactional_engine.dispose()


def test_sqlite_rejects_a_bulk_fenced_stamp() -> None:
    with Backend.sqlite() as database:
        SqlStore(database, entry_records={})
        with database.engine.begin() as connection:
            connection.execute(
                sqlalchemy.text("INSERT INTO _httk_store_metadata (key, value) VALUES ('write_profile', 'bulk-fenced')")
            )
        with pytest.raises(StorageLayoutUpgradeRequiredError):
            SqlStore(database)


def test_duckdb_rejects_a_bulk_fenced_stamp() -> None:
    pytest.importorskip("duckdb_engine")
    with Backend.duckdb() as database:
        SqlStore(database, entry_records={})
        with database.engine.begin() as connection:
            connection.execute(
                sqlalchemy.text("INSERT INTO _httk_store_metadata (key, value) VALUES ('write_profile', 'bulk-fenced')")
            )
        with pytest.raises(StorageLayoutUpgradeRequiredError):
            SqlStore(database)


def test_disposed_database_refuses_late_lifecycle_registration_and_mutation(tmp_path: Path) -> None:
    path = tmp_path / "disposed.sqlite"
    database = Backend.sqlite(path, degraded=True)
    store = SqlStore(database, entry_records={})
    store.save(RoleLeaf("one"))
    database.dispose()
    with pytest.raises(RuntimeError, match="disposed Backend"):
        database.add_dispose_callback(lambda: None)
    with pytest.raises(RuntimeError, match="disposed"):
        store.save(RoleLeaf("two"))
    other_database = Backend.sqlite(path, degraded=True)
    other = SqlStore(other_database)
    assert other.save(RoleLeaf("three")) == 2
    other_database.dispose()


def test_pre_registration_dispose_cannot_strand_a_degraded_lease(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "pre-registration.sqlite"
    database = Backend.sqlite(path, degraded=True)
    original = database.add_dispose_callback

    def dispose_after_registration(callback, *, generation=None):
        registered = original(callback, generation=generation)
        database.dispose()
        return registered

    monkeypatch.setattr(database, "add_dispose_callback", dispose_after_registration)
    with pytest.raises(RuntimeError, match="disposed"):
        SqlStore(database, entry_records={})
    with Backend.sqlite(path, degraded=True) as recovered_database:
        recovered = SqlStore(recovered_database, entry_records={})
        assert recovered.save(RoleLeaf("recovered")) == 1
        with recovered_database.engine.connect() as connection:
            assert (
                connection.execute(
                    sqlalchemy.text("SELECT count(*) FROM _httk_store_metadata WHERE key = 'lease'")
                ).scalar_one()
                == 1
            )


def test_dispose_waits_for_the_degraded_mutation_lock(tmp_path: Path) -> None:
    path = tmp_path / "dispose-race.sqlite"
    database = Backend.sqlite(path, degraded=True)
    store = SqlStore(database, entry_records={})
    entered = threading.Event()
    allow_write = threading.Event()

    def pause(point: str) -> bool:
        if point == "dirty-upsert:role_leaf":
            entered.set()
            assert allow_write.wait(timeout=5)
        return False

    store._degraded_fault_hook = pause
    writer = threading.Thread(target=lambda: store.save(RoleLeaf("blocked-release")))
    writer.start()
    assert entered.wait(timeout=5)
    disposer = threading.Thread(target=database.dispose)
    disposer.start()
    assert disposer.is_alive()
    with database.engine.connect() as connection:
        assert (
            connection.execute(
                sqlalchemy.text("SELECT count(*) FROM _httk_store_metadata WHERE key = 'lease'")
            ).scalar_one()
            == 1
        )
    allow_write.set()
    writer.join(timeout=5)
    disposer.join(timeout=5)
    assert not writer.is_alive() and not disposer.is_alive()


def _dirty_keys(database: Backend) -> set[str]:
    with database.engine.connect() as connection:
        return {
            str(key)
            for key in connection.execute(
                sqlalchemy.text("SELECT key FROM _httk_store_metadata WHERE key LIKE 'dirty:%'")
            ).scalars()
        }


def _counter_values(database: Backend) -> dict[str, int]:
    with database.engine.connect() as connection:
        if (
            connection.execute(
                sqlalchemy.text("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = '_httk_sid_counters'")
            ).first()
            is None
        ):
            return {}
        return {
            str(name): int(next_sid)
            for name, next_sid in connection.execute(
                sqlalchemy.text("SELECT table_name, next_sid FROM _httk_sid_counters")
            )
        }


def _assert_one_way_invariant(database: Backend) -> None:
    """Every visible root has its leaf and both child-element sequences."""
    with database.engine.connect() as connection:
        roots = connection.execute(sqlalchemy.text("SELECT sid, leaf_sid FROM crash_root")).all()
        for root_sid, leaf_sid in roots:
            assert connection.execute(
                sqlalchemy.text("SELECT 1 FROM crash_leaf WHERE sid = :sid"), {"sid": leaf_sid}
            ).first()
            assert (
                connection.execute(
                    sqlalchemy.text("SELECT count(*) FROM crash_root_values WHERE crash_root_sid = :sid"),
                    {"sid": root_sid},
                ).scalar_one()
                == 2
            )
            assert (
                connection.execute(
                    sqlalchemy.text("SELECT count(*) FROM crash_leaf_atoms WHERE crash_leaf_sid = :sid"),
                    {"sid": leaf_sid},
                ).scalar_one()
                == 2
            )


def _assert_no_crash_residue(database: Backend) -> None:
    """Physical post-fsck assertion, before any recovery write can hide residue."""
    with database.engine.connect() as connection:
        assert (
            connection.execute(
                sqlalchemy.text(
                    "SELECT count(*) FROM crash_leaf l "
                    "WHERE NOT EXISTS (SELECT 1 FROM crash_root r WHERE r.leaf_sid = l.sid)"
                )
            ).scalar_one()
            == 0
        )
        assert (
            connection.execute(
                sqlalchemy.text(
                    "SELECT count(*) FROM crash_leaf_atoms a "
                    "WHERE NOT EXISTS (SELECT 1 FROM crash_leaf l WHERE l.sid = a.crash_leaf_sid)"
                )
            ).scalar_one()
            == 0
        )
        assert (
            connection.execute(
                sqlalchemy.text(
                    "SELECT count(*) FROM crash_root_values v "
                    "WHERE NOT EXISTS (SELECT 1 FROM crash_root r WHERE r.sid = v.crash_root_sid)"
                )
            ).scalar_one()
            == 0
        )


@pytest.mark.parametrize("point", _tiered_crash_points(_CRASH_POINTS))
def test_degraded_crash_battery_recovers_every_ordering_step(tmp_path: Path, point: str) -> None:
    """Hard-stop after every permanentization write leaves only recoverable state."""
    path = tmp_path / f"crash-{point.replace(':', '-')}.sqlite"
    crashed = CrashRoot(CrashLeaf("crashed", ["Li", "O"]), ["one", "two"])
    trigger = CrashRoot(CrashLeaf("trigger", ["Na", "Cl"]), ["three", "four"])
    original_database = Backend.sqlite(path, degraded=True)
    original = SqlStore(original_database, entry_records={})
    original._degraded_fault_hook = lambda observed: observed == point
    with pytest.raises(_DegradedWriteCrash, match=point):
        original.save(crashed)
    _assert_one_way_invariant(original_database)
    counters_before = _counter_values(original_database)
    dirty_before = _dirty_keys(original_database)

    # A fresh owner observes exactly the crash residue, steals the stale lease,
    # and proves that its first write invokes targeted cleanup for every dirty
    # table left by the stopped operation.
    reopened_database = Backend.sqlite(path, degraded=True)
    reopened = SqlStore(reopened_database)
    reopened.steal_lease()
    report = reopened.fsck(known_types=(CrashRoot,))
    assert not report.violations
    _assert_no_crash_residue(reopened_database)
    sweeps: list[str] = []
    real_sweep = reopened._targeted_dirty_sweep

    def observing_sweep(connection, table):
        sweeps.append(table.name)
        return real_sweep(connection, table)

    reopened._targeted_dirty_sweep = observing_sweep
    reopened.save(trigger)
    assert set(sweeps) == {key.removeprefix("dirty:") for key in dirty_before}
    assert not _dirty_keys(reopened_database)
    _assert_one_way_invariant(reopened_database)
    counters_after = _counter_values(reopened_database)
    for table_name, next_sid in counters_before.items():
        assert counters_after[table_name] > next_sid

    with reopened_database.engine.connect() as connection:
        stored = [
            reopened.fetch(CrashRoot, int(sid))
            for sid in connection.execute(sqlalchemy.text("SELECT sid FROM crash_root ORDER BY sid")).scalars()
        ]
    expected = [trigger]
    if crashed in stored:
        expected.insert(0, crashed)
    with Backend.sqlite() as reference_database:
        reference = SqlStore(reference_database, entry_records={})
        for record in expected:
            reference.save(record)
        assert stored == [reference.fetch(CrashRoot, sid) for sid in range(1, len(expected) + 1)]
    original_database.dispose()
    reopened_database.dispose()


@pytest.mark.parametrize("point", _tiered_crash_points(_ROOT_COUNTER_POINTS))
def test_degraded_crash_battery_covers_root_counter_lifecycle(tmp_path: Path, point: str) -> None:
    """A root without dependencies exercises its own counter create/init path."""
    path = tmp_path / f"root-counter-{point.replace(':', '-')}.sqlite"
    database = Backend.sqlite(path, degraded=True)
    store = SqlStore(database, entry_records={})
    store._degraded_fault_hook = lambda observed: observed == point
    with pytest.raises(_DegradedWriteCrash, match=point):
        store.save(CrashSolo(["one", "two"]))
    reopened_database = Backend.sqlite(path, degraded=True)
    reopened = SqlStore(reopened_database)
    reopened.steal_lease()
    report = reopened.fsck(known_types=(CrashSolo,))
    assert not report.violations
    with reopened_database.engine.connect() as connection:
        assert (
            connection.execute(
                sqlalchemy.text(
                    "SELECT count(*) FROM crash_solo_values v "
                    "WHERE NOT EXISTS (SELECT 1 FROM crash_solo p "
                    "WHERE p.sid = v.crash_solo_sid) "
                )
            ).scalar_one()
            == 0
        )
    next_sid = reopened.save(CrashSolo(["next"]))
    assert next_sid >= 1
    database.dispose()
    reopened_database.dispose()


@pytest.mark.parametrize("finalize", ["parity", "deferred"])
def test_bulk_preserves_main_and_dependency_roles(finalize: str) -> None:
    with Backend.sqlite() as database:
        store = SqlStore(database, entry_records={})
        with store.bulk_ingest(finalize=finalize) as bulk:
            bulk.save(RoleRoot(RoleLeaf("dependency")))
        assert _role(database, "role_root", 1) == 1
        assert _role(database, "role_leaf", 1) == 0


def test_transactional_save_has_no_p3_round_trips_except_dedup_promotion() -> None:
    """The role value changes INSERT payloads, not transactional statement count."""
    with Backend.sqlite() as database:
        store = SqlStore(database, entry_records={})
        store.ensure_tables(StatementRecord, RoleRoot)
        statements: list[str] = []

        def record(_connection, _cursor, statement, _parameters, _context, _many):
            statements.append(statement)

        event.listen(database.engine, "before_cursor_execute", record)
        try:
            store.save(StatementRecord("fresh"))
            # content-id SELECT, INSERT..RETURNING, then the sanctioned same-transaction
            # logical_id fill UPDATE (own sid); role is only an INSERT value.
            assert len(statements) == 3
            statements.clear()
            store.save(StatementRecord("fresh"))
            assert len(statements) == 1  # content-id SELECT hit; it is already main, so no UPDATE.
            statements.clear()
            leaf = RoleLeaf("dependency")
            store.save(RoleRoot(leaf))
            statements.clear()
            store.save(leaf)
            assert len(statements) == 2  # dedup SELECT plus the required dep→main promotion UPDATE.
            assert statements[1].lstrip().upper().startswith("UPDATE")
        finally:
            event.remove(database.engine, "before_cursor_execute", record)


@pytest.mark.parametrize("finalize", ["parity", "deferred"])
def test_transactional_bulk_save_issues_no_main_database_statement_per_record(finalize: str) -> None:
    """P3 role propagation remains entirely in the transactional bulk encoder."""
    with Backend.sqlite() as database:
        store = SqlStore(database, entry_records={})
        if finalize == "parity":
            # This pre-existing parity bookkeeping initializes per-table sid
            # bounds on first save; it is deliberately not a P3 lease/dirty
            # round trip.
            store.ensure_tables(RoleRoot)
        statements: list[str] = []

        def record(_connection, _cursor, statement, _parameters, _context, _many):
            statements.append(statement)

        with store.bulk_ingest(finalize=finalize) as bulk:
            event.listen(database.engine, "before_cursor_execute", record)
            try:
                bulk.save(RoleRoot(RoleLeaf("bulk")))
                if finalize == "parity":
                    assert len(statements) == 4
                    assert all("_httk_store_metadata" not in statement for statement in statements)
                else:
                    assert not statements
            finally:
                event.remove(database.engine, "before_cursor_execute", record)


def test_degraded_content_metadata_conflict_does_not_promote() -> None:
    with Backend.sqlite(degraded=True) as database:
        store = SqlStore(database, entry_records={})
        leaf = MetadataLeaf("same", "dependency")
        store.save(MetadataRoot(leaf))
        sid = store.sid_of(leaf)
        assert sid is not None and _role(database, "metadata_leaf", sid) == 0
        with pytest.raises(EntryMetadataConflictError, match="note"):
            store.save(MetadataLeaf("same", "rejected"))
        assert _role(database, "metadata_leaf", sid) == 0


@pytest.mark.extended
@pytest.mark.parametrize("point", ("content-dedup-select:role_leaf", "content-promotion-update:role_leaf"))
def test_degraded_crash_battery_covers_content_promotion_window(tmp_path: Path, point: str) -> None:
    path = tmp_path / f"promotion-{point.replace(':', '-')}.sqlite"
    database = Backend.sqlite(path, degraded=True)
    store = SqlStore(database, entry_records={})
    leaf = RoleLeaf("dependency")
    store.save(RoleRoot(leaf))
    sid = store.sid_of(leaf)
    assert sid is not None
    store._degraded_fault_hook = lambda observed: observed == point
    with pytest.raises(_DegradedWriteCrash, match=point):
        store.save(leaf)
    reopened_database = Backend.sqlite(path, degraded=True)
    reopened = SqlStore(reopened_database)
    reopened.steal_lease()
    assert reopened.fsck(known_types=(RoleRoot,)).violations == ()
    assert _role(reopened_database, "role_leaf", sid) == (0 if point.startswith("content-dedup") else 1)
    database.dispose()
    reopened_database.dispose()


@pytest.mark.extended
def test_degraded_crash_battery_covers_dispatch_write(tmp_path: Path) -> None:
    path = tmp_path / "dispatch.sqlite"
    database = Backend.sqlite(path, degraded=True)
    store = SqlStore(database, entry_records={CrashDispatchFamily: (CrashDispatchA, CrashDispatchB)})
    dispatch = entry_dispatch_table_name(store.entry_layout[0].name)
    store._degraded_fault_hook = lambda observed: observed == f"dispatch-row-write:{dispatch}"
    with pytest.raises(_DegradedWriteCrash, match="dispatch-row-write"):
        store.save(CrashDispatchA("entry"))
    reopened_database = Backend.sqlite(path, degraded=True)
    reopened = SqlStore(reopened_database)
    reopened.steal_lease()
    assert reopened.fsck(known_types=(CrashDispatchA,)).violations == ()
    with reopened_database.engine.connect() as connection:
        assert connection.execute(sqlalchemy.text(f'SELECT count(*) FROM "{dispatch}"')).scalar_one() == 1
    database.dispose()
    reopened_database.dispose()


def test_fsck_uses_physical_dispatch_presence_for_empty_and_missing_families() -> None:
    with Backend.sqlite() as database:
        store = SqlStore(database, entry_records={CrashDispatchFamily: (CrashDispatchA, CrashDispatchB)})
        # Fresh families have neither backing nor dispatch table; fsck must not
        # turn this ordinary lazy-DDL state into a failure.
        assert store.fsck().violations == ()
        store.save(CrashDispatchA("backing"))
        dispatch = entry_dispatch_table_name(store.entry_layout[0].name)
        with database.engine.begin() as connection:
            connection.execute(sqlalchemy.text(f'DROP TABLE "{dispatch}"'))
        report = store.fsck(known_types=(CrashDispatchA,))
        assert any("is missing while backing rows exist" in item for item in report.violations)


@pytest.mark.parametrize(
    "leaf_type, leaf, table",
    [(RoleLeaf, RoleLeaf("bulk-content"), "role_leaf"), (ValueLeaf, ValueLeaf("bulk-value"), "value_leaf")],
)
def test_populated_serial_bulk_promotes_existing_main(leaf_type, leaf, table: str) -> None:
    with Backend.sqlite() as database:
        store = SqlStore(database, entry_records={})
        root = RoleRoot(leaf) if leaf_type is RoleLeaf else ValueRoot(leaf)
        store.save(root)
        sid = store.sid_of(leaf)
        assert sid is not None and _role(database, table, sid) == 0
        with store.bulk_ingest() as bulk:
            bulk.save(leaf)
        assert _role(database, table, sid) == 1


def test_fsck_is_immediate_read_only_when_requested_and_reaches_child_fixpoint(tmp_path: Path) -> None:
    with Backend.sqlite() as database:
        store = SqlStore(database, entry_records={})
        store.save(CrashRoot(CrashLeaf("orphan", ["a", "b"]), ["x", "y"]))
        with database.engine.begin() as connection:
            connection.execute(sqlalchemy.text("DELETE FROM crash_root"))
        statements: list[str] = []

        def record(_connection, _cursor, statement, _parameters, _context, _many):
            statements.append(statement.upper())

        event.listen(database.engine, "before_cursor_execute", record)
        try:
            report = store.fsck(repair=False, collect_garbage=False, known_types=(CrashRoot,))
        finally:
            event.remove(database.engine, "before_cursor_execute", record)
        assert report.violations == ()
        assert any("BEGIN IMMEDIATE" in statement for statement in statements)
        assert not any(statement.lstrip().startswith(("INSERT", "UPDATE", "DELETE")) for statement in statements)
        store.fsck(known_types=(CrashRoot,))
        with database.engine.connect() as connection:
            assert connection.execute(sqlalchemy.text("SELECT count(*) FROM crash_leaf")).scalar_one() == 0
            assert connection.execute(sqlalchemy.text("SELECT count(*) FROM crash_leaf_atoms")).scalar_one() == 0

    with Backend.sqlite(tmp_path / "verify-only.sqlite", degraded=True) as degraded_database:
        degraded = SqlStore(degraded_database, entry_records={})
        with degraded_database.engine.connect() as connection:
            before = connection.execute(
                sqlalchemy.text("SELECT key, value FROM _httk_store_metadata ORDER BY key")
            ).all()
        assert degraded.fsck(repair=False, collect_garbage=False, known_types=(RoleLeaf,)).violations == ()
        with degraded_database.engine.connect() as connection:
            after = connection.execute(
                sqlalchemy.text("SELECT key, value FROM _httk_store_metadata ORDER BY key")
            ).all()
        assert after == before
        assert not any(key == "lease" for key, _ in after)


def test_fsck_unattributed_refusal_and_invalid_role_repair() -> None:
    with Backend.sqlite() as database:
        store = SqlStore(database, entry_records={})
        sid = store.save(RoleLeaf("role"))
        with database.engine.begin() as connection:
            connection.execute(sqlalchemy.text("PRAGMA ignore_check_constraints = ON"))
            connection.execute(sqlalchemy.text("UPDATE role_leaf SET _httk_role = 7 WHERE sid = :sid"), {"sid": sid})
            connection.execute(sqlalchemy.text("PRAGMA ignore_check_constraints = OFF"))
        report = store.fsck(repair=False, collect_garbage=False, known_types=(RoleLeaf,))
        assert any("invalid _httk_role 7" in item for item in report.violations)
        assert _role(database, "role_leaf", sid) == 7
        store.fsck(collect_garbage=False, known_types=(RoleLeaf,))
        assert _role(database, "role_leaf", sid) == 0
        with database.engine.begin() as connection:
            connection.execute(sqlalchemy.text("CREATE TABLE unrelated_application (value INTEGER)"))
        report = store.fsck(known_types=(RoleLeaf,))
        assert any("unrelated_application" in item for item in report.violations)
        assert _role(database, "role_leaf", sid) == 0
