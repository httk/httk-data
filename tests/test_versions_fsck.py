"""Phase 5 versioned-mode fsck invariant coverage.

In ``store_timestamps="versioned"`` mode fsck verifies the half-open lifetime
bookkeeping and reports (never repairs) any violation: ``ts_end`` and
``replaced_by_sid`` are set together (pairing), a closed interval is non-empty
(ordering), each ``replaced_by_sid`` references an existing same-table row that
does not start after the predecessor closed (lineage), and — on backends
without partial unique indexes — no author-Unique value repeats among current
rows. A future ``ts_end`` is clamped alongside ``ts_start``. The garbage sweep
must never delete superseded family rows, their children, or their held-only
dependencies. Corruption is injected with raw SQL to bypass the save-side guards.
"""

from dataclasses import dataclass
from typing import Annotated, ClassVar

import sqlalchemy
from httk.core.storage import StorageInfo, Unique

from httk.store.db import Database, SqlStore
from httk.store.storage_layout import EntryFamilyDeclaration, EntryRecordDeclaration

# --------------------------------------------------------------------- records


@dataclass(frozen=True)
class VRec:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(dedup="content_id", storage_name="vfsck_rec")

    name: Annotated[str, Unique()]
    payload: str


class VFamily:
    """Application-owned single-backing family (deliberately unregistered)."""


VLAYOUT = EntryFamilyDeclaration(
    name="test-fsck-family",
    family=VFamily,
    records=(EntryRecordDeclaration(name="test-fsck-rec", record=VRec),),
)
_VTABLE = "vfsck_rec"


@dataclass(frozen=True)
class GcDep:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(dedup="content_id", storage_name="fsck_gc_dep")

    tag: str


@dataclass(frozen=True)
class GcEntry:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(dedup="content_id", storage_name="fsck_gc_entry")

    name: Annotated[str, Unique()]
    dep: GcDep
    tags: list[str]


class GcFamily:
    """Application-owned single-backing family with a dependency and children."""


GCLAYOUT = EntryFamilyDeclaration(
    name="test-fsck-gc-family",
    family=GcFamily,
    records=(EntryRecordDeclaration(name="test-fsck-gc-entry", record=GcEntry),),
)


# --------------------------------------------------------------------- helpers


def _store(database: Database) -> SqlStore:
    store = SqlStore(database, entry_families=(VLAYOUT,), store_timestamps="versioned")
    store._clock = lambda: 1_000_000_000
    return store


def _replaced(store: SqlStore) -> tuple[int, int]:
    """Save then replace one entry, returning ``(old_sid, new_sid)``."""
    old = store.save(VRec(name="x", payload="v1"))
    store._clock = lambda: 2_000_000_000
    new = store.replace(VRec(name="x", payload="v1"), VRec(name="x", payload="v2"))
    return old, new


def _corrupt(database: Database, sql: str) -> None:
    with database.engine.begin() as connection:
        connection.execute(sqlalchemy.text(sql))


def _violations(store: SqlStore) -> tuple[str, ...]:
    return store.fsck(repair=False, collect_garbage=False).violations


# --------------------------------------------------------------------- clean pass


def test_clean_versioned_store_passes_fsck() -> None:
    with Database.sqlite() as database:
        store = _store(database)
        _replaced(store)
        assert _violations(store) == ()


# --------------------------------------------------------------------- pairing


def test_ts_end_without_replaced_by_is_reported() -> None:
    with Database.sqlite() as database:
        store = _store(database)
        old, _ = _replaced(store)
        _corrupt(database, f"UPDATE {_VTABLE} SET replaced_by_sid = NULL WHERE sid = {old}")
        assert _violations(store) == (f"table '{_VTABLE}' sid {old} has ts_end 2000000 but no replaced_by_sid",)


def test_replaced_by_without_ts_end_is_reported() -> None:
    with Database.sqlite() as database:
        store = _store(database)
        old, new = _replaced(store)
        _corrupt(database, f"UPDATE {_VTABLE} SET replaced_by_sid = {old}, ts_end = NULL WHERE sid = {new}")
        assert _violations(store) == (f"table '{_VTABLE}' sid {new} has replaced_by_sid {old} but no ts_end",)


# --------------------------------------------------------------------- ordering


def test_zero_length_interval_is_reported() -> None:
    with Database.sqlite() as database:
        store = _store(database)
        old, _ = _replaced(store)
        _corrupt(database, f"UPDATE {_VTABLE} SET ts_end = ts_start WHERE sid = {old}")
        violations = _violations(store)
        assert any("is not after ts_start" in violation for violation in violations)


# --------------------------------------------------------------------- lineage


def test_replaced_by_missing_sid_is_reported() -> None:
    with Database.sqlite() as database:
        store = _store(database)
        old, _ = _replaced(store)
        _corrupt(database, f"UPDATE {_VTABLE} SET replaced_by_sid = 999999 WHERE sid = {old}")
        assert _violations(store) == (f"table '{_VTABLE}' sid {old} replaced_by_sid 999999 references a missing row",)


def test_successor_starting_after_predecessor_closed_is_reported() -> None:
    with Database.sqlite() as database:
        store = _store(database)
        old, new = _replaced(store)
        # Move the close earlier than the successor's start (1_500_000 < 2_000_000).
        _corrupt(database, f"UPDATE {_VTABLE} SET ts_end = 1500000 WHERE sid = {old}")
        assert _violations(store) == (
            f"table '{_VTABLE}' sid {old} successor {new} ts_start 2000000 is after ts_end 1500000",
        )


# --------------------------------------------------------------------- future ts_end


def test_future_ts_end_is_reported_and_clamped() -> None:
    with Database.sqlite() as database:
        store = _store(database)
        old, _ = _replaced(store)
        _corrupt(database, f"UPDATE {_VTABLE} SET ts_end = 10000000000 WHERE sid = {old}")
        store._clock = lambda: 1_500_000_000
        reported = store.fsck(repair=False, collect_garbage=False).violations
        assert any("ts_end" in violation and "exceeds" in violation for violation in reported)
        clamped = store.fsck(repair=True, collect_garbage=False, clamp_future_timestamps=True).violations
        assert any("ts_end" in violation and "clamped" in violation for violation in clamped)
        with database.engine.connect() as connection:
            now_units = connection.execute(
                sqlalchemy.text(f"SELECT ts_end FROM {_VTABLE} WHERE sid = {old}")
            ).scalar_one()
        assert now_units == 1_500_000  # clock 1_500_000_000 ns / 1000 ns resolution


# --------------------------------------------------------------------- unique-among-current (DuckDB)


def test_duplicate_current_unique_value_reported_on_duckdb() -> None:
    # DuckDB has no partial unique indexes, so unique-among-current is a fsck
    # check rather than an engine guarantee; inject a second current row with the
    # same author-Unique value, bypassing the save-side transaction guard.
    with Database.duckdb() as database:
        store = SqlStore(database, entry_families=(VLAYOUT,), store_timestamps="versioned")
        assert not store.backend_facts.supports_partial_unique_indexes
        store._clock = lambda: 1_000_000_000
        orig = store.save(VRec(name="dup", payload="v1"))
        _corrupt(
            database,
            f"INSERT INTO {_VTABLE} (sid, _httk_role, ts_start, ts_end, replaced_by_sid, content_id, name, payload) "
            "VALUES (999, 1, 500000, NULL, NULL, 'fake_cid', 'dup', 'v2')",
        )
        current = store.fsck(repair=False, collect_garbage=False, exclusive=True).violations
        assert current == (f"table '{_VTABLE}' column 'name' value 'dup' appears in 2 current rows",)
        # Close the injected duplicate (consistent lineage back onto the original)
        # and it no longer occupies a current slot.
        _corrupt(database, f"UPDATE {_VTABLE} SET ts_end = 1000000, replaced_by_sid = {orig} WHERE sid = 999")
        assert store.fsck(repair=False, collect_garbage=False, exclusive=True).violations == ()


# --------------------------------------------------------------------- gc preservation


def test_gc_preserves_superseded_rows_children_and_held_dependencies() -> None:
    with Database.sqlite() as database:
        store = SqlStore(database, entry_families=(GCLAYOUT,), store_timestamps="versioned")
        store._clock = lambda: 1_000_000_000
        old = store.save(GcEntry(name="x", dep=GcDep("depA"), tags=["a", "b"]))
        store._clock = lambda: 2_000_000_000
        # The successor references a DIFFERENT dependency, so depA is held only by
        # the (superseded) old entry.
        store.replace(
            GcEntry(name="x", dep=GcDep("depA"), tags=["a", "b"]), GcEntry(name="x", dep=GcDep("depB"), tags=["c"])
        )
        # A genuinely orphaned role-0 dependency row (referenced by nobody).
        _corrupt(
            database,
            "INSERT INTO fsck_gc_dep (sid, _httk_role, ts_start, content_id, tag) "
            "VALUES (999, 0, 1000, 'orphan_cid', 'orphan')",
        )
        assert store.fsck(collect_garbage=True).violations == ()
        with database.engine.connect() as connection:
            deps = set(connection.execute(sqlalchemy.text("SELECT tag FROM fsck_gc_dep")).scalars())
            child_count = connection.execute(
                sqlalchemy.text(f"SELECT COUNT(*) FROM fsck_gc_entry_tags WHERE fsck_gc_entry_sid = {old}")
            ).scalar_one()
        # depA held only by superseded old survives; depB survives; orphan swept.
        assert deps == {"depA", "depB"}
        assert child_count == 2  # the two child tags of the superseded old entry
