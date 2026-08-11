"""Deferred-finalize contracts beyond the shared bulk regression suites.

The existing serial and parallel suites exercise the broad record surface with
``finalize='auto'``.  These focused cases make the selection and the
stage-side guarantees explicit, including the native DuckDB CSV hand-off.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Annotated, ClassVar

import pytest
import sqlalchemy
from httk.core.storage import IdentitySkip, StorageInfo, content_id
from test_db_bulk import (
    Author,
    BulkCalcA,
    BulkCalcB,
    BulkCalcFamily,
    Leaf,
    Node,
    NoneRec,
    OptionalChildRoundTrip,
    Root,
    Sample,
    _database_of,
    _require_bulk,
    _root,
    _stream,
    _table_stats,
    make_sample,
)

from httk.store.db.layout import StoreUnderConstructionError, actual_schema_objects
from httk.store.db.mapping import backing_dispatch_column_name, entry_dispatch_table_name
from httk.store.store_common import EntryMetadataConflictError

# These modules fork their own worker processes; the loadgroup scheduler
# keeps them on one xdist worker so their memory use never stacks.
pytestmark = pytest.mark.xdist_group("bulk-heavy")


@dataclass(frozen=True)
class CsvStageRecord:
    """CSV values which distinguish every quoting/null conversion boundary."""

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="bulk_deferred_csv_stage")

    text: str
    empty: str
    optional: str | None
    payload: bytes


@dataclass(frozen=True)
class PrivateNoneParent:
    """A duplicate parent whose private child list is outside its identity."""

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="bulk_deferred_private_none")

    key: str
    events: Annotated[list[NoneRec], IdentitySkip()]


@dataclass(frozen=True)
class UnsupportedDeferredSkip:
    """A declared unsupported shape shared with the parallel restriction."""

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="bulk_deferred_unsupported_skip")

    name: str
    link: Annotated[UnsupportedDeferredSkip | None, IdentitySkip()] = None


@dataclass(frozen=True)
class MutualValueA:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="bulk_deferred_mutual_a", dedup="by_value")

    value: int
    link: MutualValueB | None = None


@dataclass(frozen=True)
class MutualValueB:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="bulk_deferred_mutual_b", dedup="by_value")

    value: int
    link: MutualValueA | None = None


@dataclass(frozen=True)
class RecursiveTree:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="bulk_deferred_recursive_tree")

    name: str
    children: list[RecursiveTree]


@dataclass(frozen=True)
class PublicByValue:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="bulk_deferred_public_byvalue", dedup="by_value")

    leaf: Leaf


@dataclass(frozen=True)
class ChecksumRecord:
    """Ordinary identifiers containing ``check`` must not look like constraints."""

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="bulk_deferred_checksum")

    checksum: str
    checked: int


CALC_FAMILY = {BulkCalcFamily: (BulkCalcA, BulkCalcB)}


def _logical_rows(store) -> dict[str, list[tuple[object, ...]]]:
    """Stable physical rows; dense sids are deterministic for one profile/run."""
    database = _database_of(store)
    result: dict[str, list[tuple[object, ...]]] = {}
    with database.engine.connect() as connection:
        for name, table in sorted(store._metadata.tables.items()):
            if name.startswith("_httk_"):
                continue
            columns = [column.name for column in table.columns]
            order = ", ".join(f'"{column}"' for column in columns)
            rows = connection.execute(sqlalchemy.text(f'SELECT * FROM "{name}" ORDER BY {order}')).all()
            result[name] = [tuple(row) for row in rows]
    return result


def _reference_graph(store) -> dict[str, tuple[str, ...] | None]:
    """Sample -> Author references translated through content IDs, not sids."""
    database = _database_of(store)
    with database.engine.connect() as connection:
        rows = connection.execute(
            sqlalchemy.text(
                "SELECT s.content_id, a.content_id FROM bulk_sample s "
                "LEFT JOIN bulk_author a ON a.sid = s.reference_sid ORDER BY s.content_id"
            )
        ).all()
    return {str(source): (str(target),) if target is not None else None for source, target in rows}


def _child_graph(store) -> list[tuple[str, int, str]]:
    """Sample child author rows, parent/element normalized to content IDs."""
    database = _database_of(store)
    with database.engine.connect() as connection:
        return [
            tuple(row)
            for row in connection.execute(
                sqlalchemy.text(
                    "SELECT s.content_id, c.authors_index, a.content_id "
                    "FROM bulk_sample_authors c JOIN bulk_sample s ON s.sid = c.bulk_sample_sid "
                    "JOIN bulk_author a ON a.sid = c.authors_sid "
                    "ORDER BY s.content_id, c.authors_index"
                )
            ).all()
        ]


def _dispatch_graph(store) -> dict[str, tuple[str, str]]:
    """Dispatch identity mapped to the selected backing's content identity."""
    table = entry_dispatch_table_name("test-db-bulk-calculations")
    a_column = backing_dispatch_column_name("test-db-bulk-calc-a")
    b_column = backing_dispatch_column_name("test-db-bulk-calc-b")
    database = _database_of(store)
    with database.engine.connect() as connection:
        rows = connection.execute(
            sqlalchemy.text(
                f'SELECT d.content_id, a.content_id, b.content_id FROM "{table}" d '
                f'LEFT JOIN bulk_calc_a a ON a.sid = d."{a_column}" '
                f'LEFT JOIN bulk_calc_b b ON b.sid = d."{b_column}" ORDER BY d.content_id'
            )
        ).all()
    return {str(key): ("a", str(a)) if a is not None else ("b", str(b)) for key, a, b in rows}


@pytest.mark.extended
@pytest.mark.parametrize("workers", [1, 2])
def test_deferred_equivalent_to_parity(store_factory, workers):
    """Parity and stage-side builds agree modulo physical sid relabeling."""
    parity = store_factory(entry_records=CALC_FAMILY)
    deferred = store_factory(entry_records=CALC_FAMILY)
    _require_bulk(parity)
    parity_ids: list[int] = []
    deferred_ids: list[int] = []
    with parity.bulk_ingest(workers=workers, finalize="parity") as parity_bulk:
        for value in _stream():
            parity_ids.append(parity_bulk.save(value))
    with deferred.bulk_ingest(workers=workers, finalize="deferred") as deferred_bulk:
        for value in _stream():
            deferred_ids.append(deferred_bulk.save(value))
    assert _table_stats(parity, _database_of(parity)) == _table_stats(deferred, _database_of(deferred))
    assert _reference_graph(parity) == _reference_graph(deferred)
    assert _child_graph(parity) == _child_graph(deferred)
    assert _dispatch_graph(parity) == _dispatch_graph(deferred)
    parity_open = store_factory.reopen(parity)
    deferred_open = store_factory.reopen(deferred)
    for record_type, index, value in (
        (Author, 0, Author("Ada", 1852)),
        (Sample, 8, make_sample()),
        (OptionalChildRoundTrip, 13, OptionalChildRoundTrip("c", ["note"])),
        (Root, 14, _root("one")),
    ):
        assert parity_open.fetch(record_type, parity_bulk.resolved_sid(record_type, parity_ids[index])) == value
        assert deferred_open.fetch(record_type, deferred_bulk.resolved_sid(record_type, deferred_ids[index])) == value
    key = content_id(BulkCalcA("alpha", 1))
    assert parity_open.fetch_entry(BulkCalcFamily, key) == deferred_open.fetch_entry(BulkCalcFamily, key)


@pytest.mark.extended
@pytest.mark.parametrize("workers", [1, 2])
def test_deferred_none_descendant_of_losing_parent_is_orphaned(store_factory, workers):
    """Only the first duplicate parent's private ``dedup='none'`` child survives."""
    store = store_factory()
    _require_bulk(store)
    with store.bulk_ingest(workers=workers, finalize="deferred", verify_metadata=False) as bulk:
        first = bulk.save(PrivateNoneParent("same", [NoneRec("kept")]))
        bulk.save(PrivateNoneParent("same", [NoneRec("orphan")]))
    assert _table_stats(store, _database_of(store))["bulk_deferred_private_none"][0] == 1
    assert _table_stats(store, _database_of(store))["bulk_none_rec"][0] == 1
    assert store_factory.reopen(store).fetch(
        PrivateNoneParent, bulk.resolved_sid(PrivateNoneParent, first)
    ) == PrivateNoneParent("same", [NoneRec("kept")])


@pytest.mark.extended
@pytest.mark.parametrize("workers", [1, 2])
def test_deferred_by_value_transitive_fixpoint(store_factory, workers):
    """The final maps converge through the self-referential by-value SCC."""
    store = store_factory()
    _require_bulk(store)
    stream = [Node(2, Node(1)), Node(2, Node(1)), Node(3, Node(2, Node(1)))]
    with store.bulk_ingest(workers=workers, finalize="deferred") as bulk:
        ids = [bulk.save(value) for value in stream]
    assert _table_stats(store, _database_of(store))["bulk_node"][0] == 3
    assert bulk.resolved_sid(Node, ids[0]) == bulk.resolved_sid(Node, ids[1])
    assert store_factory.reopen(store).fetch(Node, bulk.resolved_sid(Node, ids[2])) == stream[2]


@pytest.mark.extended
@pytest.mark.parametrize("workers", [2])
def test_mutual_by_value_profiles_reach_the_same_fixpoint(store_factory, workers):
    """Parity now revisits mutually dependent by-value tables just like deferred."""
    leaf_one = MutualValueA(0)
    leaf_two = MutualValueA(0)
    stream = [MutualValueA(2, MutualValueB(1, leaf_one)), MutualValueA(2, MutualValueB(1, leaf_two))]
    results = []
    for finalize in ("parity", "deferred"):
        store = store_factory()
        _require_bulk(store)
        with store.bulk_ingest(workers=workers, finalize=finalize) as bulk:
            for value in stream:
                bulk.save(value)
        results.append(_table_stats(store, _database_of(store)))
    assert results[0] == results[1]
    assert results[0]["bulk_deferred_mutual_a"][0] == 2
    assert results[0]["bulk_deferred_mutual_b"][0] == 1


@pytest.mark.extended
@pytest.mark.parametrize("workers", [1, 2])
def test_recursive_child_elements_survive_deferred_reachability(store_factory, workers):
    """A child-element/ownership SCC retains recursive descendants."""
    value = RecursiveTree("root", [RecursiveTree("leaf", [])])
    for finalize in ("parity", "deferred"):
        store = store_factory()
        _require_bulk(store)
        with store.bulk_ingest(workers=workers, finalize=finalize) as bulk:
            sid = bulk.save(value)
        assert store_factory.reopen(store).fetch(RecursiveTree, bulk.resolved_sid(RecursiveTree, sid)) == value


@pytest.mark.extended
@pytest.mark.parametrize("workers", [1, 2])
def test_single_dispatch_backing_is_valid_for_both_profiles(store_factory, workers):
    """A family dispatch may legitimately stage only one of its declared backings."""
    for finalize in ("parity", "deferred"):
        store = store_factory(entry_records=CALC_FAMILY)
        _require_bulk(store)
        with store.bulk_ingest(workers=workers, finalize=finalize) as bulk:
            sid = bulk.save(BulkCalcA("only", 1))
        reopened = store_factory.reopen(store)
        assert reopened.fetch(BulkCalcA, bulk.resolved_sid(BulkCalcA, sid)) == BulkCalcA("only", 1)
        assert reopened.fetch_entry(BulkCalcFamily, content_id(BulkCalcA("only", 1))) == BulkCalcA("only", 1)


@pytest.mark.extended
@pytest.mark.parametrize("workers", [1, 2])
def test_deferred_is_deterministic_for_identical_input(store_factory, workers):
    """Determinism is per backend/profile/worker count and input order."""
    first = store_factory(entry_records=CALC_FAMILY)
    second = store_factory(entry_records=CALC_FAMILY)
    _require_bulk(first)
    for store in (first, second):
        store._clock = lambda: 1_700_000_000_000_000_000
        with store.bulk_ingest(workers=workers, finalize="deferred") as bulk:
            for value in _stream():
                bulk.save(value)
    assert _logical_rows(first) == _logical_rows(second)


@pytest.mark.parametrize("finalize", ["deferred", "auto"])
def test_serial_deferred_public_sids_and_retry_contract(store_factory, finalize):
    """Serial provisional IDs deduplicate publicly and resolve after a clean retry."""
    store = store_factory()
    _require_bulk(store)
    with pytest.raises(EntryMetadataConflictError), store.bulk_ingest(finalize=finalize) as bulk:
        first = bulk.save(_root("same"))
        again = bulk.save(_root("same"))
        assert first == again
        bulk.save(_root("same", note="changed"))
    with store.bulk_ingest(finalize=finalize) as bulk:
        first = bulk.save(Author("retry", 1))
        again = bulk.save(Author("retry", 1))
        assert first == again
    assert bulk.resolved_sid(Author, first) == bulk.resolved_sid(Author, again)
    assert store_factory.reopen(store).fetch(Author, bulk.resolved_sid(Author, first)) == Author("retry", 1)
    with _database_of(store).engine.connect() as connection:
        assert (
            connection.execute(
                sqlalchemy.text("SELECT count(*) FROM _httk_store_metadata WHERE key = 'ingest_state'")
            ).scalar_one()
            == 0
        )


def test_serial_deferred_by_value_duplicates_share_public_sid(store_factory):
    """Public serial IDs normalize referenced occurrence sids for by-value parents."""
    store = store_factory()
    _require_bulk(store)
    with store.bulk_ingest(finalize="deferred") as bulk:
        first = bulk.save(PublicByValue(Leaf(7, "metadata")))
        second = bulk.save(PublicByValue(Leaf(7, "metadata")))
        assert first == second
    assert bulk.resolved_sid(PublicByValue, first) == bulk.resolved_sid(PublicByValue, second)


def test_sqlite_deferred_columns_named_check_ingest_without_a_check_constraint(store_factory):
    """SQLite catalog validation tokenizes CHECK instead of matching identifier substrings."""
    store = store_factory()
    _require_bulk(store)
    if _database_of(store).engine.dialect.name != "sqlite":
        pytest.skip("SQLite CHECK parser")
    with store.bulk_ingest(finalize="deferred") as bulk:
        sid = bulk.save(ChecksumRecord("digest", 1))
    assert store_factory.reopen(store).fetch(ChecksumRecord, bulk.resolved_sid(ChecksumRecord, sid)) == ChecksumRecord(
        "digest", 1
    )


def test_sqlite_check_clause_parser_detects_a_real_mismatch():
    """A genuine CHECK expression still compares unequal after extraction."""
    from httk.store.db.bulk import BulkIngest

    actual = BulkIngest._sqlite_check_clauses('CREATE TABLE x (checksum TEXT, checked INTEGER, CHECK (checked > 0))')
    assert actual == [BulkIngest._physical_check("checked > 0")]
    assert actual != [BulkIngest._physical_check("checked >= 0")]


@pytest.mark.parametrize("finalize", ["deferred", "auto"])
def test_empty_serial_deferred_ingest_is_a_successful_noop(store_factory, finalize):
    """A marker-only serial deferred context clears cleanly without a save."""
    store = store_factory()
    _require_bulk(store)
    with store.bulk_ingest(finalize=finalize):
        pass
    store_factory.reopen(store)


def test_duckdb_deferred_stage_detaches_before_the_next_ingest(store_factory):
    """A successful stage attachment cannot poison a later pooled connection."""
    store = store_factory()
    _require_bulk(store)
    if _database_of(store).engine.dialect.name != "duckdb":
        pytest.skip("DuckDB attachment lifecycle")
    with store.bulk_ingest(finalize="deferred") as bulk:
        bulk.save(Author("first", 1))
    database = _database_of(store)
    with database.engine.begin() as connection:
        connection.execute(sqlalchemy.text('DROP TABLE "bulk_author"'))
    reopened = store_factory.reopen(store)
    with reopened.bulk_ingest(finalize="deferred") as bulk:
        bulk.save(Author("second", 2))
    assert reopened.fetch(Author, bulk.resolved_sid(Author, 1)) == Author("second", 2)


def test_parallel_precreated_empty_store_is_marked(store_factory):
    """SQLite's permitted pre-created empty tables are still fenced during ingest."""
    store = store_factory()
    _require_bulk(store)
    if _database_of(store).engine.dialect.name != "sqlite":
        pytest.skip("SQLite permits this parallel pre-created-table shape")
    store.ensure_tables(Author)
    with store.bulk_ingest(workers=2, finalize="parity") as bulk:
        with _database_of(store).engine.connect() as connection:
            assert (
                connection.execute(
                    sqlalchemy.text("SELECT count(*) FROM _httk_store_metadata WHERE key = 'ingest_state'")
                ).scalar_one()
                == 1
            )
        bulk.save(Author("marked", 1))


def test_marker_is_cleared_when_post_marker_connection_start_fails(store_factory, monkeypatch):
    """A failure after marker commit restores the untouched catalog and clears its fence."""
    store = store_factory()
    _require_bulk(store)
    engine = _database_of(store).engine
    original = engine.connect
    calls = 0

    def fail_main_connection(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:  # probe, marker transaction, then owning ingest connection
            raise OSError("simulated post-marker connect failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(engine, "connect", fail_main_connection)
    with pytest.raises(OSError, match="post-marker"):
        store.bulk_ingest(finalize="deferred").__enter__()
    with original() as connection:
        assert (
            connection.execute(
                sqlalchemy.text("SELECT count(*) FROM _httk_store_metadata WHERE key = 'ingest_state'")
            ).scalar_one()
            == 0
        )


def test_deferred_selection_and_unsupported_shape_error(store_factory):
    """Auto uses deferred only for a fresh serial store; unsupported shapes name the parity remedy."""
    store = store_factory()
    _require_bulk(store)
    with store.bulk_ingest(finalize="auto") as bulk:
        assert bulk._finalize_profile == "deferred"
        bulk.save(Author("fresh", 1))
    with pytest.raises(RuntimeError, match="physically empty"), store.bulk_ingest(finalize="deferred"):
        pass
    with store.bulk_ingest(finalize="auto") as bulk:
        assert bulk._finalize_profile == "parity"
        bulk.save(Author("populated", 2))

    parallel = store_factory()
    with parallel.bulk_ingest(workers=2, finalize="auto") as bulk:
        assert bulk._finalize_profile == "parity"
        bulk.save(Author("parallel", 3))

    unsupported = UnsupportedDeferredSkip("a", UnsupportedDeferredSkip("b"))
    rejected = store_factory()
    with (
        pytest.raises(ValueError, match="self-referential") as error,
        rejected.bulk_ingest(finalize="deferred") as bulk,
    ):
        bulk.save(unsupported)
    assert 'finalize="parity"' in str(error.value)


def test_deferred_crash_window_marker_rejects_reopen(store_factory):
    """Simulated process loss before marker clear leaves the store fenced off."""
    store = store_factory()
    _require_bulk(store)
    bulk = store.bulk_ingest(finalize="deferred")
    bulk.__enter__()
    try:
        bulk.save(Author("interrupted", 1))
        # Deliberately bypass __exit__: this is the pre-marker-clear crash window.
        bulk._release_connection(bulk._connection)
        bulk._connection = None
        store._release_bulk_context()
        with pytest.raises(StoreUnderConstructionError):
            store_factory.reopen(store)
    finally:
        if bulk._serial_stage is not None:
            bulk._serial_stage.close()
            bulk._serial_stage = None
        # Test cleanup only; a real crash intentionally leaves the marker.
        with _database_of(store).engine.begin() as connection:
            connection.execute(sqlalchemy.text("DELETE FROM _httk_store_metadata WHERE key = 'ingest_state'"))


def test_serial_deferred_never_imports_pyarrow(store_factory, monkeypatch):
    """The external serial stage remains dependency-free, including on DuckDB."""
    store = store_factory()
    _require_bulk(store)
    seen: list[str] = []
    original = importlib.import_module

    def tracked(name, package=None):
        if name.startswith("pyarrow"):
            seen.append(name)
            raise AssertionError("serial deferred must not import pyarrow")
        return original(name, package)

    monkeypatch.setattr(importlib, "import_module", tracked)
    with store.bulk_ingest(workers=1, finalize="deferred") as bulk:
        bulk.save(Author("no-arrow", 1))
    assert seen == []


def test_duckdb_serial_csv_stage_round_trips_edge_values(store_factory):
    """COPY preserves null/empty, newlines, quotes, Unicode, and bytes."""
    store = store_factory()
    _require_bulk(store)
    if _database_of(store).engine.dialect.name != "duckdb":
        pytest.skip("DuckDB CSV stage path")
    value = CsvStageRecord('line one\n"quoted", μ', "", None, b"\x00\xff\n")
    with store.bulk_ingest(finalize="deferred") as bulk:
        sid = bulk.save(value)
    assert store_factory.reopen(store).fetch(CsvStageRecord, bulk.resolved_sid(CsvStageRecord, sid)) == value
    with _database_of(store).engine.connect() as connection:
        assert not any(name.startswith("_httk_deferred_") for name in actual_schema_objects(connection))


@pytest.mark.extended
def test_deferred_track_sids_false_keeps_logical_result_without_resolution(store_factory):
    """The bounded-memory mode drops only the optional public sid contract."""
    store = store_factory()
    _require_bulk(store)
    value = Author("untracked", 2026)
    with store.bulk_ingest(finalize="deferred", track_sids=False) as bulk:
        provisional = bulk.save(value)
    with pytest.raises(RuntimeError, match="track_sids=False"):
        bulk.resolved_sid(Author, provisional)
    with _database_of(store).engine.connect() as connection:
        assert connection.execute(sqlalchemy.text("SELECT count(*) FROM bulk_author")).scalar_one() == 1
