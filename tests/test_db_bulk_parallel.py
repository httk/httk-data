"""Equivalence and contracts for ``bulk_ingest(workers>1)`` across SQLite and DuckDB.

Every test proves the parallel encode-plus-shard-merge path against a serial
``save()`` loop or the serial ``workers=1`` build: a parallel-built store must
agree with a serially built one on per-table counts, content-id sets, the
records reconstructed through a reopen, entry dispatch, and ``resolved_sid``
resolution. MongoStore has no ``bulk_ingest`` and is skipped through the shared
``store_factory`` fixture; the parallel path itself only runs on SqlStore.

The record classes, streams, and small helpers are shared with
``tests.test_db_bulk`` so the two suites exercise the same schema surface.
"""

import math
import os
import queue as queue_mod
import signal
import tempfile
from dataclasses import dataclass
from fractions import Fraction
from typing import Annotated, ClassVar

import pytest
import sqlalchemy
from httk.core.storage import IdentitySkip, StorageInfo, content_id
from test_db_bulk import (
    Author,
    BulkCalcA,
    BulkCalcB,
    BulkCalcFamily,
    ByValParent,
    ContentParent,
    Elem,
    Leaf,
    MetaScalar,
    Node,
    NoneRec,
    OptionalChildRoundTrip,
    Root,
    Sample,
    ValidatedRecord,
    _database_of,
    _physical_counts,
    _require_bulk,
    _root,
    _stream,
    _table_stats,
    make_sample,
)

# These modules fork their own worker processes; the loadgroup scheduler
# keeps them on one xdist worker so their memory use never stacks.
pytestmark = pytest.mark.xdist_group("bulk-heavy")
from httk.store.db import Database, SqlStore
from httk.store.db.layout import METADATA_TABLE_NAME, actual_schema_objects
from httk.store.store_common import EntryDispatchIntegrityError, EntryMetadataConflictError

CALC_FAMILY = {BulkCalcFamily: (BulkCalcA, BulkCalcB)}


# --- record shapes exercising the parallel metadata-verification restriction (H2) and float semantics (M2)


@dataclass(frozen=True)
class FloatMeta:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="bulkp_float_meta")

    key: str
    measure: Annotated[float | None, IdentitySkip()]


@dataclass(frozen=True)
class FracMeta:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="bulkp_frac_meta")

    key: str
    frac: Annotated[Fraction, IdentitySkip()]


@dataclass(frozen=True)
class TwoFloatMeta:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="bulkp_two_float_meta")

    key: str
    alpha: Annotated[float | None, IdentitySkip()]  # schema-order first
    beta: Annotated[float | None, IdentitySkip()]


@dataclass(frozen=True)
class SelfRefSkip:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(
        storage_name="bulkp_self_ref_skip", identity_name="tests.bulkp.SelfRefSkip"
    )

    name: str
    link: Annotated["SelfRefSkip | None", IdentitySkip()] = None


@dataclass(frozen=True)
class SkipChildElem:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="bulkp_skip_child_elem")

    text: str


@dataclass(frozen=True)
class SkipChildContainer:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(
        storage_name="bulkp_skip_child_container", identity_name="tests.bulkp.SkipChildContainer"
    )

    name: str
    children: Annotated[list[SkipChildElem], IdentitySkip()]


@dataclass(frozen=True)
class NoneWithMeta:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(
        storage_name="bulkp_none_with_meta", dedup="none", identity_name="tests.bulkp.NoneWithMeta"
    )

    value: str
    note: Annotated[str, IdentitySkip()]


@dataclass(frozen=True)
class DescendHolder:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="bulkp_descend_holder")

    name: str
    leaf: NoneWithMeta


def _has_application_rows(store, database) -> bool:
    with database.engine.connect() as connection:
        for name, kinds in actual_schema_objects(connection).items():
            if "table" not in kinds or name == METADATA_TABLE_NAME:
                continue
            if connection.execute(sqlalchemy.text(f'SELECT count(*) FROM "{name}"')).scalar_one():
                return True
    return False


def _big_duplicated_stream(rounds: int = 180) -> list[object]:
    """A long stream with heavy cross-record duplication, orphans, and self-references."""
    stream: list[object] = []
    for i in range(rounds):
        stream.append(Author(f"A{i % 17}", 1800 + i % 17))
        stream.append(make_sample(formula=f"F{i % 9}", spacegroup=200 + i % 9))
        stream.append(ContentParent(f"cp{i % 13}", NoneRec(f"evt{i % 13}")))  # content parent + none descendant
        stream.append(ByValParent(i % 11, [Elem(f"e{i % 11}")]))  # by_value parent + child elements
        stream.append(Node(i % 7, Node(i % 5)))  # self-referential by_value chain
        stream.append(_root(f"r{i % 6}"))  # references + child list, identity-skip metadata
        stream.append(OptionalChildRoundTrip(f"o{i % 4}", None if i % 2 else ["n"]))
    return stream


# --------------------------------------------------------------------- tests


@pytest.mark.parametrize("workers", [2, pytest.param(4, marks=pytest.mark.extended)])
def test_parallel_matches_serial_mixed_stream(store_factory, workers):
    """The mixed object stream built in parallel equals the serial build (counts, ids, records, dispatch)."""
    serial = store_factory(entry_records=CALC_FAMILY)
    _require_bulk(serial)
    parallel = store_factory(entry_records=CALC_FAMILY)

    for obj in _stream():
        serial.save(obj)
    tokens: list[int] = []
    with parallel.bulk_ingest(workers=workers) as bulk:
        for obj in _stream():
            tokens.append(bulk.save(obj))

    assert _table_stats(parallel, _database_of(parallel)) == _table_stats(serial, _database_of(serial))

    reopened = store_factory.reopen(parallel)
    assert reopened.fetch(Author, bulk.resolved_sid(Author, tokens[0])) == Author("Ada", 1852)
    assert reopened.fetch(Sample, bulk.resolved_sid(Sample, tokens[8])) == make_sample()
    assert reopened.fetch(Sample, bulk.resolved_sid(Sample, tokens[10])) == make_sample(
        formula="NaCl", weight=1.25, reference=None
    )
    assert reopened.fetch(OptionalChildRoundTrip, bulk.resolved_sid(OptionalChildRoundTrip, tokens[11])).notes is None
    assert reopened.fetch(OptionalChildRoundTrip, bulk.resolved_sid(OptionalChildRoundTrip, tokens[13])).notes == [
        "note"
    ]
    assert reopened.fetch(Root, bulk.resolved_sid(Root, tokens[14])) == Root(
        "one", Leaf(1, "leaf metadata"), [Leaf(1, "leaf metadata"), Leaf(2)], _root().modified
    )
    assert reopened.fetch_entry(BulkCalcFamily, content_id(BulkCalcA("alpha", 1))) == BulkCalcA("alpha", 1)
    assert reopened.fetch_entry(BulkCalcFamily, content_id(BulkCalcB("beta", "kind-b"))) == BulkCalcB("beta", "kind-b")


@pytest.mark.parametrize(
    ("rounds", "workers"),
    [(12, 2), pytest.param(180, 6, marks=pytest.mark.extended)],
)
def test_parallel_matches_serial_large_duplicated_stream(store_factory, rounds, workers):
    """A long, heavily duplicated stream (cross-worker collapse + orphan sweep) matches the serial build."""
    stream = _big_duplicated_stream(rounds)
    serial = store_factory()
    _require_bulk(serial)
    parallel = store_factory()
    for obj in stream:
        serial.save(obj)
    with parallel.bulk_ingest(workers=workers, chunk_size=31) as bulk:
        for obj in stream:
            bulk.save(obj)
    assert _physical_counts(_database_of(parallel)) == _physical_counts(_database_of(serial))
    assert _table_stats(parallel, _database_of(parallel)) == _table_stats(serial, _database_of(serial))


def test_parallel_returned_token_resolves_and_dedups(store_factory):
    """Cross-worker duplicates resolve to one survivor; resolution is unavailable until the context exits."""
    store = store_factory()
    _require_bulk(store)
    with store.bulk_ingest(workers=4) as bulk:
        tokens = [bulk.save(Author("Ada", 1852)) for _ in range(20)]
        other = bulk.save(Author("Grace", 1906))
        with pytest.raises(RuntimeError, match="resolved_sid"):
            bulk.resolved_sid(Author, tokens[0])
    resolved = {bulk.resolved_sid(Author, token) for token in tokens}
    assert len(resolved) == 1  # every duplicate collapses to the same stored sid
    assert bulk.resolved_sid(Author, other) not in resolved
    reopened = store_factory.reopen(store)
    assert reopened.fetch(Author, resolved.pop()) == Author("Ada", 1852)
    with pytest.raises(KeyError):
        bulk.resolved_sid(Author, 10_000_019)  # a token this ingest never returned


def test_parallel_cross_worker_content_conflict_raises(store_factory):
    """A cross-worker content-id hit with conflicting identity-excluded metadata aborts and leaves no rows."""
    store = store_factory()
    _require_bulk(store)
    database = _database_of(store)
    with pytest.raises(EntryMetadataConflictError, match="note"), store.bulk_ingest(workers=4) as bulk:
        for _ in range(40):
            bulk.save(MetaScalar("k", "note1"))
        bulk.save(MetaScalar("k", "note2"))  # conflicting metadata, same content id
    assert not _has_application_rows(store, database)


def test_parallel_cross_worker_nested_conflict_raises(store_factory):
    """A cross-worker nested (descend) metadata conflict aborts the whole ingest."""
    store = store_factory()
    _require_bulk(store)
    with pytest.raises(EntryMetadataConflictError), store.bulk_ingest(workers=4) as bulk:
        for _ in range(40):
            bulk.save(_root("one", note="leaf metadata"))
        bulk.save(_root("one", note="conflicting"))  # Leaf.note differs under an equal Root content id
    assert not _has_application_rows(store, _database_of(store))


def test_parallel_metadata_conflict_ignored_when_disabled(store_factory):
    """With verify_metadata disabled a conflicting cross-worker hit deduplicates without raising."""
    store = store_factory()
    _require_bulk(store)
    with store.bulk_ingest(workers=4, verify_metadata=False) as bulk:
        for _ in range(20):
            bulk.save(MetaScalar("k", "note1"))
        bulk.save(MetaScalar("k", "note2"))
    assert _physical_counts(_database_of(store))["bulk_meta_scalar"] == 1


def test_parallel_dispatch_resolves_and_detects_conflict(store_factory):
    """Entry-family dispatch rows merge across workers and reconstruct through fetch_entry."""
    store = store_factory(entry_records=CALC_FAMILY)
    _require_bulk(store)
    with store.bulk_ingest(workers=4) as bulk:
        for _ in range(10):
            bulk.save(BulkCalcA("alpha", 1))
            bulk.save(BulkCalcB("beta", "kind-b"))
    reopened = store_factory.reopen(store)
    assert reopened.fetch_entry(BulkCalcFamily, content_id(BulkCalcA("alpha", 1))) == BulkCalcA("alpha", 1)
    assert reopened.fetch_entry(BulkCalcFamily, content_id(BulkCalcB("beta", "kind-b"))) == BulkCalcB("beta", "kind-b")


def test_parallel_dispatch_conflicting_backing_raises(store_factory, monkeypatch):
    """Two backings claiming one dispatch content id abort the ingest (cross-worker dispatch integrity)."""
    store = store_factory(entry_records=CALC_FAMILY)
    _require_bulk(store)
    # Force every content id to collide (the workers inherit this patch through fork).
    shared = content_id(BulkCalcA("alpha", 1))
    monkeypatch.setattr("httk.store.store_common.SaveProjection.content_id", lambda self, rt, src: shared)
    with pytest.raises(EntryDispatchIntegrityError, match="conflicting backing"), store.bulk_ingest(workers=2) as bulk:
        bulk.save(BulkCalcA("alpha", 1))
        bulk.save(BulkCalcB("beta", "kind-b"))  # same forced dispatch content id, different backing
    assert not _has_application_rows(store, _database_of(store))


def test_parallel_worker_failure_is_atomic(store_factory):
    """A worker-side validator failure propagates faithfully and leaves the store untouched."""
    store = store_factory()
    _require_bulk(store)
    database = _database_of(store)
    with pytest.raises(ValueError, match="validator rejected"), store.bulk_ingest(workers=4) as bulk:
        for i in range(30):
            bulk.save(Author(f"A{i}", i))
        bulk.save(ValidatedRecord(-1))  # the validator raises inside a worker
    assert not _has_application_rows(store, database)
    # The store is usable afterwards.
    sid = store.save(Author("Boole", 1854))
    assert store_factory.reopen(store).fetch(Author, sid) == Author("Boole", 1854)


def test_parallel_requires_empty_store(store_factory):
    """Opening workers>1 on a store that already holds rows fails fast (both backends direct to workers=1)."""
    store = store_factory()
    _require_bulk(store)
    store.save(Author("Ada", 1852))
    with pytest.raises(RuntimeError, match="workers=1"), store.bulk_ingest(workers=4):
        pass


def test_parallel_duckdb_rejects_precreated_tables(store_factory):
    """On DuckDB, an empty pre-created application table is refused (its FKs block the in-place merge)."""
    store = store_factory()
    _require_bulk(store)
    database = _database_of(store)
    if database.engine.dialect.name != "duckdb":
        pytest.skip("DuckDB-specific pre-existing-table rejection")
    store.ensure_tables(Author)
    with pytest.raises(RuntimeError, match="no pre-existing application tables"), store.bulk_ingest(workers=4):
        pass


def test_parallel_sqlite_allows_precreated_empty_tables(store_factory):
    """On SQLite, ensure_tables then a parallel build of distinct records succeeds (FKs are never enforced)."""
    store = store_factory()
    _require_bulk(store)
    if _database_of(store).engine.dialect.name != "sqlite":
        pytest.skip("SQLite-specific pre-created-table flow")
    store.ensure_tables(Author)
    with store.bulk_ingest(workers=3) as bulk:
        tokens = [bulk.save(Author(f"A{i}", 1900 + i)) for i in range(30)]
    reopened = store_factory.reopen(store)
    assert reopened.fetch(Author, bulk.resolved_sid(Author, tokens[7])) == Author("A7", 1907)
    assert _physical_counts(_database_of(store))["bulk_author"] == 30


def test_parallel_self_referential_by_value(store_factory):
    """A self-referential by_value chain deduplicates to the serial result across workers."""
    serial = store_factory()
    _require_bulk(serial)
    parallel = store_factory()
    chain = [Node(2, Node(1)) for _ in range(25)] + [Node(3, Node(2, Node(1))) for _ in range(25)]
    for obj in chain:
        serial.save(obj)
    with parallel.bulk_ingest(workers=5) as bulk:
        for obj in chain:
            bulk.save(obj)
    assert _physical_counts(_database_of(parallel)) == _physical_counts(_database_of(serial))


def test_parallel_invalid_workers_rejected(store_factory):
    """A non-positive worker count is rejected before any work begins."""
    store = store_factory()
    _require_bulk(store)
    with pytest.raises(ValueError, match="workers"):
        store.bulk_ingest(workers=0)


# --------------------------------------------------------------------- corrections-batch tests


def test_parallel_unpicklable_task_fails_fast(store_factory):
    """An unpicklable object aborts ``save`` promptly (synchronous pickle) and leaves the store untouched."""
    store = store_factory()
    _require_bulk(store)
    database = _database_of(store)
    unpicklable = OptionalChildRoundTrip("x", None)
    object.__setattr__(unpicklable, "_thread_lock", __import__("threading").Lock())  # locks do not pickle
    with pytest.raises((TypeError, Exception)) as raised, store.bulk_ingest(workers=2) as bulk:
        bulk.save(Author("Ada", 1852))
        bulk.save(unpicklable)  # pickling this raises synchronously inside save
    assert "pickle" in type(raised.value).__module__ or isinstance(raised.value, TypeError)
    assert not _has_application_rows(store, database)


def test_parallel_cross_worker_conflict_is_deterministic(store_factory):
    """Round-robin routing sends two colliding records to different workers, so the conflict is deterministic."""
    store = store_factory()
    _require_bulk(store)
    with pytest.raises(EntryMetadataConflictError, match="note"), store.bulk_ingest(workers=2) as bulk:
        bulk.save(MetaScalar("k", "note1"))  # token 0 -> worker 0
        bulk.save(MetaScalar("k", "note2"))  # token 1 -> worker 1 (guaranteed a different worker)
    assert not _has_application_rows(store, _database_of(store))


@pytest.mark.parametrize(
    "record, reason",
    [
        (SelfRefSkip("a", SelfRefSkip("b")), "self-referential"),
        (SkipChildContainer("c", [SkipChildElem("x")]), "child sequence"),
        (DescendHolder("d", NoneWithMeta("v", "note")), "non-deduplicated"),
    ],
)
def test_parallel_rejects_unsupported_metadata_shapes(store_factory, record, reason):
    """Unsupported identity-excluded metadata shapes fail fast in parallel but work serially."""
    store = store_factory()
    _require_bulk(store)
    with pytest.raises(ValueError, match=reason), store.bulk_ingest(workers=2) as bulk:
        bulk.save(record)
    assert not _has_application_rows(store, _database_of(store))

    # The same record stores cleanly on the serial per-record path.
    serial_store = store_factory()
    sid = serial_store.save(record)
    reopened = store_factory.reopen(serial_store)
    assert reopened.fetch(type(record), sid) == record


def test_parallel_unsupported_shape_allowed_without_verification(store_factory):
    """Disabling verify_metadata lifts the fail-fast metadata restriction."""
    store = store_factory()
    _require_bulk(store)
    with store.bulk_ingest(workers=2, verify_metadata=False) as bulk:
        sid = bulk.save(SelfRefSkip("a", SelfRefSkip("b")))
    reopened = store_factory.reopen(store)
    assert reopened.fetch(SelfRefSkip, bulk.resolved_sid(SelfRefSkip, sid)) == SelfRefSkip("a", SelfRefSkip("b"))


def test_parallel_worker_hard_death_aborts(store_factory):
    """A worker killed mid-ingest is detected promptly and the store is left untouched."""
    store = store_factory()
    _require_bulk(store)
    database = _database_of(store)
    with (
        pytest.raises(RuntimeError, match="exited unexpectedly|without reporting"),
        store.bulk_ingest(workers=2) as bulk,
    ):
        for i in range(5):
            bulk.save(Author(f"A{i}", 1900 + i))
        os.kill(bulk._controller._processes[0].pid, signal.SIGKILL)
        for i in range(2000):  # subsequent dispatches observe the dead worker and abort
            bulk.save(Author(f"B{i}", i))
    assert not _has_application_rows(store, database)


@pytest.mark.parametrize("measure", [(-0.0, 0.0)])
def test_parallel_signed_zero_is_not_a_conflict(store_factory, measure):
    """-0.0 and 0.0 in an identity-excluded float column deduplicate without a conflict (matches serial)."""
    store = store_factory()
    _require_bulk(store)
    with store.bulk_ingest(workers=2) as bulk:
        bulk.save(FloatMeta("k", measure[0]))  # token 0 -> worker 0
        bulk.save(FloatMeta("k", measure[1]))  # token 1 -> worker 1
    assert _physical_counts(_database_of(store))["bulkp_float_meta"] == 1


def test_parallel_nan_metadata_conflicts(store_factory):
    """NaN never equals itself: two NaN identity-excluded floats under one content id conflict (both backends)."""
    store = store_factory()
    _require_bulk(store)
    with pytest.raises(EntryMetadataConflictError, match="measure"), store.bulk_ingest(workers=2) as bulk:
        bulk.save(FloatMeta("k", math.nan))  # token 0 -> worker 0
        bulk.save(FloatMeta("k", math.nan))  # token 1 -> worker 1
    assert not _has_application_rows(store, _database_of(store))


@pytest.mark.parametrize("worker_index", [0, 1])
def test_parquet_untracked_worker_does_not_retain_dedup_indexes(worker_index):
    """Untracked Parquet staging delegates all deduplication to the set-wise finalizer."""
    pytest.importorskip("pyarrow")
    from httk.store.db.bulk_parallel import _WorkerConfig, _WorkerEncoder

    database = Database.duckdb()
    try:
        store = SqlStore(database, entry_records={})
        with tempfile.TemporaryDirectory() as shard_dir:
            encoder = _WorkerEncoder(
                store,
                worker_index,
                _WorkerConfig(
                    chunk_size=1_000,
                    shard_dir=shard_dir,
                    backend="parquet",
                    track_sids=False,
                    spill_deferred_auxiliary=True,
                ),
            )
            for index in range(100):
                encoder.save(index, Author(f"content-{index}", 1900 + index), None)
                encoder.save(100 + index, ByValParent(index, []), None)
            assert sum(map(len, encoder._content_index.values())) == 0
            assert sum(map(len, encoder._value_index.values())) == 0
    finally:
        database.dispose()


def test_parallel_child_values_and_references_match_serial(store_factory):
    """Reconstructed child lists and remapped references equal the serial build's, element for element."""
    serial = store_factory()
    _require_bulk(serial)
    parallel = store_factory()
    samples = [
        make_sample(formula=f"F{i % 3}", authors=[Author("Ada", 1852), Author(f"W{i % 4}", 1900 + i % 4)])
        for i in range(60)
    ]
    for obj in samples:
        serial.save(obj)
    tokens = []
    with parallel.bulk_ingest(workers=4) as bulk:
        for obj in samples:
            tokens.append(bulk.save(obj))

    serial_reopened = store_factory.reopen(serial)
    parallel_reopened = store_factory.reopen(parallel)
    # A directly reconstructed record equals the source, children and reference included.
    chosen = samples[17]
    reconstructed = parallel_reopened.fetch(Sample, bulk.resolved_sid(Sample, tokens[17]))
    assert reconstructed == chosen
    assert [author for author in reconstructed.authors] == chosen.authors  # child reference list, in order
    assert reconstructed.reference == chosen.reference  # remapped scalar reference
    # Whole-store equivalence to the serial build.
    assert _table_stats(parallel, _database_of(parallel)) == _table_stats(serial, _database_of(serial))
    assert serial_reopened.fetch(Sample, 1).authors[0] == Author("Ada", 1852)


def test_parallel_sqlite_file_store_detaches_shards(tmp_path):
    """A parallel build on a pooled file-backed SQLite store detaches its shards and removes the shard directory."""
    path = tmp_path / "store.sqlite"
    database = Database.sqlite(path)
    try:
        store = SqlStore(database, entry_records={})
        with store.bulk_ingest(workers=3) as bulk:
            shard_dir = bulk._controller._temp.name
            for i in range(40):
                bulk.save(Author(f"A{i}", 1900 + i))
        # The shards were detached (on the exact connection that attached them,
        # held until after commit) and their directory removed once the ingest
        # released the connection.
        assert bulk._parallel_attached == []
        assert not os.path.exists(shard_dir)
        # Subsequent serial writes and reads across the pool keep working.
        store.save(Author("extra", 1))
        assert store.fetch(Author, 1) == Author("A0", 1900)
    finally:
        database.dispose()
    reopened_database = Database.sqlite(path)
    try:
        assert SqlStore(reopened_database).fetch(Author, 1) == Author("A0", 1900)
    finally:
        reopened_database.dispose()


def test_parallel_shard_path_with_quote(tmp_path):
    """A database (hence shard) path containing a single quote must not break or inject the merge SQL."""
    directory = tmp_path / "o'brien"
    directory.mkdir()
    for backend, factory in (("sqlite", Database.sqlite), ("duckdb", Database.duckdb)):
        if backend == "duckdb":
            pytest.importorskip("duckdb_engine")
        database = factory(directory / f"store.{backend}")
        try:
            store = SqlStore(database, entry_records={})
            tokens = []
            with store.bulk_ingest(workers=2) as bulk:
                for i in range(20):
                    tokens.append(bulk.save(Author(f"A{i}", 1900 + i)))
            assert store.fetch(Author, bulk.resolved_sid(Author, tokens[5])) == Author("A5", 1905)
        finally:
            database.dispose()


def test_parallel_on_progress_rejected(store_factory):
    """on_progress is not observable in parallel mode and is rejected at construction."""
    store = store_factory()
    _require_bulk(store)
    with pytest.raises(ValueError, match="on_progress"):
        store.bulk_ingest(workers=2, on_progress=lambda records, rows: None)


# --------------------------------------------------------------------- residual-corrections tests


def test_parallel_h2_rejection_survives_catch_then_retry(store_factory):
    """Catching the fail-fast rejection and re-saving the same type still raises (no seen-set bypass)."""
    store = store_factory()
    _require_bulk(store)
    database = _database_of(store)
    with store.bulk_ingest(workers=2) as bulk:
        with pytest.raises(ValueError, match="self-referential"):
            bulk.save(SelfRefSkip("a", SelfRefSkip("b")))
        with pytest.raises(ValueError, match="self-referential"):
            bulk.save(SelfRefSkip("c", SelfRefSkip("d")))  # retry of the same type is re-validated
    # Nothing was created or committed (classification runs before any DDL).
    assert not _has_application_rows(store, database)


def test_parallel_exact_fraction_metadata_conflicts(store_factory):
    """Two Fractions with the same float approximation but different exact value conflict (exact channel kept)."""
    store = store_factory()
    _require_bulk(store)
    big = 2**53  # float(2**53) == float(2**53 + 1): the exact text channel is the only differentiator
    with pytest.raises(EntryMetadataConflictError, match="frac"), store.bulk_ingest(workers=2) as bulk:
        bulk.save(FracMeta("k", Fraction(big)))  # token 0 -> worker 0
        bulk.save(FracMeta("k", Fraction(big + 1)))  # token 1 -> worker 1
    assert not _has_application_rows(store, _database_of(store))


def test_parallel_equal_fraction_metadata_deduplicates(store_factory):
    """Equal exact Fractions under one content id deduplicate without a conflict."""
    store = store_factory()
    _require_bulk(store)
    with store.bulk_ingest(workers=2) as bulk:
        bulk.save(FracMeta("k", Fraction(2**53)))
        bulk.save(FracMeta("k", Fraction(2**53)))
    assert _physical_counts(_database_of(store))["bulkp_frac_meta"] == 1


def test_parallel_finish_aborts_on_dead_worker_with_full_queue(store_factory, monkeypatch):
    """A worker killed with a saturated queue cannot deadlock the stop-sentinel send; the ingest aborts."""
    from httk.store.db import bulk_parallel

    monkeypatch.setattr(bulk_parallel, "_QUEUE_MAXSIZE", 2)
    store = store_factory()
    _require_bulk(store)
    database = _database_of(store)
    with (
        pytest.raises(RuntimeError, match="exited|without reporting|signal completion|abort"),
        store.bulk_ingest(workers=2) as bulk,
    ):
        for i in range(4):
            bulk.save(Author(f"A{i}", 1900 + i))
        controller = bulk._controller
        os.kill(controller._processes[0].pid, signal.SIGKILL)
        # Saturate the dead worker's queue so the stop sentinel cannot be delivered
        # to it — the old blocking sentinel put would hang here forever.
        filled = 0
        while filled < 50:
            try:
                controller._queues[0].put(b"x", timeout=0.2)
                filled += 1
            except queue_mod.Full:
                break
    assert not _has_application_rows(store, database)


def test_parallel_sqlite_works_with_foreign_key_enforcement_enabled(tmp_path):
    """The FK-free physical schema makes parallel merge independent of PRAGMA foreign_keys."""
    from sqlalchemy import event

    database = Database.sqlite(tmp_path / "fk.sqlite")

    @event.listens_for(database.engine, "connect")
    def _enable_foreign_keys(dbapi_connection, _record):
        dbapi_connection.execute("PRAGMA foreign_keys=ON")

    store = SqlStore(database, entry_records={})
    try:
        with store.bulk_ingest(workers=2) as bulk:
            provisional = bulk.save(Author("A", 1900))
        assert bulk.resolved_sid(Author, provisional) == 1
        assert store.fetch(Author, 1) == Author("A", 1900)
    finally:
        database.dispose()


def test_parallel_healthy_manifest_survives_full_queue_health_poll(store_factory):
    """A worker that reported and exited during sentinel delivery keeps its manifest (not discarded nor read as a crash)."""
    import time

    from httk.store.db import bulk_parallel

    store = store_factory()
    _require_bulk(store)
    backend = _database_of(store).engine.dialect.name
    controller = bulk_parallel.ParallelController(store, workers=2, chunk_size=100_000, backend=backend)
    controller.start()
    try:
        controller.dispatch(0, Author("A0", 1900), None)  # routed to worker 0
        controller._queues[0].put(None)  # worker 0 completes its task and exits cleanly
        deadline = time.time() + 30
        while controller._processes[0].exitcode is None and time.time() < deadline:
            time.sleep(0.01)
        assert controller._processes[0].exitcode is not None
        time.sleep(0.1)  # let worker 0's reported result reach the pipe
        # A health poll while worker 0 has exited-and-reported must not abort, and
        # must cache its manifest (the old code discarded it and cried crash).
        controller._raise_if_worker_broken()
        assert 0 in controller._results_cache
        manifests = controller.finish()  # worker 1 takes its sentinel and completes
        assert len(manifests) == 2
        assert any(0 in manifest.token_sid for manifest in manifests)
    finally:
        controller.close()


@pytest.mark.parametrize("run", [0, 1])
def test_parallel_nan_attribution_is_schema_order_first(store_factory, run):
    """When several identity-excluded floats are NaN, the conflict is attributed to the schema-order-first field."""
    store = store_factory()
    _require_bulk(store)
    with pytest.raises(EntryMetadataConflictError, match=r"TwoFloatMeta\.alpha:"), store.bulk_ingest(workers=2) as bulk:
        bulk.save(TwoFloatMeta("k", math.nan, math.nan))  # token 0 -> worker 0
        bulk.save(TwoFloatMeta("k", math.nan, math.nan))  # token 1 -> worker 1
    assert not _has_application_rows(store, _database_of(store))
