"""P3 ClickHouse deferred-build equivalence and failure-closed coverage."""

import datetime
import math
import uuid
from contextlib import contextmanager

import pytest
import sqlalchemy
from conftest import clickhouse_test_uri
from sqlalchemy import text
from test_db_bulk import Author, ByValParent, _stream, _table_stats
from test_db_bulk_deferred import (
    CALC_FAMILY,
    Leaf,
    MutualValueA,
    MutualValueB,
    NoneRec,
    OptionalChildRoundTrip,
    PrivateNoneParent,
    Root,
    _child_graph,
    _dispatch_graph,
    _logical_rows,
    _reference_graph,
)
from test_db_bulk_parallel import TwoFloatMeta

from httk.store.backend.sql import Backend, SqlStore
from httk.store.backend.sql.bulk import BulkIngest
from httk.store.backend.clickhouse.support import ClickHouseBulkIntegrityError
from httk.store.backend.sql.layout import StoreUnderConstructionError
from httk.store.store_common import EntryMetadataConflictError


def _rows_without_batch_timestamp(store) -> dict[str, list[tuple[object, ...]]]:
    """``_logical_rows`` with the store-managed ``store_timestamp`` column dropped.

    Two independent ingests each stamp their own batch timestamp, so that column
    is a legitimate replay variant; row identity/count/content is what proves the
    exactly-once landing.
    """
    rows = _logical_rows(store)
    result: dict[str, list[tuple[object, ...]]] = {}
    for name, table in store._metadata.tables.items():
        if name not in rows:
            continue
        drop = {index for index, column in enumerate(table.columns) if column.name == "store_timestamp"}
        result[name] = [tuple(value for index, value in enumerate(row) if index not in drop) for row in rows[name]]
    return result


@contextmanager
def _clickhouse_bulk_database(uri):
    """Yield one isolated ClickHouse database with its deployment bootstrap table."""
    source = sqlalchemy.engine.make_url(uri)
    name = f"httk_p3_bulk_{uuid.uuid4().hex}"
    admin = sqlalchemy.create_engine(source.set(database="default"))
    database = None
    try:
        with admin.begin() as connection:
            present = connection.execute(
                text("SELECT count() FROM system.tables WHERE database = 'default' AND name = '_httk_bootstrap'")
            ).scalar_one()
            if not present:
                pytest.skip("ClickHouse deployment table _httk_bootstrap is absent")
            connection.execute(text(f"CREATE DATABASE {name}"))
        bootstrap = sqlalchemy.create_engine(source.set(database=name))
        try:
            with bootstrap.begin() as connection:
                connection.execute(
                    text(
                        "CREATE TABLE _httk_bootstrap (key String, value String) "
                        "ENGINE=KeeperMap('/_httk_bootstrap') PRIMARY KEY key"
                    )
                )
        finally:
            bootstrap.dispose()
        database = Backend.clickhouse(source, database=name)
        yield database
    finally:
        if database is not None:
            database.dispose()
        with admin.begin() as connection:
            connection.execute(text(f"DROP DATABASE IF EXISTS {name}"))
        admin.dispose()


@pytest.fixture
def clickhouse_bulk_database():
    with _clickhouse_bulk_database(clickhouse_test_uri()) as database:
        yield database


def _dense(store: SqlStore, database: Backend) -> None:
    with database.engine.connect() as connection:
        for name, table in store._metadata.tables.items():
            if "sid" not in table.c:
                continue
            count, low, high = connection.execute(text(f'SELECT count(), min(sid), max(sid) FROM "{name}"')).one()
            assert not count or (low, high) == (1, count)


@pytest.mark.extended
@pytest.mark.parametrize("workers", [1, 2])
def test_clickhouse_deferred_equivalent_to_duckdb(clickhouse_bulk_database, workers):
    """The shared mixed deferred corpus has backend-independent logical output."""
    pytest.importorskip("duckdb_engine")
    duck_database = Backend.duckdb()
    try:
        duck = SqlStore(duck_database, entry_records=CALC_FAMILY)
        click = SqlStore(clickhouse_bulk_database, entry_records=CALC_FAMILY)
        with duck.bulk_ingest(workers=workers, finalize="deferred") as bulk:
            for value in _stream():
                bulk.save(value)
        click_sids = []
        with click.bulk_ingest(workers=workers, finalize="deferred") as bulk:
            for value in _stream():
                click_sids.append(bulk.save(value))
        assert _table_stats(duck, duck_database) == _table_stats(click, clickhouse_bulk_database)
        assert _reference_graph(duck) == _reference_graph(click)
        assert _child_graph(duck) == _child_graph(click)
        assert _dispatch_graph(duck) == _dispatch_graph(click)
        _dense(duck, duck_database)
        _dense(click, clickhouse_bulk_database)
        reopened = SqlStore(
            Backend.clickhouse(
                clickhouse_bulk_database.engine.url, database=clickhouse_bulk_database.engine.url.database
            ),
            entry_records=CALC_FAMILY,
        )
        try:
            for index, value in enumerate(_stream()):
                assert reopened.fetch(type(value), bulk.resolved_sid(type(value), click_sids[index])) == value
            assert (
                reopened.fetch(type(_stream()[11]), bulk.resolved_sid(type(_stream()[11]), click_sids[11])).notes
                is None
            )
            assert (
                reopened.fetch(type(_stream()[12]), bulk.resolved_sid(type(_stream()[12]), click_sids[12])).notes == []
            )
        finally:
            reopened._database.dispose()
    finally:
        duck_database.dispose()


@pytest.mark.extended
def test_clickhouse_nan_conflict_and_mutual_fixpoint(clickhouse_bulk_database):
    """NaN conflict semantics and cyclic by-value fixpoints match deferred DuckDB."""
    click = SqlStore(clickhouse_bulk_database, entry_records={})
    with pytest.raises(EntryMetadataConflictError, match="TwoFloatMeta.alpha"), click.bulk_ingest(workers=2) as bulk:
        bulk.save(TwoFloatMeta("key", math.nan, math.nan))
        bulk.save(TwoFloatMeta("key", math.nan, math.nan))
    leaf_one, leaf_two = MutualValueA(0), MutualValueA(0)
    roots = [MutualValueA(2, MutualValueB(1, leaf_one)), MutualValueA(2, MutualValueB(1, leaf_two))]
    with click.bulk_ingest(workers=2) as bulk:
        sids = [bulk.save(value) for value in roots]
    with clickhouse_bulk_database.engine.connect() as connection:
        assert connection.execute(text("SELECT count() FROM bulk_deferred_mutual_a")).scalar_one() == 2
        assert connection.execute(text("SELECT count() FROM bulk_deferred_mutual_b")).scalar_one() == 1
    reopened = SqlStore(
        Backend.clickhouse(clickhouse_bulk_database.engine.url, database=clickhouse_bulk_database.engine.url.database),
        entry_records={},
    )
    try:
        assert reopened.fetch(MutualValueA, bulk.resolved_sid(MutualValueA, sids[0])) == roots[0]
        assert reopened.fetch(MutualValueA, bulk.resolved_sid(MutualValueA, sids[1])) == roots[1]
    finally:
        reopened._database.dispose()


def test_clickhouse_marker_residue_rejects_reopen(clickhouse_bulk_database):
    store = SqlStore(clickhouse_bulk_database, entry_records={})
    bulk = store.bulk_ingest(finalize="deferred")
    bulk.__enter__()
    try:
        with pytest.raises(StoreUnderConstructionError):
            SqlStore(
                Backend.clickhouse(
                    clickhouse_bulk_database.engine.url, database=clickhouse_bulk_database.engine.url.database
                )
            )
    finally:
        bulk._release_connection(bulk._connection)
        bulk._connection = None
        bulk._release_bulk_ownership()
        store._release_bulk_context()


@pytest.mark.parametrize("workers", [1, 2])
def test_clickhouse_untracked_save_returns_provisional_value_without_resolution(clickhouse_bulk_database, workers):
    store = SqlStore(clickhouse_bulk_database, entry_records={})
    with store.bulk_ingest(workers=workers, track_sids=False) as bulk:
        sid = bulk.save(MutualValueA(1))
    assert isinstance(sid, int)
    with pytest.raises(RuntimeError, match="track_sids=False"):
        bulk.resolved_sid(MutualValueA, sid)


def test_clickhouse_serial_untracked_parquet_stage_keeps_no_dedup_indexes(clickhouse_bulk_database):
    """One hundred unique content/by-value records leave no client dedup map in bounded mode."""
    store = SqlStore(clickhouse_bulk_database, entry_records={})
    with store.bulk_ingest(workers=1, track_sids=False) as bulk:
        for index in range(100):
            bulk.save(Author(f"author-{index}", 1900 + index))
            bulk.save(ByValParent(index, []))
        assert bulk._serial_stage is not None
        encoder = bulk._serial_stage._encoder
        assert sum(map(len, encoder._content_index.values())) == 0
        assert sum(map(len, encoder._value_index.values())) == 0


class _InjectedCrash(BaseException):
    """Model process loss: marker recovery must remain fail-closed."""


@pytest.mark.extended
@pytest.mark.parametrize(
    "seam",
    [
        "_after_clickhouse_stage_load",
        "_after_clickhouse_projection",
        "_after_clickhouse_cleanup",
        "_after_clickhouse_physical_validation",
        "_before_clickhouse_marker_clear",
    ],
)
def test_clickhouse_crash_seams_preserve_marker(clickhouse_bulk_database, monkeypatch, seam):
    store = SqlStore(clickhouse_bulk_database, entry_records={})

    def crash(*_args):
        raise _InjectedCrash(seam)

    monkeypatch.setattr(BulkIngest, seam, crash)
    with pytest.raises(_InjectedCrash), store.bulk_ingest(workers=1) as bulk:
        bulk.save(MutualValueA(1))
    with clickhouse_bulk_database.engine.connect() as connection:
        assert connection.execute(
            text("SELECT count() FROM _httk_store_metadata WHERE key = 'ingest_state'")
        ).scalar_one()
    fresh = Backend.clickhouse(
        clickhouse_bulk_database.engine.url, database=clickhouse_bulk_database.engine.url.database
    )
    try:
        with pytest.raises(StoreUnderConstructionError):
            SqlStore(fresh)
    finally:
        fresh.dispose()


@pytest.mark.extended
@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("content", "content_id uniqueness"),
        ("sid", "sid uniqueness/density"),
        ("role", "_httk_role domain"),
        ("dispatch", "dispatch exactly-one"),
        ("dispatch_pk", r"unique \('content_id',\)"),
    ],
)
def test_clickhouse_generated_integrity_rejects_raw_corruption(clickhouse_bulk_database, monkeypatch, case, message):
    """The post-projection seam proves every non-enforcing ClickHouse invariant."""
    store = SqlStore(clickhouse_bulk_database, entry_records=CALC_FAMILY)

    def corrupt() -> None:
        with clickhouse_bulk_database.engine.begin() as connection:
            if case == "content":
                connection.execute(
                    text(
                        "INSERT INTO bulk_author "
                        "(sid, _httk_role, store_timestamp, logical_id, content_id, name, year) "
                        "SELECT 4, _httk_role, store_timestamp, logical_id, content_id, name, year "
                        "FROM bulk_author LIMIT 1"
                    )
                )
            elif case == "sid":
                connection.execute(text("ALTER TABLE bulk_author DELETE WHERE sid = 1 SETTINGS mutations_sync = 1"))
            elif case == "role":
                connection.execute(
                    text("ALTER TABLE bulk_author UPDATE _httk_role = 7 WHERE sid = 1 SETTINGS mutations_sync = 1")
                )
            elif case == "dispatch":
                dispatch = next(name for name in store._metadata.tables if name.startswith("_httk_entry_dispatch_"))
                columns = [column.name for column in store._table(dispatch).columns if column.name.endswith("_sid")]
                connection.execute(
                    text(
                        f'ALTER TABLE "{dispatch}" UPDATE "{columns[0]}" = 991, "{columns[1]}" = 992 '
                        "WHERE content_id = (SELECT content_id FROM "
                        f'"{dispatch}" LIMIT 1) SETTINGS mutations_sync = 1'
                    )
                )
            else:
                dispatch = next(name for name in store._metadata.tables if name.startswith("_httk_entry_dispatch_"))
                columns = [column.name for column in store._table(dispatch).columns if column.name.endswith("_sid")]
                key = connection.execute(text(f'SELECT content_id FROM "{dispatch}" LIMIT 1')).scalar_one()
                connection.execute(
                    text(
                        f'INSERT INTO "{dispatch}" (content_id, "{columns[0]}", "{columns[1]}") '
                        "VALUES (:key, 2147483647, NULL)"
                    ),
                    {"key": key},
                )

    monkeypatch.setattr(BulkIngest, "_before_clickhouse_integrity_verification", lambda _: corrupt())
    with pytest.raises(ClickHouseBulkIntegrityError, match=message), store.bulk_ingest(workers=1) as bulk:
        for value in _stream():
            bulk.save(value)
    with clickhouse_bulk_database.engine.connect() as connection:
        assert connection.execute(
            text("SELECT count() FROM _httk_store_metadata WHERE key = 'ingest_state'")
        ).scalar_one()
    fresh = Backend.clickhouse(
        clickhouse_bulk_database.engine.url, database=clickhouse_bulk_database.engine.url.database
    )
    try:
        with pytest.raises(StoreUnderConstructionError):
            SqlStore(fresh)
    finally:
        fresh.dispose()


@pytest.mark.extended
@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("default", "column types/nullability/defaults"),
        ("metadata_engine", "metadata table _httk_store_metadata failed physical validation"),
    ],
)
def test_clickhouse_physical_validation_rejects_defaults_and_metadata_shape(
    clickhouse_bulk_database, monkeypatch, case, message
):
    """The final pre-marker-clear pass rejects catalog corruption, not just row corruption."""
    store = SqlStore(clickhouse_bulk_database, entry_records={})

    def corrupt() -> None:
        with clickhouse_bulk_database.engine.begin() as connection:
            if case == "default":
                connection.execute(text("ALTER TABLE bulk_deferred_mutual_a MODIFY COLUMN value Int64 DEFAULT 0"))
            else:
                connection.execute(text("DROP TABLE _httk_store_metadata"))
                connection.execute(
                    text("CREATE TABLE _httk_store_metadata (key String, value String) ENGINE = MergeTree ORDER BY key")
                )

    monkeypatch.setattr(BulkIngest, "_after_clickhouse_cleanup", lambda _: corrupt())
    with pytest.raises(RuntimeError, match=message), store.bulk_ingest(workers=1) as bulk:
        bulk.save(MutualValueA(1))


class _DisconnectingLoader:
    """A real loader client which drops exactly one response before or after send."""

    def __init__(self, client, target: str, lands: bool):
        self._client = client
        self._target = target
        self._lands = lands
        self._triggered = False

    def query(self, *args, **kwargs):
        return self._client.query(*args, **kwargs)

    def insert_arrow(self, table, arrow):
        if table == self._target and not self._triggered:
            self._triggered = True
            if self._lands:
                self._client.insert_arrow(table, arrow)
            raise OSError("simulated response disconnect")
        return self._client.insert_arrow(table, arrow)

    def close(self):
        self._client.close()


@pytest.mark.extended
@pytest.mark.parametrize(
    "target",
    ["_httk_stage_bulk_optional_child", "_httk_stage_bulk_optional_child_notes", "_httk_stage__httk_roots"],
)
@pytest.mark.parametrize("lands", [True, False], ids=["accepted", "not-landed"])
def test_clickhouse_loader_disconnect_is_exactly_once(clickhouse_bulk_database, monkeypatch, target, lands):
    """Count verification neither replays accepted parent/child/root shards nor loses a rejected one."""
    from httk.store.backend.clickhouse import support as clickhouse

    uri = clickhouse_bulk_database.engine.url
    corpus = [OptionalChildRoundTrip("row", ["child"])]
    with _clickhouse_bulk_database(uri) as reference_database:
        reference = SqlStore(reference_database, entry_records={})
        with reference.bulk_ingest(workers=1) as bulk:
            for value in corpus:
                bulk.save(value)
        expected = _rows_without_batch_timestamp(reference)

        original_client = clickhouse._client_for_url

        def disconnecting_client(url):
            return _DisconnectingLoader(original_client(url), target, lands)

        monkeypatch.setattr(clickhouse, "_client_for_url", disconnecting_client)
        store = SqlStore(clickhouse_bulk_database, entry_records={})
        with store.bulk_ingest(workers=1) as bulk:
            for value in corpus:
                bulk.save(value)
        # Row identity/count/content proves exactly-once; the store-managed
        # store_timestamp is excluded because a separate ingest legitimately
        # stamps its own batch timestamp (a replay would re-stamp it too).
        assert _rows_without_batch_timestamp(store) == expected


@pytest.mark.extended
@pytest.mark.parametrize("workers", [1, 2])
@pytest.mark.parametrize("chunk_size", [1, None], ids=["one", "default"])
def test_deferred_role_promotion_is_max_across_chunks_workers_and_backends(
    clickhouse_bulk_database, monkeypatch, workers, chunk_size
):
    """A record first nested then saved as a root keeps role=1 across explicit/default flush boundaries."""
    pytest.importorskip("duckdb_engine")
    effective_chunk_size = 3 if chunk_size is None else chunk_size
    if chunk_size is None:
        # Exercise the public default-argument route without constructing
        # 100,000 filler objects solely to trip its production-sized boundary.
        monkeypatch.setitem(SqlStore.bulk_ingest.__kwdefaults__, "chunk_size", effective_chunk_size)
    duck_database = Backend.duckdb()
    leaf = Leaf(9, "metadata")
    root = Root("outer", leaf, [leaf], datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC))
    try:
        for database in (duck_database, clickhouse_bulk_database):
            store = SqlStore(database, entry_records={})
            options = {"workers": workers, "finalize": "deferred"}
            if chunk_size is not None:
                options["chunk_size"] = chunk_size
            with store.bulk_ingest(**options) as bulk:
                bulk.save(root)  # leaf is nested first (role 0)
                for _ in range(effective_chunk_size * workers - 1):
                    bulk.save(Leaf(99, "filler"))
                leaf_sid = bulk.save(leaf)  # root occurrence crosses both chunk and worker boundaries
            with database.engine.connect() as connection:
                assert connection.execute(text("SELECT _httk_role FROM bulk_leaf WHERE value = 9")).scalar_one() == 1
            reopened = SqlStore(database)
            assert reopened.fetch(type(leaf), bulk.resolved_sid(type(leaf), leaf_sid)) == leaf
    finally:
        duck_database.dispose()


@pytest.mark.extended
@pytest.mark.parametrize("workers", [1, 2])
def test_clickhouse_two_nan_diagnostics_are_one_occurrence_not_a_conflict(clickhouse_bulk_database, workers):
    """Two skipped NaN fields in one row are diagnostics for one occurrence, not a duplicate."""
    store = SqlStore(clickhouse_bulk_database, entry_records={})
    value = TwoFloatMeta("one", math.nan, math.nan)
    with store.bulk_ingest(workers=workers) as bulk:
        bulk.save(value)
    with clickhouse_bulk_database.engine.connect() as connection:
        assert connection.execute(text("SELECT count() FROM bulkp_two_float_meta")).scalar_one() == 1


@pytest.mark.extended
@pytest.mark.parametrize("boundary", ["before-rename", "after-rename", "before-drop", "after-drop"])
def test_clickhouse_map_swap_boundaries_preserve_marker_and_clean_relations(
    clickhouse_bulk_database, monkeypatch, boundary
):
    """Every durable map-swap boundary remains recoverable and leaves no working map behind."""
    store = SqlStore(clickhouse_bulk_database, entry_records={})

    def crash(_: BulkIngest, _table: str, seen: str) -> None:
        if seen == boundary:
            raise _InjectedCrash(boundary)

    monkeypatch.setattr(BulkIngest, "_before_clickhouse_map_swap", crash)
    with pytest.raises(_InjectedCrash, match=boundary), store.bulk_ingest(workers=2) as bulk:
        bulk.save(MutualValueA(1))
        bulk.save(MutualValueA(1))
    with clickhouse_bulk_database.engine.connect() as connection:
        assert connection.execute(
            text("SELECT count() FROM _httk_store_metadata WHERE key = 'ingest_state'")
        ).scalar_one()
        assert (
            connection.execute(
                text(
                    "SELECT count() FROM system.tables WHERE database = currentDatabase() AND name LIKE '_httk_deferred_%'"
                )
            ).scalar_one()
            == 0
        )


@pytest.mark.extended
def test_duckdb_manifest_roots_and_parquet_spilled_roots_keep_the_same_orphans(monkeypatch):
    """The root-sidecar scale path preserves the legacy manifest-root survivor set."""
    pytest.importorskip("duckdb_engine")
    from httk.store.backend.sql.bulk_parallel import ParallelController

    stream = [
        PrivateNoneParent("same", [NoneRec("kept")]),
        PrivateNoneParent("same", [NoneRec("orphan")]),
        PrivateNoneParent("other", [NoneRec("other")]),
    ]

    def build(*, spill: bool):
        database = Backend.duckdb()
        store = SqlStore(database, entry_records={})
        try:
            store._clock = lambda: 1_700_000_000_000_000_000
            if spill:
                with store.bulk_ingest(workers=2, chunk_size=1, finalize="deferred", verify_metadata=False) as bulk:
                    for value in stream:
                        bulk.save(value)
            else:
                original = ParallelController.__init__

                def manifest_roots(self, *args, **kwargs):
                    kwargs["spill_deferred_auxiliary"] = False
                    original(self, *args, **kwargs)

                with monkeypatch.context() as scoped:
                    scoped.setattr(ParallelController, "__init__", manifest_roots)
                    with store.bulk_ingest(workers=2, chunk_size=1, finalize="deferred", verify_metadata=False) as bulk:
                        for value in stream:
                            bulk.save(value)
            return _logical_rows(store)
        finally:
            database.dispose()

    assert build(spill=False) == build(spill=True)
