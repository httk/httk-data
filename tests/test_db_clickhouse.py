"""P1 ClickHouse engine, layout, DDL, and mutation-policy coverage."""

import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass

import pytest
import sqlalchemy
from conftest import clickhouse_test_uri
from sqlalchemy import text

from httk.store.db import Database, SqlStore
from httk.store.db import clickhouse as clickhouse_adapter
from httk.store.db.clickhouse import (
    actual_columns,
    bootstrap_fence,
    ensure_bootstrap_table,
    keeper_database_uuid,
    keeper_metadata_path,
    validate_metadata_table,
    verify_clickhouse_connection,
)
from httk.store.db.layout import STORAGE_PROTOCOL_VERSION, actual_schema_objects
from httk.store.db.mapping import table_for
from httk.store.db.schema import resolve_schema
from httk.store.db.store import StorageLayoutUpgradeRequiredError


class _RecordingConnection:
    def __init__(self) -> None:
        self.statements = []

    def execute(self, statement):
        self.statements.append(statement)


def test_clickhouse_value_conditioned_deletes_use_keeper_strict_mode() -> None:
    connection = _RecordingConnection()
    clickhouse_adapter.release_lease(connection, "lease-token")
    clickhouse_adapter.clear_ingest_marker(connection, "marker-token")

    assert len(connection.statements) == 2
    for statement in connection.statements:
        assert statement.get_execution_options()["settings"] == {"keeper_map_strict_mode": 1}


@dataclass(frozen=True)
class ClickHouseProbe:
    integer: int
    scalar: float
    title: str
    enabled: bool
    payload: bytes
    note: str | None = None


@pytest.fixture
def clickhouse_database():
    source_url = sqlalchemy.engine.make_url(clickhouse_test_uri())
    database_name = f"httk_p1_{uuid.uuid4().hex}"
    admin = sqlalchemy.create_engine(source_url.set(database="default"))
    with admin.begin() as connection:
        # The bootstrap map is deployment-global and lives in the default
        # database; each test database receives the same deployment DDL below.
        deployment_database = "default"
        present = connection.execute(
            text("SELECT count(*) FROM system.tables WHERE database = :database AND name = '_httk_bootstrap'"),
            {"database": deployment_database},
        ).scalar_one()
    if not present:
        admin.dispose()
        pytest.skip(
            "ClickHouse deployment table _httk_bootstrap is absent; deploy "
            "KeeperMap('/_httk_bootstrap') PRIMARY KEY key before live tests"
        )
    database = None
    try:
        with admin.begin() as connection:
            connection.execute(text(f"CREATE DATABASE {database_name}"))
        target_admin = sqlalchemy.create_engine(source_url.set(database=database_name))
        try:
            with target_admin.begin() as connection:
                connection.execute(
                    text(
                        "CREATE TABLE _httk_bootstrap (key String, value String) "
                        "ENGINE=KeeperMap('/_httk_bootstrap') PRIMARY KEY key"
                    )
                )
        finally:
            target_admin.dispose()
        database = Database.clickhouse(source_url, database=database_name)
        yield database
    finally:
        if database is not None:
            database.dispose()
        with admin.begin() as connection:
            connection.execute(text(f"DROP DATABASE IF EXISTS {database_name}"))
        admin.dispose()


def test_clickhouse_facts_and_keeper_round_trip(clickhouse_database: Database) -> None:
    store = SqlStore(clickhouse_database, entry_records={})
    assert tuple(int(part) for part in clickhouse_database.server_version.split(".")[:4]) >= (26, 8, 1, 1028)
    assert store.write_profile == "bulk-fenced"
    assert store.backend_facts.write_profiles == ("bulk-fenced",)
    assert store.backend_facts.metadata_backend == "keepermap"
    assert not store.backend_facts.supports_incremental_save
    with clickhouse_database.engine.connect() as connection:
        assert dict(connection.execute(text("SELECT key, value FROM _httk_store_metadata")).all()) == {
            "protocol": STORAGE_PROTOCOL_VERSION,
            "entry_declaration": '{"families":[],"format":1}',
            "write_profile": "bulk-fenced",
            "store_timestamps": "v1:1000",
        }
        database_uuid = keeper_database_uuid(connection)
        engine, engine_full = connection.execute(
            text(
                "SELECT engine, engine_full FROM system.tables "
                "WHERE database = currentDatabase() AND name = '_httk_store_metadata'"
            )
        ).one()
        assert engine == "KeeperMap"
        assert engine_full == f"KeeperMap('{keeper_metadata_path(database_uuid)}') PRIMARY KEY key"
        validate_metadata_table(connection)
    reopened = SqlStore(clickhouse_database)
    assert reopened.layout == store.layout


def test_clickhouse_catalog_and_ddl_decoration(clickhouse_database: Database) -> None:
    metadata = sqlalchemy.MetaData()
    table = table_for(resolve_schema(ClickHouseProbe), metadata)
    with clickhouse_database.engine.begin() as connection:
        table.create(connection)
        objects = actual_schema_objects(connection)
        columns = actual_columns(connection, table.name)
        engine, sorting_key = connection.execute(
            text("SELECT engine, sorting_key FROM system.tables WHERE database = currentDatabase() AND name = :name"),
            {"name": table.name},
        ).one()
    assert objects[table.name] == frozenset({"table"})
    assert engine == "MergeTree"
    assert sorting_key == "sid"
    assert columns["sid"] == "Int64"
    assert columns["integer"] == "Int64"
    assert columns["scalar"] == "Float64"
    assert columns["title"] == "String"
    assert columns["enabled"] == "Bool"
    assert columns["payload"] == "String"
    assert columns["note"] == "Nullable(String)"


def test_clickhouse_ddl_decoration_is_dialect_local() -> None:
    pytest.importorskip("clickhouse_connect")
    from sqlalchemy.dialects import sqlite

    metadata = sqlalchemy.MetaData()
    table = sqlalchemy.Table(
        "ddl_byte_stability",
        metadata,
        sqlalchemy.Column("sid", sqlalchemy.Integer, primary_key=True),
        sqlalchemy.Column("label", sqlalchemy.Text, nullable=True),
        sqlalchemy.CheckConstraint("sid > 0", name="ck_ddl_byte_stability"),
    )
    index = sqlalchemy.Index("ix_ddl_byte_stability_label", table.c.label)
    sqlite_dialect = sqlite.dialect()
    duckdb_dialect = sqlalchemy.create_engine("duckdb:///:memory:").dialect
    clickhouse_dialect = sqlalchemy.create_engine("clickhousedb://default:@127.0.0.1:28123/default").dialect
    sqlite_before = str(sqlalchemy.schema.CreateTable(table).compile(dialect=sqlite_dialect))
    duckdb_before = str(sqlalchemy.schema.CreateTable(table).compile(dialect=duckdb_dialect))
    sqlite_index_before = str(sqlalchemy.schema.CreateIndex(index).compile(dialect=sqlite_dialect))
    duckdb_index_before = str(sqlalchemy.schema.CreateIndex(index).compile(dialect=duckdb_dialect))
    sqlalchemy.schema.CreateTable(table).compile(dialect=clickhouse_dialect)
    sqlalchemy.schema.CreateIndex(index).compile(dialect=clickhouse_dialect)
    assert str(sqlalchemy.schema.CreateTable(table).compile(dialect=sqlite_dialect)) == sqlite_before
    assert str(sqlalchemy.schema.CreateTable(table).compile(dialect=duckdb_dialect)) == duckdb_before
    assert str(sqlalchemy.schema.CreateIndex(index).compile(dialect=sqlite_dialect)) == sqlite_index_before
    assert str(sqlalchemy.schema.CreateIndex(index).compile(dialect=duckdb_dialect)) == duckdb_index_before
    assert str(table.c.sid.type) == "INTEGER"
    assert str(table.c.label.type) == "TEXT"
    assert len(table.indexes) == 1
    assert not table.info.get("_httk_clickhouse_decorated")


def test_clickhouse_child_dispatch_order_and_index_absence(clickhouse_database: Database) -> None:
    metadata = sqlalchemy.MetaData()
    parent = sqlalchemy.Table(
        "ddl_parent",
        metadata,
        sqlalchemy.Column("sid", sqlalchemy.Integer, primary_key=True),
        sqlalchemy.Column("value", sqlalchemy.Integer, nullable=False),
        sqlalchemy.CheckConstraint("value >= 0", name="ck_ddl_parent_value"),
    )
    sqlalchemy.Table(
        "ddl_parent_values",
        metadata,
        sqlalchemy.Column("ddl_parent_sid", sqlalchemy.Integer, nullable=False),
        sqlalchemy.Column("value_index", sqlalchemy.Integer, nullable=False),
        sqlalchemy.Column("value", sqlalchemy.Text),
    )
    sqlalchemy.Table(
        "ddl_dispatch",
        metadata,
        sqlalchemy.Column("content_id", sqlalchemy.Text, primary_key=True),
    )
    sqlalchemy.Index("ix_ddl_parent_value", parent.c.value)
    with clickhouse_database.engine.begin() as connection:
        metadata.create_all(connection)
        rows = connection.execute(
            text(
                "SELECT name, sorting_key FROM system.tables "
                "WHERE database = currentDatabase() AND name IN "
                "('ddl_parent', 'ddl_parent_values', 'ddl_dispatch')"
            )
        ).all()
        assert dict(rows) == {
            "ddl_parent": "sid",
            "ddl_parent_values": "ddl_parent_sid, value_index",
            "ddl_dispatch": "content_id",
        }
        assert (
            connection.execute(
                text("SELECT count(*) FROM system.data_skipping_indices WHERE database = currentDatabase()")
            ).scalar_one()
            == 0
        )
        assert (
            connection.execute(
                text(
                    "SELECT count(*) FROM system.tables WHERE database = currentDatabase() "
                    "AND name = 'ddl_parent' AND create_table_query LIKE '%CHECK%'"
                )
            ).scalar_one()
            == 1
        )


def test_clickhouse_metadata_wrong_path_refuses_before_read(clickhouse_database: Database) -> None:
    with clickhouse_database.engine.begin() as connection:
        connection.execute(
            text(
                "CREATE TABLE _httk_store_metadata (key String, value String) "
                "ENGINE=KeeperMap('/wrong-metadata-path') PRIMARY KEY key"
            )
        )
    with pytest.raises(RuntimeError, match="physical validation"):
        SqlStore(clickhouse_database, entry_records={})


def test_clickhouse_metadata_merge_tree_imposter_refuses_before_read(clickhouse_database: Database) -> None:
    with clickhouse_database.engine.begin() as connection:
        connection.execute(
            text("CREATE TABLE _httk_store_metadata (key String, value String) ENGINE=MergeTree ORDER BY key")
        )
    with pytest.raises(RuntimeError, match="physical validation"):
        SqlStore(clickhouse_database, entry_records={})


def test_clickhouse_url_settings_merge_and_failed_open_verification(clickhouse_database: Database) -> None:
    url = clickhouse_database.engine.url.update_query_dict({"max_execution_time": "60"})
    opened = Database.clickhouse(url, database=url.database)
    try:
        assert opened.engine.url.query["max_execution_time"] == "60"
        assert opened.engine.url.query["join_use_nulls"] == "1"
    finally:
        opened.dispose()
    bad_engine = sqlalchemy.create_engine(clickhouse_database.engine.url.update_query_dict({"join_use_nulls": "0"}))
    try:
        with pytest.raises(RuntimeError, match="join_use_nulls"):
            Database(bad_engine)
    finally:
        bad_engine.dispose()


def test_clickhouse_pool_checkout_rechecks_poisoned_connection(clickhouse_database: Database, monkeypatch) -> None:
    import httk.store.db.clickhouse as clickhouse_adapter

    original_verify = clickhouse_adapter._raw_verify
    monkeypatch.setattr(clickhouse_adapter, "_raw_verify", lambda _: ("26.8.1.1028", 0))
    with pytest.raises(RuntimeError, match="join_use_nulls"):
        clickhouse_database.engine.connect()
    monkeypatch.setattr(clickhouse_adapter, "_raw_verify", original_verify)
    with clickhouse_database.engine.connect() as connection:
        assert connection.execute(text("SELECT getSetting('join_use_nulls')")).scalar_one() == 1


def test_database_rejects_unknown_and_backend_absent_profiles() -> None:
    with pytest.raises(ValueError, match="unknown storage write profile"):
        Database(sqlalchemy.create_engine("sqlite://"), write_profile="bogus")  # type: ignore[arg-type]
    pytest.importorskip("duckdb_engine")
    duckdb = sqlalchemy.create_engine("duckdb:///:memory:")
    try:
        with pytest.raises(ValueError, match="not supported"):
            Database(duckdb, write_profile="bulk-fenced")
    finally:
        duckdb.dispose()


@pytest.mark.parametrize("value", ["missing", None, "00000000-0000-0000-0000-000000000000", "not-a-uuid"])
def test_clickhouse_rejects_missing_nil_and_malformed_database_uuid(value: str | None) -> None:
    class Result:
        def one_or_none(self) -> tuple[str | None] | None:
            return None if value == "missing" else (value,)

    class FakeConnection:
        def execute(self, _: object) -> Result:
            return Result()

    with pytest.raises(RuntimeError, match=r"system\.databases\.uuid"):
        keeper_database_uuid(FakeConnection())  # type: ignore[arg-type]


def test_clickhouse_pool_size_one_fence_uses_caller_connection(clickhouse_database: Database) -> None:
    engine = sqlalchemy.create_engine(clickhouse_database.engine.url, pool_size=1, max_overflow=0)
    database = Database(engine, write_profile="bulk-fenced")
    try:
        SqlStore(database, entry_records={})
    finally:
        database.dispose()


def test_clickhouse_concurrent_first_open_converges_without_raw_table_exists(
    clickhouse_database: Database, monkeypatch
) -> None:
    import httk.store.db.clickhouse as clickhouse_adapter

    source_url = clickhouse_database.engine.url
    admin = sqlalchemy.create_engine(source_url.set(database="default"))
    barrier_holder: list[threading.Barrier] = []
    original_fence = clickhouse_adapter.bootstrap_fence

    @contextmanager
    def synchronized_fence(connection):
        barrier_holder[-1].wait(timeout=20)
        with original_fence(connection) as held:
            yield held

    monkeypatch.setattr(clickhouse_adapter, "bootstrap_fence", synchronized_fence)
    try:
        for iteration in range(5):
            database_name = f"httk_p1_race_{uuid.uuid4().hex}"
            first_database = second_database = None
            try:
                with admin.begin() as connection:
                    connection.execute(text(f"CREATE DATABASE {database_name}"))
                target_admin = sqlalchemy.create_engine(source_url.set(database=database_name))
                try:
                    with target_admin.begin() as connection:
                        connection.execute(
                            text(
                                "CREATE TABLE _httk_bootstrap (key String, value String) "
                                "ENGINE=KeeperMap('/_httk_bootstrap') PRIMARY KEY key"
                            )
                        )
                finally:
                    target_admin.dispose()
                first_database = Database.clickhouse(source_url, database=database_name)
                second_database = Database.clickhouse(source_url, database=database_name)
                barrier_holder.append(threading.Barrier(2))
                results: list[BaseException | None] = []
                results_lock = threading.Lock()

                def open_store(
                    database: Database,
                    iteration_results: list[BaseException | None],
                    iteration_lock,
                ) -> None:
                    try:
                        SqlStore(database, entry_records={})
                    except BaseException as error:
                        with iteration_lock:
                            iteration_results.append(error)
                    else:
                        with iteration_lock:
                            iteration_results.append(None)

                first = threading.Thread(target=open_store, args=(first_database, results, results_lock), daemon=True)
                second = threading.Thread(target=open_store, args=(second_database, results, results_lock), daemon=True)
                first.start()
                second.start()
                first.join(timeout=30)
                second.join(timeout=30)
                assert not first.is_alive() and not second.is_alive(), f"race iteration {iteration} hung"
                assert len(results) == 2
                assert all(
                    result is None
                    or isinstance(result, RuntimeError)
                    and ("bootstrap contention" in str(result) or "database UUID" in str(result))
                    for result in results
                ), [(type(result).__name__, str(result), getattr(result, "diff", None)) for result in results]
                assert sum(result is None for result in results) in {1, 2}
                with first_database.engine.connect() as connection:
                    stamps = dict(connection.execute(text("SELECT key, value FROM _httk_store_metadata")).all())
                assert stamps == {
                    "protocol": STORAGE_PROTOCOL_VERSION,
                    "entry_declaration": '{"families":[],"format":1}',
                    "write_profile": "bulk-fenced",
                    "store_timestamps": "v1:1000",
                }
                assert len(stamps) == 4
            finally:
                if first_database is not None:
                    first_database.dispose()
                if second_database is not None:
                    second_database.dispose()
                with admin.begin() as connection:
                    connection.execute(text(f"DROP DATABASE IF EXISTS {database_name}"))
    finally:
        admin.dispose()


def test_clickhouse_drop_recreate_scopes_keeper_metadata_by_new_uuid(clickhouse_database: Database) -> None:
    SqlStore(clickhouse_database, entry_records={})
    old_url = clickhouse_database.engine.url
    with clickhouse_database.engine.connect() as connection:
        old_uuid = keeper_database_uuid(connection)
    clickhouse_database.dispose()
    admin = sqlalchemy.create_engine(old_url.set(database="default"))
    try:
        with admin.begin() as connection:
            connection.execute(text(f"DROP DATABASE {old_url.database}"))
            connection.execute(text(f"CREATE DATABASE {old_url.database}"))
        target_admin = sqlalchemy.create_engine(old_url)
        try:
            with target_admin.begin() as connection:
                connection.execute(
                    text(
                        "CREATE TABLE _httk_bootstrap (key String, value String) "
                        "ENGINE=KeeperMap('/_httk_bootstrap') PRIMARY KEY key"
                    )
                )
        finally:
            target_admin.dispose()
        recreated = Database.clickhouse(old_url, database=old_url.database)
        try:
            with recreated.engine.connect() as connection:
                new_uuid = keeper_database_uuid(connection)
            assert new_uuid != old_uuid
            assert keeper_metadata_path(new_uuid) != keeper_metadata_path(old_uuid)
            SqlStore(recreated, entry_records={})
        finally:
            recreated.dispose()
    finally:
        admin.dispose()


def test_clickhouse_stale_release_cannot_delete_replacement_token(clickhouse_database: Database) -> None:
    import httk.store.db.clickhouse as clickhouse_adapter

    bootstrap = clickhouse_adapter._bootstrap_table()
    with clickhouse_database.engine.begin() as connection:
        key = keeper_database_uuid(connection)
        connection.execute(
            sqlalchemy.insert(bootstrap)
            .values(key=key, value="old-token")
            .execution_options(settings={"keeper_map_strict_mode": 1})
        )
        connection.execute(
            sqlalchemy.delete(bootstrap)
            .where(bootstrap.c.key == key, bootstrap.c.value == "old-token")
            .execution_options(settings={"keeper_map_strict_mode": 1})
        )
        connection.execute(
            sqlalchemy.insert(bootstrap)
            .values(key=key, value="new-token")
            .execution_options(settings={"keeper_map_strict_mode": 1})
        )
        connection.execute(
            sqlalchemy.delete(bootstrap)
            .where(bootstrap.c.key == key, bootstrap.c.value == "old-token")
            .execution_options(settings={"keeper_map_strict_mode": 1})
        )
        assert connection.execute(
            text("SELECT value FROM _httk_bootstrap WHERE key = :key"), {"key": key}
        ).scalar_one() == ("new-token")
        connection.execute(
            sqlalchemy.delete(bootstrap)
            .where(bootstrap.c.key == key, bootstrap.c.value == "new-token")
            .execution_options(settings={"keeper_map_strict_mode": 1})
        )


def test_clickhouse_bootstrap_contention_and_exact_release(clickhouse_database: Database) -> None:
    with clickhouse_database.engine.connect() as first, clickhouse_database.engine.connect() as second:
        database_uuid = keeper_database_uuid(first)
        with (
            bootstrap_fence(first),
            pytest.raises(RuntimeError, match="database UUID.*manual recovery"),
            bootstrap_fence(second),
        ):
            pass
        with clickhouse_database.engine.connect() as probe:
            assert (
                probe.execute(
                    text("SELECT count(*) FROM _httk_bootstrap WHERE key = :key"), {"key": database_uuid}
                ).scalar_one()
                == 0
            )


def test_clickhouse_stale_bootstrap_residue_has_manual_recovery(clickhouse_database: Database) -> None:
    import httk.store.db.clickhouse as clickhouse_adapter

    bootstrap = clickhouse_adapter._bootstrap_table()
    with clickhouse_database.engine.begin() as connection:
        database_uuid = keeper_database_uuid(connection)
        connection.execute(
            sqlalchemy.insert(bootstrap)
            .values(key=database_uuid, value="hard-crash-token")
            .execution_options(settings={"keeper_map_strict_mode": 1})
        )
    try:
        with (
            clickhouse_database.engine.connect() as connection,
            pytest.raises(RuntimeError, match="manual recovery.*hard-crash-token"),
            bootstrap_fence(connection),
        ):
            pass
    finally:
        with clickhouse_database.engine.begin() as connection:
            connection.execute(
                sqlalchemy.delete(bootstrap)
                .where(bootstrap.c.key == database_uuid, bootstrap.c.value == "hard-crash-token")
                .execution_options(settings={"keeper_map_strict_mode": 1})
            )


def test_clickhouse_partial_metadata_stamp_is_rejected(clickhouse_database: Database) -> None:
    with clickhouse_database.engine.begin() as connection:
        database_uuid = keeper_database_uuid(connection)
        path = keeper_metadata_path(database_uuid)
        connection.execute(
            text(
                "CREATE TABLE _httk_store_metadata (key String, value String) "
                f"ENGINE=KeeperMap('{path}') PRIMARY KEY key"
            )
        )
        connection.execute(
            text("INSERT INTO _httk_store_metadata (key, value) VALUES ('protocol', :value)"),
            {"value": STORAGE_PROTOCOL_VERSION},
        )
    with pytest.raises(StorageLayoutUpgradeRequiredError) as error:
        SqlStore(clickhouse_database, entry_records={})
    assert error.value.diff["declaration"]["actual"] is None


def test_clickhouse_bootstrap_wrong_engine_and_absence_refuse(clickhouse_database: Database) -> None:
    with clickhouse_database.engine.begin() as connection:
        connection.execute(text("DROP TABLE _httk_bootstrap"))
        connection.execute(
            text("CREATE TABLE _httk_bootstrap (key String, value String) ENGINE=MergeTree ORDER BY key")
        )
    with pytest.raises(RuntimeError, match="wrong engine"):
        ensure_bootstrap_table(clickhouse_database.engine)
    with clickhouse_database.engine.begin() as connection:
        connection.execute(text("DROP TABLE _httk_bootstrap"))
    with pytest.raises(RuntimeError, match="absent"):
        ensure_bootstrap_table(clickhouse_database.engine)
    with clickhouse_database.engine.begin() as connection:
        connection.execute(
            text(
                "CREATE TABLE _httk_bootstrap (key String, value String) "
                "ENGINE=KeeperMap('/_httk_bootstrap') PRIMARY KEY key"
            )
        )


def test_clickhouse_old_version_rejected() -> None:
    class FakeConnection:
        def execute(self, _: object) -> "FakeConnection":
            return self

        def one(self) -> tuple[str, int]:
            return "26.8.1.1027", 1

    with pytest.raises(RuntimeError, match="too old"):
        verify_clickhouse_connection(FakeConnection())  # type: ignore[arg-type]


@pytest.mark.parametrize("operation", ["save", "ensure_tables", "transaction", "steal_lease"])
def test_clickhouse_mutation_policy_refuses_public_operations(clickhouse_database: Database, operation: str) -> None:
    store = SqlStore(clickhouse_database, entry_records={})
    with pytest.raises(RuntimeError, match="clickhousedb.*bulk-fenced"):
        if operation == "save":
            store.save(ClickHouseProbe(1, 1.0, "x", True, b"x"))
        elif operation == "ensure_tables":
            store.ensure_tables(ClickHouseProbe)
        elif operation == "transaction":
            with store.transaction():
                pass
        else:
            store.steal_lease()


@pytest.mark.parametrize(
    ("repair", "collect_garbage"),
    [(False, False), (False, True), (True, False), (True, True)],
)
def test_clickhouse_fsck_is_refused_in_every_mode(
    clickhouse_database: Database, repair: bool, collect_garbage: bool
) -> None:
    store = SqlStore(clickhouse_database, entry_records={})
    with pytest.raises(RuntimeError, match="fsck.*clickhousedb.*bulk-fenced"):
        store.fsck(repair=repair, collect_garbage=collect_garbage)


@pytest.mark.parametrize("finalize", ["parity", "auto", "deferred"])
def test_clickhouse_bulk_profile_selection(clickhouse_database: Database, finalize: str) -> None:
    store = SqlStore(clickhouse_database, entry_records={})
    if finalize == "parity":
        with pytest.raises(RuntimeError, match="deferred-only"):
            store.bulk_ingest(finalize=finalize).__enter__()
    else:
        with store.bulk_ingest(finalize=finalize) as bulk:
            assert bulk._finalize_profile == "deferred"


def test_clickhouse_bulk_into_nonempty_store_is_refused(clickhouse_database: Database) -> None:
    store = SqlStore(clickhouse_database, entry_records={})
    with clickhouse_database.engine.begin() as connection:
        connection.execute(text("CREATE TABLE nonempty (value Int64) ENGINE=MergeTree ORDER BY value"))
        connection.execute(text("INSERT INTO nonempty VALUES (1)"))
    with (
        pytest.raises(RuntimeError, match="bulk_ingest.*clickhousedb.*bulk-fenced"),
        store.bulk_ingest(finalize="deferred"),
    ):
        pass


def test_clickhouse_unversioned_database_is_rejected(clickhouse_database: Database) -> None:
    with clickhouse_database.engine.begin() as connection:
        connection.execute(text("CREATE TABLE unversioned (value Int64) ENGINE=MergeTree ORDER BY value"))
    with pytest.raises(StorageLayoutUpgradeRequiredError) as error:
        SqlStore(clickhouse_database, entry_records={})
    assert error.value.diff["schema"]["unversioned"]["unversioned"] is True
