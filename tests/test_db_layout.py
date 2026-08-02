"""Focused SqlStore protocol/layout and entry-family dispatch coverage."""

import threading
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import pytest
import sqlalchemy
from httk.core import StorageInfo, content_id, register_entry_backing, register_entry_family
from sqlalchemy.exc import IntegrityError

import httk.data.db.store as store_module
from httk.data.db import (
    STORAGE_PROTOCOL_VERSION,
    Database,
    SqlStore,
    StorageLayoutUpgradeRequiredError,
)
from httk.data.db.layout import (
    METADATA_TABLE_NAME,
    StorageLayout,
    actual_schema_objects,
    actual_table_names,
    expected_metadata,
    normalize_entry_backings,
    validate_expected_tables,
)
from httk.data.db.mapping import DISPATCH_CONTENT_ID_COLUMN, backing_dispatch_column_name
from httk.data.db.store import EntryDispatchIntegrityError


class LayoutFamily:
    """Registered test entry family with one concrete backing."""


class MultiLayoutFamily:
    """Registered test entry family with two concrete backings."""


@dataclass(frozen=True)
class LayoutSingle:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="layout_single")

    value: str


@dataclass(frozen=True)
class LayoutFirst:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="layout_first")

    value: str


@dataclass(frozen=True)
class LayoutSecond:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="layout_second")

    value: int


@dataclass(frozen=True)
class PrivateLayoutRecord:
    value: str


@dataclass(frozen=True)
class CheckRecord:
    value: str


@dataclass(frozen=True)
class WeirdNamedRecord:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="weird(name")

    value: str


class UnregisteredFamily:
    pass


register_entry_family(name="test-layout-single-family", family=f"{__name__}:LayoutFamily")
register_entry_backing(
    name="test-layout-single-backing",
    family_name="test-layout-single-family",
    record=f"{__name__}:LayoutSingle",
)
register_entry_family(name="test-layout-multi-family", family=f"{__name__}:MultiLayoutFamily")
register_entry_backing(
    name="test-layout-first-backing",
    family_name="test-layout-multi-family",
    record=f"{__name__}:LayoutFirst",
)
register_entry_backing(
    name="test-layout-second-backing",
    family_name="test-layout-multi-family",
    record=f"{__name__}:LayoutSecond",
)


@pytest.fixture
def database() -> Iterator[Database]:
    with Database.sqlite() as database:
        yield database


def _tables(database: Database) -> set[str]:
    with database.engine.connect() as connection:
        return set(connection.execute(sqlalchemy.text("SELECT name FROM sqlite_master WHERE type = 'table'")).scalars())


def _multi_layout() -> tuple[StorageLayout, sqlalchemy.MetaData, sqlalchemy.Table]:
    layout = normalize_entry_backings({MultiLayoutFamily: (LayoutFirst, LayoutSecond)})
    metadata = expected_metadata(layout)
    dispatch_name = layout.families[0].dispatch_table_name
    assert dispatch_name is not None
    return layout, metadata, metadata.tables[dispatch_name]


def test_empty_database_requires_declaration_and_verify_is_read_only(database: Database) -> None:
    with pytest.raises(TypeError, match="entry_backings"):
        SqlStore(database)
    with pytest.raises(StorageLayoutUpgradeRequiredError) as error:
        SqlStore(database, entry_backings={}, layout_mode="verify")
    assert set(error.value.diff) == {"protocol", "declaration", "schema"}
    assert not _tables(database)

    created = SqlStore(database, entry_backings={})
    assert created.entry_layout == ()
    reopened = SqlStore(database)
    assert reopened.entry_layout == ()
    assert STORAGE_PROTOCOL_VERSION == "v2.0.2"


def test_protocol_and_explicit_declaration_mismatches_have_structured_diffs(database: Database) -> None:
    SqlStore(database, entry_backings={})
    with database.engine.begin() as connection:
        connection.execute(sqlalchemy.text("UPDATE _httk_store_metadata SET value = 'old' WHERE key = 'protocol'"))
    with pytest.raises(StorageLayoutUpgradeRequiredError) as error:
        SqlStore(database)
    assert error.value.diff["protocol"] == {"expected": STORAGE_PROTOCOL_VERSION, "actual": "old"}

    with Database.sqlite() as second_database:
        SqlStore(second_database, entry_backings={LayoutFamily: LayoutSingle})
        with pytest.raises(StorageLayoutUpgradeRequiredError) as mismatch:
            SqlStore(second_database, entry_backings={})
        assert "declaration" in mismatch.value.diff

    with Database.sqlite() as malformed_database:
        SqlStore(malformed_database, entry_backings={})
        with malformed_database.engine.begin() as connection:
            connection.execute(
                sqlalchemy.text("UPDATE _httk_store_metadata SET value = '{' WHERE key = 'entry_declaration'")
            )
        with pytest.raises(StorageLayoutUpgradeRequiredError) as malformed:
            SqlStore(malformed_database)
        assert malformed.value.diff["declaration"]["actual"] == "{"

    with Database.sqlite() as overlap_database:
        SqlStore(overlap_database, entry_backings={LayoutFamily: LayoutSingle})
        with overlap_database.engine.begin() as connection:
            connection.execute(
                sqlalchemy.text(
                    "UPDATE _httk_store_metadata SET value = '[\"layout_single\"]' WHERE key = 'unclaimed_tables'"
                )
            )
        with pytest.raises(StorageLayoutUpgradeRequiredError) as overlap:
            SqlStore(overlap_database)
        assert overlap.value.diff["declaration"]["unclaimed_tables"]["overlap_with_protocol_tables"] == (
            "layout_single",
        )


def test_nonempty_unversioned_database_is_never_adopted(database: Database) -> None:
    metadata = expected_metadata(normalize_entry_backings({LayoutFamily: LayoutSingle}))
    with database.engine.begin() as connection:
        for table in metadata.sorted_tables:
            if table.name != METADATA_TABLE_NAME:
                table.create(connection)
        connection.execute(sqlalchemy.text("CREATE TABLE unrelated_layout_table (value INTEGER)"))
    tables_before = _tables(database)
    with pytest.raises(StorageLayoutUpgradeRequiredError) as explicit:
        SqlStore(database, entry_backings={LayoutFamily: LayoutSingle})
    assert explicit.value.diff["protocol"] == {"expected": STORAGE_PROTOCOL_VERSION, "actual": None}
    assert explicit.value.diff["schema"]["layout_single"]["unversioned"] is True
    assert explicit.value.diff["schema"]["unrelated_layout_table"]["unversioned"] is True
    assert _tables(database) == tables_before

    with pytest.raises(StorageLayoutUpgradeRequiredError) as implicit:
        SqlStore(database)
    assert implicit.value.diff["declaration"] == {"expected": "explicit entry_backings", "actual": None}
    assert _tables(database) == tables_before

    with Database.sqlite() as collision_database:
        with collision_database.engine.begin() as connection:
            connection.execute(sqlalchemy.text("CREATE TABLE private_layout_record (value TEXT)"))
        with pytest.raises(StorageLayoutUpgradeRequiredError) as collision:
            SqlStore(collision_database, entry_backings={})
        assert collision.value.diff["schema"]["private_layout_record"]["unversioned"]
        assert METADATA_TABLE_NAME not in _tables(collision_database)


@pytest.mark.parametrize("dialect", ("sqlite", "duckdb"))
def test_view_only_unversioned_database_is_not_physically_empty(dialect: str) -> None:
    if dialect == "duckdb":
        pytest.importorskip("duckdb_engine")
        database = Database.duckdb()
    else:
        database = Database.sqlite()
    with database:
        with database.engine.begin() as connection:
            connection.execute(sqlalchemy.text("CREATE VIEW preexisting_view AS SELECT 7 AS value"))
            assert actual_schema_objects(connection)["preexisting_view"] == frozenset(("view",))
        for mode in ("ensure", "verify"):
            with pytest.raises(StorageLayoutUpgradeRequiredError) as error:
                SqlStore(database, entry_backings={}, layout_mode=mode)
            assert error.value.diff["schema"]["preexisting_view"] == {
                "unversioned": True,
                "object_type": "view",
                "message": "a nonempty database without SqlStore metadata cannot be adopted",
            }
        with database.engine.connect() as connection:
            assert connection.execute(sqlalchemy.text("SELECT value FROM preexisting_view")).scalar_one() == 7
            assert METADATA_TABLE_NAME not in actual_schema_objects(connection)


def test_duckdb_schema_discovery_preserves_table_sequence_name_collisions() -> None:
    pytest.importorskip("duckdb_engine")
    with Database.duckdb() as database:
        with database.engine.begin() as connection:
            connection.execute(sqlalchemy.text("CREATE TABLE same_name (value INTEGER)"))
            connection.execute(sqlalchemy.text("CREATE SEQUENCE same_name"))
            objects = actual_schema_objects(connection)
            tables = actual_table_names(connection)
        assert objects["same_name"] == frozenset(("sequence", "table"))
        assert "same_name" in tables


def test_marked_duckdb_rejects_sequence_sharing_managed_table_name() -> None:
    pytest.importorskip("duckdb_engine")
    with Database.duckdb() as database:
        SqlStore(database, entry_backings={})
        with database.engine.begin() as connection:
            connection.execute(sqlalchemy.text(f"CREATE SEQUENCE {METADATA_TABLE_NAME}"))
        with pytest.raises(StorageLayoutUpgradeRequiredError) as error:
            SqlStore(database)
        assert error.value.diff["schema"][METADATA_TABLE_NAME] == {
            "object_type": ("sequence", "table"),
            "message": "a managed table name is also used by an unexpected schema object",
        }


def test_unversioned_rows_are_refused_before_content_identity_is_trusted(database: Database) -> None:
    metadata = expected_metadata(normalize_entry_backings({LayoutFamily: LayoutSingle}))
    with database.engine.begin() as connection:
        for table in metadata.sorted_tables:
            if table.name != METADATA_TABLE_NAME:
                table.create(connection)
        connection.execute(sqlalchemy.text("INSERT INTO layout_single (content_id, value) VALUES ('wrong', 'value')"))
    with pytest.raises(StorageLayoutUpgradeRequiredError) as error:
        SqlStore(database, entry_backings={LayoutFamily: LayoutSingle})
    assert error.value.diff["schema"]["layout_single"]["unversioned"] is True
    assert METADATA_TABLE_NAME not in _tables(database)


def test_schema_diff_reports_all_physical_constraint_categories(database: Database) -> None:
    layout = normalize_entry_backings({LayoutFamily: LayoutSingle})
    metadata = expected_metadata(layout)
    with database.engine.begin() as connection:
        connection.execute(
            sqlalchemy.text(
                "CREATE TABLE layout_single (value INTEGER NOT NULL, content_id TEXT NOT NULL, sid INTEGER, "
                "extra INTEGER, PRIMARY KEY (value), FOREIGN KEY (value) REFERENCES other_table (id), "
                "UNIQUE (value), CHECK (value > 0))"
            )
        )
        connection.execute(sqlalchemy.text("CREATE INDEX wrong_layout_index ON layout_single (value)"))
        difference = validate_expected_tables(connection, metadata, table_names=("layout_single",))
    table_difference = difference["layout_single"]
    assert isinstance(table_difference, Mapping)
    assert {"column_order", "columns", "primary_key", "foreign_keys", "unique", "indexes", "checks"} <= set(
        table_difference
    )


def test_sqlite_schema_diff_rejects_partial_ordered_collated_indexes() -> None:
    layout = normalize_entry_backings({LayoutFamily: LayoutSingle})
    metadata = expected_metadata(layout)
    table = metadata.tables["layout_single"]
    index = next(iter(table.indexes))
    assert index.name is not None
    with Database.sqlite() as partial_database:
        SqlStore(partial_database, entry_backings={LayoutFamily: LayoutSingle})
        with partial_database.engine.begin() as connection:
            connection.execute(sqlalchemy.text(f'DROP INDEX "{index.name}"'))
            connection.execute(
                sqlalchemy.text(
                    f'CREATE UNIQUE INDEX "{index.name}" ON layout_single (content_id) WHERE content_id IS NOT NULL'
                )
            )
        with pytest.raises(StorageLayoutUpgradeRequiredError) as partial:
            SqlStore(partial_database)
        assert "indexes" in partial.value.diff["schema"]["layout_single"]
        assert "unique" in partial.value.diff["schema"]["layout_single"]

    with Database.sqlite() as ordered_database:
        SqlStore(ordered_database, entry_backings={LayoutFamily: LayoutSingle})
        with ordered_database.engine.begin() as connection:
            connection.execute(sqlalchemy.text(f'DROP INDEX "{index.name}"'))
            connection.execute(
                sqlalchemy.text(f'CREATE UNIQUE INDEX "{index.name}" ON layout_single (content_id COLLATE NOCASE DESC)')
            )
        with pytest.raises(StorageLayoutUpgradeRequiredError) as ordered:
            SqlStore(ordered_database)
        assert "indexes" in ordered.value.diff["schema"]["layout_single"]


def test_sqlite_schema_diff_preserves_check_grouping_and_foreign_key_actions() -> None:
    _layout, metadata, dispatch = _multi_layout()
    first_column = backing_dispatch_column_name("test-layout-first-backing")
    second_column = backing_dispatch_column_name("test-layout-second-backing")
    expected_check = (
        f"(CASE WHEN {first_column} IS NOT NULL THEN 1 ELSE 0 END + "
        f"CASE WHEN {second_column} IS NOT NULL THEN 1 ELSE 0 END) = 1"
    )
    regrouped_check = (
        f"CASE WHEN {first_column} IS NOT NULL THEN 1 ELSE 0 END + "
        f"(CASE WHEN {second_column} IS NOT NULL THEN 1 ELSE 0 END = 1)"
    )
    with Database.sqlite() as check_database:
        with check_database.engine.begin() as connection:
            for table in metadata.sorted_tables:
                if table.name not in {METADATA_TABLE_NAME, dispatch.name}:
                    table.create(connection)
            ddl = str(sqlalchemy.schema.CreateTable(dispatch).compile(dialect=connection.dialect))
            assert expected_check in ddl
            connection.execute(sqlalchemy.text(ddl.replace(expected_check, regrouped_check)))
            difference = validate_expected_tables(connection, metadata, table_names=(dispatch.name,))
        check_difference = difference[dispatch.name]
        assert isinstance(check_difference, Mapping)
        assert "checks" in check_difference

    _, action_metadata, action_dispatch = _multi_layout()
    with Database.sqlite() as action_database:
        with action_database.engine.begin() as connection:
            for table in action_metadata.sorted_tables:
                if table.name not in {METADATA_TABLE_NAME, action_dispatch.name}:
                    table.create(connection)
            ddl = str(sqlalchemy.schema.CreateTable(action_dispatch).compile(dialect=connection.dialect))
            reference = "REFERENCES layout_first (sid)"
            assert reference in ddl
            connection.execute(sqlalchemy.text(ddl.replace(reference, f"{reference} ON DELETE CASCADE", 1)))
            difference = validate_expected_tables(connection, action_metadata, table_names=(action_dispatch.name,))
        action_difference = difference[action_dispatch.name]
        assert isinstance(action_difference, Mapping)
        assert "foreign_keys" in action_difference


def test_sqlite_schema_scanning_ignores_keywords_and_parentheses_inside_quotes() -> None:
    metadata = sqlalchemy.MetaData()
    sqlalchemy.Table(
        "quoted_sql_text",
        metadata,
        sqlalchemy.Column("check_record", sqlalchemy.Text, server_default=sqlalchemy.text("'CHECK(fake(value))'")),
    )
    with Database.sqlite() as database:
        store = SqlStore(database, entry_backings={})
        with database.engine.begin() as connection:
            metadata.create_all(connection)
            assert validate_expected_tables(connection, metadata) == {}

        check_record = CheckRecord("ordinary")
        check_sid = store.save(check_record)
        weird_record = WeirdNamedRecord("parenthesized table name")
        weird_sid = store.save(weird_record)

        reopened = SqlStore(database)
        assert reopened.fetch(CheckRecord, check_sid) == check_record
        assert reopened.fetch(WeirdNamedRecord, weird_sid) == weird_record

    string_check = sqlalchemy.MetaData()
    sqlalchemy.Table(
        "string_check",
        string_check,
        sqlalchemy.Column("value", sqlalchemy.Text),
        sqlalchemy.CheckConstraint("value != '(1)'"),
    )
    with Database.sqlite() as string_database:
        with string_database.engine.begin() as connection:
            connection.execute(sqlalchemy.text("CREATE TABLE string_check (value TEXT, CHECK (value != '1'))"))
            difference = validate_expected_tables(connection, string_check)
        string_difference = difference["string_check"]
        assert isinstance(string_difference, Mapping)
        assert "checks" in string_difference


def test_sqlite_foreign_key_comparison_includes_match_and_deferral() -> None:
    explicit = sqlalchemy.MetaData()
    sqlalchemy.Table("fk_parent", explicit, sqlalchemy.Column("sid", sqlalchemy.Integer, primary_key=True))
    sqlalchemy.Table(
        "fk_child",
        explicit,
        sqlalchemy.Column("sid", sqlalchemy.Integer, primary_key=True),
        sqlalchemy.Column(
            "parent_sid",
            sqlalchemy.Integer,
            sqlalchemy.ForeignKey(
                "fk_parent.sid",
                match="FULL",
                deferrable=True,
                initially="DEFERRED",
            ),
        ),
    )
    default = sqlalchemy.MetaData()
    sqlalchemy.Table("fk_parent", default, sqlalchemy.Column("sid", sqlalchemy.Integer, primary_key=True))
    sqlalchemy.Table(
        "fk_child",
        default,
        sqlalchemy.Column("sid", sqlalchemy.Integer, primary_key=True),
        sqlalchemy.Column("parent_sid", sqlalchemy.Integer, sqlalchemy.ForeignKey("fk_parent.sid")),
    )
    with Database.sqlite() as database:
        with database.engine.begin() as connection:
            explicit.create_all(connection)
            assert validate_expected_tables(connection, explicit) == {}
            difference = validate_expected_tables(connection, default, table_names=("fk_child",))
        child_difference = difference["fk_child"]
        assert isinstance(child_difference, Mapping)
        assert "foreign_keys" in child_difference

    with Database.sqlite() as inline_database:
        with inline_database.engine.begin() as connection:
            default.tables["fk_parent"].create(connection)
            connection.execute(
                sqlalchemy.text(
                    "CREATE TABLE fk_child (sid INTEGER PRIMARY KEY, parent_sid INTEGER "
                    "REFERENCES fk_parent(sid) DEFERRABLE INITIALLY DEFERRED)"
                )
            )
            inline_difference = validate_expected_tables(connection, default, table_names=("fk_child",))
        inline_child_difference = inline_difference["fk_child"]
        assert isinstance(inline_child_difference, Mapping)
        assert "foreign_keys" in inline_child_difference


def test_sqlite_foreign_key_comparison_preserves_constraint_grouping() -> None:
    expected = sqlalchemy.MetaData()
    sqlalchemy.Table("group_parent", expected, sqlalchemy.Column("sid", sqlalchemy.Integer, primary_key=True))
    sqlalchemy.Table(
        "group_child",
        expected,
        sqlalchemy.Column("a", sqlalchemy.Integer, sqlalchemy.ForeignKey("group_parent.sid")),
        sqlalchemy.Column("b", sqlalchemy.Integer, sqlalchemy.ForeignKey("group_parent.sid")),
    )
    with Database.sqlite() as database:
        with database.engine.begin() as connection:
            expected.tables["group_parent"].create(connection)
            connection.execute(
                sqlalchemy.text(
                    "CREATE TABLE group_child (a INTEGER, b INTEGER, "
                    "FOREIGN KEY (a, b) REFERENCES group_parent (sid, sid))"
                )
            )
            difference = validate_expected_tables(connection, expected, table_names=("group_child",))
        child_difference = difference["group_child"]
        assert isinstance(child_difference, Mapping)
        assert "foreign_keys" in child_difference


def test_reserved_prefix_tables_are_rejected_before_and_after_marking(database: Database) -> None:
    with database.engine.begin() as connection:
        connection.execute(sqlalchemy.text("CREATE TABLE _httk_unknown (value INTEGER)"))
    with pytest.raises(StorageLayoutUpgradeRequiredError) as unversioned:
        SqlStore(database, entry_backings={})
    assert unversioned.value.diff["schema"]["_httk_unknown"]["reserved"] is True
    assert METADATA_TABLE_NAME not in _tables(database)

    with Database.sqlite() as marked_database:
        SqlStore(marked_database, entry_backings={})
        with marked_database.engine.begin() as connection:
            connection.execute(sqlalchemy.text("CREATE TABLE _httk_unknown (value INTEGER)"))
        with pytest.raises(StorageLayoutUpgradeRequiredError) as marked:
            SqlStore(marked_database)
        assert marked.value.diff["schema"]["_httk_unknown"]["reserved"] is True

    with Database.sqlite() as stale_declaration_database:
        SqlStore(stale_declaration_database, entry_backings={})
        with stale_declaration_database.engine.begin() as connection:
            connection.execute(
                sqlalchemy.text(
                    "UPDATE _httk_store_metadata SET value = '[\"_httk_absent\"]' WHERE key = 'unclaimed_tables'"
                )
            )
        with pytest.raises(StorageLayoutUpgradeRequiredError) as stale:
            SqlStore(stale_declaration_database)
        assert "reserved _httk_" in stale.value.diff["declaration"]["error"]


def test_failed_empty_initialization_leaves_no_partial_layout(
    database: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_stamp(
        self: SqlStore,
        connection: sqlalchemy.Connection,
        layout: object,
        unclaimed: tuple[str, ...],
    ) -> None:
        raise RuntimeError("stamp failure")

    monkeypatch.setattr(SqlStore, "_stamp_layout", fail_stamp)
    with pytest.raises(RuntimeError, match="stamp failure"):
        SqlStore(database, entry_backings={LayoutFamily: LayoutSingle})
    assert not _tables(database)


def test_concurrent_first_initialization_loser_does_not_drop_winner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "layout-race.sqlite"
    barrier = threading.Barrier(2)
    original = store_module.actual_table_names
    thread_state = threading.local()

    def synchronized_names(connection: sqlalchemy.Connection) -> frozenset[str]:
        names = original(connection)
        if threading.current_thread().name.startswith("layout-init") and not getattr(thread_state, "observed", False):
            thread_state.observed = True
            barrier.wait(timeout=10)
        return names

    monkeypatch.setattr(store_module, "actual_table_names", synchronized_names)
    outcomes: list[BaseException | None] = []
    outcomes_lock = threading.Lock()

    def initialize() -> None:
        database = Database.sqlite(path)
        try:
            SqlStore(database, entry_backings={LayoutFamily: LayoutSingle})
        except BaseException as error:
            outcome: BaseException | None = error
        else:
            outcome = None
        finally:
            database.dispose()
        with outcomes_lock:
            outcomes.append(outcome)

    threads = [threading.Thread(target=initialize, name=f"layout-init-{index}") for index in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=15)
        assert not thread.is_alive()
    assert len(outcomes) == 2
    assert sum(outcome is None for outcome in outcomes) == 1
    assert sum(outcome is not None for outcome in outcomes) == 1

    with Database.sqlite(path) as reopened:
        assert SqlStore(reopened).entry_layout[0].family is LayoutFamily


def test_registry_normalization_and_single_backing_dispatch_free_storage(database: Database) -> None:
    with pytest.raises(ValueError, match="registered"):
        SqlStore(database, entry_backings={UnregisteredFamily: LayoutSingle})

    store = SqlStore(database, entry_backings={LayoutFamily: LayoutSingle})
    family = store.entry_layout[0]
    assert family.backing_names == ("test-layout-single-backing",)
    assert family.dispatch_table_name is None
    assert not any(name.startswith("_httk_entry_dispatch") for name in _tables(database))
    record = LayoutSingle("single")
    assert store.fetch_entry(LayoutFamily, content_id(record)) is None
    sid = store.save(record)
    assert store.fetch_entry(LayoutFamily, content_id(record)) is record
    assert store.fetch_by_content_id(LayoutSingle, content_id(record)) is record
    assert sid == store.sid_of(record)


def test_multi_backing_dispatch_is_exactly_one_and_atomic(database: Database) -> None:
    store = SqlStore(database, entry_backings={MultiLayoutFamily: (LayoutFirst, LayoutSecond)})
    family = store.entry_layout[0]
    assert family.dispatch_table_name is not None
    first = LayoutFirst("first")
    second = LayoutSecond(2)
    first_sid = store.save(first)
    second_sid = store.save(second)
    assert store.save(first) == first_sid
    assert store.fetch_entry(MultiLayoutFamily, content_id(first)) is first
    assert store.fetch_entry(MultiLayoutFamily, content_id(second)) is second

    first_column = backing_dispatch_column_name("test-layout-first-backing")
    second_column = backing_dispatch_column_name("test-layout-second-backing")
    table = family.dispatch_table_name
    with store._write_connection() as connection:
        with pytest.raises(EntryDispatchIntegrityError, match="already maps backing sid"):
            store._save_entry_dispatch(connection, family, LayoutFirst, first_sid, "conflicting-backing-sid")
        assert connection.execute(sqlalchemy.select(store._table(table))).first() is not None
    with database.engine.begin() as connection:
        with pytest.raises(IntegrityError):
            connection.execute(
                sqlalchemy.text(
                    f"INSERT INTO {table} ({DISPATCH_CONTENT_ID_COLUMN}, {first_column}, {second_column}) "
                    "VALUES ('both', :first, :second)"
                ),
                {"first": first_sid, "second": second_sid},
            )
        connection.execute(sqlalchemy.text("PRAGMA ignore_check_constraints = ON"))
        connection.execute(
            sqlalchemy.text("INSERT INTO layout_second (content_id, value) VALUES ('orphaned-corruption-row', 99)")
        )
        orphan_sid = connection.execute(
            sqlalchemy.text("SELECT sid FROM layout_second WHERE content_id = 'orphaned-corruption-row'")
        ).scalar_one()
        connection.execute(
            sqlalchemy.text(f"UPDATE {table} SET {second_column} = :sid WHERE {DISPATCH_CONTENT_ID_COLUMN} = :key"),
            {"sid": orphan_sid, "key": content_id(first)},
        )
        connection.execute(sqlalchemy.text("PRAGMA ignore_check_constraints = OFF"))
    with pytest.raises(EntryDispatchIntegrityError, match="2 backing"):
        store.fetch_entry(MultiLayoutFamily, content_id(first))

    with Database.sqlite() as rollback_database:
        rollback_store = SqlStore(rollback_database, entry_backings={MultiLayoutFamily: (LayoutFirst, LayoutSecond)})
        rollback_record = LayoutFirst("rollback")
        with pytest.raises(RuntimeError), rollback_store.transaction():
            rollback_store.save(rollback_record)
            raise RuntimeError("force rollback")
        assert rollback_store.fetch_entry(MultiLayoutFamily, content_id(rollback_record)) is None


def test_fetch_entry_detects_missing_and_mismatched_multi_backing_dispatch(database: Database) -> None:
    store = SqlStore(database, entry_backings={MultiLayoutFamily: (LayoutFirst, LayoutSecond)})
    record = LayoutFirst("fetch-integrity")
    sid = store.save(record)
    key = content_id(record)
    family = store.entry_layout[0]
    assert family.dispatch_table_name is not None
    first_column = backing_dispatch_column_name("test-layout-first-backing")
    with database.engine.begin() as connection:
        connection.execute(
            sqlalchemy.delete(store._table(family.dispatch_table_name)).where(
                store._table(family.dispatch_table_name).c[DISPATCH_CONTENT_ID_COLUMN] == key
            )
        )
    with pytest.raises(EntryDispatchIntegrityError, match="missing"):
        store.fetch_entry(MultiLayoutFamily, key)

    with database.engine.begin() as connection:
        connection.execute(
            sqlalchemy.insert(store._table(family.dispatch_table_name)).values(
                {DISPATCH_CONTENT_ID_COLUMN: key, first_column: sid}
            )
        )
        connection.execute(
            sqlalchemy.text("UPDATE layout_first SET content_id = 'changed-content-id' WHERE sid = :sid"),
            {"sid": sid},
        )
    with pytest.raises(EntryDispatchIntegrityError, match="whose content_id"):
        store.fetch_entry(MultiLayoutFamily, key)


def test_marked_reopen_audits_missing_mismatched_and_invalid_dispatch_rows() -> None:
    declaration = {MultiLayoutFamily: (LayoutFirst, LayoutSecond)}
    with Database.sqlite() as missing_database:
        missing_store = SqlStore(missing_database, entry_backings=declaration)
        record = LayoutFirst("missing-marked")
        missing_store.save(record)
        family = missing_store.entry_layout[0]
        assert family.dispatch_table_name is not None
        with missing_database.engine.begin() as connection:
            connection.execute(sqlalchemy.text(f'DELETE FROM "{family.dispatch_table_name}"'))
        with pytest.raises(StorageLayoutUpgradeRequiredError) as missing:
            SqlStore(missing_database)
        assert "missing_dispatch_rows" in missing.value.diff["schema"][family.dispatch_table_name]["data"]

    with Database.sqlite() as mismatch_database:
        mismatch_store = SqlStore(mismatch_database, entry_backings=declaration)
        record = LayoutFirst("mismatch-marked")
        sid = mismatch_store.save(record)
        family = mismatch_store.entry_layout[0]
        assert family.dispatch_table_name is not None
        with mismatch_database.engine.begin() as connection:
            connection.execute(
                sqlalchemy.text("UPDATE layout_first SET content_id = 'different' WHERE sid = :sid"), {"sid": sid}
            )
        with pytest.raises(StorageLayoutUpgradeRequiredError) as mismatch:
            SqlStore(mismatch_database)
        assert (
            "orphaned_or_mismatched_dispatch_rows" in mismatch.value.diff["schema"][family.dispatch_table_name]["data"]
        )

    with Database.sqlite() as invalid_database:
        invalid_store = SqlStore(invalid_database, entry_backings=declaration)
        first_sid = invalid_store.save(LayoutFirst("invalid-first"))
        second_sid = invalid_store.save(LayoutSecond(88))
        family = invalid_store.entry_layout[0]
        assert family.dispatch_table_name is not None
        first_column = backing_dispatch_column_name("test-layout-first-backing")
        second_column = backing_dispatch_column_name("test-layout-second-backing")
        with invalid_database.engine.begin() as connection:
            connection.execute(sqlalchemy.text("PRAGMA ignore_check_constraints = ON"))
            connection.execute(
                sqlalchemy.text(f'DELETE FROM "{family.dispatch_table_name}" WHERE {second_column} = :second_sid'),
                {"second_sid": second_sid},
            )
            connection.execute(
                sqlalchemy.text(
                    f'UPDATE "{family.dispatch_table_name}" SET {second_column} = :second_sid '
                    f"WHERE {first_column} = :first_sid"
                ),
                {"first_sid": first_sid, "second_sid": second_sid},
            )
            connection.execute(sqlalchemy.text("PRAGMA ignore_check_constraints = OFF"))
        with pytest.raises(StorageLayoutUpgradeRequiredError) as invalid:
            SqlStore(invalid_database)
        assert "invalid_dispatch_rows" in invalid.value.diff["schema"][family.dispatch_table_name]["data"]


def test_private_tables_are_created_only_by_ensure_and_content_identity_is_store_local(database: Database) -> None:
    ensured = SqlStore(database, entry_backings={})
    record = PrivateLayoutRecord("private")
    first_sid = ensured.save(record)
    assert ensured.fetch_by_content_id(PrivateLayoutRecord, content_id(record)) == record

    with Database.sqlite() as verify_database:
        SqlStore(verify_database, entry_backings={})
        verified = SqlStore(verify_database, layout_mode="verify")
        with pytest.raises(StorageLayoutUpgradeRequiredError) as error:
            verified.fetch_by_content_id(PrivateLayoutRecord, content_id(record))
        assert "private_layout_record" in error.value.diff["schema"]

    with Database.sqlite() as second_database:
        second = SqlStore(second_database, entry_backings={})
        second.save(PrivateLayoutRecord("before"))
        second_sid = second.save(record)
        assert first_sid != second_sid
        assert content_id(second.fetch(PrivateLayoutRecord, second_sid)) == content_id(record)


def test_duckdb_reopens_and_dispatches_when_available() -> None:
    pytest.importorskip("duckdb_engine")
    with Database.duckdb() as database:
        store = SqlStore(database, entry_backings={MultiLayoutFamily: (LayoutFirst, LayoutSecond)})
        record = LayoutSecond(7)
        sid = store.save(record)
        assert store.fetch_entry(MultiLayoutFamily, content_id(record)) == record
        family = store.entry_layout[0]
        with store._write_connection() as connection:
            with pytest.raises(EntryDispatchIntegrityError, match="already maps backing sid"):
                store._save_entry_dispatch(connection, family, LayoutSecond, sid, "duckdb-conflicting-backing-sid")
            assert connection.execute(sqlalchemy.select(store._table(family.dispatch_table_name))).first() is not None
        assert SqlStore(database).fetch_entry(MultiLayoutFamily, content_id(record)) == record


def test_duckdb_private_table_validation_cache_recovers_after_rollback() -> None:
    pytest.importorskip("duckdb_engine")
    with Database.duckdb() as database:
        store = SqlStore(database, entry_backings={})
        with pytest.raises(RuntimeError), store.transaction():
            store.save(PrivateLayoutRecord("rolled-back"))
            raise RuntimeError("force rollback")
        replacement = PrivateLayoutRecord("replacement")
        sid = store.save(replacement)
        assert store.fetch(PrivateLayoutRecord, sid) == replacement
