"""Focused SqlStore protocol/layout and entry-family dispatch coverage."""

import os
import subprocess
import sys
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import pytest
import sqlalchemy
from httk.core.register import register_entry_family, register_entry_record
from httk.core.storage import StorageInfo, content_id

from httk.store import EntryFamilyDeclaration, EntryLayoutBindingError, EntryRecordDeclaration, storage_layout
from httk.store.db import (
    STORAGE_PROTOCOL_VERSION,
    BackendFacts,
    Database,
    SqlStore,
    StorageLayoutUpgradeRequiredError,
    StoreUnderConstructionError,
)
from httk.store.db.layout import (
    METADATA_TABLE_NAME,
    StorageLayout,
    actual_schema_objects,
    actual_table_names,
    backend_facts_for_dialect,
    declaration_json,
    expected_metadata,
    normalize_entry_records,
)
from httk.store.db.mapping import entry_dispatch_table_name


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


class LocalLayoutFamily:
    """Application-owned family which is deliberately not registered."""


@dataclass(frozen=True)
class LocalLayoutRecord:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="local_layout_record")

    value: str


LOCAL_LAYOUT = EntryFamilyDeclaration(
    name="test-local-layout-family",
    family=LocalLayoutFamily,
    records=(
        EntryRecordDeclaration(
            name="test-local-layout-record",
            record=LocalLayoutRecord,
        ),
    ),
)


register_entry_family(name="test-layout-single-family", family=f"{__name__}:LayoutFamily")
register_entry_record(
    name="test-layout-single-backing",
    family="test-layout-single-family",
    record=f"{__name__}:LayoutSingle",
)
register_entry_family(name="test-layout-multi-family", family=f"{__name__}:MultiLayoutFamily")
register_entry_record(
    name="test-layout-first-backing",
    family="test-layout-multi-family",
    record=f"{__name__}:LayoutFirst",
)
register_entry_record(
    name="test-layout-second-backing",
    family="test-layout-multi-family",
    record=f"{__name__}:LayoutSecond",
)
register_entry_record(name="test-layout-unbound-record", record=f"{__name__}:PrivateLayoutRecord")


@pytest.fixture
def database() -> Iterator[Database]:
    with Database.sqlite() as database:
        yield database


def _tables(database: Database) -> set[str]:
    with database.engine.connect() as connection:
        return set(connection.execute(sqlalchemy.text("SELECT name FROM sqlite_master WHERE type = 'table'")).scalars())


def test_family_store_rejects_record_without_registered_family() -> None:
    with pytest.raises(ValueError, match="no registered family"):
        normalize_entry_records({LayoutFamily: PrivateLayoutRecord})


def test_registry_normalization_does_not_resolve_unrelated_lazy_entries(monkeypatch: pytest.MonkeyPatch) -> None:
    """Declaration validation matches supplied classes without importing every registry plugin."""
    family_ref = f"{__name__}:LayoutFamily"
    record_ref = f"{__name__}:LayoutSingle"
    monkeypatch.setattr(storage_layout, "known_entry_families", lambda: ["selected", "unrelated"])
    monkeypatch.setattr(
        storage_layout,
        "entry_family_info",
        lambda name: (family_ref if name == "selected" else "unloaded.optional:Family", None),
    )
    monkeypatch.setattr(storage_layout, "known_entry_records", lambda: ["selected-record", "unrelated-record"])
    monkeypatch.setattr(
        storage_layout,
        "entry_record_info",
        lambda name: (record_ref if name == "selected-record" else "unloaded.optional:Record", "selected", None),
    )
    monkeypatch.setattr(
        storage_layout,
        "resolve_entry_family",
        lambda name: pytest.fail(f"unexpected lazy family resolution: {name}"),
        raising=False,
    )
    monkeypatch.setattr(
        storage_layout,
        "resolve_entry_record",
        lambda name: pytest.fail(f"unexpected lazy record resolution: {name}"),
        raising=False,
    )

    layout = normalize_entry_records({LayoutFamily: LayoutSingle})

    assert layout.declaration == {"selected": ("selected-record",)}


def _multi_layout() -> tuple[StorageLayout, sqlalchemy.MetaData, sqlalchemy.Table]:
    layout = normalize_entry_records({MultiLayoutFamily: (LayoutFirst, LayoutSecond)})
    metadata = expected_metadata(layout)
    dispatch_name = entry_dispatch_table_name(layout.families[0].name)
    return layout, metadata, metadata.tables[dispatch_name]


def test_declaration_json_stamp_is_byte_stable() -> None:
    # This exact string guards cross-backend stamp stability.
    layout = normalize_entry_records({LayoutFamily: LayoutSingle, MultiLayoutFamily: (LayoutFirst, LayoutSecond)})
    assert declaration_json(layout) == (
        '{"families":[{"definition_id":null,"family":"test-layout-multi-family","records":['
        '{"definition_id":null,"record":"test-layout-first-backing"},{"definition_id":null,'
        '"record":"test-layout-second-backing"}]},{"definition_id":null,"family":'
        '"test-layout-single-family","records":[{"definition_id":null,"record":'
        '"test-layout-single-backing"}]}],"format":2}'
    )


def test_storage_layout_import_does_not_import_sqlalchemy() -> None:
    code = "import httk.store.storage_layout\nimport sys\nassert 'sqlalchemy' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True, env=dict(os.environ))


def test_common_save_and_paging_imports_do_not_import_sqlalchemy() -> None:
    code = (
        "import httk.store.store_common\n"
        "import httk.store.query.paging_tokens\n"
        "import sys\n"
        "assert 'sqlalchemy' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True, env=dict(os.environ))


def test_empty_database_requires_declaration_and_stamps_metadata_only(database: Database) -> None:
    with pytest.raises(TypeError, match="entry_records"):
        SqlStore(database)

    store = SqlStore(database, entry_records={})
    assert store.entry_layout == ()
    with database.engine.connect() as connection:
        names = actual_table_names(connection)
        assert names == {METADATA_TABLE_NAME}
        declaration = connection.execute(
            sqlalchemy.text("SELECT value FROM _httk_store_metadata WHERE key = 'entry_declaration'")
        ).scalar_one()
        assert declaration == '{"families":[],"format":2}'
    assert STORAGE_PROTOCOL_VERSION == "v2.5.0"
    assert SqlStore(database).entry_layout == ()


def test_application_owned_declaration_needs_no_registry_and_rebinds_on_reopen(database: Database) -> None:
    store = SqlStore(database, entry_families=(LOCAL_LAYOUT,))
    record = LocalLayoutRecord("private")
    sid = store.save(record)

    layout = store.entry_layout[0]
    assert layout.name == "test-local-layout-family"
    assert layout.family is LocalLayoutFamily
    assert layout.definition_id is None
    assert layout.record_names == ("test-local-layout-record",)
    assert layout.records == (LocalLayoutRecord,)
    assert layout.record_definition_ids == (None,)

    reopened = SqlStore(database, entry_families=(LOCAL_LAYOUT,))
    assert reopened.fetch(LocalLayoutRecord, sid) == record
    assert reopened.fetch_entry(LocalLayoutFamily, content_id(record)) == record

    with pytest.raises(EntryLayoutBindingError, match="entry_families"):
        SqlStore(database)


def test_registered_and_application_owned_declarations_compose(database: Database) -> None:
    store = SqlStore(
        database,
        entry_records={LayoutFamily: LayoutSingle},
        entry_families=(LOCAL_LAYOUT,),
    )
    assert tuple(layout.family for layout in store.entry_layout) == (LayoutFamily, LocalLayoutFamily)


def test_stamp_trust_reopens_with_missing_or_changed_record_tables(database: Database) -> None:
    store = SqlStore(database, entry_records={LayoutFamily: LayoutSingle})
    store.save(LayoutSingle("present"))
    with database.engine.begin() as connection:
        connection.execute(sqlalchemy.text("DROP TABLE layout_single"))
    assert SqlStore(database).fetch_by_content_id(LayoutSingle, content_id(LayoutSingle("missing"))) is None


def test_multi_record_store_reopens_with_its_dispatch_table(database: Database) -> None:
    store = SqlStore(database, entry_records={MultiLayoutFamily: (LayoutFirst, LayoutSecond)})
    sid = store.save(LayoutFirst("first"))

    reopened = SqlStore(database)

    assert reopened.fetch(LayoutFirst, sid) == LayoutFirst("first")

    dispatch_name = entry_dispatch_table_name(store.entry_layout[0].name)
    with database.engine.begin() as connection:
        connection.execute(sqlalchemy.text(f'DROP TABLE "{dispatch_name}"'))
        connection.execute(sqlalchemy.text(f'CREATE VIEW "{dispatch_name}" AS SELECT "first" AS content_id'))
    with pytest.raises(StorageLayoutUpgradeRequiredError):
        SqlStore(database)


def test_second_store_does_not_cache_an_uncommitted_table(tmp_path: Path) -> None:
    database = Database.sqlite(tmp_path / "cache.sqlite")
    first = SqlStore(database, entry_records={})
    second = SqlStore(database)
    try:
        with first.transaction():
            first.ensure_tables(LayoutSingle)
            assert second.fetch_by_content_id(LayoutSingle, "missing") is None
            # The shared cache may only ever reflect committed catalog state. SQLite's
            # legacy transaction mode may autocommit DDL, so the table being visible is
            # a legal outcome — the contract is cache ⊆ committed catalog, not invisibility.
            if "layout_single" in second._tables_present:
                with database.engine.connect() as connection:
                    assert "layout_single" in actual_table_names(connection)
    finally:
        database.dispose()


def test_fresh_store_reads_are_empty_and_do_not_create_record_tables(database: Database) -> None:
    store = SqlStore(database, entry_records={LayoutFamily: LayoutSingle})
    with database.engine.connect() as connection:
        before = actual_table_names(connection)

    assert store.fetch_by_content_id(LayoutSingle, "missing") is None
    assert store.fetch_entry(LayoutFamily, "missing") is None
    assert store.sid_of(LayoutSingle("missing")) is None
    searcher = store.searcher()
    variable = searcher.variable(LayoutSingle)
    searcher.output(variable, "record")
    assert searcher.count() == 0
    assert list(searcher) == []

    with database.engine.connect() as connection:
        assert actual_table_names(connection) == before


def test_read_candidate_metadata_is_memoized_per_class_set(database: Database, monkeypatch: pytest.MonkeyPatch) -> None:
    store = SqlStore(database, entry_records={LayoutFamily: LayoutSingle})
    calls: list[frozenset[type]] = []
    original = SqlStore._candidate_metadata

    def spy(self: SqlStore, classes: object) -> object:
        materialized = tuple(classes)  # type: ignore[call-overload]
        calls.append(frozenset(materialized))
        return original(self, materialized)

    monkeypatch.setattr(SqlStore, "_candidate_metadata", spy)
    assert store.fetch_by_content_id(LayoutSingle, "missing") is None
    assert len(calls) == 1
    for _ in range(5):
        assert store.fetch_by_content_id(LayoutSingle, "missing") is None
    # The first read builds candidate metadata; later reads of the same
    # class-set reuse the memoized name set and never rebuild it.
    assert len(calls) == 1


def test_warm_read_memo_does_not_block_table_creation_on_write(database: Database) -> None:
    store = SqlStore(database, entry_records={LayoutFamily: LayoutSingle})
    # Warm the read memo for {LayoutSingle} while its table is still absent.
    assert store.fetch_by_content_id(LayoutSingle, "missing") is None
    store.save(LayoutSingle("kept"))
    # The write path creates the table despite the warm read memo, and the
    # next read (memoized name set, live presence) finds the new row.
    key = content_id(LayoutSingle("kept"))
    fetched = store.fetch_by_content_id(LayoutSingle, key)
    assert fetched is not None and fetched.value == "kept"


@pytest.mark.parametrize("old_protocol", ["v2.1.0", "v2.2.0", "v2.3.0", "v2.4.0"])
def test_protocol_and_explicit_declaration_mismatches_have_structured_diffs(
    database: Database, old_protocol: str
) -> None:
    SqlStore(database, entry_records={})
    with database.engine.begin() as connection:
        # This is the prior persisted protocol, not an arbitrary malformed value.
        connection.execute(
            sqlalchemy.text("UPDATE _httk_store_metadata SET value = :protocol WHERE key = 'protocol'"),
            {"protocol": old_protocol},
        )
    with pytest.raises(StorageLayoutUpgradeRequiredError) as error:
        SqlStore(database)
    assert error.value.diff["protocol"] == {"expected": STORAGE_PROTOCOL_VERSION, "actual": old_protocol}


def test_backend_facts_are_resolved_and_frozen(database: Database) -> None:
    store = SqlStore(database, entry_records={})
    assert isinstance(store.backend_facts, BackendFacts)
    assert store.backend_facts.serial_stage_format == "sqlite"
    assert store.backend_facts.parallel_shard_format == "sqlite"
    assert store.backend_facts.supports_deferred_finalize
    assert store.backend_facts.supports_degraded
    assert backend_facts_for_dialect("sqlite") == store.backend_facts
    with pytest.raises(ValueError, match="does not support dialect"):
        backend_facts_for_dialect("unknown")


def test_duckdb_main_catalog_ignores_unrelated_attached_database(tmp_path: Path) -> None:
    """An attached database is not part of this store's physical layout scan."""
    pytest.importorskip("duckdb_engine")
    database = Database.duckdb(tmp_path / "main.duckdb")
    attached_path = tmp_path / "legitimate.duckdb"
    attached = Database.duckdb(attached_path)
    try:
        with attached.engine.begin() as connection:
            connection.execute(sqlalchemy.text("CREATE TABLE empty_table (value INTEGER)"))
        attached.dispose()
        from sqlalchemy import event

        @event.listens_for(database.engine, "connect")
        def attach_unrelated(dbapi_connection, _connection_record):
            dbapi_connection.execute(f"ATTACH '{attached_path}' AS legitimate")

        store = SqlStore(database, entry_records={})
        with database.engine.connect() as connection:
            assert "empty_table" not in actual_schema_objects(connection)
        with store.bulk_ingest(finalize="deferred") as bulk:
            bulk.save(LayoutSingle("main"))
        with database.engine.connect() as connection:
            assert "legitimate" in set(
                connection.execute(sqlalchemy.text("SELECT database_name FROM duckdb_databases()")).scalars()
            )
            assert connection.execute(sqlalchemy.text("SELECT count(*) FROM legitimate.empty_table")).scalar_one() == 0
    finally:
        database.dispose()


def test_marker_rejects_read_and_write_opens(database: Database) -> None:
    SqlStore(database, entry_records={})
    with database.engine.begin() as connection:
        connection.execute(
            sqlalchemy.text("INSERT INTO _httk_store_metadata (key, value) VALUES ('ingest_state', 'bulk-ingest')")
        )
    for supplied in (None, {}):
        with pytest.raises(StoreUnderConstructionError, match="dropped and re-ingested"):
            SqlStore(database, entry_records=supplied) if supplied is not None else SqlStore(database)


def test_marker_with_partial_application_table_is_still_rejected(database: Database) -> None:
    SqlStore(database, entry_records={})
    with database.engine.begin() as connection:
        connection.execute(
            sqlalchemy.text("INSERT INTO _httk_store_metadata (key, value) VALUES ('ingest_state', 'bulk-ingest')")
        )
        connection.execute(sqlalchemy.text("CREATE TABLE partial_ingest (sid INTEGER PRIMARY KEY)"))
    with pytest.raises(StoreUnderConstructionError):
        SqlStore(database)


def test_unknown_metadata_key_is_rejected(database: Database) -> None:
    SqlStore(database, entry_records={})
    with database.engine.begin() as connection:
        connection.execute(sqlalchemy.text("INSERT INTO _httk_store_metadata (key, value) VALUES ('mystery', 'value')"))
    with pytest.raises(StorageLayoutUpgradeRequiredError):
        SqlStore(database)


def test_reserved_prefix_tables_are_rejected_before_and_after_marking(database: Database) -> None:
    with database.engine.begin() as connection:
        connection.execute(sqlalchemy.text("CREATE TABLE _httk_unknown (value INTEGER)"))
    with pytest.raises(StorageLayoutUpgradeRequiredError):
        SqlStore(database, entry_records={})
    assert METADATA_TABLE_NAME not in _tables(database)

    with Database.sqlite() as marked_database:
        SqlStore(marked_database, entry_records={})
        with marked_database.engine.begin() as connection:
            connection.execute(sqlalchemy.text("CREATE TABLE _httk_unknown (value INTEGER)"))
        with pytest.raises(StorageLayoutUpgradeRequiredError) as marked:
            SqlStore(marked_database)
        assert marked.value.diff["schema"]["_httk_unknown"]["reserved"] is True


def test_failed_empty_initialization_leaves_no_partial_layout(
    database: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_stamp(self: SqlStore, connection: sqlalchemy.Connection, layout: object) -> None:
        raise RuntimeError("stamp failure")

    monkeypatch.setattr(SqlStore, "_stamp_layout", fail_stamp)
    with pytest.raises(RuntimeError, match="stamp failure"):
        SqlStore(database, entry_records={LayoutFamily: LayoutSingle})
    assert not _tables(database)


def test_concurrent_first_initialization_loser_does_not_drop_winner(tmp_path: Path) -> None:
    path = tmp_path / "layout-race.sqlite"
    start = threading.Barrier(2)
    outcomes: list[BaseException | None] = []
    outcomes_lock = threading.Lock()

    def initialize() -> None:
        database = Database.sqlite(path)
        try:
            start.wait(timeout=10)
            SqlStore(database, entry_records={LayoutFamily: LayoutSingle})
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
    assert sum(outcome is not None for outcome in outcomes) <= 1

    with Database.sqlite(path) as reopened:
        store = SqlStore(reopened)
        assert tuple(item.family for item in store.entry_layout) == (LayoutFamily,)
        assert _tables(reopened) == {METADATA_TABLE_NAME}


def test_registry_normalization_and_single_record_dispatch_free_storage(database: Database) -> None:
    with pytest.raises(ValueError, match="registered"):
        SqlStore(database, entry_records={UnregisteredFamily: LayoutSingle})

    store = SqlStore(database, entry_records={LayoutFamily: LayoutSingle})
    family = store.entry_layout[0]
    assert family.record_names == ("test-layout-single-backing",)
    assert not hasattr(family, "dispatch_table_name")
    record = LayoutSingle("single")
    assert store.fetch_entry(LayoutFamily, content_id(record)) is None
    sid = store.save(record)
    assert store.fetch_entry(LayoutFamily, content_id(record)) is record
    assert store.fetch_by_content_id(LayoutSingle, content_id(record)) is record
    assert sid == store.sid_of(record)
