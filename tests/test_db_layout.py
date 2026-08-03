"""Focused SqlStore protocol/layout and entry-family dispatch coverage."""

import threading
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import pytest
import sqlalchemy
from httk.core.register import register_entry_family, register_entry_record
from httk.core.storage import StorageInfo, content_id

from httk.data.db import (
    STORAGE_PROTOCOL_VERSION,
    Database,
    SqlStore,
    StorageLayoutUpgradeRequiredError,
)
from httk.data.db.layout import (
    METADATA_TABLE_NAME,
    StorageLayout,
    actual_table_names,
    expected_metadata,
    normalize_entry_records,
)


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


def _multi_layout() -> tuple[StorageLayout, sqlalchemy.MetaData, sqlalchemy.Table]:
    layout = normalize_entry_records({MultiLayoutFamily: (LayoutFirst, LayoutSecond)})
    metadata = expected_metadata(layout)
    dispatch_name = layout.families[0].dispatch_table_name
    assert dispatch_name is not None
    return layout, metadata, metadata.tables[dispatch_name]


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
        assert declaration == '{"families":[],"format":1}'
    assert STORAGE_PROTOCOL_VERSION == "v2.0.3"
    assert SqlStore(database).entry_layout == ()


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

    dispatch_name = store.entry_layout[0].dispatch_table_name
    assert dispatch_name is not None
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


def test_protocol_and_explicit_declaration_mismatches_have_structured_diffs(database: Database) -> None:
    SqlStore(database, entry_records={})
    with database.engine.begin() as connection:
        connection.execute(sqlalchemy.text("UPDATE _httk_store_metadata SET value = 'old' WHERE key = 'protocol'"))
    with pytest.raises(StorageLayoutUpgradeRequiredError) as error:
        SqlStore(database)
    assert error.value.diff["protocol"] == {"expected": STORAGE_PROTOCOL_VERSION, "actual": "old"}


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
    assert family.dispatch_table_name is None
    record = LayoutSingle("single")
    assert store.fetch_entry(LayoutFamily, content_id(record)) is None
    sid = store.save(record)
    assert store.fetch_entry(LayoutFamily, content_id(record)) is record
    assert store.fetch_by_content_id(LayoutSingle, content_id(record)) is record
    assert sid == store.sid_of(record)
