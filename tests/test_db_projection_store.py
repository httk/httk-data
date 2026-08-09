"""SQL-physical projection mechanics and SQL query-count guarantees."""

import contextlib
import datetime
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Annotated, ClassVar
from zoneinfo import ZoneInfo

import pytest
import sqlalchemy
from httk.core.storage import IdentitySkip, StorageInfo, stored_property

from httk.data.db import Database, EntryMetadataConflictError, SqlStore

_calls: dict[tuple[str, int], int] = {}


@dataclass(frozen=True)
class LeafView:
    value: int
    note: str | None = None


@dataclass(frozen=True)
class RootView:
    name: str
    primary: LeafView
    related: list[LeafView]
    history: tuple[LeafView, ...]
    modified: datetime.datetime
    note: str | None = None


@dataclass(frozen=True)
class LeafRecord:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(
        storage_name="projection_leaf", identity_name="tests.projection.LeafRecord"
    )
    __httk_canonical_source__: ClassVar[type] = LeafView

    value: int
    note: Annotated[str | None, IdentitySkip()] = None

    @classmethod
    def __httk_project__(cls, source: LeafView) -> Mapping[str, object]:
        key = ("leaf", id(source))
        _calls[key] = _calls.get(key, 0) + 1
        return {"value": source.value, "note": source.note}


@dataclass(frozen=True)
class RootRecord:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(
        storage_name="projection_root", identity_name="tests.projection.RootRecord"
    )
    __httk_canonical_source__: ClassVar[type] = RootView

    name: str
    primary: LeafRecord
    related: list[LeafRecord]
    history: tuple[LeafRecord, ...]
    modified: Annotated[datetime.datetime, IdentitySkip()]
    note: Annotated[str | None, IdentitySkip()] = None

    @classmethod
    def __httk_project__(cls, source: RootView) -> Mapping[str, object]:
        key = ("root", id(source))
        _calls[key] = _calls.get(key, 0) + 1
        return {
            "name": source.name,
            "primary": source.primary,
            "related": source.related,
            "history": source.history,
            "modified": source.modified,
            "note": source.note,
        }


@dataclass(frozen=True)
class SummaryRecord:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(
        storage_name="projection_summary", identity_name="tests.projection.SummaryRecord"
    )
    __httk_canonical_source__: ClassVar[type] = RootView

    name: str
    primary_value: int

    @classmethod
    def __httk_project__(cls, source: RootView) -> Mapping[str, object]:
        return {"name": source.name, "primary_value": source.primary.value}


@dataclass(frozen=True)
class DerivedView:
    value: int

    @property
    def doubled(self) -> int:
        return self.value * 2


@dataclass(frozen=True)
class DerivedRecord:
    __httk_canonical_source__: ClassVar[type] = DerivedView

    value: int

    @stored_property
    def doubled(self) -> int:
        return self.value * 2

    @classmethod
    def __httk_project__(cls, source: DerivedView) -> Mapping[str, object]:
        return {"value": source.value}


@dataclass(frozen=True)
class TupleRecord:
    __httk_canonical_source__: ClassVar[type] = tuple

    value: int

    @classmethod
    def __httk_project__(cls, source: tuple[int, ...]) -> Mapping[str, object]:
        return {"value": source[0]}


@dataclass(frozen=True)
class CycleRecord:
    name: str
    link: Annotated["CycleRecord | None", IdentitySkip()] = None


@dataclass(frozen=True)
class RecursiveMetadataRecord:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="projection_recursive_metadata")

    value: int
    child: "RecursiveMetadataRecord | None" = None
    note: Annotated[str, IdentitySkip()] = "stored"


@dataclass(frozen=True)
class RecursiveNoMetadataRecord:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="projection_recursive_no_metadata")

    value: int
    child: "RecursiveNoMetadataRecord | None" = None


@dataclass(frozen=True)
class IdentityChildContainer:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="projection_identity_child_container")

    children: Annotated[list[RecursiveNoMetadataRecord], IdentitySkip()]


@dataclass(frozen=True)
class NoDedupLeaf:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="projection_no_dedup_leaf", dedup="none")

    value: str


@dataclass(frozen=True)
class ByValueHolder:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="projection_by_value_holder", dedup="by_value")

    leaf: NoDedupLeaf
    value: str


@dataclass(frozen=True)
class RaceHolder:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="projection_race_holder")

    leaf: NoDedupLeaf
    value: str


@dataclass(frozen=True)
class SummaryReference:
    target: SummaryRecord
    value: str


@dataclass(frozen=True)
class FoldMetadata:
    value: str
    instants: Annotated[tuple[datetime.datetime, ...], IdentitySkip()]


@dataclass(frozen=True)
class MetadataProbeRecord:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="projection_metadata_probe")

    value: int
    metadata: Annotated[str, IdentitySkip()]
    constructed: ClassVar[int] = 0

    def __post_init__(self) -> None:
        type(self).constructed += 1


@dataclass(frozen=True)
class NoMetadataProbeRecord:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="projection_no_metadata_probe")

    value: int


@dataclass(frozen=True)
class ValidationProbeView:
    value: int


@dataclass(frozen=True)
class ValidationProbeRecord:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="projection_validation_probe")
    __httk_canonical_source__: ClassVar[type] = ValidationProbeView

    value: int
    calls: ClassVar[list[int]] = []

    @classmethod
    def __httk_project__(cls, source: ValidationProbeView) -> Mapping[str, object]:
        return {"value": source.value}

    @classmethod
    def __httk_validate__(cls, source: "ValidationProbeRecord") -> None:
        cls.calls.append(source.value)


ValidationProbeView.__httk_storage_record__ = ValidationProbeRecord


@dataclass(frozen=True)
class LazySourceRecord:
    value: int


@dataclass(frozen=True)
class LazyAlternateRecord:
    __httk_canonical_source__: ClassVar[type] = LazySourceRecord

    value: int

    @classmethod
    def __httk_project__(cls, source: LazySourceRecord) -> Mapping[str, object]:
        return {"value": source.value}


LeafView.__httk_storage_record__ = LeafRecord
RootView.__httk_storage_record__ = RootRecord


def _root(name: str = "one") -> RootView:
    shared = LeafView(1, "leaf metadata")
    return RootView(
        name,
        shared,
        [shared, LeafView(2)],
        (LeafView(3),),
        datetime.datetime(2026, 8, 1, 12, tzinfo=datetime.timezone(datetime.timedelta(hours=2))),
    )


@pytest.fixture(params=["sqlite", "duckdb"])
def projection_database(request):
    if request.param == "duckdb":
        pytest.importorskip("duckdb_engine")
        manager = Database.duckdb()
    else:
        manager = Database.sqlite()
    with manager as database:
        yield database


def _count(database: Database, table_name: str) -> int:
    with database.engine.connect() as connection:
        return connection.execute(sqlalchemy.text(f"SELECT COUNT(*) FROM {table_name}")).scalar_one()


def test_same_content_identity_has_store_local_sids():
    source = _root()
    with Database.sqlite() as first_database, Database.sqlite() as second_database:
        first = SqlStore(first_database, entry_records={})
        second = SqlStore(second_database, entry_records={})
        first_sid = first.save(source)
        second.save(_root("other"))
        second_sid = second.save(source)

        assert first_sid == 1
        assert second_sid == 2
        assert first.sid_of(source) == first_sid
        assert second.sid_of(source) == second_sid


def test_sid_of_queries_reopened_database(tmp_path):
    path = tmp_path / "projection.sqlite"
    source = _root()
    database = Database.sqlite(path)
    sid = SqlStore(database, entry_records={}).save(source)
    database.dispose()

    with Database.sqlite(path) as reopened:
        assert SqlStore(reopened).sid_of(_root()) == sid


def test_recursive_metadata_free_type_has_no_dedup_metadata_query():
    leaf = RecursiveNoMetadataRecord(3)
    middle = RecursiveNoMetadataRecord(2, leaf)
    root = RecursiveNoMetadataRecord(1, middle)

    with Database.sqlite() as database:
        store = SqlStore(database, entry_records={})
        root_sid = store.save(root)
        statements: list[str] = []

        def count_select(_conn, _cursor, statement, _parameters, _context, _executemany):
            if statement.lstrip().upper().startswith("SELECT"):
                statements.append(statement)

        sqlalchemy.event.listen(database.engine, "before_cursor_execute", count_select)
        try:
            assert store.save(root) == root_sid
        finally:
            sqlalchemy.event.remove(database.engine, "before_cursor_execute", count_select)

    assert len(statements) == 1


def test_dedup_hit_with_no_identity_skip_has_no_metadata_query():
    with Database.sqlite() as database:
        store = SqlStore(database, entry_records={})
        store.save(NoMetadataProbeRecord(1))
        statements: list[str] = []

        def count_select(_conn, _cursor, statement, _parameters, _context, _executemany):
            if statement.lstrip().upper().startswith("SELECT"):
                statements.append(statement)

        sqlalchemy.event.listen(database.engine, "before_cursor_execute", count_select)
        try:
            assert store.save(NoMetadataProbeRecord(1)) == 1
        finally:
            sqlalchemy.event.remove(database.engine, "before_cursor_execute", count_select)

    assert len(statements) == 1


def test_dedup_hit_checks_metadata_without_reconstructing_or_selecting_the_graph():
    MetadataProbeRecord.constructed = 0
    with Database.sqlite() as database:
        store = SqlStore(database, entry_records={})
        store.save(MetadataProbeRecord(1, "metadata"))
        second = MetadataProbeRecord(1, "metadata")
        statements: list[str] = []

        def count_select(_conn, _cursor, statement, _parameters, _context, _executemany):
            if statement.lstrip().upper().startswith("SELECT"):
                statements.append(statement)

        sqlalchemy.event.listen(database.engine, "before_cursor_execute", count_select)
        try:
            assert store.save(second) == 1
        finally:
            sqlalchemy.event.remove(database.engine, "before_cursor_execute", count_select)

    assert MetadataProbeRecord.constructed == 2
    assert len(statements) == 2  # content-id lookup plus the one planned metadata column
    assert all("projection_metadata_probe.value" not in statement for statement in statements)


def test_insert_race_winner_still_checks_metadata(monkeypatch):
    source = _root()
    with Database.sqlite() as database:
        store = SqlStore(database, entry_records={})
        store.save(source)
        original_execute = sqlalchemy.Connection.execute
        missed_preflight = False

        class NoRow:
            @staticmethod
            def first():
                return None

        def miss_root_once(connection, statement, *args, **kwargs):
            nonlocal missed_preflight
            tables = statement.get_final_froms() if isinstance(statement, sqlalchemy.sql.Select) else ()
            if not missed_preflight and any(table.name == "projection_root" for table in tables):
                missed_preflight = True
                return NoRow()
            return original_execute(connection, statement, *args, **kwargs)

        monkeypatch.setattr(sqlalchemy.Connection, "execute", miss_root_once)
        with pytest.raises(EntryMetadataConflictError, match="note"):
            store.save(replace(source, note="racing metadata"))

        assert missed_preflight


@pytest.mark.parametrize("explicit_transaction", [False, True])
def test_by_value_reuse_discards_nested_no_dedup_insert(projection_database, monkeypatch, explicit_transaction):
    store = SqlStore(projection_database, entry_records={})
    leaf = NoDedupLeaf("same")
    winner = ByValueHolder(leaf, "holder")
    winner_sid = store.save(winner)
    leaf_sid = store.sid_of(leaf)
    assert leaf_sid is not None
    original_parent_row = store._parent_row

    def reuse_original_leaf(connection, schema, source, projected, projection, path):
        values = original_parent_row(connection, schema, source, projected, projection, path)
        if schema.cls is ByValueHolder:
            values["leaf_sid"] = leaf_sid
        return values

    monkeypatch.setattr(store, "_parent_row", reuse_original_leaf)
    context = store.transaction() if explicit_transaction else contextlib.nullcontext()
    with context:
        assert store.save(ByValueHolder(leaf, "holder")) == winner_sid
    assert _count(projection_database, "projection_no_dedup_leaf") == 1


@pytest.mark.parametrize("explicit_transaction", [False, True])
def test_content_race_loss_discards_nested_no_dedup_insert(projection_database, monkeypatch, explicit_transaction):
    store = SqlStore(projection_database, entry_records={})
    winner = RaceHolder(NoDedupLeaf("same"), "holder")
    winner_sid = store.save(winner)
    original_execute = sqlalchemy.Connection.execute
    missed_preflight = False

    class NoRow:
        @staticmethod
        def first():
            return None

    def miss_root_once(connection, statement, *args, **kwargs):
        nonlocal missed_preflight
        tables = statement.get_final_froms() if isinstance(statement, sqlalchemy.sql.Select) else ()
        if not missed_preflight and any(table.name == "projection_race_holder" for table in tables):
            missed_preflight = True
            return NoRow()
        return original_execute(connection, statement, *args, **kwargs)

    monkeypatch.setattr(sqlalchemy.Connection, "execute", miss_root_once)
    context = store.transaction() if explicit_transaction else contextlib.nullcontext()
    with context:
        assert store.save(RaceHolder(NoDedupLeaf("same"), "holder")) == winner_sid
    assert _count(projection_database, "projection_no_dedup_leaf") == 1
    assert missed_preflight


def test_reference_comparison_uses_declared_record_target():
    source = _root()
    with Database.sqlite() as database:
        store = SqlStore(database, entry_records={})
        store.save(source)
        store.save(_root("other"), as_record=SummaryRecord)
        store.save(source, as_record=SummaryRecord)
        store.save(SummaryReference(source, "match"))

        searcher = store.searcher()
        reference = searcher.variable(SummaryReference)
        searcher.add(reference.target == source)
        searcher.output(reference, "reference")

        assert [row[0][0].value for row in searcher] == ["match"]


def test_alternate_record_sids_are_cached_per_record_type():
    source = _root()
    with Database.sqlite() as database:
        store = SqlStore(database, entry_records={})
        default_sid = store.save(source)
        store.save(_root("other"), as_record=SummaryRecord)
        summary_sid = store.save(source, as_record=SummaryRecord)

        assert default_sid == 1
        assert summary_sid == 2
        assert store.sid_of(source) == default_sid
        assert store.sid_of(source, as_record=SummaryRecord) == summary_sid


def test_projected_stored_property_is_read_from_source():
    with Database.sqlite() as database:
        store = SqlStore(database, entry_records={})
        sid = store.save(DerivedView(4), as_record=DerivedRecord)
        assert store.fetch(DerivedRecord, sid).doubled == 8


def test_projected_source_must_expose_record_stored_properties():
    class IncompleteDerivedView(DerivedView):
        @property
        def doubled(self) -> int:
            raise AttributeError

    with Database.sqlite() as database, pytest.raises(TypeError, match="expose derived stored property 'doubled'"):
        SqlStore(database, entry_records={}).save(IncompleteDerivedView(4), as_record=DerivedRecord)


def test_sql_reopened_store_checks_datetime_metadata_against_utc_instants(tmp_path):
    path = tmp_path / "fold.sqlite"
    zone = ZoneInfo("America/New_York")
    first = datetime.datetime(2026, 11, 1, 1, 30, tzinfo=zone, fold=0)
    second = datetime.datetime(2026, 11, 1, 1, 30, tzinfo=zone, fold=1)
    source = FoldMetadata("fold", (first,))
    database = Database.sqlite(path)
    store = SqlStore(database, entry_records={})
    store.save(source)
    database.dispose()

    with Database.sqlite(path) as reopened, pytest.raises(EntryMetadataConflictError, match="instants"):
        SqlStore(reopened).save(FoldMetadata("fold", (second,)))


def test_lazy_row_alternate_sid_uses_requested_record_target():
    with Database.sqlite() as database:
        store = SqlStore(database, entry_records={})
        source_sid = store.save(LazySourceRecord(7))
        store.save(LazySourceRecord(8), as_record=LazyAlternateRecord)
        searcher = store.searcher()
        source = searcher.variable(LazySourceRecord)
        searcher.output(source, "source")
        row = next(iter(searcher))[0][0]
        alternate_sid = store.save(row, as_record=LazyAlternateRecord)

        assert source_sid != alternate_sid
        assert store.sid_of(row) == source_sid
        assert store.sid_of(row, as_record=LazyAlternateRecord) == alternate_sid


def test_concrete_record_save_returns_same_live_record():
    record = RootRecord(
        "record",
        LeafRecord(4),
        [LeafRecord(5)],
        (LeafRecord(6),),
        datetime.datetime(2026, 8, 1, tzinfo=datetime.UTC),
    )
    with Database.sqlite() as database:
        store = SqlStore(database, entry_records={})
        sid = store.save(record)
        assert store.fetch(RootRecord, sid) is record
