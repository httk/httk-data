"""Parity and batching checks for lazy SQL rows."""

import copy
import gc
import pickle
from dataclasses import asdict, dataclass, field, replace
from typing import Annotated, ClassVar

import pytest
import sqlalchemy
from httk.core.storage import Skip, StorageInfo

from httk.data.db import Database, SchemaError, SqlStore, StaleResultError, content_id, resolve_schema
from httk.data.db.rows import RowHydrator, row_class


@dataclass(frozen=True)
class ParityRecord:
    name: str
    number: int


@dataclass(frozen=True)
class BatchRecord:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(dedup="none")

    name: str
    values: list[str]


@dataclass(frozen=True)
class ValidatedRecord:
    name: str
    calls: ClassVar[list[str]] = []

    def __post_init__(self) -> None:
        self.calls.append(self.name)


@dataclass(frozen=True, slots=True)
class SlotsRecord:
    name: str


@dataclass(frozen=True)
class FidelityRecord:
    included: int
    ignored: int = field(compare=False, hash=False, repr=False)


@dataclass(frozen=True)
class FidelitySubclass(FidelityRecord):
    extra: int = 0


@dataclass(frozen=True)
class SkippedRecord:
    name: str
    ignored: Annotated[list[str], Skip()] = field(default_factory=list)


@dataclass(frozen=True)
class IdentityChild:
    name: str


@dataclass(frozen=True)
class IdentityParent:
    child: IdentityChild


@dataclass(frozen=True)
class CycleRecord:
    next: "CycleRecord | None"


@dataclass(frozen=True)
class PresenceCollision:
    values: list[str] | None = None
    values_present: bool = False


@dataclass(frozen=True)
class CustomEqRecord:
    value: int

    def __eq__(self, other: object) -> bool:
        return isinstance(other, CustomEqRecord) and self.value == other.value


@dataclass(frozen=True)
class CustomHashRecord:
    value: int

    def __hash__(self) -> int:
        return 1


@pytest.fixture(params=["sqlite", "duckdb"])
def database(request):
    if request.param == "duckdb":
        pytest.importorskip("duckdb_engine")
        with Database.duckdb() as db:
            yield db
    else:
        with Database.sqlite() as db:
            yield db


def _row(store: SqlStore, cls: type, name: str = "A"):
    searcher = store.searcher()
    variable = searcher.variable(cls)
    searcher.output(variable, "record")
    if name != "A":
        searcher.add(variable.name == name)
    return next(iter(searcher))[0][0]


def test_equality_is_symmetric(database):
    store = SqlStore(database, entry_records={})
    eager = ParityRecord("A", 1)
    store.save(eager)
    row = _row(store, ParityRecord)
    assert eager == row
    assert row == eager
    assert row != ParityRecord("B", 1)


def test_optional_child_presence_name_collision_is_rejected():
    with pytest.raises(SchemaError, match="values_present.*presence column"):
        resolve_schema(PresenceCollision)


def test_hash_parity(database):
    store = SqlStore(database, entry_records={})
    eager = ParityRecord("A", 1)
    store.save(eager)
    assert hash(_row(store, ParityRecord)) == hash(eager)
    unhashable = BatchRecord("B", ["x"])
    store.save(unhashable)
    with pytest.raises(TypeError):
        hash(_row(store, BatchRecord, "B"))


def test_dataclass_compare_hash_and_repr_flags_are_preserved(database):
    store = SqlStore(database, entry_records={})
    eager = FidelityRecord(1, 2)
    store.save(eager)
    row = _row(store, FidelityRecord)
    other = FidelityRecord(1, 99)
    assert row == other
    assert other == row
    assert hash(row) == hash(other)
    assert repr(row) == "FidelityRecord(included=1)"
    assert row != FidelitySubclass(1, 2, 0)


def test_custom_eq_and_hash_are_rejected():
    with pytest.raises(SchemaError, match=r"CustomEqRecord.*custom __eq__"):
        row_class(CustomEqRecord)
    with pytest.raises(SchemaError, match=r"CustomHashRecord.*custom __hash__"):
        row_class(CustomHashRecord)


def test_schema_and_content_id_parity(database):
    store = SqlStore(database, entry_records={})
    eager = ParityRecord("A", 1)
    store.save(eager)
    row = _row(store, ParityRecord)
    assert resolve_schema(type(row)).table_name == resolve_schema(ParityRecord).table_name
    assert content_id(row) == content_id(eager)


def test_dataclass_replace_runs_init_and_post_init(database):
    ValidatedRecord.calls.clear()
    store = SqlStore(database, entry_records={})
    eager = ValidatedRecord("A")
    store.save(eager)
    row = _row(store, ValidatedRecord)
    ValidatedRecord.calls.clear()
    replaced = replace(row, name="B")
    assert type(replaced) is type(row)
    assert replaced.name == "B"
    assert replaced.sid is None
    assert ValidatedRecord.calls == ["B"]


def test_repr_matches_base_dataclass(database):
    store = SqlStore(database, entry_records={})
    eager = ParityRecord("A", 1)
    store.save(eager)
    assert repr(_row(store, ParityRecord)) == repr(eager)


@pytest.mark.parametrize("operation", [copy.copy, copy.deepcopy, pickle.dumps])
def test_copy_deepcopy_and_pickle_are_explicitly_rejected(database, operation):
    store = SqlStore(database, entry_records={})
    eager = ParityRecord("A", 1)
    store.save(eager)
    with pytest.raises(TypeError, match="materialize"):
        operation(_row(store, ParityRecord))


def test_save_lazy_row_deduplicates_like_eager(database):
    store = SqlStore(database, entry_records={})
    eager = ParityRecord("A", 1)
    sid = store.save(eager)
    row = _row(store, ParityRecord)
    assert store.save(row) == sid
    assert content_id(row) == content_id(eager)
    assert type(store.fetch(ParityRecord, sid)) is ParityRecord


def test_eager_materialization_reuses_live_nested_identity(database):
    store = SqlStore(database, entry_records={})
    child = IdentityChild("child")
    child_sid = store.save(child)
    parent_sid = store.save(IdentityParent(child))
    live_child = store.fetch(IdentityChild, child_sid)
    fetched = store.fetch(IdentityParent, parent_sid)
    assert fetched.child is live_child


def test_skip_default_factory_is_available_on_lazy_rows(database):
    store = SqlStore(database, entry_records={})
    store.save(SkippedRecord("A"))
    row = _row(store, SkippedRecord)
    assert row.ignored == []
    assert row.ignored is row.ignored
    assert row.__dict__["ignored"] == []
    assert asdict(row)["ignored"] == []


def test_eager_cycles_raise_instead_of_returning_lazy_rows(database):
    if database.engine.dialect.name == "duckdb":
        pytest.skip("DuckDB rejects the self-referencing foreign-key fixture")
    store = SqlStore(database, entry_records={})
    store.ensure_tables(CycleRecord)
    schema = resolve_schema(CycleRecord)
    table = store._table(schema.table_name)
    with database.engine.begin() as connection:
        connection.execute(table.insert().values(sid=1, content_id="cycle", next_sid=None))
        connection.execute(table.update().where(table.c.sid == 1).values(next_sid=1))
    with pytest.raises(SchemaError, match=r"cyclic eager hydration.*CycleRecord.*sid 1"):
        store.fetch(CycleRecord, 1)


def test_sid_of_lazy_row_is_store_local(database):
    store = SqlStore(database, entry_records={})
    other = SqlStore(database)
    sid = store.save(ParityRecord("A", 1))
    row = _row(store, ParityRecord)
    assert store.sid_of(row) == sid
    assert other.sid_of(row) is None


def test_slots_dataclasses_are_rejected():
    with pytest.raises(SchemaError, match="slots"):
        row_class(SlotsRecord)


def test_iteration_has_no_child_query_until_field_access(database):
    store = SqlStore(database, entry_records={})
    for index in range(3):
        store.save(BatchRecord(str(index), ["x", "y"]))
    statements: list[str] = []

    def count(_conn, _cursor, statement, _parameters, _context, _executemany):
        statements.append(statement)

    sqlalchemy.event.listen(database.engine, "before_cursor_execute", count)
    try:
        searcher = store.searcher()
        variable = searcher.variable(BatchRecord)
        searcher.output(variable, "record")
        rows = list(searcher)
        child_table = "batch_record_values"
        assert not any(child_table in statement for statement in statements)
        assert rows[0][0][0].values == ["x", "y"]
        assert sum(child_table in statement for statement in statements) == 1
    finally:
        sqlalchemy.event.remove(database.engine, "before_cursor_execute", count)


def test_child_batches_once_per_chunk(database):
    store = SqlStore(database, entry_records={})
    for index in range(1500):
        store.save(BatchRecord(str(index), [str(index)]))
    statements: list[str] = []

    def count(_conn, _cursor, statement, _parameters, _context, _executemany):
        if statement.lstrip().upper().startswith("SELECT"):
            statements.append(statement)

    sqlalchemy.event.listen(database.engine, "before_cursor_execute", count)
    try:
        searcher = store.searcher()
        variable = searcher.variable(BatchRecord)
        searcher.output(variable, "record")
        rows = list(searcher)
        for index in (0, 1, 500, 501, 1000, 1499):
            assert rows[index][0][0].values == [str(index)]
        assert len(statements) <= 8  # 1 outer + 3 parent + 3 child, with one slack statement
    finally:
        sqlalchemy.event.remove(database.engine, "before_cursor_execute", count)


def test_stale_result_is_reported_at_hydration(database):
    store = SqlStore(database, entry_records={})
    sid = store.save(ParityRecord("A", 1))
    searcher = store.searcher()
    variable = searcher.variable(ParityRecord)
    searcher.output(variable, "record")
    results = iter(searcher)
    with database.engine.begin() as connection:
        connection.execute(sqlalchemy.text(f"DELETE FROM parity_record WHERE sid = {sid}"))
    with pytest.raises(StaleResultError, match=r"ParityRecord.*sid"):
        next(results)


def test_weak_chunk_is_rehydrated(database):
    store = SqlStore(database, entry_records={})
    for index in range(501):
        store.save(BatchRecord(str(index), [str(index)]))
    statements: list[str] = []

    def count(_conn, _cursor, statement, _parameters, _context, _executemany):
        if "FROM batch_record" in statement:
            statements.append(statement)

    sqlalchemy.event.listen(database.engine, "before_cursor_execute", count)
    try:
        hydrator = RowHydrator(store, BatchRecord, range(1, 502))
        first = hydrator.row(1)
        second = hydrator.row(2)
        _ = first.name
        _ = second.name
        before = len(statements)
        del first, second
        gc.collect()
        assert hydrator.row(3).name == "2"
        assert len(statements) > before
    finally:
        sqlalchemy.event.remove(database.engine, "before_cursor_execute", count)


def test_fetch_one_without_child_or_reference_uses_one_statement(database):
    store = SqlStore(database, entry_records={})
    sid = store.save(ParityRecord("A", 1))
    statements: list[str] = []

    def count(_conn, _cursor, statement, _parameters, _context, _executemany):
        if statement.lstrip().upper().startswith("SELECT"):
            statements.append(statement)

    sqlalchemy.event.listen(database.engine, "before_cursor_execute", count)
    try:
        assert store.fetch(ParityRecord, sid) == ParityRecord("A", 1)
        assert len(statements) == 1
    finally:
        sqlalchemy.event.remove(database.engine, "before_cursor_execute", count)


def test_fetch_one_with_child_uses_at_most_two_statements(database):
    store = SqlStore(database, entry_records={})
    sid = store.save(BatchRecord("A", ["x"]))
    statements: list[str] = []

    def count(_conn, _cursor, statement, _parameters, _context, _executemany):
        if statement.lstrip().upper().startswith("SELECT"):
            statements.append(statement)

    sqlalchemy.event.listen(database.engine, "before_cursor_execute", count)
    try:
        assert store.fetch(BatchRecord, sid).values == ["x"]
        assert len(statements) <= 2
    finally:
        sqlalchemy.event.remove(database.engine, "before_cursor_execute", count)
