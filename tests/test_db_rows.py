"""Parity and batching checks for lazy SQL rows."""

import copy
import gc
import pickle
from dataclasses import asdict, dataclass, field, replace
from typing import Annotated, ClassVar

import pytest
import sqlalchemy
from httk.core.storage import Skip, StorageInfo, content_id

from httk.store.backend.schema import SchemaError, resolve_schema
from httk.store.backend.sql import Backend, SqlStore, StaleResultError
from httk.store.backend.sql.rows import ExpiredLazyRecordError, RowHydrator, is_lazy_row, row_class


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
        with Backend.duckdb() as db:
            yield db
    else:
        with Backend.sqlite() as db:
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
    with pytest.raises(TypeError, match=r"materialize with store\.fetch\(\.\.\., eager=True\) first"):
        operation(_row(store, ParityRecord))


def test_save_lazy_row_deduplicates_like_eager(database):
    store = SqlStore(database, entry_records={})
    eager = ParityRecord("A", 1)
    sid = store.save(eager)
    row = _row(store, ParityRecord)
    assert store.save(row) == sid
    assert content_id(row) == content_id(eager)
    # The default fetch is lazy; eager=True materializes the exact base type.
    assert is_lazy_row(store.fetch(ParityRecord, sid))
    assert type(store.fetch(ParityRecord, sid, eager=True)) is ParityRecord


def test_eager_materialization_reuses_live_nested_identity(database):
    store = SqlStore(database, entry_records={})
    child = IdentityChild("child")
    child_sid = store.save(child)
    parent_sid = store.save(IdentityParent(child))
    # Eager nested reuse applies to *materialized* objects only: both fetches
    # must be eager, else the parent's eager hydration skips the lazily fetched
    # child (the type-guard treats a proxy hit as a miss) and re-materializes.
    live_child = store.fetch(IdentityChild, child_sid, eager=True)
    fetched = store.fetch(IdentityParent, parent_sid, eager=True)
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
        connection.execute(
            table.insert().values(
                sid=1, content_id="cycle", _httk_role=1, store_timestamp=0, logical_id=1, next_sid=None
            )
        )
        connection.execute(table.update().where(table.c.sid == 1).values(next_sid=1))
    with pytest.raises(SchemaError, match=r"cyclic eager hydration.*CycleRecord.*sid 1"):
        store.fetch(CycleRecord, 1, eager=True)


def test_default_fetch_of_cyclic_graph_succeeds_with_proxies(database):
    if database.engine.dialect.name == "duckdb":
        pytest.skip("DuckDB rejects the self-referencing foreign-key fixture")
    store = SqlStore(database, entry_records={})
    store.ensure_tables(CycleRecord)
    schema = resolve_schema(CycleRecord)
    table = store._table(schema.table_name)
    with database.engine.begin() as connection:
        connection.execute(
            table.insert().values(
                sid=1, content_id="cycle", _httk_role=1, store_timestamp=0, logical_id=1, next_sid=None
            )
        )
        connection.execute(table.update().where(table.c.sid == 1).values(next_sid=1))
    # The lazy default never eagerly walks the cycle; each hop is a fresh proxy.
    row = store.fetch(CycleRecord, 1)
    assert is_lazy_row(row)
    assert is_lazy_row(row.next)
    assert is_lazy_row(row.next.next)
    assert row.next.sid == 1


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


@pytest.mark.parametrize("records", [502, pytest.param(1500, marks=pytest.mark.extended)])
def test_child_batches_once_per_chunk(database, records: int):
    store = SqlStore(database, entry_records={})
    for index in range(records):
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
        for index in (0, 1, 500, 501, records - 1):
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


def test_fetch_many_preserves_input_order_and_shares_identity(database):
    store = SqlStore(database, entry_records={})
    # `records` stays live so the save-time identity cache holds each instance;
    # the batch must therefore return those very objects, not fresh hydrations.
    records = [ParityRecord(name, number) for number, name in enumerate("ABCDE")]
    sids = [store.save(record) for record in records]
    shuffled = [sids[index] for index in (3, 0, 4, 1, 2)]
    fetched = store.fetch_many(ParityRecord, shuffled)
    assert [record.name for record in fetched] == ["D", "A", "E", "B", "C"]
    assert fetched[1] is records[0]  # the identity pre-filter returned the live object


def test_fetch_many_returns_cached_object_for_deleted_row(database):
    store = SqlStore(database, entry_records={})
    record = ParityRecord("A", 1)
    sid = store.save(record)
    with database.engine.begin() as connection:
        connection.execute(sqlalchemy.text(f"DELETE FROM parity_record WHERE sid = {sid}"))
    # Parity with fetch: a live-cached instance survives its row's deletion.
    assert store.fetch(ParityRecord, sid) is record
    assert store.fetch_many(ParityRecord, [sid])[0] is record


@pytest.mark.parametrize("eager", [False, True])
def test_fetch_many_raises_key_error_not_stale_for_absent_sid(database, eager: bool):
    store = SqlStore(database, entry_records={})
    present = store.save(ParityRecord("A", 1))
    absent = store.save(ParityRecord("B", 2))
    store._clear_identity_caches()
    with database.engine.begin() as connection:
        connection.execute(sqlalchemy.text(f"DELETE FROM parity_record WHERE sid = {absent}"))
    with pytest.raises(KeyError):
        store.fetch_many(ParityRecord, [present, absent], eager=eager)


def test_save_proxy_under_new_sid_does_not_miscache_for_dedup_none(database):
    store = SqlStore(database, entry_records={})
    orig_sid = store.save(BatchRecord("A", ["x"]))
    store._clear_identity_caches()
    proxy = store.fetch(BatchRecord, orig_sid)
    assert is_lazy_row(proxy)
    new_sid = store.save(proxy)  # dedup="none" mints a fresh sid, not orig_sid
    assert new_sid != orig_sid
    # The proxy (reading its original row) stays cached under its OWN sid only.
    assert store.fetch(BatchRecord, orig_sid) is proxy
    refetched = store.fetch(BatchRecord, new_sid)
    assert refetched is not proxy
    assert refetched.sid == new_sid  # would be orig_sid if the proxy were miscached
    assert refetched.values == ["x"]


def test_eager_fetch_many_prefilter_skips_a_cached_proxy(database):
    store = SqlStore(database, entry_records={})
    sid = store.save(ParityRecord("A", 1))
    store._clear_identity_caches()
    proxy = store.fetch(ParityRecord, sid)  # caches a proxy under (cls, sid)
    assert is_lazy_row(proxy)
    materialized = store.fetch_many(ParityRecord, [sid], eager=True)[0]
    # The eager prefilter must treat the proxy hit as a miss and re-materialize,
    # then overwrite the cache slot (materialized-wins).
    assert type(materialized) is ParityRecord
    assert store.fetch(ParityRecord, sid) is materialized


@pytest.mark.parametrize("records", [502, pytest.param(1500, marks=pytest.mark.extended)])
def test_fetch_many_batches_children_per_chunk(database, records: int):
    store = SqlStore(database, entry_records={})
    sids = [store.save(BatchRecord(str(index), [str(index)])) for index in range(records)]
    # Make the cold-cache condition deterministic: the weak identity caches
    # would usually have dropped the save-time temporaries already, but that
    # depends on GC timing rather than on anything this test asserts.
    store._clear_identity_caches()
    statements: list[str] = []

    def count(_conn, _cursor, statement, _parameters, _context, _executemany):
        if statement.lstrip().upper().startswith("SELECT"):
            statements.append(statement)

    sqlalchemy.event.listen(database.engine, "before_cursor_execute", count)
    try:
        fetched = store.fetch_many(BatchRecord, sids, eager=True)
    finally:
        sqlalchemy.event.remove(database.engine, "before_cursor_execute", count)
    assert [record.values for record in fetched] == [[str(index)] for index in range(records)]
    chunks = -(-records // 500)
    # One parent plus one child SELECT per 500-row chunk, far below the
    # per-row 2 x N of the unbatched fetch path.  The lower bound keeps the
    # assertion meaningful: a fully cached run would issue zero SELECTs and
    # pass the upper bound vacuously.
    assert chunks <= len(statements) <= 2 * chunks + 1


@pytest.mark.parametrize("records", [502, pytest.param(1500, marks=pytest.mark.extended)])
def test_lazy_fetch_many_batches_children_per_chunk(database, records: int):
    store = SqlStore(database, entry_records={})
    sids = [store.save(BatchRecord(str(index), [str(index)])) for index in range(records)]
    store._clear_identity_caches()
    fetched = store.fetch_many(BatchRecord, sids)
    assert all(is_lazy_row(record) for record in fetched)
    child_statements: list[str] = []

    def count(_conn, _cursor, statement, _parameters, _context, _executemany):
        if statement.lstrip().upper().startswith("SELECT") and "batch_record_values" in statement:
            child_statements.append(statement)

    # Count child SELECTs in a window AROUND the deferred .values accesses: the
    # lazy path must still batch children once per 500-row chunk, merely later.
    sqlalchemy.event.listen(database.engine, "before_cursor_execute", count)
    try:
        assert [record.values for record in fetched] == [[str(index)] for index in range(records)]
    finally:
        sqlalchemy.event.remove(database.engine, "before_cursor_execute", count)
    chunks = -(-records // 500)
    assert 1 <= len(child_statements) <= chunks


def test_repeated_default_fetch_returns_same_proxy_while_alive(database):
    store = SqlStore(database, entry_records={})
    sid = store.save(ParityRecord("A", 1))
    store._clear_identity_caches()
    first = store.fetch(ParityRecord, sid)
    assert is_lazy_row(first)
    assert store.fetch(ParityRecord, sid) is first


def test_dead_proxy_is_rehydrated_on_next_default_fetch(database):
    store = SqlStore(database, entry_records={})
    sid = store.save(ParityRecord("A", 1))
    store._clear_identity_caches()
    first = store.fetch(ParityRecord, sid)
    first_id = id(first)
    # Proxies sit in a reference cycle with their pinned chunk; only the cyclic
    # GC reclaims them, so force a collection before asserting re-hydration.
    del first
    gc.collect()
    second = store.fetch(ParityRecord, sid)
    assert is_lazy_row(second)
    assert second == ParityRecord("A", 1)
    assert id(second) != first_id


def test_materialized_wins_eager_then_lazy_returns_instance(database):
    store = SqlStore(database, entry_records={})
    sid = store.save(ParityRecord("A", 1))
    store._clear_identity_caches()
    materialized = store.fetch(ParityRecord, sid, eager=True)
    assert type(materialized) is ParityRecord
    # A live materialized instance takes precedence over creating a proxy.
    assert store.fetch(ParityRecord, sid) is materialized


def test_lazy_then_eager_replaces_cache_slot_with_fresh_instance(database):
    store = SqlStore(database, entry_records={})
    sid = store.save(ParityRecord("A", 1))
    store._clear_identity_caches()
    proxy = store.fetch(ParityRecord, sid)
    assert is_lazy_row(proxy)
    materialized = store.fetch(ParityRecord, sid, eager=True)
    assert type(materialized) is ParityRecord
    # The eager fetch overwrote the proxy's cache slot; the later default fetch
    # returns the materialized instance, not the still-live proxy.
    assert store.fetch(ParityRecord, sid) is materialized
    assert materialized == proxy


def test_save_lazy_row_registers_it_for_default_fetch(database):
    store = SqlStore(database, entry_records={})
    store.save(ParityRecord("A", 1))
    row = _row(store, ParityRecord)
    sid = store.save(row)
    assert store.fetch(ParityRecord, sid) is row


def test_no_child_or_reference_sql_until_access(database):
    store = SqlStore(database, entry_records={})
    sid = store.save(BatchRecord("A", ["x", "y"]))
    store._clear_identity_caches()
    statements: list[str] = []

    def count(_conn, _cursor, statement, _parameters, _context, _executemany):
        if statement.lstrip().upper().startswith("SELECT"):
            statements.append(statement)

    sqlalchemy.event.listen(database.engine, "before_cursor_execute", count)
    try:
        row = store.fetch(BatchRecord, sid)
        assert is_lazy_row(row)
        # Only the parent SELECT so far; the child table is untouched.
        assert not any("batch_record_values" in statement for statement in statements)
        assert len(statements) == 1
        assert row.values == ["x", "y"]
        assert sum("batch_record_values" in statement for statement in statements) == 1
    finally:
        sqlalchemy.event.remove(database.engine, "before_cursor_execute", count)


def test_child_of_lazy_record_is_lazy(database):
    store = SqlStore(database, entry_records={})
    child_sid = store.save(IdentityChild("child"))
    parent_sid = store.save(IdentityParent(store.fetch(IdentityChild, child_sid)))
    store._clear_identity_caches()
    parent = store.fetch(IdentityParent, parent_sid)
    assert is_lazy_row(parent)
    assert is_lazy_row(parent.child)
    assert parent.child.name == "child"


def test_lazy_fetch_many_one_parent_select_per_chunk(database):
    store = SqlStore(database, entry_records={})
    sids = [store.save(ParityRecord(str(index), index)) for index in range(502)]
    store._clear_identity_caches()
    statements: list[str] = []

    def count(_conn, _cursor, statement, _parameters, _context, _executemany):
        if statement.lstrip().upper().startswith("SELECT") and "parity_record" in statement:
            statements.append(statement)

    sqlalchemy.event.listen(database.engine, "before_cursor_execute", count)
    try:
        fetched = store.fetch_many(ParityRecord, sids)
    finally:
        sqlalchemy.event.remove(database.engine, "before_cursor_execute", count)
    assert all(is_lazy_row(record) for record in fetched)
    # Two 500-row chunks -> two parent SELECTs; no per-row queries.
    assert len(statements) == 2


def test_lazy_fetch_many_fully_cached_issues_no_sql(database):
    store = SqlStore(database, entry_records={})
    sids = [store.save(ParityRecord(str(index), index)) for index in range(3)]
    store._clear_identity_caches()
    proxies = store.fetch_many(ParityRecord, sids)  # populate the cache with proxies
    statements: list[str] = []

    def count(_conn, _cursor, statement, _parameters, _context, _executemany):
        statements.append(statement)

    sqlalchemy.event.listen(database.engine, "before_cursor_execute", count)
    try:
        again = store.fetch_many(ParityRecord, sids)
    finally:
        sqlalchemy.event.remove(database.engine, "before_cursor_execute", count)
    assert again == proxies
    assert statements == []


def test_lazy_and_eager_fetch_diverge_on_post_init(database):
    ValidatedRecord.calls.clear()
    store = SqlStore(database, entry_records={})
    sid = store.save(ValidatedRecord("A"))
    store._clear_identity_caches()
    ValidatedRecord.calls.clear()
    lazy = store.fetch(ValidatedRecord, sid)
    assert lazy.name == "A"
    # A lazy row exposes stored/codec values without re-running __post_init__.
    assert ValidatedRecord.calls == []
    store._clear_identity_caches()
    assert store.fetch(ValidatedRecord, sid, eager=True).name == "A"
    assert ValidatedRecord.calls == ["A"]


def test_rolled_back_transaction_expires_scalar_child_and_reference(database):
    store = SqlStore(database, entry_records={})
    scalar_sid = store.save(ParityRecord("A", 1))
    child_sid = store.save(BatchRecord("A", ["x"]))
    parent_sid = store.save(IdentityParent(IdentityChild("child")))
    store._clear_identity_caches()
    escaped: dict[str, object] = {}
    with pytest.raises(RuntimeError, match="boom"), store.transaction():
        escaped["scalar"] = store.fetch(ParityRecord, scalar_sid)
        escaped["child"] = store.fetch(BatchRecord, child_sid)
        escaped["reference"] = store.fetch(IdentityParent, parent_sid)
        # Read a scalar and a reference BEFORE the rollback: the pre-memo
        # expiry check must still fire for these already-read fields.
        assert escaped["scalar"].name == "A"
        assert is_lazy_row(escaped["reference"].child)
        raise RuntimeError("boom")
    with pytest.raises(ExpiredLazyRecordError, match=r"ParityRecord sid \d+.*rolled back"):
        _ = escaped["scalar"].name
    with pytest.raises(ExpiredLazyRecordError):
        _ = escaped["child"].values
    with pytest.raises(ExpiredLazyRecordError):
        _ = escaped["reference"].child


def test_child_read_inside_rolled_back_transaction_expires_only_that_field(database):
    store = SqlStore(database, entry_records={})
    sid = store.save(BatchRecord("A", ["x"]))
    store._clear_identity_caches()
    # The chunk is born OUTSIDE any transaction; its scalar is read now.
    row = store.fetch(BatchRecord, sid)
    assert row.name == "A"
    with pytest.raises(RuntimeError, match="boom"), store.transaction():
        # The deferred child read executes on the rolled-back transaction's
        # connection, so it captures that transaction's token per-read.
        assert row.values == ["x"]
        raise RuntimeError("boom")
    # The scalar (parent rows, no token) is unaffected; only the child expires.
    assert row.name == "A"
    with pytest.raises(ExpiredLazyRecordError):
        _ = row.values


def test_committed_transaction_proxies_keep_working(database):
    store = SqlStore(database, entry_records={})
    sid = store.save(ParityRecord("A", 1))
    store._clear_identity_caches()
    with store.transaction():
        row = store.fetch(ParityRecord, sid)
    assert row.name == "A"
    assert row.number == 1


def test_outside_transaction_proxy_unaffected_by_unrelated_rollback(database):
    store = SqlStore(database, entry_records={})
    sid = store.save(ParityRecord("A", 1))
    store._clear_identity_caches()
    row = store.fetch(ParityRecord, sid)
    assert row.name == "A"
    with pytest.raises(RuntimeError, match="boom"), store.transaction():
        store.save(ParityRecord("B", 2))
        raise RuntimeError("boom")
    assert row.name == "A"
    assert row.number == 1


def test_fetch_by_content_id_lazy_default_and_eager(database):
    store = SqlStore(database, entry_records={})
    key = content_id(ParityRecord("A", 1))
    store.save(ParityRecord("A", 1))
    store._clear_identity_caches()
    assert is_lazy_row(store.fetch_by_content_id(ParityRecord, key))
    store._clear_identity_caches()
    assert type(store.fetch_by_content_id(ParityRecord, key, eager=True)) is ParityRecord


def test_referring_lazy_default_and_eager(database):
    store = SqlStore(database, entry_records={})
    child_sid = store.save(IdentityChild("child"))
    store.save(IdentityParent(store.fetch(IdentityChild, child_sid, eager=True)))
    store._clear_identity_caches()
    child = store.fetch(IdentityChild, child_sid, eager=True)
    lazy = store.referring(IdentityParent, field="child", to=child)
    assert lazy and all(is_lazy_row(record) for record in lazy)
    store._clear_identity_caches()
    child = store.fetch(IdentityChild, child_sid, eager=True)
    eager = store.referring(IdentityParent, field="child", to=child, eager=True)
    assert eager and all(type(record) is IdentityParent for record in eager)


def test_deferred_reference_deletion_raises_stale_result(database):
    store = SqlStore(database, entry_records={})
    parent_sid = store.save(IdentityParent(IdentityChild("child")))
    child_sid = store.sid_of(IdentityChild("child"))
    store._clear_identity_caches()
    parent = store.fetch(IdentityParent, parent_sid)
    with database.engine.begin() as connection:
        connection.execute(sqlalchemy.text(f"DELETE FROM identity_child WHERE sid = {child_sid}"))
    with pytest.raises(StaleResultError, match="IdentityChild"):
        _ = parent.child
