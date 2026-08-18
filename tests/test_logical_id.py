"""Store-managed ``logical_id`` lineage column, write-path fill, and replace/history."""

from dataclasses import dataclass
from typing import ClassVar

import pytest
import sqlalchemy
from httk.core import PropertyDefinition, load_entry_type_definition
from httk.core.register import register_entry_family, register_entry_record
from httk.core.storage import StorageInfo, StoredPropertyProjection

from httk.store.backend.schema import SchemaError, resolve_schema
from httk.store.backend.sql import Backend, EntryReplacementError, SqlStore, stored_property_sql_plan
from httk.store.backend.sql.mapping import LOGICAL_ID_COLUMN, sqlalchemy_metadata
from httk.store.backend.sql.optimade import optimade_filter_searcher

LOGICAL_ID_CALC_DEFINITION = "https://schemas.optimade.org/defs/v1.3/entrytypes/optimade/calculations"


class LogicalIdCalculation:
    type = "calculations"
    definition_id = LOGICAL_ID_CALC_DEFINITION

    @staticmethod
    def entry_type_definition():
        return load_entry_type_definition(LOGICAL_ID_CALC_DEFINITION).extended(
            {
                "_httk_logical_id": PropertyDefinition.from_simple(
                    "_httk_logical_id",
                    description="The store-managed lineage id, served for tests.",
                    fulltype="integer",
                ),
            }
        )


def _logical_id_query(context, operator, literal):
    return context.compare(context.field("logical_id"), operator, context.constant(literal))


@dataclass(frozen=True)
class LogicalIdBacking:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="logical_id_backing")

    label: str

    __httk_stored_properties__: ClassVar = {
        "immutable_id": StoredPropertyProjection(
            response=lambda record: record.label,
            query=lambda context, operator, literal: context.compare(
                context.field("label"), operator, context.constant(literal)
            ),
            sort=lambda context: context.field("label"),
        ),
        # The store manages the value (response overridden by the plan), but the
        # query and sort reach the lineage column through the query context.
        "_httk_logical_id": StoredPropertyProjection(
            response=lambda _record: None,
            query=_logical_id_query,
            sort=lambda context: context.field("logical_id"),
        ),
    }


register_entry_family(
    name="test-logical-id-calculations",
    family=f"{__name__}:LogicalIdCalculation",
    definition_id=LOGICAL_ID_CALC_DEFINITION,
)
register_entry_record(
    name="test-logical-id-backing",
    family="test-logical-id-calculations",
    record=f"{__name__}:LogicalIdBacking",
)


@dataclass(frozen=True)
class Widget:
    value: int


@dataclass(frozen=True)
class ValueWidget:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(dedup="by_value")
    value: int


@dataclass(frozen=True)
class NoneWidget:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(dedup="none")
    value: int


@dataclass(frozen=True)
class Leaf:
    n: int


@dataclass(frozen=True)
class Holder:
    leaf: Leaf
    label: str


def _skip_non_sql(store: object) -> None:
    if not isinstance(store, SqlStore):
        pytest.skip("logical_id is a SQL-layer feature; the Mongo store is handled by a sibling packet")


def _rows(store: SqlStore, table_name: str) -> list[tuple[int, int]]:
    with store._database.engine.connect() as connection:
        return [
            (int(sid), int(logical_id))
            for sid, logical_id in connection.execute(
                sqlalchemy.text(f"SELECT sid, logical_id FROM {table_name} ORDER BY sid")
            ).all()
        ]


def _logical_id(store: SqlStore, table_name: str, sid: int) -> int:
    with store._database.engine.connect() as connection:
        return int(
            connection.execute(
                sqlalchemy.text(f"SELECT logical_id FROM {table_name} WHERE sid = :sid"),
                {"sid": sid},
            ).scalar_one()
        )


def test_parent_table_has_unconditional_logical_id_column_and_index():
    table = sqlalchemy_metadata([resolve_schema(Widget)]).tables["widget"]
    assert isinstance(table.c[LOGICAL_ID_COLUMN].type, sqlalchemy.BigInteger)
    assert not table.c[LOGICAL_ID_COLUMN].nullable
    assert any(index.name == "ix_widget_logical_id" for index in table.indexes)
    # Unlike store_timestamp, the column is unconditional: still present when
    # timestamps are disabled.
    disabled = sqlalchemy_metadata([resolve_schema(Widget)], store_timestamps=False).tables["widget"]
    assert LOGICAL_ID_COLUMN in disabled.c


def test_logical_id_is_a_reserved_field_name():
    @dataclass(frozen=True)
    class BadRecord:
        logical_id: int

    with pytest.raises(SchemaError, match="reserved"):
        resolve_schema(BadRecord)


def test_fresh_save_logical_id_equals_own_sid(store_factory):
    store = store_factory()
    _skip_non_sql(store)
    sid = store.save(Widget(1))
    assert _logical_id(store, "widget", sid) == sid


def test_nested_records_keep_their_own_sid_lineage(store_factory):
    store = store_factory()
    _skip_non_sql(store)
    store.save(Holder(Leaf(7), "a"))
    leaf_rows = _rows(store, "leaf")
    assert leaf_rows
    assert all(sid == logical_id for sid, logical_id in leaf_rows)


def test_replace_shares_lineage_history_and_leaves_both_rows(store_factory):
    store = store_factory()
    _skip_non_sql(store)
    first = Widget(1)
    a = store.save(first)
    b = store.replace(first, Widget(2))
    assert b != a
    assert _logical_id(store, "widget", b) == _logical_id(store, "widget", a) == a

    # A plain fetch/search still returns both the replaced and the replacement.
    assert store.fetch(Widget, a).value == 1
    assert store.fetch(Widget, b).value == 2
    searcher = store.searcher()
    variable = searcher.variable(Widget)
    searcher.output(variable, "record")
    assert sorted(row[0][0].value for row in searcher) == [1, 2]

    # history is the lineage in sid order, from either member.
    assert tuple(w.value for w in store.history(store.fetch(Widget, b))) == (1, 2)
    assert tuple(w.value for w in store.history(store.fetch(Widget, a))) == (1, 2)

    # A chained replacement keeps the original lineage.
    c = store.replace(store.fetch(Widget, b), Widget(3))
    assert _logical_id(store, "widget", c) == a
    assert tuple(w.value for w in store.history(store.fetch(Widget, c))) == (1, 2, 3)


def test_idempotent_re_replace_returns_existing_sid_without_a_new_row(store_factory):
    store = store_factory()
    _skip_non_sql(store)
    first = Widget(1)
    a = store.save(first)
    b = store.replace(first, Widget(2))
    assert len(_rows(store, "widget")) == 2

    # Same replacement content again: idempotent no-op on the same lineage.
    assert store.replace(store.fetch(Widget, b), Widget(2)) == b
    # Replacing with the predecessor's own content is likewise a no-op.
    assert store.replace(first, Widget(1)) == a
    assert len(_rows(store, "widget")) == 2


def test_cross_lineage_dedup_collision_raises_content_id(store_factory):
    store = store_factory()
    _skip_non_sql(store)
    predecessor = Widget(1)
    store.save(predecessor)
    store.save(Widget(99))  # an independent lineage
    with pytest.raises(EntryReplacementError, match="logical_id"):
        store.replace(predecessor, Widget(99))


def test_by_value_replacement_and_cross_lineage_collision(store_factory):
    store = store_factory()
    _skip_non_sql(store)
    predecessor = ValueWidget(1)
    a = store.save(predecessor)
    b = store.replace(predecessor, ValueWidget(2))
    assert _logical_id(store, "value_widget", b) == a
    store.save(ValueWidget(50))  # independent lineage
    with pytest.raises(EntryReplacementError):
        store.replace(predecessor, ValueWidget(50))


def test_dedup_none_replacement_always_inserts_a_new_row(store_factory):
    store = store_factory()
    _skip_non_sql(store)
    predecessor = NoneWidget(1)
    a = store.save(predecessor)
    b = store.replace(predecessor, NoneWidget(1))  # equal content, but dedup="none"
    assert b != a
    assert _logical_id(store, "none_widget", b) == a
    assert tuple(w.value for w in store.history(store.fetch(NoneWidget, b))) == (1, 1)


def test_replace_and_history_reject_a_never_stored_object(store_factory):
    store = store_factory()
    _skip_non_sql(store)
    store.save(Widget(1))
    with pytest.raises(ValueError, match="has not been stored or fetched"):
        store.replace(Widget(777), Widget(778))
    with pytest.raises(ValueError, match="has not been stored or fetched"):
        store.history(Widget(777))


def test_replace_rejects_a_cross_table_object(store_factory):
    store = store_factory()
    _skip_non_sql(store)
    widget = Widget(1)
    store.save(widget)
    with pytest.raises(ValueError, match="cannot replace a record"):
        store.replace(widget, ValueWidget(1))


def test_logical_id_with_store_timestamps_disabled():
    with Backend.sqlite() as database:
        store = SqlStore(database, entry_records={}, store_timestamps=False)
        first = Widget(1)
        a = store.save(first)
        assert _logical_id(store, "widget", a) == a
        b = store.replace(first, Widget(2))
        assert _logical_id(store, "widget", b) == a
        assert tuple(w.value for w in store.history(store.fetch(Widget, b))) == (1, 2)


def test_logical_id_degraded_write_profile():
    with Backend.sqlite(degraded=True) as database:
        store = SqlStore(database, entry_records={})
        first = Widget(1)
        a = store.save(first)
        assert _logical_id(store, "widget", a) == a
        b = store.replace(first, Widget(2))
        assert _logical_id(store, "widget", b) == a
        assert tuple(w.value for w in store.history(store.fetch(Widget, b))) == (1, 2)
        # Idempotent re-replace on the degraded path deduplicates without a new row.
        assert store.replace(store.fetch(Widget, b), Widget(2)) == b
        assert len(_rows(store, "widget")) == 2


def test_logical_id_searcher_output_filter_and_sort(store_factory):
    store = store_factory()
    _skip_non_sql(store)
    first = Widget(1)
    a = store.save(first)
    b = store.replace(first, Widget(2))
    store.save(Widget(100))  # an independent lineage; its logical_id is its own sid

    # logical_id is projectable as a scalar output (the served value channel).
    exposed = store.searcher()
    variable = exposed.variable(Widget)
    exposed.output(variable.sid, "sid")
    exposed.output(variable.logical_id, "logical_id")
    lineage = {int(row[0][0]): int(row[0][1]) for row in exposed}
    assert lineage[a] == a
    assert lineage[b] == a

    # logical_id is filterable, selecting a whole lineage.
    filtered = store.searcher()
    filtered_variable = filtered.variable(Widget)
    filtered.add(filtered_variable.logical_id == a)
    filtered.output(filtered_variable, "record")
    assert sorted(row[0][0].value for row in filtered) == [1, 2]

    # logical_id is sortable; the independent, higher-sid lineage sorts last.
    ascending = store.searcher()
    asc = ascending.variable(Widget)
    ascending.output(asc, "record")
    ascending.add_sort(asc.logical_id)
    assert [row[0][0].value for row in ascending][-1] == 100


def test_optimade_filter_searcher_selects_lineage_by_logical_id(store_factory):
    store = store_factory()
    _skip_non_sql(store)
    first = Widget(1)
    a = store.save(first)
    store.replace(first, Widget(2))
    store.save(Widget(100))  # an independent lineage

    searcher = optimade_filter_searcher(store, Widget, f"_httk_logical_id = {a}")
    assert {item[0][0].value for item in searcher} == {1, 2}


def test_stored_property_plan_serves_filters_and_sorts_logical_id(store_factory):
    store = store_factory(entry_records={LogicalIdCalculation: (LogicalIdBacking,)})
    _skip_non_sql(store)
    first = LogicalIdBacking("first")
    a = store.save(first)
    store.replace(first, LogicalIdBacking("second"))
    store.save(LogicalIdBacking("independent"))  # a separate lineage
    plan = stored_property_sql_plan(store, LogicalIdCalculation)

    # The served row carries the lineage id, unconditional and unscaled.
    rows = {row["immutable_id"]: row["_httk_logical_id"] for row in plan.records()}
    assert rows["first"] == a
    assert rows["second"] == a
    assert rows["independent"] != a

    # Filtering by _httk_logical_id selects the whole lineage.
    searchers = plan.filter_searchers(f"_httk_logical_id = {a}")
    assert sorted(result[0][0].label for searcher in searchers for result in searcher) == ["first", "second"]

    # Sorting by _httk_logical_id compiles and orders by the lineage id.
    ordered = plan.filter_searchers("_httk_logical_id >= 0", sort=(("_httk_logical_id", False),))
    assert [result[0][0].label for searcher in ordered for result in searcher][-1] == "independent"
