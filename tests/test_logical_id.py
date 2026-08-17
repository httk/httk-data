"""Store-managed ``logical_id`` lineage column, write-path fill, and replace/history."""

from dataclasses import dataclass
from typing import ClassVar

import pytest
import sqlalchemy
from httk.core.storage import StorageInfo

from httk.store.db import Database, EntryReplacementError, SchemaError, SqlStore
from httk.store.db.mapping import LOGICAL_ID_COLUMN, sqlalchemy_metadata
from httk.store.db.schema import resolve_schema


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
    with Database.sqlite() as database:
        store = SqlStore(database, entry_records={}, store_timestamps=False)
        first = Widget(1)
        a = store.save(first)
        assert _logical_id(store, "widget", a) == a
        b = store.replace(first, Widget(2))
        assert _logical_id(store, "widget", b) == a
        assert tuple(w.value for w in store.history(store.fetch(Widget, b))) == (1, 2)


def test_logical_id_degraded_write_profile():
    with Database.sqlite(degraded=True) as database:
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
