"""Live ``logical_id`` lineage, replace/history and ``only_latest`` coverage for :mod:`httk.store.mongo`.

Every test needs a real replica-set MongoDB and is env-gated through the
``mongo_test_database`` fixture (skips when ``HTTK_TEST_MONGODB_URI`` is unset).
The Mongo end-to-end behaviour is therefore verified by CI, not by a machine
without a MongoDB deployment.
"""

from dataclasses import dataclass
from dataclasses import replace as dataclasses_replace
from typing import ClassVar

import pytest
from httk.core.storage import StorageInfo
from test_db_stored_properties import FIRST, CalculationEntry

from httk.store.db.schema import resolve_schema
from httk.store.db.store import EntryReplacementError
from httk.store.mongo import MongoStore
from httk.store.mongo.mapping import collection_name_for


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


@dataclass(frozen=True)
class WidgetRef:
    widget: Widget
    tag: str


def _store(database, **kwargs) -> MongoStore:
    return MongoStore(database, entry_records={}, **kwargs)


def _table(cls: type) -> str:
    return collection_name_for(resolve_schema(cls))


def _logical_id(database, cls: type, sid: int) -> int:
    document = database.database[_table(cls)].find_one({"_id": sid}, {"logical_id": 1})
    assert document is not None
    return int(document["logical_id"])


def _count(database, cls: type) -> int:
    return database.database[_table(cls)].count_documents({})


def _search_values(store: MongoStore, cls: type, **searcher_kwargs) -> list[int]:
    searcher = store.searcher(**searcher_kwargs)
    variable = searcher.variable(cls)
    searcher.output(variable, "record")
    return sorted(row[0][0].value for row in searcher)


def test_fresh_save_logical_id_equals_own_sid(mongo_test_database) -> None:
    store = _store(mongo_test_database)
    sid = store.save(Widget(1))
    assert _logical_id(mongo_test_database, Widget, sid) == sid


def test_nested_records_keep_their_own_sid_lineage(mongo_test_database) -> None:
    store = _store(mongo_test_database)
    store.save(Holder(Leaf(7), "a"))
    leaves = mongo_test_database.database[_table(Leaf)]
    documents = list(leaves.find({}, {"_id": 1, "logical_id": 1}))
    assert documents
    assert all(int(document["_id"]) == int(document["logical_id"]) for document in documents)


def test_replace_shares_lineage_history_and_leaves_both_documents(mongo_test_database) -> None:
    store = _store(mongo_test_database)
    first = Widget(1)
    a = store.save(first)
    b = store.replace(first, Widget(2))
    assert b != a
    assert _logical_id(mongo_test_database, Widget, b) == _logical_id(mongo_test_database, Widget, a) == a

    # A plain fetch/search still returns both the replaced and the replacement.
    assert store.fetch(Widget, a).value == 1
    assert store.fetch(Widget, b).value == 2
    assert _search_values(store, Widget) == [1, 2]

    # history is the lineage in sid order, from either member.
    assert tuple(w.value for w in store.history(store.fetch(Widget, b))) == (1, 2)
    assert tuple(w.value for w in store.history(store.fetch(Widget, a))) == (1, 2)

    # A chained replacement keeps the original lineage.
    c = store.replace(store.fetch(Widget, b), Widget(3))
    assert _logical_id(mongo_test_database, Widget, c) == a
    assert tuple(w.value for w in store.history(store.fetch(Widget, c))) == (1, 2, 3)


def test_idempotent_re_replace_returns_existing_sid_without_a_new_document(mongo_test_database) -> None:
    store = _store(mongo_test_database)
    first = Widget(1)
    a = store.save(first)
    b = store.replace(first, Widget(2))
    assert _count(mongo_test_database, Widget) == 2

    # Same replacement content again: idempotent no-op on the same lineage.
    assert store.replace(store.fetch(Widget, b), Widget(2)) == b
    # Replacing with the predecessor's own content is likewise a no-op.
    assert store.replace(first, Widget(1)) == a
    assert _count(mongo_test_database, Widget) == 2


def test_cross_lineage_dedup_collision_raises_content_id(mongo_test_database) -> None:
    store = _store(mongo_test_database)
    predecessor = Widget(1)
    store.save(predecessor)
    store.save(Widget(99))  # an independent lineage
    with pytest.raises(EntryReplacementError, match="logical_id"):
        store.replace(predecessor, Widget(99))


def test_by_value_replacement_and_cross_lineage_collision(mongo_test_database) -> None:
    store = _store(mongo_test_database)
    predecessor = ValueWidget(1)
    a = store.save(predecessor)
    b = store.replace(predecessor, ValueWidget(2))
    assert _logical_id(mongo_test_database, ValueWidget, b) == a
    store.save(ValueWidget(50))  # independent lineage
    with pytest.raises(EntryReplacementError):
        store.replace(predecessor, ValueWidget(50))


def test_dedup_none_replacement_always_inserts_a_new_document(mongo_test_database) -> None:
    store = _store(mongo_test_database)
    predecessor = NoneWidget(1)
    a = store.save(predecessor)
    b = store.replace(predecessor, NoneWidget(1))  # equal content, but dedup="none"
    assert b != a
    assert _logical_id(mongo_test_database, NoneWidget, b) == a
    assert tuple(w.value for w in store.history(store.fetch(NoneWidget, b))) == (1, 1)


def test_replace_and_history_reject_a_never_stored_object(mongo_test_database) -> None:
    store = _store(mongo_test_database)
    store.save(Widget(1))
    with pytest.raises(ValueError, match="has not been stored or fetched"):
        store.replace(Widget(777), Widget(778))
    with pytest.raises(ValueError, match="has not been stored or fetched"):
        store.history(Widget(777))


def test_replace_rejects_a_cross_collection_object(mongo_test_database) -> None:
    store = _store(mongo_test_database)
    widget = Widget(1)
    store.save(widget)
    with pytest.raises(ValueError, match="cannot replace a record"):
        store.replace(widget, ValueWidget(1))


def test_only_latest_returns_only_the_latest_of_each_lineage(mongo_test_database) -> None:
    store = _store(mongo_test_database)
    first = Widget(1)
    store.save(first)
    b = store.replace(first, Widget(2))
    store.replace(store.fetch(Widget, b), Widget(3))
    store.save(Widget(100))  # an independent single-row lineage

    # A plain search returns every row of every lineage.
    assert _search_values(store, Widget) == [1, 2, 3, 100]
    # only_latest keeps just the highest-sid row of each lineage.
    assert _search_values(store, Widget, only_latest=True) == [3, 100]


def test_only_latest_with_as_of_returns_the_middle_row(mongo_test_database) -> None:
    store = _store(mongo_test_database)
    first = Widget(1)
    store._clock = lambda: 1_000_000
    store.save(first)  # v1
    store._clock = lambda: 3_000_000
    b = store.replace(first, Widget(2))  # v2
    store._clock = lambda: 5_000_000
    store.replace(store.fetch(Widget, b), Widget(3))  # v3

    # A cutoff between v2 and v3 hides v3 entirely; without only_latest both
    # earlier rows remain visible.
    assert _search_values(store, Widget, as_of=4_000_000) == [1, 2]
    # only_latest as of that cutoff yields the latest row not past the cutoff:
    # v3 is excluded by as_of, so v2 is the latest of the lineage.
    assert _search_values(store, Widget, as_of=4_000_000, only_latest=True) == [2]


def test_only_latest_leaves_referenced_records_unfiltered(mongo_test_database) -> None:
    store = _store(mongo_test_database)
    widget = Widget(1)
    store.save(widget)
    store.save(WidgetRef(widget, "pin"))
    store.replace(widget, Widget(2))  # the reference stays pinned to the replaced row

    searcher = store.searcher(only_latest=True)
    variable = searcher.variable(WidgetRef)
    searcher.output(variable, "record")
    records = [row[0][0] for row in searcher]
    assert len(records) == 1
    # The root WidgetRef is latest-filtered, but its pinned reference still
    # resolves the replaced widget rather than the lineage's latest row.
    assert records[0].widget.value == 1


def test_variable_logical_id_filters_and_outputs(mongo_test_database) -> None:
    store = _store(mongo_test_database)
    first = Widget(1)
    a = store.save(first)
    b = store.replace(first, Widget(2))
    store.save(Widget(100))  # an independent lineage whose logical_id is its own sid

    # logical_id is projectable as an output (a top-level document field).
    searcher = store.searcher()
    variable = searcher.variable(Widget)
    searcher.output(variable.sid, "sid")
    searcher.output(variable.logical_id, "logical_id")
    lineage = {int(row[0][0]): int(row[0][1]) for row in searcher}
    assert lineage[a] == a
    assert lineage[b] == a

    # logical_id is also filterable, selecting a whole lineage.
    filtered = store.searcher()
    filtered_variable = filtered.variable(Widget)
    filtered.add(filtered_variable.logical_id == a)
    filtered.output(filtered_variable, "record")
    assert sorted(row[0][0].value for row in filtered) == [1, 2]


def test_stored_property_plan_threads_only_latest(mongo_test_database) -> None:
    from test_db_stored_properties import GenericCalculationFirst, GenericCalculationSecond

    store = MongoStore(
        mongo_test_database,
        entry_records={CalculationEntry: (GenericCalculationFirst, GenericCalculationSecond)},
    )
    store.save(FIRST)
    store.replace(FIRST, dataclasses_replace(FIRST, label="first-replaced"))  # same lineage, distinct content
    plan = store.stored_property_plan(CalculationEntry)

    def _candidate_count(**kwargs) -> int:
        streams = plan.candidate_searchers(None, **kwargs)
        return sum(1 for stream in streams for _row in stream.searcher)

    assert _candidate_count() == 2
    assert _candidate_count(only_latest=True) == 1
