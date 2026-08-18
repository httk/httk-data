"""``logical_id`` parity across every bulk-ingest path.

A bulk-ingested row is always its own lineage root, so ``logical_id`` must
equal that row's *final* stored sid after any dedup collapse or parallel
sid compaction.  These tests pin that invariant for the sequential, parallel
(compacting merge) and deferred (pure-SQL merge) paths, and — crucially —
prove that ingesting into a store that already holds ``replace()`` lineages
(rows whose ``logical_id != sid``) never disturbs those pre-existing rows.
The parallel/deferred paths only run on ``SqlStore``; Mongo/Postgres are
skipped through the shared ``store_factory`` fixture.
"""

from dataclasses import dataclass
from typing import ClassVar

import pytest
import sqlalchemy
from httk.core.storage import StorageInfo

from httk.store.backend.sql import SqlStore

# --------------------------------------------------------------------- records


@dataclass(frozen=True)
class Widget:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="lidbulk_widget")

    value: int


@dataclass(frozen=True)
class ValueWidget:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="lidbulk_value", dedup="by_value")

    value: int


@dataclass(frozen=True)
class Leaf:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="lidbulk_leaf")

    n: int


@dataclass(frozen=True)
class Holder:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="lidbulk_holder")

    leaf: Leaf
    label: str


# --------------------------------------------------------------------- helpers


def _skip_non_sql(store: object) -> None:
    if not isinstance(store, SqlStore):
        pytest.skip("logical_id is a SQL-layer feature; the Mongo store is handled by a sibling packet")


def _rows(store: SqlStore, table: str) -> list[tuple[int, int]]:
    with store._database.engine.connect() as connection:
        return [
            (int(sid), int(logical_id))
            for sid, logical_id in connection.execute(
                sqlalchemy.text(f'SELECT sid, logical_id FROM "{table}" ORDER BY sid')
            ).all()
        ]


def _logical_id(store: SqlStore, table: str, sid: int) -> int:
    with store._database.engine.connect() as connection:
        return int(
            connection.execute(
                sqlalchemy.text(f'SELECT logical_id FROM "{table}" WHERE sid = :sid'),
                {"sid": sid},
            ).scalar_one()
        )


def _bulk_kwargs(mode: str) -> dict[str, object]:
    if mode == "sequential":
        return {}
    if mode == "parallel":
        return {"workers": 2}
    return {"workers": 2, "finalize": "deferred"}


_MODES = ["sequential", "parallel", "deferred"]


# --------------------------------------------------------------------- tests


@pytest.mark.parametrize("mode", _MODES)
def test_bulk_row_logical_id_equals_own_final_sid(store_factory, mode):
    store = store_factory()
    _skip_non_sql(store)
    with store.bulk_ingest(**_bulk_kwargs(mode)) as bulk:
        for value in range(6):
            bulk.save(Widget(value))
    rows = _rows(store, "lidbulk_widget")
    assert len(rows) == 6
    assert all(sid == logical_id for sid, logical_id in rows)


@pytest.mark.parametrize("mode", _MODES)
def test_parallel_referrers_stay_intact_and_children_root_their_lineage(store_factory, mode):
    store = store_factory()
    _skip_non_sql(store)
    tokens = []
    with store.bulk_ingest(**_bulk_kwargs(mode)) as bulk:
        for index in range(10):
            tokens.append((index, bulk.save(Holder(Leaf(index), f"h{index}"))))

    # Both the parent and its referenced-child table root every lineage at their
    # own final sid, even after compaction renumbered the child sids.
    for table in ("lidbulk_leaf", "lidbulk_holder"):
        rows = _rows(store, table)
        assert rows
        assert all(sid == logical_id for sid, logical_id in rows)

    # Referrer integrity end to end: every holder still resolves its own leaf.
    reopened = store_factory.reopen(store)
    for index, token in tokens:
        holder = reopened.fetch(Holder, bulk.resolved_sid(Holder, token))
        assert holder.leaf.n == index
        assert holder.label == f"h{index}"


@pytest.mark.parametrize("mode", _MODES)
def test_by_value_dedup_collapses_to_one_row_with_logical_id(store_factory, mode):
    store = store_factory()
    _skip_non_sql(store)
    with store.bulk_ingest(**_bulk_kwargs(mode)) as bulk:
        for _ in range(5):
            bulk.save(ValueWidget(7))
    rows = _rows(store, "lidbulk_value")
    # logical_id must not participate in value identity, or dedup would never
    # collapse (every fresh row carries a distinct logical_id).
    assert len(rows) == 1
    sid, logical_id = rows[0]
    assert sid == logical_id


def test_preexisting_replacement_lineage_survives_bulk(store_factory):
    # Only the sequential path appends into a pre-populated store; the parallel
    # and deferred merges both require a physically empty store.
    store = store_factory()
    _skip_non_sql(store)
    first = Widget(1)
    a = store.save(first)
    b = store.replace(first, Widget(2))
    assert b != a
    assert _logical_id(store, "lidbulk_widget", a) == a
    assert _logical_id(store, "lidbulk_widget", b) == a

    with store.bulk_ingest() as bulk:
        for value in range(100, 104):
            bulk.save(Widget(value))

    # The pre-existing replacement lineage is untouched by the bulk fill.
    assert _logical_id(store, "lidbulk_widget", a) == a
    assert _logical_id(store, "lidbulk_widget", b) == a
    # Every newly ingested row roots its own lineage.
    fresh = [(sid, logical_id) for sid, logical_id in _rows(store, "lidbulk_widget") if sid not in (a, b)]
    assert len(fresh) == 4
    assert all(sid == logical_id for sid, logical_id in fresh)


def test_content_dedup_against_existing_replaced_row_leaves_it_untouched(store_factory):
    # Content-id dedup against a pre-existing row is a sequential-path concern;
    # parallel/deferred require an empty store.
    store = store_factory()
    _skip_non_sql(store)
    first = Widget(1)
    a = store.save(first)
    b = store.replace(first, Widget(2))  # row b holds content_id(Widget(2)), logical_id == a
    assert _logical_id(store, "lidbulk_widget", b) == a
    before = _rows(store, "lidbulk_widget")

    with store.bulk_ingest() as bulk:
        token = bulk.save(Widget(2))  # collides on content_id with the existing row b

    # No new row for the colliding content, and the existing (differently
    # rooted) row is untouched.
    assert _rows(store, "lidbulk_widget") == before
    assert _logical_id(store, "lidbulk_widget", b) == a
    assert bulk.resolved_sid(Widget, token) == b
