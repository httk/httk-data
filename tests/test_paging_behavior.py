"""Backend-neutral page traversal and cursor rejection behavior."""

from dataclasses import dataclass
from functools import cmp_to_key

import pytest
from clickhouse_read_support import bulk_store

from httk.data import ContinuationToken, PageOrder, PaginationCursorError, UnsupportedQueryError

pytestmark = pytest.mark.xdist_group("clickhouse_read_corpus")


@dataclass(frozen=True)
class PageRecord:
    bucket: int | None
    score: int | None
    label: str
    tags: list[str]


@dataclass(frozen=True)
class ChangedPageRecord:
    bucket: int | None
    score: int | None
    label: str
    revision: int


ROWS = (
    PageRecord(2, 1, "b-1", ["common", "red"]),
    PageRecord(1, 2, "a-2", ["common"]),
    PageRecord(None, 1, "null-1", ["common", "blue"]),
    PageRecord(2, 1, "b-2", ["red"]),
    PageRecord(1, None, "a-null", ["common", "blue"]),
    PageRecord(None, None, "null-null", []),
    PageRecord(2, 3, "b-3", ["common"]),
    PageRecord(1, 2, "a-2-second", ["common", "red"]),
)


@pytest.fixture(autouse=True)
def _require_query_support(store_factory):
    """Paging behavior is deferred for backends without a searcher."""
    if not hasattr(store_factory(), "searcher") or not getattr(store_factory(), "supports_page", True):
        pytest.skip("backend has no query support yet")


@pytest.fixture
def paging_store(store_factory):
    store = store_factory()
    for row in ROWS:
        store.save(row)
    return store


@pytest.fixture(scope="module")
def clickhouse_paging_store():
    with bulk_store(ROWS) as store:
        yield store


def test_clickhouse_bulk_paging_behavior(clickhouse_paging_store):
    result = results(clickhouse_paging_store)
    order = (PageOrder("bucket", nulls="first"), PageOrder("score", descending=True, nulls="last"))
    page = result.page(size=2, order_by=order)
    found = labels(page)
    while page.next is not None:
        page = result.page(size=2, order_by=order, cursor=page.next)
        found.extend(labels(page))
    assert found == expected_labels(order)


def results(store, *, common_only: bool = False):
    searcher = store.searcher()
    record = searcher.variable(PageRecord)
    if common_only:
        searcher.add(record.tags.has("common"))
        searcher.add(record.tags.has_only("common", "red", "blue"))
    return searcher.results(record=record, bucket=record.bucket, score=record.score, label=record.label)


def labels(page) -> list[str]:
    return [row.label for row in page.rows]


def all_pages(result, order: tuple[PageOrder, ...], size: int = 2) -> list[str]:
    page = result.page(size=size, order_by=order)
    found = labels(page)
    while page.next is not None:
        page = result.page(size=size, order_by=order, cursor=page.next)
        found.extend(labels(page))
    return found


def expected_labels(order: tuple[PageOrder, ...]) -> list[str]:
    def compare(left: tuple[int, PageRecord], right: tuple[int, PageRecord]) -> int:
        for item in order:
            left_value = getattr(left[1], item.name)
            right_value = getattr(right[1], item.name)
            if left_value is None or right_value is None:
                if left_value is right_value:
                    continue
                if left_value is None:
                    return -1 if item.nulls == "first" else 1
                return 1 if item.nulls == "first" else -1
            if left_value != right_value:
                comparison = -1 if left_value < right_value else 1
                return -comparison if item.descending else comparison
        return -1 if left[0] < right[0] else 1 if left[0] > right[0] else 0

    return [row.label for _index, row in sorted(enumerate(ROWS, start=1), key=cmp_to_key(compare))]


def backwards_to_start(result, page, order: tuple[PageOrder, ...], size: int) -> list[str]:
    chunks = [labels(page)]
    while page.previous is not None:
        page = result.page(size=size, order_by=order, cursor=page.previous)
        chunks.append(labels(page))
    return [label for chunk in reversed(chunks) for label in chunk]


def test_empty_store_page_is_empty(store_factory):
    result_store = store_factory()
    searcher = result_store.searcher()
    record = searcher.variable(PageRecord)
    page = searcher.results(record=record).page(size=2, order_by=(), include_total=True)
    assert page.rows == ()
    assert page.next is None
    assert page.previous is None
    assert page.total == 0


def test_pages_are_duplicate_free_stable_and_reversible(paging_store):
    result = results(paging_store)
    order = (PageOrder("bucket", nulls="first"), PageOrder("score", descending=True, nulls="last"))
    first = result.page(size=2, order_by=order)
    second = result.page(size=2, order_by=order, cursor=first.next)
    assert first.previous is None
    assert second.previous is not None
    assert labels(result.page(size=2, order_by=order, cursor=second.previous)) == labels(first)
    paged = all_pages(result, order)
    assert paged == ["null-1", "null-null", "a-2", "a-2-second", "a-null", "b-3", "b-1", "b-2"]
    assert len(paged) == len(set(paged)) == len(ROWS)
    assert backwards_to_start(result, second, order, 2) == paged[:4]


@pytest.mark.parametrize(
    ("order", "expected"),
    [
        (
            (PageOrder("bucket", nulls="last"),),
            ["a-2", "a-null", "a-2-second", "b-1", "b-2", "b-3", "null-1", "null-null"],
        ),
        (
            (PageOrder("bucket", descending=True, nulls="first"),),
            ["null-1", "null-null", "b-1", "b-2", "b-3", "a-2", "a-null", "a-2-second"],
        ),
    ],
)
def test_null_order_and_duplicate_primary_values(paging_store, order, expected):
    assert all_pages(results(paging_store), order) == expected


@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("nulls", ["first", "last"])
@pytest.mark.parametrize("size", [1, 2, 3])
def test_composite_order_traverses_forward_and_backward(paging_store, descending, nulls, size):
    order = (
        PageOrder("bucket", descending=descending, nulls=nulls),
        PageOrder("score", descending=not descending, nulls="last" if nulls == "first" else "first"),
    )
    result = results(paging_store)
    page = result.page(size=size, order_by=order)
    forward = labels(page)
    while page.next is not None:
        page = result.page(size=size, order_by=order, cursor=page.next)
        forward.extend(labels(page))
    assert forward == expected_labels(order)
    assert backwards_to_start(result, page, order, size) == expected_labels(order)


def test_empty_order_uses_insertion_order(paging_store):
    result = results(paging_store)
    forward = all_pages(result, (), size=3)
    assert forward == [row.label for row in ROWS]
    page = result.page(size=3, order_by=())
    while page.next is not None:
        page = result.page(size=3, order_by=(), cursor=page.next)
    assert backwards_to_start(result, page, (), 3) == forward
    with pytest.raises(UnsupportedQueryError, match="at most 32"):
        result.page(size=2, order_by=tuple(PageOrder("bucket") for _ in range(33)))


def test_grouped_child_filter_pages_one_root_per_match(paging_store):
    assert all_pages(results(paging_store, common_only=True), (PageOrder("bucket", nulls="last"),)) == [
        "a-2",
        "a-null",
        "a-2-second",
        "b-1",
        "b-3",
        "null-1",
    ]


def test_cursor_rejects_changed_query_and_tampering(paging_store):
    order = (PageOrder("bucket"),)
    result = results(paging_store)
    token = result.page(size=2, order_by=order).next
    assert token is not None

    with pytest.raises(PaginationCursorError, match="different query"):
        results(paging_store, common_only=True).page(size=2, order_by=order, cursor=token)
    with pytest.raises(PaginationCursorError, match="different query"):
        result.page(size=2, order_by=(PageOrder("score"),), cursor=token)
    with pytest.raises(PaginationCursorError):
        result.page(size=2, order_by=order, cursor=ContinuationToken(f"{token}x"))
    with pytest.raises(PaginationCursorError, match="ContinuationToken"):
        result.page(size=2, order_by=order, cursor=str(token))


def test_cursor_rejects_changed_schema(paging_store):
    order = (PageOrder("bucket"),)
    token = results(paging_store).page(size=2, order_by=order).next
    assert token is not None
    searcher = paging_store.searcher()
    changed = searcher.variable(ChangedPageRecord)
    changed_result = searcher.results(bucket=changed.bucket, score=changed.score, label=changed.label)
    with pytest.raises(PaginationCursorError, match="different query"):
        changed_result.page(size=2, order_by=order, cursor=token)


@pytest.mark.parametrize(
    ("records", "steps"),
    [(200, 20), pytest.param(2_000, 100, marks=pytest.mark.extended)],
)
def test_deep_page_traversal_remains_seekable(store_factory, records: int, steps: int):
    paging_store = store_factory()
    for index in range(records):
        paging_store.save(PageRecord(index, index, f"r-{index}", ["common"]))
    result = results(paging_store)
    order = (PageOrder("bucket"),)
    page = result.page(size=7, order_by=order)
    for _ in range(steps):
        assert page.next is not None
        page = result.page(size=7, order_by=order, cursor=page.next)
        assert len(page.rows) == 7
    first = steps * 7
    assert labels(page) == [f"r-{index}" for index in range(first, first + 7)]
