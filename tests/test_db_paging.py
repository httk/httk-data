"""Continuation-page coverage for the optional SQL result-set capability."""

import base64
import datetime
import json
from dataclasses import dataclass
from functools import cmp_to_key

import pytest
from clickhouse_read_support import CLICKHOUSE_PARAM, bulk_store

from httk.data import ContinuationToken, PageOrder, PaginationCursorError, UnsupportedQueryError
from httk.data.db import Database, SqlStore
from httk.data.db.paging import _decode_continuation, _encode_continuation

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


@pytest.fixture(scope="module", params=["sqlite", "duckdb", CLICKHOUSE_PARAM])
def store(request):
    if request.param == "clickhousedb":
        with bulk_store(ROWS) as value:
            yield value
        return
    if request.param == "duckdb":
        pytest.importorskip("duckdb_engine")
        manager = Database.duckdb()
    else:
        manager = Database.sqlite()
    with manager as database:
        value = SqlStore(database, entry_records={})
        with value.transaction():
            for row in ROWS:
                value.save(row)
        yield value


def results(store: SqlStore, *, common_only: bool = False):
    searcher = store.searcher()
    record = searcher.variable(PageRecord)
    if common_only:
        searcher.add(record.tags.has("common"))
        # This is a post/HAVING child-list filter.  Keeping it beside the
        # existential predicate proves paging preserves the grouped inner
        # match relation rather than merely supporting a child-table join.
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
    """The public lexicographic order, independently of the SQL expression."""

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
        # Inserted test records have monotonically increasing root sids.
        return -1 if left[0] < right[0] else 1 if left[0] > right[0] else 0

    return [row.label for _index, row in sorted(enumerate(ROWS, start=1), key=cmp_to_key(compare))]


def backwards_to_start(result, page, order: tuple[PageOrder, ...], size: int) -> list[str]:
    chunks = [labels(page)]
    while page.previous is not None:
        page = result.page(size=size, order_by=order, cursor=page.previous)
        chunks.append(labels(page))
    return [label for chunk in reversed(chunks) for label in chunk]


def test_token_rejects_changed_schema_and_dialect():
    order = (PageOrder("bucket"),)
    with Database.sqlite() as sqlite_database:
        sqlite_store = SqlStore(sqlite_database, entry_records={})
        with sqlite_store.transaction():
            for row in ROWS:
                sqlite_store.save(row)
        token = results(sqlite_store).page(size=2, order_by=order).next
        assert token is not None

        changed_searcher = sqlite_store.searcher()
        changed = changed_searcher.variable(ChangedPageRecord)
        changed_result = changed_searcher.results(bucket=changed.bucket, score=changed.score, label=changed.label)
        with pytest.raises(PaginationCursorError, match="different query"):
            changed_result.page(size=2, order_by=order, cursor=token)

        pytest.importorskip("duckdb_engine")
        with Database.duckdb() as duckdb_database:
            duckdb_store = SqlStore(duckdb_database, entry_records={})
            with duckdb_store.transaction():
                for row in ROWS:
                    duckdb_store.save(row)
            with pytest.raises(PaginationCursorError, match="different query"):
                results(duckdb_store).page(size=2, order_by=order, cursor=token)


def test_token_is_canonical_urlsafe_and_works_with_fresh_equivalent_handle(store):
    order = (PageOrder("bucket", nulls="first"), PageOrder("score", descending=True))
    first = results(store).page(size=2, order_by=order)
    assert first.next is not None
    assert "=" not in first.next
    payload = json.loads(base64.urlsafe_b64decode(str(first.next) + "=" * (-len(first.next) % 4)))
    assert set(payload) == {"a", "d", "f", "s", "v"}
    assert "SELECT" not in str(payload)

    fresh = SqlStore(store._database)
    second = results(fresh).page(size=2, order_by=order, cursor=first.next)
    assert labels(second) == ["a-2", "a-2-second"]


def test_page_validation_profile_and_no_implicit_materialization_or_count(store, monkeypatch):
    result = results(store)
    order = (PageOrder("bucket"),)
    monkeypatch.setattr(result, "_ensure", lambda: (_ for _ in ()).throw(AssertionError("must not materialize")))
    monkeypatch.setattr(result, "__len__", lambda: (_ for _ in ()).throw(AssertionError("must not count")))
    page = result.page(size=2, order_by=order)
    assert len(page.rows) == 2

    with pytest.raises(TypeError, match="bool"):
        result.page(size=True, order_by=order)
    with pytest.raises(ValueError, match="between"):
        result.page(size=10_001, order_by=order)
    with pytest.raises(UnsupportedQueryError, match="duplicate"):
        result.page(size=2, order_by=(PageOrder("bucket"), PageOrder("bucket", descending=True)))
    with pytest.raises(UnsupportedQueryError, match="object projection"):
        result.page(size=2, order_by=(PageOrder("record"),))

    searcher = store.searcher()
    record = searcher.variable(PageRecord)
    searcher.add_sort(record.bucket)
    with pytest.raises(UnsupportedQueryError, match="add_sort"):
        searcher.results(bucket=record.bucket).page(size=2, order_by=order)
    searcher = store.searcher()
    record = searcher.variable(PageRecord)
    searcher.add_offset(1)
    with pytest.raises(UnsupportedQueryError, match="offset"):
        searcher.results(bucket=record.bucket).page(size=2, order_by=order)
    searcher = store.searcher()
    record = searcher.variable(PageRecord)
    searcher.set_limit(2)
    with pytest.raises(UnsupportedQueryError, match="limit"):
        searcher.results(bucket=record.bucket).page(size=2, order_by=order)


def test_page_statement_is_seek_based_and_total_is_opt_in(store, monkeypatch):
    result = results(store)
    order = (PageOrder("bucket"),)
    first = result.page(size=2, order_by=order)
    assert first.next is not None
    fingerprint = result._page_fingerprint(result._page_keys(order))
    from httk.data.db.paging import _decode_continuation

    cursor = _decode_continuation(first.next, fingerprint=fingerprint, anchors=1)
    statement, _raw_width = result._page_statement(result._page_keys(order), cursor, 2)
    assert statement._offset_clause is None
    assert statement._limit_clause is not None
    statement_text = (
        str(statement.compile(dialect=store._database.engine.dialect))
        if store._database.engine.dialect.name == "clickhousedb"
        else str(statement)
    )
    assert "WHERE" in statement_text

    calls = 0
    original = result._plan.count

    def count():
        nonlocal calls
        calls += 1
        return original()

    monkeypatch.setattr(result._plan, "count", count)
    assert result.page(size=2, order_by=order).total is None
    assert calls == 0
    assert result.page(size=2, order_by=order, include_total=True).total == len(ROWS)
    assert calls == 1


@pytest.mark.extended
def test_extended_deep_page_has_a_bounded_seek_statement():
    with Database.sqlite() as database:
        store = SqlStore(database, entry_records={})
        with store.transaction():
            for index in range(10_000):
                store.save(PageRecord(index, index, f"r-{index}", ["common"]))
        result = results(store)
        order = (PageOrder("bucket"),)
        page = result.page(size=7, order_by=order)
        for _ in range(500):
            assert page.next is not None
            page = result.page(size=7, order_by=order, cursor=page.next)
        assert labels(page) == [f"r-{index}" for index in range(3500, 3507)]

        keys = result._page_keys(order)
        fingerprint = result._page_fingerprint(keys)
        cursor = _decode_continuation(page.next, fingerprint=fingerprint, anchors=len(keys))
        statement, _raw_width = result._page_statement(keys, cursor, 7)
        assert statement._offset_clause is None
        assert statement._limit_clause.value == 8
        assert "WHERE" in str(statement)


def test_malformed_nonfinite_cursor_payload_is_rejected(store):
    result = results(store)
    order = (PageOrder("bucket"),)
    token = result.page(size=2, order_by=order).next
    assert token is not None
    payload = json.loads(base64.urlsafe_b64decode(str(token) + "=" * (-len(token) % 4)))
    payload["a"][0] = {"t": "float", "v": "inf"}
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    malformed = ContinuationToken(base64.urlsafe_b64encode(raw).decode().rstrip("="))
    with pytest.raises(PaginationCursorError, match="invalid anchor"):
        result.page(size=2, order_by=order, cursor=malformed)


def test_continuation_codec_roundtrips_supported_anchor_scalars():
    fingerprint = "a" * 64
    anchors = (
        None,
        True,
        -7,
        1.25,
        "text",
        b"",
        b"bytes",
        datetime.date(2026, 7, 30),
        datetime.datetime(2026, 7, 30, 12, 34, 56, tzinfo=datetime.UTC),
    )
    token = _encode_continuation(direction="forward", anchors=anchors, sid=3, fingerprint=fingerprint)
    decoded = _decode_continuation(token, fingerprint=fingerprint, anchors=len(anchors))
    assert decoded.direction == "forward"
    assert decoded.anchors == anchors
    assert decoded.sid == 3
