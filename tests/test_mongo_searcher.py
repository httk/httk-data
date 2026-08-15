"""Live MongoDB tests for the packet-4a searcher profile."""

import base64
import json
from dataclasses import dataclass
from fractions import Fraction
from types import SimpleNamespace

import pytest

from httk.store.db import Database, SqlStore
from httk.store.mongo import MongoSearcher
from httk.store.query import MultipleResultsError, NoResultError, PageOrder, PaginationCursorError, UnsupportedQueryError
from httk.store.query.paging_tokens import _decode_continuation, _encode_continuation, _encode_payload, _plan_fingerprint


@dataclass(frozen=True)
class MongoQueryRecord:
    label: str
    rank: int | None
    note: str | None = None
    energy: Fraction = Fraction(0)


def _query(mongo_test_database):
    from httk.store.mongo import MongoStore

    store = MongoStore(mongo_test_database, entry_records={})
    records = [
        MongoQueryRecord("50% Mg", None, None, Fraction(-1, 3)),
        MongoQueryRecord("5012 Mg", 2, "present", Fraction(1, 2)),
        MongoQueryRecord("a_b", 1, None, Fraction(3, 2)),
        MongoQueryRecord("axb", 3, "other", Fraction(7, 8)),
    ]
    for record in records:
        store.save(record)
    searcher = MongoSearcher(store)
    variable = searcher.variable(MongoQueryRecord)
    searcher.output(variable, "record")
    return store, searcher, variable


def _labels(searcher):
    return [result.values[0].label for result in searcher]


def _paging_results(store):
    """Build the scalar page plan used by the Mongo paging edge cases."""
    searcher = store.searcher()
    record = searcher.variable(MongoQueryRecord)
    return searcher.results(rank=record.rank, label=record.label)


def test_scalar_comparisons_are_three_valued(mongo_test_database):
    _store, searcher, variable = _query(mongo_test_database)
    searcher.add(variable.rank > 1)
    assert _labels(searcher) == ["5012 Mg", "axb"]

    _store, searcher, variable = _query(mongo_test_database)
    searcher.add(variable.note == variable.note)
    assert _labels(searcher) == ["5012 Mg", "axb"]

    _store, searcher, variable = _query(mongo_test_database)
    searcher.add(~(variable.note == "present"))
    assert _labels(searcher) == ["axb"]


def test_scalar_is_in_none_contract_and_negation(mongo_test_database):
    _store, searcher, variable = _query(mongo_test_database)
    searcher.add(variable.note.is_in(None, "present"))
    assert set(_labels(searcher)) == {"50% Mg", "5012 Mg", "a_b"}

    _store, searcher, variable = _query(mongo_test_database)
    searcher.add(~variable.note.is_in(None, "present"))
    assert _labels(searcher) == ["axb"]

    _store, searcher, variable = _query(mongo_test_database)
    searcher.add(~variable.note.is_in("present"))
    assert set(_labels(searcher)) == {"axb"}


@pytest.mark.parametrize(
    ("mode", "text", "expected"),
    [("contains", "50%", {"50% Mg"}), ("startswith", "a_", {"a_b"}), ("endswith", "b", {"a_b", "axb"})],
)
def test_string_matching_is_literal_and_case_sensitive(mongo_test_database, mode, text, expected):
    _store, searcher, variable = _query(mongo_test_database)
    expression = getattr(variable.label, mode)(text)
    searcher.add(expression)
    assert set(_labels(searcher)) == expected


def test_sort_null_rank_limit_offset_count_and_scalar_results(mongo_test_database):
    _store, searcher, variable = _query(mongo_test_database)
    searcher.add_sort(variable.rank, nulls="first")
    assert _labels(searcher) == ["50% Mg", "a_b", "5012 Mg", "axb"]

    _store, searcher, variable = _query(mongo_test_database)
    searcher.add_sort(variable.rank, descending=True, nulls="first")
    assert _labels(searcher) == ["50% Mg", "axb", "5012 Mg", "a_b"]
    assert searcher.count() == 4
    searcher.set_limit(2)
    searcher.add_offset(1)
    assert _labels(searcher) == ["axb", "5012 Mg"]
    assert searcher.count() == 4

    _store, searcher, variable = _query(mongo_test_database)
    result = searcher.results(label=variable.label, energy=variable.energy)
    assert list(result.scalars("label")) == ["50% Mg", "5012 Mg", "a_b", "axb"]
    assert list(result.scalars("energy")) == [Fraction(-1, 3), Fraction(1, 2), Fraction(3, 2), Fraction(7, 8)]


def test_result_set_edges_and_disconnected_variables(mongo_test_database):
    _store, searcher, variable = _query(mongo_test_database)
    result = searcher.results(record=variable)
    assert result.first() is not None
    with pytest.raises(MultipleResultsError):
        result.one()

    _store, searcher, variable = _query(mongo_test_database)
    searcher.add(variable.label == "does-not-exist")
    result = searcher.results(record=variable)
    assert result.first() is None
    with pytest.raises(NoResultError):
        result.one()

    other = searcher.variable(MongoQueryRecord)
    searcher.output(other, "other")
    with pytest.raises(UnsupportedQueryError, match="disconnected cartesian"):
        list(searcher)


@dataclass(frozen=True)
class MongoLeaf:
    code: str


@dataclass(frozen=True)
class MongoBranch:
    leaf: MongoLeaf | None


@dataclass(frozen=True)
class MongoRoot:
    name: str
    branch: MongoBranch | None
    tags: list[str] | None


def test_deep_reference_chain_missing_links_and_reference_join(mongo_test_database):
    """Lookups preserve roots with missing links without treating them as matches."""
    from httk.store.mongo import MongoStore

    store = MongoStore(mongo_test_database, entry_records={})
    leaf = MongoLeaf("present")
    branch = MongoBranch(leaf)
    missing_leaf = MongoBranch(None)
    present = MongoRoot("present", branch, ["allowed"])
    missing_middle = MongoRoot("missing-middle", None, ["allowed"])
    missing_deep = MongoRoot("missing-deep", missing_leaf, ["allowed"])
    for item in (present, missing_middle, missing_deep):
        store.save(item)

    searcher = store.searcher()
    root = searcher.variable(MongoRoot)
    searcher.output(root, "root")
    searcher.add(root.branch.leaf.code == "present")
    assert [row.values[0].name for row in searcher] == ["present"]
    assert searcher.count() == len(list(searcher))

    searcher = store.searcher()
    root = searcher.variable(MongoRoot)
    joined_branch = searcher.variable(MongoBranch)
    searcher.add(root.branch == joined_branch)
    searcher.output(root, "root")
    searcher.output(joined_branch, "branch")
    rows = list(searcher)
    assert [(row.values[0].name, row.values[1]) for row in rows] == [
        ("present", branch),
        ("missing-deep", missing_leaf),
    ]
    assert searcher.count() == len(rows)


def test_negated_composed_child_sets_with_null_elements(mongo_test_database):
    """An embedded null is an outsider and does not break composed negation."""
    from httk.store.mongo import MongoStore

    store = MongoStore(mongo_test_database, entry_records={})
    clean = MongoRoot("clean", None, ["allowed"])
    null_element = MongoRoot("null-element", None, ["allowed"])
    empty = MongoRoot("empty", None, [])
    for item in (clean, null_element, empty):
        store.save(item)
    sid = store.sid_of(null_element)
    assert sid is not None
    collection = store._database.database["mongo_root"]
    collection.update_one(
        {"_id": sid},
        {"$set": {"f.tags": [{"tags": "allowed"}, {"tags": None}]}},
        bypass_document_validation=True,
    )

    searcher = store.searcher()
    root = searcher.variable(MongoRoot)
    searcher.output(root, "root")
    searcher.add(~(root.tags.has_any("allowed") & root.tags.has_only("allowed")))
    rows = list(searcher)
    assert {row.values[0].name for row in rows} == {"null-element", "empty"}
    assert searcher.count() == len(rows)


@dataclass(frozen=True)
class MongoChildComparisonRecord:
    name: str
    numbers: list[int] | None
    words: list[str] | None


def _child_comparison_store(mongo_test_database):
    from httk.store.mongo import MongoStore

    store = MongoStore(mongo_test_database, entry_records={})
    for record in (
        MongoChildComparisonRecord("mixed", [1, 2], ["alpha", "beta"]),
        MongoChildComparisonRecord("two", [2], ["bravo"]),
        MongoChildComparisonRecord("high", [3], ["omega"]),
        MongoChildComparisonRecord("empty", [], []),
        MongoChildComparisonRecord("absent", None, None),
    ):
        store.save(record)
    return store


def _child_comparison_names(store, build):
    searcher = store.searcher()
    variable = searcher.variable(MongoChildComparisonRecord)
    searcher.output(variable, "record")
    searcher.add(build(variable))
    rows = list(searcher)
    assert searcher.count() == len(rows)
    return {row.values[0].name for row in rows}


@pytest.mark.parametrize(
    ("build", "expected"),
    [
        (lambda record: record.numbers < 2, {"mixed"}),
        (lambda record: record.numbers <= 2, {"mixed", "two"}),
        (lambda record: record.numbers > 2, {"high"}),
        (lambda record: record.numbers >= 2, {"mixed", "two", "high"}),
        # This is existential: the ``1`` witnesses [1, 2] != 2.
        (lambda record: record.numbers != 2, {"mixed", "high"}),
        (lambda record: ~(record.numbers < 2), {"two", "high", "empty", "absent"}),
        (lambda record: ~(record.numbers <= 2), {"high", "empty", "absent"}),
        (lambda record: ~(record.numbers > 2), {"mixed", "two", "empty", "absent"}),
        (lambda record: ~(record.numbers >= 2), {"empty", "absent"}),
        (lambda record: ~(record.numbers != 2), {"two", "empty", "absent"}),
    ],
)
def test_child_rich_comparisons_are_existential(mongo_test_database, build, expected):
    assert _child_comparison_names(_child_comparison_store(mongo_test_database), build) == expected


@pytest.mark.parametrize(
    ("build", "expected"),
    [
        (lambda record: record.words.contains("ph"), {"mixed"}),
        (lambda record: record.words.startswith("br"), {"two"}),
        (lambda record: record.words.endswith("ta"), {"mixed"}),
        (lambda record: ~record.words.contains("ph"), {"two", "high", "empty", "absent"}),
    ],
)
def test_child_string_operations_match_element_values(mongo_test_database, build, expected):
    assert _child_comparison_names(_child_comparison_store(mongo_test_database), build) == expected


def test_join_predicates_below_not_or_or_are_rejected(mongo_test_database):
    """A conditional equality cannot safely establish a physical lookup."""
    from httk.store.mongo import MongoStore

    store = MongoStore(mongo_test_database, entry_records={})
    branch = MongoBranch(None)
    store.save(MongoRoot("flag", branch, []))

    for build in (
        lambda root, other: ~(root.branch == other),
        lambda root, other: (root.branch == other) | (root.name == "flag"),
    ):
        searcher = store.searcher()
        root = searcher.variable(MongoRoot)
        other = searcher.variable(MongoBranch)
        searcher.add(build(root, other))
        searcher.output(root, "root")
        with pytest.raises(UnsupportedQueryError, match="disconnected cartesian"):
            list(searcher)


def test_paging_rejects_sql_tokens_in_both_directions(mongo_test_database):
    """The backend tag in a plan fingerprint makes cursors non-interchangeable."""
    from httk.store.mongo import MongoStore

    records = [
        MongoQueryRecord("one", 1),
        MongoQueryRecord("two", 2),
        MongoQueryRecord("three", 3),
    ]
    mongo_store = MongoStore(mongo_test_database, entry_records={})
    for record in records:
        mongo_store.save(record)
    mongo_result = _paging_results(mongo_store)
    order = (PageOrder("rank"),)

    with Database.sqlite() as database:
        sql_store = SqlStore(database, entry_records={})
        with sql_store.transaction():
            for record in records:
                sql_store.save(record)
        sql_searcher = sql_store.searcher()
        sql_record = sql_searcher.variable(MongoQueryRecord)
        sql_result = sql_searcher.results(rank=sql_record.rank, label=sql_record.label)
        sql_token = sql_result.page(size=1, order_by=order).next
        mongo_token = mongo_result.page(size=1, order_by=order).next
        assert sql_token is not None
        assert mongo_token is not None
        with pytest.raises(PaginationCursorError, match="different query"):
            mongo_result.page(size=1, order_by=order, cursor=sql_token)
        with pytest.raises(PaginationCursorError, match="different query"):
            sql_result.page(size=1, order_by=order, cursor=mongo_token)


def test_paging_preserves_preexisting_rows_across_an_insert(mongo_test_database):
    """A keyset continuation neither repeats nor loses rows that existed at its anchor."""
    store, _searcher, _variable = _query(mongo_test_database)
    result = _paging_results(store)
    order = (PageOrder("rank", nulls="last"),)
    first = result.page(size=2, order_by=order)
    seen = [row.label for row in first.rows]

    store.save(MongoQueryRecord("inserted", 2, "later"))
    page = first
    while page.next is not None:
        page = result.page(size=2, order_by=order, cursor=page.next)
        seen.extend(row.label for row in page.rows)

    preexisting = {"50% Mg", "5012 Mg", "a_b", "axb"}
    assert preexisting <= set(seen)
    assert len(seen) == len(set(seen))


def test_paging_boundary_tokens_reverse_from_last_to_first(mongo_test_database):
    """The first/last boundaries do not synthesize unusable continuation tokens."""
    store, _searcher, _variable = _query(mongo_test_database)
    result = _paging_results(store)
    order = (PageOrder("rank", nulls="last"),)
    first = result.page(size=2, order_by=order)
    assert first.previous is None
    page = first
    while page.next is not None:
        page = result.page(size=2, order_by=order, cursor=page.next)
    assert page.next is None
    backwards = [row.label for row in page.rows]
    while page.previous is not None:
        page = result.page(size=2, order_by=order, cursor=page.previous)
        backwards[0:0] = [row.label for row in page.rows]
    assert page.previous is None
    assert backwards == ["a_b", "5012 Mg", "axb", "50% Mg"]


def test_verifier_is_authoritative_for_all_result_consumers(mongo_test_database):
    """Verification precedes every read consumer, including windows and totals."""
    store, searcher, variable = _query(mongo_test_database)
    verifier = lambda document: str(document["f"].get("label", "")).endswith("Mg")
    searcher.set_row_verifier(verifier, "label-suffix/Mg/v1")
    assert _labels(searcher) == ["50% Mg", "5012 Mg"]
    assert searcher.count() == 2

    result = searcher.results(label=variable.label, rank=variable.rank)
    assert [row.label for row in result] == ["50% Mg", "5012 Mg"]
    assert len(result) == 2
    assert result.first() is not None
    assert result.first().label == "50% Mg"
    assert list(result.scalars("label")) == ["50% Mg", "5012 Mg"]
    order = (PageOrder("rank", nulls="last"),)
    first = result.page(size=1, order_by=order)
    assert [row.label for row in first.rows] == ["5012 Mg"]
    assert first.next is not None
    second = result.page(size=1, order_by=order, cursor=first.next)
    assert [row.label for row in second.rows] == ["50% Mg"]
    assert second.next is None
    assert result.page(size=1, order_by=order, include_total=True).total == 2

    windowed = store.searcher()
    windowed_record = windowed.variable(MongoQueryRecord)
    windowed.set_row_verifier(verifier, "label-suffix/Mg/v1")
    windowed.add_sort(windowed_record.rank)
    windowed.add_offset(1)
    windowed.set_limit(1)
    assert windowed.count() == 2
    assert all("$skip" not in stage and "$limit" not in stage for stage in windowed._pipeline())
    one = windowed.results(label=windowed_record.label)
    assert [row.label for row in one] == ["50% Mg"]
    assert len(one) == 1
    assert one.first() is not None
    assert one.first().label == "50% Mg"
    assert one.one().label == "50% Mg"
    assert list(one.scalars()) == ["50% Mg"]

    rejected = store.searcher()
    rejected_record = rejected.variable(MongoQueryRecord)
    rejected.set_row_verifier(lambda _document: False, "reject-all/v1")
    empty = rejected.results(label=rejected_record.label, rank=rejected_record.rank)
    assert list(empty) == []
    assert len(empty) == 0
    assert empty.first() is None
    with pytest.raises(NoResultError):
        empty.one()
    page = empty.page(size=1, order_by=order, include_total=True)
    assert page.rows == ()
    assert page.next is None
    assert page.previous is None
    assert page.total == 0


def test_verifier_identity_rejects_cross_plan_tokens_and_bare_callbacks(mongo_test_database):
    """Two verifier identities cannot share a cursor even with one server pipeline."""
    store, _searcher, _variable = _query(mongo_test_database)
    order = (PageOrder("rank", nulls="last"),)

    first_search = store.searcher()
    first_record = first_search.variable(MongoQueryRecord)
    first_search.set_row_verifier(lambda _document: True, "logical-ast/one")
    first = first_search.results(rank=first_record.rank, label=first_record.label)
    token = first.page(size=1, order_by=order).next
    assert token is not None

    second_search = store.searcher()
    second_record = second_search.variable(MongoQueryRecord)
    second_search.set_row_verifier(lambda _document: True, "logical-ast/two")
    second = second_search.results(rank=second_record.rank, label=second_record.label)
    with pytest.raises(PaginationCursorError, match="different query"):
        second.page(size=1, order_by=order, cursor=token)

    unsafe = store.searcher()
    unsafe_record = unsafe.variable(MongoQueryRecord)
    unsafe._row_verifier = lambda _document: True
    with pytest.raises(ValueError, match="identity"):
        unsafe.results(rank=unsafe_record.rank)


def test_verifier_identity_rejects_empty_payloads(mongo_test_database):
    """The empty identity is reserved for verifier-less fingerprints."""
    from httk.store.mongo import MongoStore

    for identity in ("", b""):
        store = MongoStore(mongo_test_database, entry_records={})
        searcher = store.searcher()
        searcher.variable(MongoQueryRecord)
        with pytest.raises(ValueError, match="must not be empty"):
            searcher.set_row_verifier(lambda _document: True, identity)


def test_paging_backend_discriminator_is_load_bearing(mongo_test_database):
    """Changing only the fingerprint backend context invalidates a re-encoded token."""
    store, _searcher, _variable = _query(mongo_test_database)
    result = _paging_results(store)
    order = (PageOrder("rank", nulls="last"),)
    keys = result._page_keys(order)
    fingerprint = result._page_fingerprint(keys)
    context = result._page_fingerprint_payload(keys)
    changed_context = {**context, "backend": "sqlite"}
    changed_fingerprint = _plan_fingerprint(changed_context)
    assert context["backend"] == "mongodb"
    assert changed_fingerprint != fingerprint

    token = result.page(size=1, order_by=order).next
    assert token is not None
    decoded = _decode_continuation(token, fingerprint=fingerprint, anchors=len(keys))
    changed_token = _encode_continuation(
        direction=decoded.direction,
        anchors=decoded.anchors,
        sid=decoded.sid,
        fingerprint=changed_fingerprint,
    )
    with pytest.raises(PaginationCursorError, match="different query"):
        result.page(size=1, order_by=order, cursor=changed_token)


def test_type_tampered_anchor_is_a_clean_cursor_error(mongo_test_database):
    """A string swapped for an integer anchor fails as a clean PaginationCursorError."""
    store, _searcher, _variable = _query(mongo_test_database)
    result = _paging_results(store)
    order = (PageOrder("rank", nulls="last"),)
    token = result.page(size=1, order_by=order).next
    assert token is not None
    payload = json.loads(base64.urlsafe_b64decode(str(token) + "=" * (-len(token) % 4)))
    assert payload["a"][0]["t"] == "int"
    payload["a"][0] = {"t": "str", "v": "not-an-integer"}
    tampered = type(token)(_encode_payload(payload))
    with pytest.raises(PaginationCursorError, match="incompatible"):
        result.page(size=1, order_by=order, cursor=tampered)


def test_page_anchor_python_type_derives_from_the_order_column_kind():
    """The Mongo anchor-type helper maps a key's stored column kind (no server needed)."""
    from httk.store.db.codecs import codec_named
    from httk.store.db.schema import resolve_schema
    from httk.store.mongo.results import _anchor_python_type

    schema = resolve_schema(MongoQueryRecord)
    rank = schema.field("rank")  # int | None -> scalar int column
    label = schema.field("label")  # str -> scalar str column
    energy = schema.field("energy")  # Fraction -> encoded (codec query column)
    assert _anchor_python_type(SimpleNamespace(_spec=rank, _codec=None)) is int
    assert _anchor_python_type(SimpleNamespace(_spec=label, _codec=None)) is str
    codec = codec_named(energy.codec_name)
    assert _anchor_python_type(SimpleNamespace(_spec=energy, _codec=codec)) in {int, float, str}
