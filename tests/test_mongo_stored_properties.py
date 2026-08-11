"""Live Mongo coverage for stored-property plans and exact verification."""

from dataclasses import dataclass, replace
from fractions import Fraction

import pytest
from httk.core import FracScalar
from httk.core.storage import content_id
from test_db_stored_properties import (
    FIRST,
    SECOND,
    CalculationEntry,
    GenericCalculationFirst,
    GenericCalculationSecond,
)

from httk.store.mongo import MongoStore
from httk.store.mongo.evaluator import canonical_predicate, evaluate
from httk.store.mongo.stored_properties import (
    MongoStoredPropertyConfigurationError,
    _MongoQueryContext,
    _response_json_value,
)


@dataclass(frozen=True)
class _EvaluatorRecord:
    value: float | None
    values: list[float]
    assemblies: list[str] | None = None
    attached: list[str] | None = None
    exact_values: list[FracScalar] | None = None


@dataclass(frozen=True)
class _ResponseDataclass:
    value: Fraction


class _ResponseFloat:
    def to_float(self) -> float:
        return 1.25


@pytest.fixture
def plan(mongo_test_database):
    store = MongoStore(
        mongo_test_database,
        entry_records={CalculationEntry: (GenericCalculationFirst, GenericCalculationSecond)},
    )
    store.save(FIRST)
    store.save(SECOND)
    return store.stored_property_plan(CalculationEntry)


def _records(searchers):
    return [result[0][0] for searcher in searchers for result in searcher]


@pytest.mark.parametrize(
    ("filter_string", "expected"),
    (
        ('immutable_id = "one-third"', {"first"}),
        ('immutable_id = "composition"', {"first"}),
        ('immutable_id = "null-comment"', {"first"}),
        ('immutable_id = "nested"', {"first"}),
        ('immutable_id = "when-known"', {"second"}),
        ('immutable_id = "score-gt"', {"second"}),
        ("last_modified IS UNKNOWN", {"first"}),
        ("last_modified IS KNOWN", {"second"}),
        ('last_modified = "2026-08-02T08:30:00Z"', {"second"}),
    ),
)
def test_mongo_plan_matches_the_sql_stored_property_scenarios(plan, filter_string, expected):
    assert {record.label for record in _records(plan.filter_searchers(filter_string))} == expected


def test_candidate_streams_are_id_only_and_verified(plan):
    streams = plan.candidate_searchers('immutable_id = "composition"', sort=(("immutable_id", False),))
    assert len(streams) == 2
    rows = [row for stream in streams for row in stream.searcher]
    assert [row[0][0] for row in rows]
    assert all(isinstance(row[0][1], str) for row in rows)
    assert len(rows) == 1


def test_public_id_prefix_filters_sorts_and_candidate_streams(plan):
    prefix = "mongo:"
    first_id = prefix + content_id(FIRST)
    assert [
        record.label for record in _records(plan.filter_searchers(f'id = "{first_id}"', public_id_prefix=prefix))
    ] == ["first"]
    assert not _records(plan.filter_searchers(f'id = "other:{content_id(FIRST)}"', public_id_prefix=prefix))

    additional = replace(FIRST, label="another-first")
    plan.store.save(additional)
    streams = plan.candidate_searchers(sort=(("id", False),), public_id_prefix=prefix)
    first_stream = next(stream for stream in streams if stream.backing is GenericCalculationFirst)
    rows = [result[0] for result in first_stream.searcher]
    assert {row[1] for row in rows} == {content_id(FIRST), content_id(additional)}
    assert [row[2] for row in rows] == sorted(prefix + row[1] for row in rows)


def test_evaluator_preserves_unknown_and_canonical_exact_constants():
    context = _MongoQueryContext(GenericCalculationFirst)
    unknown = context.when_known(context.not_(context.is_null(context.field("comment"))), context.always_true())
    assert evaluate(unknown, FIRST) is None

    exact = context.scaled_exact_equal(
        context.field("energy"), context.constant(Fraction(1)), context.constant(Fraction(1, 3)), context.constant(1)
    )
    assert evaluate(exact, FIRST) is True
    assert "1/3" in canonical_predicate(exact)


def test_evaluator_resolves_store_timestamp_from_the_candidate_sid():
    store = type("Store", (), {"store_timestamps": True, "store_timestamp_resolution": 1000})()
    context = _MongoQueryContext(GenericCalculationFirst, store)
    predicate = context.compare(context.field("store_timestamp"), "<=", context.constant(1_000_499))
    assert evaluate(predicate, FIRST, store_timestamp_resolver=lambda: 1000) is True
    assert evaluate(predicate, FIRST, store_timestamp_resolver=lambda: 1001) is False


def test_evaluator_nullable_exact_distinct_and_exists_semantics():
    context = _MongoQueryContext(_EvaluatorRecord)
    record = _EvaluatorRecord(None, [-0.0, 0.0, float("nan"), float("nan")], assemblies=None, attached=[])

    ordinary = context.equal(context.field("value"), context.constant(1.0))
    assert evaluate(ordinary, record) is None
    assert evaluate(context.not_(ordinary), record) is None
    assert evaluate(context.equal(context.field("value"), context.null()), record) is True

    zero = _EvaluatorRecord(-0.0, record.values)
    assert evaluate(context.exact_equal(context.field("value"), context.constant(0.0)), zero) is False

    values = context.scope("values")
    distinct = context.equal(context.distinct_count(values, values.field("value")), context.constant(3))
    assert evaluate(distinct, record) is True

    null_witness = _EvaluatorRecord(None, [None])  # type: ignore[list-item]
    exists = context.exists(values, context.equal(values.field("value"), context.constant(1.0)))
    assert evaluate(exists, null_witness) is False
    assert evaluate(context.not_(exists), null_witness) is True


def test_optional_child_presence_and_response_serialization(plan):
    context = _MongoQueryContext(_EvaluatorRecord)
    assert evaluate(
        context.equal(context.field("assemblies_present"), context.constant(False)), _EvaluatorRecord(0.0, [])
    )
    assert evaluate(
        context.equal(context.field("attached_present"), context.constant(True)), _EvaluatorRecord(0.0, [], attached=[])
    )
    assert evaluate(
        context.equal(context.field("exact_values_present"), context.constant(False)), _EvaluatorRecord(0.0, [])
    )
    assert evaluate(
        context.equal(context.field("exact_values_present"), context.constant(True)),
        _EvaluatorRecord(0.0, [], exact_values=[FracScalar(1, denom=3)]),
    )

    assert _response_json_value(_ResponseDataclass(Fraction(1, 3))) == {"value": 1 / 3}
    assert _response_json_value(_ResponseFloat()) == 1.25
    with pytest.raises(TypeError, match="string keys"):
        _response_json_value({1: "not-public-json"})

    backing = plan._backings[0]
    projection = backing.projections["immutable_id"]
    exact_sort = replace(projection, sort=lambda context: context.field("energy"))
    incompatible = replace(backing, projections={**backing.projections, "immutable_id": exact_sort})
    variable = plan.store.searcher().variable(backing.backing)
    with pytest.raises(MongoStoredPropertyConfigurationError, match="canonical text channel"):
        plan._sort_field(incompatible, variable, "immutable_id", "")

    assert [
        record.label for record in _records(plan.filter_searchers('type = "calculations"', sort=(("type", False),)))
    ] == [
        "first",
        "second",
    ]
    streams = plan.candidate_searchers(sort=(("type", True), ("immutable_id", False)))
    assert all(stream.sort_count == 2 for stream in streams)
    assert all(
        result[0][2] == "calculations" and len(result[0]) == 4 for stream in streams for result in stream.searcher
    )
