"""SQL translation coverage for backing-local stored-property declarations."""

import datetime
from dataclasses import dataclass, field
from fractions import Fraction
from typing import ClassVar

import pytest
import sqlalchemy
from clickhouse_read_support import CLICKHOUSE_PARAM, bulk_store
from httk.core.register import register_entry_family, register_entry_record
from httk.core.storage import QueryLiteralError, StorageInfo, StoredPropertyProjection

from httk.store.db import (
    Database,
    SqlStore,
    StoredPropertySqlConfigurationError,
    stored_property_sql_plan,
)
from httk.store.query.optimade_filters import FilterTranslationError

pytestmark = pytest.mark.xdist_group("clickhouse_read_corpus")

CALCULATIONS_DEFINITION = "https://schemas.optimade.org/defs/v1.3/entrytypes/optimade/calculations"
FILES_DEFINITION = "https://schemas.optimade.org/defs/v1.2/entrytypes/optimade/files"


class CalculationEntry:
    type = "calculations"
    definition_id = CALCULATIONS_DEFINITION


class FileEntry:
    type = "files"
    definition_id = FILES_DEFINITION


@dataclass(frozen=True)
class CompositionPart:
    symbol: str
    amount: Fraction
    ratios: list[Fraction] = field(default_factory=list)


def _immutable_id_query(ctx, operator: str, literal: object):
    if operator != "=":
        raise QueryLiteralError("generic calculation examples support only equality")
    if literal == "one-third":
        return ctx.exact_equal(ctx.field("energy"), ctx.constant(Fraction(1, 3)))
    if literal == "composition":
        left = ctx.scope("parts")
        right = ctx.scope("parts")
        a = ctx.filtered(left, ctx.equal(left.field("symbol"), ctx.constant("A")))
        b = ctx.filtered(right, ctx.equal(right.field("symbol"), ctx.constant("B")))
        all_parts = ctx.scope("parts")
        return ctx.and_(
            ctx.equal(ctx.count(a), ctx.constant(1)),
            ctx.equal(ctx.count(b), ctx.constant(1)),
            ctx.equal(ctx.distinct_count(all_parts, all_parts.field("symbol")), ctx.constant(2)),
            ctx.exists(
                a,
                ctx.exists(
                    b,
                    ctx.scaled_exact_equal(
                        left.field("amount"), ctx.constant(1), right.field("amount"), ctx.constant(2)
                    ),
                ),
            ),
        )
    if literal == "null-comment":
        return ctx.is_null(ctx.field("comment"))
    if literal == "nested":
        # A nested child scope can be used directly from the root context:
        # its local FROM tree retains the parent child-target correlation.
        ratios = ctx.scope("parts").scope("ratios")
        return ctx.exists(ratios, ctx.exact_equal(ratios.field("value"), ctx.constant(Fraction(1, 3))))
    if literal == "filtered-count-nested":
        parts = ctx.scope("parts")
        ratios = parts.scope("ratios")
        nested = ctx.exists(ratios, ctx.exact_equal(ratios.field("value"), ctx.constant(Fraction(1, 3))))
        return ctx.equal(ctx.count(ctx.filtered(parts, nested)), ctx.constant(0))
    if literal == "filtered-count-value-nested":
        parts = ctx.scope("parts")
        ratios = parts.scope("ratios")
        filtered = ctx.filtered(
            parts,
            ctx.exact_equal(ratios.field("value"), ctx.constant(Fraction(1, 3))),
        )
        return ctx.equal(ctx.count(filtered), ctx.constant(0))
    if literal == "filtered-count-single":
        parts = ctx.scope("parts")
        filtered = ctx.filtered(parts, ctx.exact_equal(parts.field("symbol"), ctx.constant("A")))
        return ctx.equal(ctx.count(filtered), ctx.constant(1))
    if literal == "filtered-distinct-count-nested":
        parts = ctx.scope("parts")
        ratios = parts.scope("ratios")
        nested = ctx.exists(ratios, ctx.exact_equal(ratios.field("value"), ctx.constant(Fraction(1, 3))))
        filtered = ctx.filtered(parts, nested)
        return ctx.equal(ctx.distinct_count(filtered, filtered.field("symbol")), ctx.constant(0))
    if literal == "boolean-nested":
        ratios = ctx.scope("parts").scope("ratios")
        nested = ctx.exists(ratios, ctx.exact_equal(ratios.field("value"), ctx.constant(Fraction(1, 3))))
        return ctx.when_known(ctx.always_true(), nested)
    if literal == "boolean-combinators-nested":
        ratios = ctx.scope("parts").scope("ratios")
        nested = ctx.exists(ratios, ctx.exact_equal(ratios.field("value"), ctx.constant(Fraction(1, 3))))
        return ctx.and_(ctx.or_(ctx.always_true(), nested), ctx.not_(nested))
    if literal == "unused-nested":
        ctx.scope("parts").scope("ratios")
        return ctx.always_true()
    if literal == "when-known":
        return ctx.when_known(
            ctx.not_(ctx.is_null(ctx.field("comment"))),
            ctx.exact_equal(ctx.field("energy"), ctx.constant(Fraction(7, 9))),
        )
    if literal == "empty-and":
        return ctx.and_()
    if literal == "empty-or":
        return ctx.or_()
    if literal == "score-gt":
        return ctx.compare(ctx.field("score"), ">", ctx.constant(1.5))
    raise QueryLiteralError(f"unknown generic calculation literal {literal!r}")


def _immutable_id_sort(ctx):
    return ctx.field("score")


def _last_modified_query(ctx, operator: str, literal: object):
    value = ctx.field("modified")
    if operator == "IS_UNKNOWN":
        return ctx.is_null(value)
    if operator == "IS_KNOWN":
        return ctx.not_(ctx.is_null(value))
    return ctx.compare(value, operator, ctx.constant(literal))


@dataclass(frozen=True)
class GenericCalculationFirst:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="stored_property_first")

    label: str
    energy: Fraction
    comment: str | None
    score: float
    parts: list[CompositionPart] = field(default_factory=list)

    __httk_stored_properties__: ClassVar = {
        "immutable_id": StoredPropertyProjection(
            response=lambda record: record.label,
            query=_immutable_id_query,
            sort=_immutable_id_sort,
        ),
    }


@dataclass(frozen=True)
class GenericCalculationSecond:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="stored_property_second")

    label: str
    energy: Fraction
    comment: str | None
    modified: datetime.datetime
    score: float
    parts: list[CompositionPart] = field(default_factory=list)

    __httk_stored_properties__: ClassVar = {
        "immutable_id": StoredPropertyProjection(
            response=lambda record: record.label,
            query=_immutable_id_query,
            sort=_immutable_id_sort,
        ),
        "last_modified": StoredPropertyProjection(
            response=lambda record: record.modified,
            query=_last_modified_query,
            sort=lambda context: context.field("modified"),
        ),
    }


@dataclass(frozen=True)
class IncompleteFile:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="stored_property_incomplete_file")

    url: str

    __httk_stored_properties__: ClassVar = {
        "url": StoredPropertyProjection(response=lambda record: record.url),
    }


@dataclass(frozen=True)
class BadCalculation:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="stored_property_bad_calculation")

    label: str

    __httk_stored_properties__: ClassVar = {
        "id": StoredPropertyProjection(response=lambda record: record.label),
    }


class BadFamily:
    type = "calculations"
    definition_id = CALCULATIONS_DEFINITION


register_entry_family(
    name="test-stored-properties-calculations",
    family=f"{__name__}:CalculationEntry",
    definition_id=CALCULATIONS_DEFINITION,
)
register_entry_record(
    name="test-stored-properties-calculation-first",
    family="test-stored-properties-calculations",
    record=f"{__name__}:GenericCalculationFirst",
)
register_entry_record(
    name="test-stored-properties-calculation-second",
    family="test-stored-properties-calculations",
    record=f"{__name__}:GenericCalculationSecond",
)
register_entry_family(
    name="test-stored-properties-files",
    family=f"{__name__}:FileEntry",
    definition_id=FILES_DEFINITION,
)
register_entry_record(
    name="test-stored-properties-incomplete-file",
    family="test-stored-properties-files",
    record=f"{__name__}:IncompleteFile",
)
register_entry_family(
    name="test-stored-properties-bad-family",
    family=f"{__name__}:BadFamily",
    definition_id=CALCULATIONS_DEFINITION,
)
register_entry_record(
    name="test-stored-properties-bad-backing",
    family="test-stored-properties-bad-family",
    record=f"{__name__}:BadCalculation",
)


FIRST = GenericCalculationFirst(
    "first",
    Fraction(1, 3),
    None,
    0.5,
    [
        CompositionPart("A", Fraction(2, 3), [Fraction(2, 3)]),
        CompositionPart("B", Fraction(1, 3), [Fraction(1, 3)]),
    ],
)
SECOND = GenericCalculationSecond(
    "second",
    Fraction(7, 9),
    "known",
    datetime.datetime(2026, 8, 2, 10, 30, tzinfo=datetime.timezone(datetime.timedelta(hours=2))),
    2.0,
    [
        CompositionPart("A", Fraction(1, 2), [Fraction(1, 2)]),
        CompositionPart("B", Fraction(1, 2), [Fraction(1, 2)]),
    ],
)


@pytest.fixture(scope="module", params=("sqlite", "duckdb", CLICKHOUSE_PARAM))
def plan(request):
    if request.param == "clickhousedb":
        with bulk_store(
            (FIRST, SECOND),
            entry_records={CalculationEntry: (GenericCalculationFirst, GenericCalculationSecond)},
        ) as store:
            yield stored_property_sql_plan(store, CalculationEntry)
        return
    if request.param == "duckdb":
        pytest.importorskip("duckdb_engine")
        database = Database.duckdb()
    else:
        database = Database.sqlite()
    with database:
        store = SqlStore(
            database,
            entry_records={CalculationEntry: (GenericCalculationFirst, GenericCalculationSecond)},
        )
        store.save(FIRST)
        store.save(SECOND)
        yield stored_property_sql_plan(store, CalculationEntry)


def _records(searchers):
    return [result[0][0] for searcher in searchers for result in searcher]


def test_plan_projects_concrete_backings_and_nullable_missing_properties(plan):
    rows = list(plan.records())
    assert {row["immutable_id"] for row in rows} == {"first", "second"}
    assert {row["type"] for row in rows} == {"calculations"}
    assert all(isinstance(row["id"], str) and len(row["id"]) == 64 for row in rows)
    # ``last_modified`` is absent from the first backing but mapped by the
    # second, so nullable response absence remains a concrete SQL NULL.
    assert {row["last_modified"] for row in rows} == {None, "2026-08-02T08:30:00+00:00"}


@pytest.mark.parametrize(
    ("filter_string", "expected"),
    (
        ('id CONTAINS ""', {"first", "second"}),
        ('type = "calculations"', {"first", "second"}),
        ('immutable_id = "one-third"', {"first"}),
        ('immutable_id = "null-comment"', {"first"}),
        ('immutable_id = "nested"', {"first"}),
        ('immutable_id = "when-known"', {"second"}),
        ('immutable_id = "score-gt"', {"second"}),
        ("last_modified IS UNKNOWN", {"first"}),
        ("last_modified IS KNOWN", {"second"}),
        ('NOT last_modified = "2026-01-01T00:00:00Z"', {"second"}),
        ('last_modified = "2026-08-02T08:30:00Z"', {"second"}),
        ('last_modified = "2026-08-02T10:30:00+02:00"', {"second"}),
        ('last_modified > "2026-08-02T08:29:59Z"', {"second"}),
    ),
)
def test_plan_translates_callback_operations_and_nullable_absence(plan, filter_string, expected):
    if plan.store._database.engine.dialect.name == "clickhousedb" and filter_string == 'immutable_id = "nested"':
        with pytest.raises(Exception, match="beyond one immediate scope"):
            plan.filter_searchers(filter_string)
        return
    assert {record.label for record in _records(plan.filter_searchers(filter_string))} == expected


def test_empty_boolean_combinators_follow_the_query_context_contract(plan):
    assert {record.label for record in _records(plan.filter_searchers('immutable_id = "empty-and"'))} == {
        "first",
        "second",
    }
    assert _records(plan.filter_searchers('immutable_id = "empty-or"')) == []


def test_fraction_equality_compiles_against_canonical_exact_column(plan):
    searcher = plan.filter_searchers('immutable_id = "one-third"')[0]
    statement = searcher._base_select(
        [searcher._outputs[0].element],
        [searcher._variables[0]._alias.c["sid"]],
    )
    rendered = str(statement.compile(dialect=plan.store._database.engine.dialect))
    assert "energy_exact" in rendered
    assert "energy =" not in rendered


def test_dialect_registers_exact_fraction_comparison_function(plan):
    with plan.store._database._engine.connect() as connection:
        equal = connection.execute(
            sqlalchemy.select(sqlalchemy.func.httk_fraction_scaled_equal("1/3", "2", "2/3", "1"))
        ).scalar_one()
        unequal = connection.execute(
            sqlalchemy.select(sqlalchemy.func.httk_fraction_scaled_equal("1/3", "1", "2/3", "1"))
        ).scalar_one()
    assert bool(equal)
    assert not bool(unequal)


@pytest.mark.parametrize("dialect", ("sqlite", "duckdb"))
def test_exact_fraction_function_installs_on_existing_connections_once(dialect):
    if dialect == "duckdb":
        pytest.importorskip("duckdb_engine")
        engine = sqlalchemy.create_engine("duckdb:///:memory:")
    else:
        engine = sqlalchemy.create_engine("sqlite://")
    try:
        # Populate the pool before it is wrapped. ``Database`` must install on
        # the subsequent checkout, and a second wrapper must stay idempotent.
        with engine.connect() as connection:
            connection.execute(sqlalchemy.text("SELECT 1"))
        Database(engine)
        Database(engine)
        with engine.connect() as connection:
            value = connection.execute(
                sqlalchemy.text("SELECT httk_fraction_scaled_equal('1/3', '2', '2/3', '1')")
            ).scalar_one()
        assert bool(value)
    finally:
        engine.dispose()


@pytest.mark.parametrize("dialect", ("sqlite", "duckdb"))
def test_exact_fraction_function_is_safe_across_simultaneous_and_reconnected_pool_connections(tmp_path, dialect):
    if dialect == "duckdb":
        pytest.importorskip("duckdb_engine")
        engine = sqlalchemy.create_engine(f"duckdb:///{tmp_path / 'fractions.duckdb'}", pool_size=2, max_overflow=0)
    else:
        engine = sqlalchemy.create_engine(f"sqlite:///{tmp_path / 'fractions.sqlite'}", pool_size=2, max_overflow=0)
    try:
        Database(engine)
        query = sqlalchemy.text("SELECT httk_fraction_scaled_equal('1/3', '2', '2/3', '1')")
        with engine.connect() as first:
            assert bool(first.execute(query).scalar_one())
            with engine.connect() as second:
                assert bool(second.execute(query).scalar_one())
        engine.dispose()
        with engine.connect() as reconnected:
            assert bool(reconnected.execute(query).scalar_one())
    finally:
        engine.dispose()


def test_sort_uses_each_backing_sort_mapping(plan):
    assert _records(plan.filter_searchers('type = "calculations"', sort=(("immutable_id", True),))) == [FIRST, SECOND]


def test_invalid_callback_literal_is_a_filter_type_error(plan):
    with pytest.raises(FilterTranslationError) as caught:
        plan.filter_searchers('immutable_id = "not-a-literal"')
    assert caught.value.category == "type-mismatch"


@pytest.mark.parametrize("literal", ("2026-08-02 08:30:00Z", "2026-W31-7T08:30:00Z"))
def test_timestamp_filters_reject_non_rfc3339_iso_forms(plan, literal):
    with pytest.raises(FilterTranslationError) as caught:
        plan.filter_searchers(f'last_modified = "{literal}"')
    assert caught.value.category == "type-mismatch"


def test_unconfigured_family_and_missing_nonnullable_response_are_configuration_errors():
    with Database.sqlite() as database:
        unconfigured = SqlStore(database, entry_records={})
        with pytest.raises(StoredPropertySqlConfigurationError, match="not configured"):
            stored_property_sql_plan(unconfigured, CalculationEntry)

    with Database.sqlite() as database:
        store = SqlStore(database, entry_records={FileEntry: IncompleteFile})
        with pytest.raises(StoredPropertySqlConfigurationError, match="name"):
            stored_property_sql_plan(store, FileEntry)


def test_backings_cannot_override_intrinsic_id_or_type():
    with Database.sqlite() as database:
        store = SqlStore(database, entry_records={BadFamily: BadCalculation})
        with pytest.raises(StoredPropertySqlConfigurationError, match="intrinsic"):
            stored_property_sql_plan(store, BadFamily)
