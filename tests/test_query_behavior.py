"""Backend-neutral search and stored-property query behavior.

These tests assert public query results and three-valued/set-operation
semantics.  SQL rendering, aliases, grouping flags, and other implementation
details remain in the SQL-only test modules.
"""

from dataclasses import dataclass
from fractions import Fraction
from typing import ClassVar

import pytest
from httk.core.storage import StorageInfo, stored_property
from test_db_searcher import ALL_FORMULAS, ALL_LABELS, LABELS, RECORDS, REF_A, Rec, Reference, Tag
from test_db_stored_properties import FIRST, SECOND, CalculationEntry, GenericCalculationFirst, GenericCalculationSecond


@pytest.fixture(autouse=True)
def _require_query_support(store_factory):
    """Query behavior is deferred for backends without a searcher."""
    if not hasattr(store_factory(), "searcher"):
        pytest.skip("backend has no query support yet")


@pytest.fixture
def query_store(store_factory):
    store = store_factory()
    for record in RECORDS:
        store.save(record)
    for tag in (
        Tag(RECORDS[0], "quality", "good"),
        Tag(RECORDS[0], "source", "exp"),
        Tag(RECORDS[2], "quality", "bad"),
    ):
        store.save(tag)
    for label in LABELS:
        store.save(label)
    return store


def rec_searcher(store):
    searcher = store.searcher()
    variable = searcher.variable(Rec)
    searcher.output(variable, "rec")
    return searcher, variable


def formulas(searcher) -> set[str]:
    return {item[0][0].formula for item in searcher}


def label_searcher(store):
    searcher = store.searcher()
    variable = searcher.variable(type(LABELS[0]))
    searcher.output(variable, "label")
    return searcher, variable


def texts(searcher) -> set[str]:
    return {item[0][0].text for item in searcher}


def stored_property_plan(store, family):
    """Use the store-level property-plan hook, with the current SQL adapter."""
    plan_factory = getattr(store, "stored_property_plan", None)
    if plan_factory is not None:
        return plan_factory(family)
    # The SQL store predates the backend-neutral hook; keep this compatibility
    # adapter local to the test until the next backend supplies that hook.
    from httk.data.db import stored_property_sql_plan

    return stored_property_sql_plan(store, family)


@pytest.mark.parametrize(
    ("build", "expected"),
    [
        (lambda v: v.formula == "NaCl", {"NaCl"}),
        (lambda v: v.formula != "NaCl", ALL_FORMULAS - {"NaCl"}),
        (lambda v: v.spacegroup < 221, {"SrCaTiO", "X"}),
        (lambda v: v.spacegroup <= 221, {"CaTiO3", "SrCaTiO", "X"}),
        (lambda v: v.spacegroup > 221, {"NaCl", "MgO", "CaO"}),
        (lambda v: v.spacegroup >= 225, {"NaCl", "MgO", "CaO"}),
        (lambda v: v.energy > Fraction(0), {"NaCl", "SrCaTiO", "X"}),
        (lambda v: v.energy == Fraction(1, 2), {"NaCl"}),
        (lambda v: v.energy != Fraction(1, 2), ALL_FORMULAS - {"NaCl"}),
        (lambda v: v.energy < Fraction(0), {"CaTiO3", "MgO"}),
        (lambda v: v.energy <= Fraction(-1, 3), {"CaTiO3", "MgO"}),
        (lambda v: v.energy >= Fraction(7, 8), {"SrCaTiO", "X"}),
        (lambda v: v.formula.contains("aTi"), {"CaTiO3", "SrCaTiO"}),
        (lambda v: v.formula.startswith("Ca"), {"CaTiO3", "CaO"}),
        (lambda v: v.formula.endswith("O"), {"MgO", "CaO", "SrCaTiO"}),
        (lambda v: v.formula.is_in("NaCl", "MgO"), {"NaCl", "MgO"}),
    ],
)
def test_operator_results(query_store, build, expected):
    searcher, variable = rec_searcher(query_store)
    searcher.add(build(variable))
    assert formulas(searcher) == expected


def test_boolean_combinators_and_counts(query_store):
    searcher, v = rec_searcher(query_store)
    searcher.add((v.spacegroup == 225) & v.formula.startswith("M"))
    assert formulas(searcher) == {"MgO"}

    searcher, v = rec_searcher(query_store)
    searcher.add((v.formula == "X") | (v.formula == "NaCl"))
    assert formulas(searcher) == {"X", "NaCl"}

    searcher, v = rec_searcher(query_store)
    searcher.add(~(v.spacegroup == 225))
    assert formulas(searcher) == {"CaTiO3", "SrCaTiO", "X"}
    assert searcher.count() == 3


@pytest.mark.parametrize(
    ("build", "expected"),
    [
        (lambda v: v.text.contains("50%"), {"50% Mg", "Mg 50%"}),
        (lambda v: v.text.contains("a_b"), {"a_b", "Mg a_b"}),
        (lambda v: v.text.startswith("50%"), {"50% Mg"}),
        (lambda v: v.text.startswith("a_b"), {"a_b"}),
        (lambda v: v.text.endswith("50%"), {"Mg 50%"}),
        (lambda v: v.text.endswith("a_b"), {"a_b", "Mg a_b"}),
    ],
)
def test_literal_string_matching_is_literal(query_store, build, expected):
    searcher, variable = label_searcher(query_store)
    searcher.add(build(variable))
    assert texts(searcher) == expected
    assert searcher.count() == len(expected)


def test_reference_chain_filters(query_store):
    searcher, v = rec_searcher(query_store)
    searcher.add(v.ref.doi == "10.1/a")
    assert formulas(searcher) == {"CaTiO3", "NaCl"}

    searcher, v = rec_searcher(query_store)
    searcher.add(v.ref.title == "Beta")
    assert formulas(searcher) == {"MgO", "SrCaTiO"}

    searcher, v = rec_searcher(query_store)
    searcher.add(v.ref == None)
    assert formulas(searcher) == {"CaO", "X"}

    searcher, v = rec_searcher(query_store)
    searcher.add(v.ref != None)
    assert formulas(searcher) == ALL_FORMULAS - {"CaO", "X"}

    searcher, v = rec_searcher(query_store)
    searcher.add(v.ref == REF_A)
    assert formulas(searcher) == {"CaTiO3", "NaCl"}

    with pytest.raises(ValueError, match="has not been stored"):
        rec_searcher(query_store)[1].ref == Reference("10.9/z", "Zeta")  # noqa: B015


def test_reference_join_and_self_join_results(query_store):
    searcher = query_store.searcher()
    tag = searcher.variable(Tag)
    record = searcher.variable(Rec)
    searcher.add(tag.rec == record)
    searcher.add(tag.tag == "quality")
    searcher.output(record, "rec")
    searcher.output(tag.value, "value")
    assert {(item[0][0].formula, item[0][1]) for item in searcher} == {("CaTiO3", "good"), ("MgO", "bad")}

    searcher = query_store.searcher()
    first = searcher.variable(Rec)
    second = searcher.variable(Rec)
    searcher.add(first.formula == "NaCl")
    searcher.add(first.spacegroup == second.spacegroup)
    searcher.add(second.formula != "NaCl")
    searcher.output(second, "rec")
    assert formulas(searcher) == {"MgO", "CaO"}


@pytest.mark.parametrize(
    ("operation", "field_value", "expected"),
    [
        ("has", None, False),
        ("has", [], False),
        ("has", ["allowed"], True),
        ("has_any", None, False),
        ("has_any", [], False),
        ("has_any", ["allowed"], True),
        ("has_only", None, True),
        ("has_only", [], True),
        ("has_only", ["outside"], False),
        ("is_in", None, True),
        ("is_in", [], True),
        ("is_in", ["outside"], False),
    ],
)
def test_optional_child_set_operation_truth_table(store_factory, operation, field_value, expected):
    @dataclass(frozen=True)
    class OptionalSetRecord:
        __httk_storage__: ClassVar[StorageInfo] = StorageInfo(dedup="none")

        name: str
        children: list[str] | None

    store = store_factory()
    store.save(OptionalSetRecord("target", field_value))
    searcher = store.searcher()
    variable = searcher.variable(OptionalSetRecord)
    searcher.output(variable, "record")
    values = ("allowed",)
    if operation == "has":
        expression = variable.children.has(values[0])
    elif operation == "has_any":
        expression = variable.children.has_any(*values)
    elif operation == "has_only":
        expression = variable.children.has_only(*values)
    else:
        expression = variable.children.is_in(*values)
    searcher.add(expression)
    assert bool(list(searcher)) is expected


@pytest.mark.parametrize("operation", ["has", "has_any", "has_only", "is_in"])
def test_child_set_operations_reject_none_at_build_time(store_factory, operation):
    @dataclass(frozen=True)
    class OptionalSetRecord:
        __httk_storage__: ClassVar[StorageInfo] = StorageInfo(dedup="none")

        name: str
        children: list[str] | None

    variable = store_factory().searcher().variable(OptionalSetRecord)
    with pytest.raises(ValueError, match="None is not a valid member of a child-field set operation"):
        if operation == "has":
            variable.children.has(None)
        elif operation == "has_any":
            variable.children.has_any(None)
        elif operation == "has_only":
            variable.children.has_only(None)
        else:
            variable.children.is_in(None)


def test_scalar_is_in_none_contract_is_unchanged(query_store):
    searcher, v = label_searcher(query_store)
    searcher.add(v.note.is_in(None, "present"))
    assert texts(searcher) == ALL_LABELS

    searcher, v = label_searcher(query_store)
    searcher.add(~v.note.is_in(None, "present"))
    assert texts(searcher) == set()

    searcher, v = label_searcher(query_store)
    searcher.add(v.note.is_in("present"))
    assert texts(searcher) == set()


def test_set_operations_and_negation_have_canonical_results(query_store):
    searcher, v = rec_searcher(query_store)
    searcher.add(v.symbols.has_any("O"))
    assert formulas(searcher) == {"CaTiO3", "MgO", "CaO", "SrCaTiO"}

    searcher, v = rec_searcher(query_store)
    searcher.add(v.symbols.has_only("O", "Ca", "Ti"))
    assert formulas(searcher) == {"CaTiO3", "CaO", "X"}

    searcher, v = rec_searcher(query_store)
    searcher.add(v.symbols.is_in("O", "Ca", "Ti"))
    assert formulas(searcher) == {"CaTiO3", "CaO", "X"}

    searcher, v = rec_searcher(query_store)
    searcher.add(~v.symbols.has_any("Ca", "Ti"))
    assert formulas(searcher) == {"NaCl", "MgO", "X"}

    searcher, v = rec_searcher(query_store)
    searcher.add(~v.symbols.has_only("O", "Ca", "Ti"))
    assert formulas(searcher) == {"NaCl", "MgO", "SrCaTiO"}

    searcher, v = rec_searcher(query_store)
    searcher.add(v.symbols.has_any("Ca") & v.symbols.has_any("Ti"))
    assert formulas(searcher) == {"CaTiO3", "SrCaTiO"}


def test_multi_member_has_any_does_not_duplicate_parents(query_store):
    searcher, v = rec_searcher(query_store)
    searcher.add(v.symbols.has_any("O", "Ca"))
    results = list(searcher)
    assert len(results) == 4
    assert {item[0][0].formula for item in results} == {"CaTiO3", "MgO", "CaO", "SrCaTiO"}


def test_set_operation_count_matches_iteration(query_store):
    for build in (
        lambda v: v.symbols.has_only("O", "Ca", "Ti"),
        lambda v: v.symbols.is_in("O", "Ca", "Ti"),
        lambda v: ~v.symbols.has_any("Ca", "Ti"),
        lambda v: ~v.symbols.has_only("O", "Ca", "Ti"),
    ):
        searcher, variable = rec_searcher(query_store)
        searcher.add(build(variable))
        assert searcher.count() == len(list(searcher))


def test_not_has_all_is_negation_of_anded_has_any(query_store):
    searcher, v = rec_searcher(query_store)
    searcher.add(~(v.symbols.has_any("Ca") & v.symbols.has_any("Ti")))
    assert formulas(searcher) == ALL_FORMULAS - {"CaTiO3", "SrCaTiO"}
    assert searcher.count() == len(list(searcher))


def test_double_not_round_trips(query_store):
    searcher, v = rec_searcher(query_store)
    searcher.add(~~v.symbols.has_any("Ca", "Ti"))
    assert formulas(searcher) == {"CaTiO3", "CaO", "SrCaTiO"}
    assert searcher.count() == len(list(searcher))


def test_not_inside_and_with_a_scalar(query_store):
    searcher, v = rec_searcher(query_store)
    searcher.add((v.spacegroup == 225) & ~v.symbols.has_any("Ca"))
    assert formulas(searcher) == {"NaCl", "MgO"}
    assert searcher.count() == len(list(searcher))


def test_not_over_a_mixed_conjunction(query_store):
    searcher, v = rec_searcher(query_store)
    searcher.add(~((v.spacegroup == 225) & v.symbols.has_any("Ca")))
    assert formulas(searcher) == ALL_FORMULAS - {"CaO"}
    assert searcher.count() == len(list(searcher))


def test_not_over_a_mixed_disjunction(query_store):
    searcher, v = rec_searcher(query_store)
    searcher.add(~((v.spacegroup == 225) | v.symbols.has_any("Ti")))
    assert formulas(searcher) == {"X"}
    assert searcher.count() == len(list(searcher))


def test_child_comparison_and_mixed_predicate_results(query_store):
    searcher, v = rec_searcher(query_store)
    searcher.add(~(v.symbols == "O"))
    assert formulas(searcher) == {"NaCl", "X"}

    searcher, v = rec_searcher(query_store)
    searcher.add(v.symbols == "O")
    assert formulas(searcher) == {"CaTiO3", "MgO", "CaO", "SrCaTiO"}

    searcher, v = rec_searcher(query_store)
    searcher.add(v.symbols.has_only("Ca", "O") | (v.symbols == "Na"))
    assert formulas(searcher) == {"CaO", "X", "NaCl"}


def test_query_order_offset_limit_and_count(query_store):
    searcher, v = rec_searcher(query_store)
    searcher.add_sort(v.formula)
    assert [item[0][0].formula for item in searcher] == ["CaO", "CaTiO3", "MgO", "NaCl", "SrCaTiO", "X"]
    searcher.set_limit(2)
    assert [item[0][0].formula for item in searcher] == ["CaO", "CaTiO3"]
    searcher.set_limit(-1)
    searcher.add_offset(2)
    assert [item[0][0].formula for item in searcher] == ["MgO", "NaCl", "SrCaTiO", "X"]
    assert searcher.count() == 6


@dataclass(frozen=True)
class StoredValueRecord:
    value: int | None

    @stored_property
    def doubled(self) -> int | None:
        return None if self.value is None else self.value * 2


def test_stored_property_predicate_preserves_three_valued_results(store_factory):
    store = store_factory()
    for value in (None, 1, 2):
        store.save(StoredValueRecord(value))
    searcher = store.searcher()
    variable = searcher.variable(StoredValueRecord)
    searcher.output(variable, "record")
    searcher.add(variable.doubled == 2)
    assert [item[0][0].value for item in searcher] == [1]

    searcher = store.searcher()
    variable = searcher.variable(StoredValueRecord)
    searcher.output(variable, "record")
    searcher.add(~(variable.doubled == 2))
    assert {item[0][0].value for item in searcher} == {2}


def test_scaled_exact_equal_stored_property_result(store_factory):
    """The callback's exact scaled comparison selects the matching backing."""
    store = store_factory(entry_records={CalculationEntry: (GenericCalculationFirst, GenericCalculationSecond)})
    store.save(FIRST)
    store.save(SECOND)
    plan = stored_property_plan(store, CalculationEntry)
    searchers = plan.filter_searchers('immutable_id = "composition"')
    assert [result[0][0].label for searcher in searchers for result in searcher] == ["first"]
