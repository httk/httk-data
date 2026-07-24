"""Tests for the query DSL (httk.data.db.searcher): operators, joins, set semantics, parity."""

from dataclasses import dataclass
from fractions import Fraction
from typing import Annotated, ClassVar

import pytest
from httk.core import FracVector, Shape, StorageInfo

from httk.data.db import Database, SchemaError, SqlStore


@dataclass(frozen=True)
class Reference:
    doi: str
    title: str


@dataclass(frozen=True)
class Rec:
    formula: str
    spacegroup: int
    energy: Fraction
    symbols: list[str]
    ref: Reference | None = None


@dataclass(frozen=True)
class Tag:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(dedup="by_value")

    rec: Rec
    tag: str
    value: str


@dataclass(frozen=True)
class Cell:
    name: str
    basis: Annotated[FracVector, Shape(3, 3)]


REF_A = Reference("10.1/a", "Alpha")
REF_B = Reference("10.1/b", "Beta")

RECORDS = [
    Rec("CaTiO3", 221, Fraction(-1, 3), ["O", "Ca", "Ti"], REF_A),
    Rec("NaCl", 225, Fraction(1, 2), ["Na", "Cl"], REF_A),
    Rec("MgO", 225, Fraction(-5, 4), ["Mg", "O"], REF_B),
    Rec("CaO", 225, Fraction(0), ["Ca", "O"], None),
    Rec("SrCaTiO", 62, Fraction(3, 2), ["O", "Ca", "Ti", "Sr"], REF_B),
    Rec("X", 1, Fraction(7, 8), [], None),
]

TAGS = [
    Tag(RECORDS[0], "quality", "good"),
    Tag(RECORDS[0], "source", "exp"),
    Tag(RECORDS[2], "quality", "bad"),
]

ALL_FORMULAS = {rec.formula for rec in RECORDS}


@pytest.fixture(params=["sqlite", "duckdb"])
def store(request):
    """A populated store per supported dialect (duckdb skips where not installed)."""
    if request.param == "duckdb":
        pytest.importorskip("duckdb_engine")
        database_manager = Database.duckdb()
    else:
        database_manager = Database.sqlite()
    with database_manager as database:
        sql_store = SqlStore(database)
        with sql_store.transaction():
            for rec in RECORDS:
                sql_store.save(rec)
            for tag in TAGS:
                sql_store.save(tag)
        yield sql_store


def formulas(searcher) -> set[str]:
    return {item[0][0].formula for item in searcher}


def rec_searcher(store):
    searcher = store.searcher()
    variable = searcher.variable(Rec)
    searcher.output(variable, "rec")
    return searcher, variable


# --------------------------------------------------------------------- operator matrix


@pytest.mark.parametrize(
    "build, expected",
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
        (lambda v: v.formula.like("%aTi%"), {"CaTiO3", "SrCaTiO"}),
        (lambda v: v.formula.startswith("Ca"), {"CaTiO3", "CaO"}),
        (lambda v: v.formula.endswith("O"), {"MgO", "CaO", "SrCaTiO"}),
        (lambda v: v.formula.is_in("NaCl", "MgO"), {"NaCl", "MgO"}),
    ],
)
def test_operator_matrix(store, build, expected):
    searcher, variable = rec_searcher(store)
    searcher.add(build(variable))
    assert formulas(searcher) == expected


def test_combinators(store):
    searcher, v = rec_searcher(store)
    searcher.add((v.spacegroup == 225) & v.formula.startswith("M"))
    assert formulas(searcher) == {"MgO"}

    searcher, v = rec_searcher(store)
    searcher.add((v.formula == "X") | (v.formula == "NaCl"))
    assert formulas(searcher) == {"X", "NaCl"}

    searcher, v = rec_searcher(store)
    searcher.add(~(v.spacegroup == 225))
    assert formulas(searcher) == {"CaTiO3", "SrCaTiO", "X"}


# --------------------------------------------------------------------- references


def test_reference_chain_attribute(store):
    searcher, v = rec_searcher(store)
    searcher.add(v.ref.doi == "10.1/a")
    assert formulas(searcher) == {"CaTiO3", "NaCl"}


def test_reference_chain_shares_one_join(store):
    searcher, v = rec_searcher(store)
    searcher.add(v.ref.doi == "10.1/b")
    searcher.add(v.ref.title == "Beta")
    assert len(v._joins) == 1  # both conditions hit the same joined alias
    assert len(v._reference_variables) == 1
    assert formulas(searcher) == {"MgO", "SrCaTiO"}


def test_reference_equals_none(store):
    searcher, v = rec_searcher(store)
    searcher.add(v.ref == None)  # noqa: E711  (the DSL builds IS NULL from == None)
    assert formulas(searcher) == {"CaO", "X"}


def test_reference_not_equals_none(store):
    searcher, v = rec_searcher(store)
    searcher.add(v.ref != None)  # noqa: E711
    assert formulas(searcher) == ALL_FORMULAS - {"CaO", "X"}


def test_reference_equals_stored_object(store):
    searcher, v = rec_searcher(store)
    searcher.add(v.ref == REF_A)
    assert formulas(searcher) == {"CaTiO3", "NaCl"}


def test_reference_equals_unknown_object_raises(store):
    _searcher, v = rec_searcher(store)
    with pytest.raises(ValueError, match="has not been stored"):
        v.ref == Reference("10.9/z", "Zeta")  # noqa: B015


def test_reference_join_across_variables(store):
    searcher = store.searcher()
    t = searcher.variable(Tag)
    v = searcher.variable(Rec)
    searcher.add(t.rec == v)
    searcher.add(t.tag == "quality")
    searcher.output(v, "rec")
    searcher.output(t.value, "value")
    results = list(searcher)
    assert {(item[0][0].formula, item[0][1]) for item in results} == {("CaTiO3", "good"), ("MgO", "bad")}
    assert all(item[1] == ("rec", "value") for item in results)


def test_self_join(store):
    searcher = store.searcher()
    a = searcher.variable(Rec)
    b = searcher.variable(Rec)
    searcher.add(a.formula == "NaCl")
    searcher.add(a.spacegroup == b.spacegroup)
    searcher.add(b.formula != "NaCl")
    searcher.output(b, "rec")
    assert formulas(searcher) == {"MgO", "CaO"}


# --------------------------------------------------------------------- child set operations


def translate(searcher, expression):
    """The httk-optimade translate_filter pattern for needs_post expressions."""
    searcher.add(expression)
    searcher.add_all(expression)


def test_has_any_where_position(store):
    searcher, v = rec_searcher(store)
    searcher.add(v.symbols.has_any("O"))
    assert formulas(searcher) == {"CaTiO3", "MgO", "CaO", "SrCaTiO"}


def test_has_any_does_not_duplicate_parents(store):
    searcher, v = rec_searcher(store)
    searcher.add(v.symbols.has_any("O", "Ca"))
    results = list(searcher)
    assert len(results) == 4  # CaTiO3 and CaO match twice each, but appear once
    assert {item[0][0].formula for item in results} == {"CaTiO3", "MgO", "CaO", "SrCaTiO"}


def test_has_only_exact_subset_superset_disjoint(store):
    searcher, v = rec_searcher(store)
    translate(searcher, v.symbols.has_only("O", "Ca", "Ti"))
    # exact ({O,Ca,Ti}) and subset ({Ca,O}) match; superset (+Sr) and
    # disjoint ({Na,Cl}, {Mg,O}) do not; the empty set matches (see below).
    assert formulas(searcher) == {"CaTiO3", "CaO", "X"}


def test_has_only_includes_empty_child_record(store):
    searcher, v = rec_searcher(store)
    translate(searcher, v.symbols.has_only("Na", "Cl"))
    # Locked-in semantics: a record with no child rows satisfies has_only
    # (the empty set is a subset of any value set), matching the reference
    # in-memory store's exact set predicate.
    assert formulas(searcher) == {"NaCl", "X"}


def test_has_any_excludes_empty_child_record(store):
    searcher, v = rec_searcher(store)
    searcher.add(v.symbols.has_any("O", "Ca", "Ti", "Sr", "Na", "Cl", "Mg"))
    assert formulas(searcher) == ALL_FORMULAS - {"X"}


def test_not_has_any_via_inv_any(store):
    searcher, v = rec_searcher(store)
    translate(searcher, ~v.symbols.has_inv_any("Ca", "Ti"))
    # NOT (symbols HAS ANY "Ca","Ti"): records with no symbol in the set,
    # including the record with no symbols at all.
    assert formulas(searcher) == {"NaCl", "MgO", "X"}


def test_not_has_only_via_inv_only(store):
    searcher, v = rec_searcher(store)
    translate(searcher, ~v.symbols.has_inv_only("O", "Ca", "Ti"))
    # NOT (symbols HAS ONLY "O","Ca","Ti"): some symbol outside the set.
    assert formulas(searcher) == {"NaCl", "MgO", "SrCaTiO"}


def test_repeated_child_access_makes_fresh_joins(store):
    # v1 semantics: each attribute access mints a fresh child-table alias, so
    # AND-composed set predicates on one field constrain independent rows.
    searcher, v = rec_searcher(store)
    first = v.symbols
    second = v.symbols
    assert len(v._joins) == 2  # a fresh child alias per attribute access
    assert first._element is not second._element
    searcher.add(first.has_any("Ti"))
    assert formulas(searcher) == {"CaTiO3", "SrCaTiO"}


def test_has_all_pattern_via_anded_has_any(store):
    # The OPTIMADE translation layer implements non-inverted HAS ALL as
    # has_any(a) & has_any(b) in WHERE position; that requires the fresh
    # aliases above (on one shared alias it would be unsatisfiable).
    searcher, v = rec_searcher(store)
    searcher.add(v.symbols.has_any("Ca") & v.symbols.has_any("Ti"))
    assert formulas(searcher) == {"CaTiO3", "SrCaTiO"}


def test_has_all_pattern_no_false_positives(store):
    searcher, v = rec_searcher(store)
    searcher.add(v.symbols.has_any("Na") & v.symbols.has_any("Ti"))
    assert formulas(searcher) == set()


# --------------------------------------------------------------------- outputs and iteration shape


def test_iteration_shape_and_object_identity(store):
    searcher = store.searcher()
    v = searcher.variable(Rec)
    searcher.output(v, "rec")
    searcher.output(v.formula, "formula")
    searcher.add(v.formula == "NaCl")
    items = list(searcher)
    assert len(items) == 1
    values, names = items[0]
    assert names == ("rec", "formula")
    assert values[0] is RECORDS[1]  # identity cache: the saved instance itself
    assert values[1] == "NaCl"
    assert items[0][0][0] is values[0]  # item[0][0] is the matched object


def test_iteration_without_outputs_raises(store):
    searcher = store.searcher()
    searcher.variable(Rec)
    with pytest.raises(ValueError, match="output"):
        iter(searcher)


def test_output_column_alongside_object_in_grouped_mode(store):
    searcher = store.searcher()
    v = searcher.variable(Rec)
    searcher.output(v, "rec")
    searcher.output(v.spacegroup, "spacegroup")
    translate(searcher, v.symbols.has_only("O", "Ca", "Ti"))
    results = {(item[0][0].formula, item[0][1]) for item in searcher}
    assert results == {("CaTiO3", 221), ("CaO", 225), ("X", 1)}


# --------------------------------------------------------------------- sorting, limit, offset, count


def test_sort_ascending_on_fraction_float_companion(store):
    searcher, v = rec_searcher(store)
    searcher.add_sort(v.energy, False)
    assert [item[0][0].formula for item in searcher] == ["MgO", "CaTiO3", "CaO", "NaCl", "X", "SrCaTiO"]


def test_sort_descending_with_secondary_key(store):
    searcher, v = rec_searcher(store)
    searcher.add_sort(v.spacegroup, True)
    searcher.add_sort(v.formula, False)
    assert [item[0][0].formula for item in searcher] == ["CaO", "MgO", "NaCl", "CaTiO3", "SrCaTiO", "X"]


def test_set_limit_and_clearing(store):
    searcher, v = rec_searcher(store)
    searcher.add_sort(v.formula, False)
    searcher.set_limit(2)
    assert [item[0][0].formula for item in searcher] == ["CaO", "CaTiO3"]
    searcher.set_limit(-1)
    assert len(list(searcher)) == 6


def test_add_offset_and_mutable_offset_attribute(store):
    searcher, v = rec_searcher(store)
    searcher.add_sort(v.formula, False)
    assert searcher.offset == 0
    searcher.add_offset(2)
    assert searcher.offset == 2
    assert [item[0][0].formula for item in searcher] == ["MgO", "NaCl", "SrCaTiO", "X"]
    searcher.add_offset(2)
    assert searcher.offset == 4
    searcher.offset = 5  # the attribute is directly writable (execution.py contract)
    assert [item[0][0].formula for item in searcher] == ["X"]


def test_offset_without_limit_after_clearing(store):
    # execution.py sets set_limit(-1) before add_offset to mean "no bound".
    searcher, v = rec_searcher(store)
    searcher.add_sort(v.formula, False)
    searcher.set_limit(-1)
    searcher.add_offset(4)
    assert [item[0][0].formula for item in searcher] == ["SrCaTiO", "X"]


def test_count_ungrouped(store):
    searcher, _v = rec_searcher(store)
    assert searcher.count() == 6


def test_count_ignores_limit_and_offset(store):
    searcher, v = rec_searcher(store)
    searcher.add(v.spacegroup == 225)
    searcher.set_limit(1)
    searcher.add_offset(1)
    assert searcher.count() == 3
    assert len(list(searcher)) == 1


def test_count_grouped(store):
    searcher, v = rec_searcher(store)
    translate(searcher, v.symbols.has_only("O", "Ca", "Ti"))
    searcher.set_limit(1)
    assert searcher.count() == 3


# --------------------------------------------------------------------- errors


def test_unknown_field_raises_attribute_error(store):
    _searcher, v = rec_searcher(store)
    with pytest.raises(AttributeError, match=r"Rec.*'nope'"):
        v.nope  # noqa: B018


def test_fixed_array_field_not_queryable():
    with Database.sqlite() as database:
        sql_store = SqlStore(database)
        searcher = sql_store.searcher()
        v = searcher.variable(Cell)
        with pytest.raises(SchemaError, match="basis"):
            v.basis  # noqa: B018


def test_iteration_without_variables_raises():
    with Database.sqlite() as database:
        searcher = SqlStore(database).searcher()
        with pytest.raises(ValueError, match="variable"):
            searcher.count()


# --------------------------------------------------------------------- parity with InMemoryStore


def _program_has_any(searcher, v):
    searcher.add(v.symbols.has_any("O", "Na"))


def _program_has_only(searcher, v):
    expression = v.symbols.has_only("O", "Ca", "Ti")
    searcher.add(expression)
    searcher.add_all(expression)


def _program_not_has_any(searcher, v):
    expression = ~v.symbols.has_inv_any("Ca", "Ti")
    searcher.add(expression)
    searcher.add_all(expression)


def _program_string_ops(searcher, v):
    searcher.add(v.formula.startswith("Ca") | v.formula.endswith("O"))
    searcher.add(v.formula.like("%a%"))


def _program_numeric_range(searcher, v):
    searcher.add((v.energy > 0.0) & (v.energy <= 1.0))


PARITY_PROGRAMS = [
    _program_has_any,
    _program_has_only,
    _program_not_has_any,
    _program_string_ops,
    _program_numeric_range,
]


def test_parity_with_in_memory_store():
    pytest.importorskip("httk.optimade")
    from httk.optimade.backend.memory_store import InMemoryStore

    memory_rows = [
        {
            "formula": rec.formula,
            "spacegroup": rec.spacegroup,
            "energy": float(rec.energy),
            "symbols": list(rec.symbols),
        }
        for rec in RECORDS
    ]
    memory_store = InMemoryStore({"recs": memory_rows})

    with Database.sqlite() as database:
        sql_store = SqlStore(database)
        for rec in RECORDS:
            sql_store.save(rec)

        for program in PARITY_PROGRAMS:
            memory_searcher = memory_store.searcher()
            memory_variable = memory_searcher.variable("recs")
            memory_searcher.output(memory_variable, "rec")
            program(memory_searcher, memory_variable)
            memory_ids = {item[0][0]["formula"] for item in memory_searcher}

            sql_searcher = sql_store.searcher()
            sql_variable = sql_searcher.variable(Rec)
            sql_searcher.output(sql_variable, "rec")
            program(sql_searcher, sql_variable)
            sql_ids = {item[0][0].formula for item in sql_searcher}

            assert sql_ids == memory_ids, program.__name__
            assert sql_searcher.count() == memory_searcher.count(), program.__name__
