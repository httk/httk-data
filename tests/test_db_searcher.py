"""Tests for the query DSL (httk.data.db.searcher): operators, joins, set semantics, parity."""

import gc
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
class Label:
    """A string field carrying LIKE metacharacters, plus a nullable companion.

    Kept apart from :class:`Rec` on purpose: the escaping and NULL-safety tests
    need values that would break a naive pattern rendering, and RECORDS must
    stay readable as the plain operator/set-semantics fixture.
    """

    text: str
    note: str | None = None


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

# `%` and `_` are LIKE metacharacters; the neutral protocol matches them
# literally, so every "wrong" row below is one a naive (unescaped) rendering
# would wrongly match.
LABELS = [
    Label("50% Mg"),
    Label("5012 Mg"),
    Label("a_b"),
    Label("axb"),
    Label("Mg 50%"),
    Label("Mg 5012"),
    Label("Mg a_b"),
    Label("Mg axb"),
]

ALL_LABELS = {label.text for label in LABELS}


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
            for label in LABELS:
                sql_store.save(label)
        yield sql_store


def formulas(searcher) -> set[str]:
    return {item[0][0].formula for item in searcher}


def rec_searcher(store):
    searcher = store.searcher()
    variable = searcher.variable(Rec)
    searcher.output(variable, "rec")
    return searcher, variable


def texts(searcher) -> set[str]:
    return {item[0][0].text for item in searcher}


def label_searcher(store):
    searcher = store.searcher()
    variable = searcher.variable(Label)
    searcher.output(variable, "label")
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
        (lambda v: v.formula.contains("aTi"), {"CaTiO3", "SrCaTiO"}),
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


# --------------------------------------------------------------- literal string matching

# contains/startswith/endswith take literal text: `%` and `_` match themselves.
# Each case pairs a query with the rows it must match; the LABELS rows that are
# *not* listed are exactly the ones a naive (unescaped LIKE) rendering would
# wrongly return.
LITERAL_MATCH_CASES = [
    (lambda v: v.text.contains("50%"), {"50% Mg", "Mg 50%"}),
    (lambda v: v.text.contains("a_b"), {"a_b", "Mg a_b"}),
    (lambda v: v.text.startswith("50%"), {"50% Mg"}),
    (lambda v: v.text.startswith("a_b"), {"a_b"}),
    (lambda v: v.text.endswith("50%"), {"Mg 50%"}),
    (lambda v: v.text.endswith("a_b"), {"a_b", "Mg a_b"}),
]


@pytest.mark.parametrize("build, expected", LITERAL_MATCH_CASES)
def test_literal_string_matching_escapes_like_metacharacters(store, build, expected):
    searcher, v = label_searcher(store)
    searcher.add(build(v))
    assert texts(searcher) == expected
    assert searcher.count() == len(expected)


def test_like_is_private_to_the_sql_backend(store):
    # The neutral protocol carries no pattern language; `like` was removed from
    # the column surface and the LIKE rendering is an implementation detail.
    _searcher, v = label_searcher(store)
    assert not hasattr(v.text, "like")
    assert hasattr(v.text, "_like")


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
    searcher.add(v.ref == None)
    assert formulas(searcher) == {"CaO", "X"}


def test_reference_not_equals_none(store):
    searcher, v = rec_searcher(store)
    searcher.add(v.ref != None)
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
    searcher.add(v.symbols.has_only("O", "Ca", "Ti"))
    # exact ({O,Ca,Ti}) and subset ({Ca,O}) match; superset (+Sr) and
    # disjoint ({Na,Cl}, {Mg,O}) do not; the empty set matches (see below).
    assert formulas(searcher) == {"CaTiO3", "CaO", "X"}


def test_has_only_includes_empty_child_record(store):
    searcher, v = rec_searcher(store)
    searcher.add(v.symbols.has_only("Na", "Cl"))
    # Locked-in semantics: a record with no child rows satisfies has_only
    # (the empty set is a subset of any value set), matching the reference
    # in-memory store's exact set predicate.
    assert formulas(searcher) == {"NaCl", "X"}


def test_has_any_excludes_empty_child_record(store):
    searcher, v = rec_searcher(store)
    searcher.add(v.symbols.has_any("O", "Ca", "Ti", "Sr", "Na", "Cl", "Mg"))
    assert formulas(searcher) == ALL_FORMULAS - {"X"}


def test_not_has_any(store):
    searcher, v = rec_searcher(store)
    searcher.add(~v.symbols.has_any("Ca", "Ti"))
    # NOT (symbols HAS ANY "Ca","Ti"): records with no symbol in the set,
    # including the record with no symbols at all. This is the exact result set
    # the removed `~has_inv_any(...)` + add_all pattern produced.
    assert formulas(searcher) == {"NaCl", "MgO", "X"}
    assert searcher.count() == len(list(searcher))


def test_not_has_only(store):
    searcher, v = rec_searcher(store)
    searcher.add(~v.symbols.has_only("O", "Ca", "Ti"))
    # NOT (symbols HAS ONLY "O","Ca","Ti"): some symbol outside the set. The
    # exact result set the removed `~has_inv_only(...)` + add_all pattern gave.
    assert formulas(searcher) == {"NaCl", "MgO", "SrCaTiO"}
    assert searcher.count() == len(list(searcher))


def test_has_only_alone_is_complete(store):
    # Previously `add(has_only(...))` on its own rendered as constant TRUE and
    # matched everything; the expression now carries its own post-filter
    # placement, so one add() is the whole condition.
    searcher, v = rec_searcher(store)
    searcher.add(v.symbols.has_only("O", "Ca", "Ti"))
    assert formulas(searcher) == {"CaTiO3", "CaO", "X"}
    assert searcher.count() == 3


def test_is_in_on_a_child_field_is_for_all(store):
    # On a child field is_in reads as "every child value is in the set" — the
    # same aggregate as has_only, and likewise complete from a single add().
    searcher, v = rec_searcher(store)
    searcher.add(v.symbols.is_in("O", "Ca", "Ti"))
    assert formulas(searcher) == {"CaTiO3", "CaO", "X"}
    assert searcher.count() == len(list(searcher))


def test_not_has_all_is_negation_of_anded_has_any(store):
    # NOT (symbols HAS ALL "Ca","Ti"): everything except the records holding
    # both. Negating a conjunction of set predicates goes through the aggregate.
    searcher, v = rec_searcher(store)
    searcher.add(~(v.symbols.has_any("Ca") & v.symbols.has_any("Ti")))
    assert formulas(searcher) == ALL_FORMULAS - {"CaTiO3", "SrCaTiO"}
    assert searcher.count() == len(list(searcher))


def test_double_not_round_trips(store):
    searcher, v = rec_searcher(store)
    searcher.add(~~v.symbols.has_any("Ca", "Ti"))
    assert formulas(searcher) == {"CaTiO3", "CaO", "SrCaTiO"}
    assert searcher.count() == len(list(searcher))


def test_not_inside_and_with_a_scalar(store):
    searcher, v = rec_searcher(store)
    searcher.add((v.spacegroup == 225) & ~v.symbols.has_any("Ca"))
    assert formulas(searcher) == {"NaCl", "MgO"}
    assert searcher.count() == len(list(searcher))


def test_not_over_a_mixed_conjunction(store):
    searcher, v = rec_searcher(store)
    searcher.add(~((v.spacegroup == 225) & v.symbols.has_any("Ca")))
    assert formulas(searcher) == ALL_FORMULAS - {"CaO"}
    assert searcher.count() == len(list(searcher))


def test_not_over_a_mixed_disjunction(store):
    searcher, v = rec_searcher(store)
    searcher.add(~((v.spacegroup == 225) | v.symbols.has_any("Ti")))
    assert formulas(searcher) == {"X"}
    assert searcher.count() == len(list(searcher))


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


# ------------------------------------------------- grouping of HAVING-referenced columns


def test_mixed_scalar_and_set_filter(store):
    # A scalar WHERE condition alongside a set condition in HAVING position.
    searcher, v = rec_searcher(store)
    searcher.add(v.formula == "CaTiO3")
    searcher.add(~v.symbols.has_any("Na", "Cl"))
    assert formulas(searcher) == {"CaTiO3"}
    assert searcher.count() == len(list(searcher))


def test_for_all_with_scalar_filter(store):
    # The OPTIMADE `nelements=3 AND elements HAS ONLY ...` shape: one expression
    # applied in both positions, so the scalar column reaches HAVING and must be
    # grouped by (a BinderException on DuckDB before the grouping fix).
    searcher, v = rec_searcher(store)
    searcher.add((v.spacegroup == 225) & v.symbols.has_only("Na", "Cl"))
    assert formulas(searcher) == {"NaCl"}
    assert searcher.count() == len(list(searcher))


def test_sort_under_grouped_mode(store):
    # ORDER BY on a root column while grouped: the sort key must join the group
    # set (a BinderException on DuckDB before the grouping fix).
    searcher, v = rec_searcher(store)
    searcher.add(v.symbols.has_any("O"))
    searcher.add_sort(v.formula, False)
    assert [item[0][0].formula for item in searcher] == ["CaO", "CaTiO3", "MgO", "SrCaTiO"]
    assert searcher.count() == len(list(searcher))


def test_reference_comparison_under_grouped_mode(store):
    # A foreign-key comparison in HAVING position groups by the foreign key.
    searcher, v = rec_searcher(store)
    searcher.add((v.ref == REF_B) & ~v.symbols.has_any("Na"))
    assert formulas(searcher) == {"MgO", "SrCaTiO"}
    assert searcher.count() == len(list(searcher))


# ------------------------------------------------------------- constant expressions


def test_always_true_and_always_false_are_constants(store):
    searcher, v = label_searcher(store)
    searcher.add(v.always_true())
    assert texts(searcher) == ALL_LABELS
    assert searcher.count() == len(ALL_LABELS)

    searcher, v = label_searcher(store)
    searcher.add(v.always_false())
    assert texts(searcher) == set()
    assert searcher.count() == 0


def test_always_true_is_null_safe_where_column_self_comparison_was_not(store):
    # Every LABELS row has a NULL `note`. `always_true()` still matches them all;
    # the `column == column` convention it replaced evaluates to NULL — not true
    # — for a NULL column, and so silently dropped exactly these rows.
    searcher, v = label_searcher(store)
    searcher.add(v.always_true())
    assert texts(searcher) == ALL_LABELS

    searcher, v = label_searcher(store)
    searcher.add(v.note == v.note)
    assert texts(searcher) == set()


def test_scalar_membership_handles_nulls_without_sql_three_valued_leaks(store):
    """``is_in`` treats NULL as an explicit member, including under ``~``."""
    searcher, v = label_searcher(store)
    searcher.add(v.note.is_in(None, "present"))
    assert texts(searcher) == ALL_LABELS

    searcher, v = label_searcher(store)
    searcher.add(~v.note.is_in(None, "present"))
    assert texts(searcher) == set()

    searcher, v = label_searcher(store)
    searcher.add(v.note.is_in("present"))
    assert texts(searcher) == set()

    searcher, v = label_searcher(store)
    searcher.add(~v.note.is_in("present"))
    assert texts(searcher) == ALL_LABELS


def test_true_handler_filter_matches_rows_with_a_null_scalar(store):
    # End to end through the OPTIMADE translation: `IS KNOWN` on a property the
    # handler table declares always-known routes through true_handler, and must
    # match even though the underlying column is NULL for every row.
    from httk.data.optimade_query import filter_searcher

    searcher = filter_searcher(
        store,
        Label,
        "note IS KNOWN",
        entry_type="labels",
        property_fulltypes={"text": "string", "note": "string"},
    )
    assert texts(searcher) == ALL_LABELS


# --------------------------------------------------------------------- outputs and iteration shape


def test_iteration_shape_and_lazy_object_equality(store):
    searcher = store.searcher()
    v = searcher.variable(Rec)
    searcher.output(v, "rec")
    searcher.output(v.formula, "formula")
    searcher.add(v.formula == "NaCl")
    items = list(searcher)
    assert len(items) == 1
    values, names = items[0]
    assert names == ("rec", "formula")
    assert values[0] == RECORDS[1]  # search rows bypass the identity cache
    assert values[1] == "NaCl"
    assert items[0][0][0] is values[0]  # item[0][0] is the matched object


def test_field_output_carries_exact_projection_ir(store):
    searcher = store.searcher()
    variable = searcher.variable(Rec)
    searcher.output(variable.energy, "energy")
    projection = searcher._outputs[0]
    assert projection.variable is variable
    assert projection.spec is variable._schema.field("energy")
    assert projection.exact_element is not None
    assert projection.exact_element.name == "energy_exact"
    assert projection.codec is not None
    assert projection.decoder is projection.codec.decode


def test_search_result_is_a_named_two_tuple(store):
    searcher = store.searcher()
    v = searcher.variable(Label)
    searcher.output(v, "label")
    searcher.output(v.text, "text")
    searcher.add(v.text == "a_b")
    (result,) = list(searcher)
    values, names = result  # unpacks as the plain 2-tuple it always was
    assert len(result) == 2
    assert names == ("label", "text")
    assert result.names == names and result.values == values
    assert values[0].text == "a_b"
    assert values[1] == "a_b"
    assert result[0][0] is values[0]


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
    searcher.add(v.symbols.has_only("O", "Ca", "Ti"))
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
    searcher.add(v.symbols.has_only("O", "Ca", "Ti"))
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
    searcher.add(v.symbols.has_only("O", "Ca", "Ti"))


def _program_not_has_any(searcher, v):
    searcher.add(~v.symbols.has_any("Ca", "Ti"))


def _program_not_has_only(searcher, v):
    searcher.add(~v.symbols.has_only("O", "Ca", "Ti"))


def _program_not_has_all(searcher, v):
    # NOT (HAS ALL "Ca","Ti") — the translation layer's HAS ALL is a
    # conjunction of has_any, so this negates a conjunction of set predicates.
    searcher.add(~(v.symbols.has_any("Ca") & v.symbols.has_any("Ti")))


def _program_not_inside_and(searcher, v):
    searcher.add((v.spacegroup == 225) & ~v.symbols.has_any("Ca"))


def _program_not_over_mixed_and(searcher, v):
    searcher.add(~((v.spacegroup == 225) & v.symbols.has_any("Ca")))


def _program_not_over_mixed_or(searcher, v):
    searcher.add(~((v.spacegroup == 225) | v.symbols.has_any("Ti")))


def _program_double_not(searcher, v):
    searcher.add(~~v.symbols.has_any("Ca", "Ti"))


def _program_string_ops(searcher, v):
    searcher.add(v.formula.startswith("Ca") | v.formula.endswith("O"))
    searcher.add(v.formula.contains("a"))


def _program_numeric_range(searcher, v):
    searcher.add((v.energy > 0.0) & (v.energy <= 1.0))


def _program_mixed_scalar_and_set(searcher, v):
    searcher.add((v.spacegroup == 225) & v.symbols.has_only("Na", "Cl"))


PARITY_PROGRAMS = [
    _program_has_any,
    _program_has_only,
    _program_not_has_any,
    _program_string_ops,
    _program_numeric_range,
    _program_mixed_scalar_and_set,
    _program_not_has_only,
    _program_not_has_all,
    _program_not_inside_and,
    _program_not_over_mixed_and,
    _program_not_over_mixed_or,
    _program_double_not,
]


def test_parity_with_in_memory_store(store):
    pytest.importorskip("httk.serve.optimade")
    from httk.serve.optimade.backend.memory_store import InMemoryStore

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

    for program in PARITY_PROGRAMS:
        memory_searcher = memory_store.searcher()
        memory_variable = memory_searcher.variable("recs")
        memory_searcher.output(memory_variable, "rec")
        program(memory_searcher, memory_variable)
        memory_ids = {item[0][0]["formula"] for item in memory_searcher}

        sql_searcher = store.searcher()
        sql_variable = sql_searcher.variable(Rec)
        sql_searcher.output(sql_variable, "rec")
        program(sql_searcher, sql_variable)
        sql_ids = {item[0][0].formula for item in sql_searcher}

        assert sql_ids == memory_ids, program.__name__
        assert sql_searcher.count() == memory_searcher.count(), program.__name__


def test_literal_string_matching_parity_with_in_memory_store(store):
    pytest.importorskip("httk.serve.optimade")
    from httk.serve.optimade.backend.memory_store import InMemoryStore

    memory_store = InMemoryStore({"labels": [{"text": label.text, "note": label.note} for label in LABELS]})

    for build, expected in LITERAL_MATCH_CASES:
        memory_searcher = memory_store.searcher()
        memory_variable = memory_searcher.variable("labels")
        memory_searcher.output(memory_variable, "label")
        memory_searcher.add(build(memory_variable))
        memory_texts = {item[0][0]["text"] for item in memory_searcher}

        sql_searcher, sql_variable = label_searcher(store)
        sql_searcher.add(build(sql_variable))
        sql_texts = texts(sql_searcher)

        assert sql_texts == memory_texts == expected
        assert sql_searcher.count() == memory_searcher.count()


def test_constant_expression_parity_with_in_memory_store(store):
    pytest.importorskip("httk.serve.optimade")
    from httk.serve.optimade.backend.memory_store import InMemoryStore

    memory_store = InMemoryStore({"labels": [{"text": label.text, "note": label.note} for label in LABELS]})
    for build, expected in [
        (lambda v: v.always_true(), ALL_LABELS),
        (lambda v: v.always_false(), set()),
    ]:
        memory_searcher = memory_store.searcher()
        memory_variable = memory_searcher.variable("labels")
        memory_searcher.output(memory_variable, "label")
        memory_searcher.add(build(memory_variable))
        memory_texts = {item[0][0]["text"] for item in memory_searcher}

        sql_searcher, sql_variable = label_searcher(store)
        sql_searcher.add(build(sql_variable))
        assert texts(sql_searcher) == memory_texts == expected


def test_search_result_names_parity_with_in_memory_store(store):
    # The reference store used to swallow output() entirely, so `values, names =
    # result` raised there while working over SQL. Both now agree on the shape.
    pytest.importorskip("httk.serve.optimade")
    from httk.serve.optimade.backend.memory_store import InMemoryStore

    memory_store = InMemoryStore({"labels": [{"text": label.text, "note": label.note} for label in LABELS]})
    memory_searcher = memory_store.searcher()
    memory_variable = memory_searcher.variable("labels")
    memory_searcher.output(memory_variable, "label")
    memory_searcher.output(memory_variable.text, "text")

    sql_searcher = store.searcher()
    sql_variable = sql_searcher.variable(Label)
    sql_searcher.output(sql_variable, "label")
    sql_searcher.output(sql_variable.text, "text")

    memory_results = list(memory_searcher)
    sql_results = list(sql_searcher)
    assert {result.names for result in memory_results} == {("label", "text")}
    assert {result.names for result in sql_results} == {("label", "text")}
    assert {result[0][1] for result in memory_results} == {result[0][1] for result in sql_results} == ALL_LABELS


def test_object_outputs_survive_reconstruction_on_every_row(store):
    """Every match is yielded even when each object output needs a real fetch.

    Reconstructing an object output issues a nested query on the searcher's own
    connection. A DuckDB connection carries only one active result set, so
    streaming the outer cursor across that nested query truncates the match set
    after the first row. The shared fixture cannot catch it — ``RECORDS`` keeps
    its instances alive, so every reconstruction hits the store's identity cache
    and never queries — hence this test saves throwaway rows and drops all
    references to them before searching.
    """
    for index in range(4):
        store.save(Rec(f"Throwaway{index}", 1, Fraction(index), ["Zz"]))
    gc.collect()  # drop the identity-cache entries, forcing real fetches below

    searcher, variable = rec_searcher(store)
    searcher.add(variable.formula.startswith("Throwaway"))
    matched = formulas(searcher)
    assert matched == {f"Throwaway{index}" for index in range(4)}
    assert searcher.count() == 4


# ------------------------------------------------- child-field comparison set semantics


def test_negated_child_comparison_means_no_row_matches(store):
    """``~(child == x)`` is "no child value is x", not "some child value is not x"."""
    searcher, v = rec_searcher(store)
    searcher.add(~(v.symbols == "O"))
    # Records containing O must be excluded even though they hold other symbols
    # too; the childless record matches (it has no symbol equal to O).
    assert formulas(searcher) == {"NaCl", "X"}

    # ... which is exactly what the equivalent set operation yields.
    reference, rv = rec_searcher(store)
    reference.add(~rv.symbols.has_any("O"))
    assert formulas(reference) == {"NaCl", "X"}


def test_plain_child_comparison_keeps_existential_meaning(store):
    """Un-negated ``child == x`` still means "some child value is x", in WHERE alone."""
    searcher, v = rec_searcher(store)
    expression = v.symbols == "O"
    assert expression.post is False  # no grouped mode forced for a plain filter
    searcher.add(expression)
    assert formulas(searcher) == {"CaTiO3", "MgO", "CaO", "SrCaTiO"}


def test_negated_child_string_predicate_means_no_row_matches(store):
    searcher, v = rec_searcher(store)
    searcher.add(~v.symbols.contains("O"))
    assert formulas(searcher) == {"NaCl", "X"}


def test_child_comparison_composed_with_for_all_reaches_having(store):
    """A plain child comparison OR-ed with a for-all form is aggregated in HAVING.

    Composition routes the whole expression into HAVING position; an
    unaggregated child column there is a hard binder error on DuckDB and an
    arbitrary-row guess on SQLite, so the child comparison must render as an
    aggregate.
    """
    searcher, v = rec_searcher(store)
    searcher.add(v.symbols.has_only("Ca", "O") | (v.symbols == "Na"))
    # has_only({Ca,O}): CaO and the empty-symbol record; == "Na": NaCl.
    assert formulas(searcher) == {"CaO", "X", "NaCl"}
    assert searcher.count() == 3


def test_is_in_is_set_derived_only_on_child_fields(store):
    """``~root.is_in(...)`` stays row-wise; only the child form needs the aggregate."""
    root_searcher, rv = rec_searcher(store)
    root_expression = ~rv.formula.is_in("NaCl", "MgO")
    assert (root_expression.set_derived, root_expression.post) == (False, False)
    root_searcher.add(root_expression)
    assert root_searcher._grouped is False  # no needless grouping
    assert formulas(root_searcher) == ALL_FORMULAS - {"NaCl", "MgO"}

    child_searcher, cv = rec_searcher(store)
    child_expression = ~cv.symbols.is_in("Ca", "O")
    assert (child_expression.set_derived, child_expression.post) == (True, True)
    child_searcher.add(child_expression)
    # is_in on a child field is the for-all reading, so its negation is
    # "not every symbol is in {Ca,O}".
    assert formulas(child_searcher) == {"CaTiO3", "NaCl", "MgO", "SrCaTiO"}
