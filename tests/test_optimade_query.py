"""Tests for the generic OPTIMADE filter translation (httk.data.optimade_query)."""

from typing import Any, Iterator

import pytest
from httk.core import parse_optimade_filter

from httk.data.optimade_query import (
    FilterTranslationError,
    filter_searcher,
    format_value,
    number_handler,
    relationship_id_handler,
    simple_property_handlers,
    translate_filter_ast,
)

# ---------------------------------------------------------------------- a minimal fake store


class FakeExpression:
    def __init__(self, tree: tuple[Any, ...]) -> None:
        self.tree = tree

    def __and__(self, other: "FakeExpression") -> "FakeExpression":
        return FakeExpression(("AND", self.tree, other.tree))

    def __or__(self, other: "FakeExpression") -> "FakeExpression":
        return FakeExpression(("OR", self.tree, other.tree))

    def __invert__(self) -> "FakeExpression":
        return FakeExpression(("NOT", self.tree))

    def __repr__(self) -> str:
        return f"FakeExpression({self.tree!r})"


class FakeColumn:
    def __init__(self, name: str) -> None:
        self.name = name

    def _binary(self, op: str, other: Any) -> FakeExpression:
        if isinstance(other, FakeColumn):
            other = ("column", other.name)
        return FakeExpression((op, ("column", self.name), other))

    def __eq__(self, other: object) -> FakeExpression:  # type: ignore[override]
        return self._binary("eq", other)

    def __ne__(self, other: object) -> FakeExpression:  # type: ignore[override]
        return self._binary("ne", other)

    def __lt__(self, other: Any) -> FakeExpression:
        return self._binary("lt", other)

    def __le__(self, other: Any) -> FakeExpression:
        return self._binary("le", other)

    def __gt__(self, other: Any) -> FakeExpression:
        return self._binary("gt", other)

    def __ge__(self, other: Any) -> FakeExpression:
        return self._binary("ge", other)

    def __hash__(self) -> int:
        return hash(self.name)

    def like(self, pattern: str) -> FakeExpression:
        return self._binary("like", pattern)

    def has_any(self, *values: Any) -> FakeExpression:
        return FakeExpression(("has_any", ("column", self.name), values))

    def has_inv_any(self, *values: Any) -> FakeExpression:
        return FakeExpression(("has_inv_any", ("column", self.name), values))

    def has_only(self, *values: Any) -> FakeExpression:
        return FakeExpression(("has_only", ("column", self.name), values))

    def has_inv_only(self, *values: Any) -> FakeExpression:
        return FakeExpression(("has_inv_only", ("column", self.name), values))


class FakeVariable:
    def __init__(self, target: Any) -> None:
        self.target = target

    def __getattr__(self, name: str) -> FakeColumn:
        return FakeColumn(name)


class FakeSearcher:
    def __init__(self, rows: list[Any]) -> None:
        self.rows = rows
        self.offset = 0
        self.variables: list[FakeVariable] = []
        self.outputs: list[tuple[FakeVariable, str]] = []
        self.expressions: list[FakeExpression] = []
        self.all_expressions: list[FakeExpression] = []

    def variable(self, target: Any) -> FakeVariable:
        variable = FakeVariable(target)
        self.variables.append(variable)
        return variable

    def output(self, variable: FakeVariable, name: str) -> None:
        self.outputs.append((variable, name))

    def add(self, expression: FakeExpression) -> None:
        self.expressions.append(expression)

    def add_all(self, expression: FakeExpression) -> None:
        self.all_expressions.append(expression)

    def count(self) -> int:
        return len(self.rows)

    def set_limit(self, limit: int) -> None:
        pass

    def add_offset(self, offset: int) -> None:
        self.offset += offset

    def add_sort(self, column: FakeColumn, descending: bool) -> None:
        pass

    def __iter__(self) -> Iterator[Any]:
        return iter([((row,),) for row in self.rows])


class FakeStore:
    def __init__(self, rows: list[Any] | None = None) -> None:
        self.rows = rows if rows is not None else []
        self.searchers: list[FakeSearcher] = []

    def searcher(self) -> FakeSearcher:
        searcher = FakeSearcher(self.rows)
        self.searchers.append(searcher)
        return searcher


# ---------------------------------------------------------------------- fixtures

FULLTYPES = {
    "id": "string",
    "nelements": "integer",
    "nsites": "integer",
    "chemical_formula_descriptive": "string",
    "elements": "list of string",
    "blob": "dict",
}

COLUMNS = {
    "nelements": "number_of_elements",
    "nsites": "number_of_sites",
    "chemical_formula_descriptive": "formula",
    "elements": "formula_symbols",
    "blob": "blob",
}

TRUE_TREE = ("eq", ("column", "hexhash"), ("column", "hexhash"))
FALSE_TREE = ("ne", ("column", "hexhash"), ("column", "hexhash"))


def make_handlers() -> dict[str, Any]:
    handlers = dict(simple_property_handlers("structures", COLUMNS, FULLTYPES))
    elements = dict(handlers["elements"])
    elements["length"] = lambda entry, op, value, sv: number_handler("number_of_elements", op, value, sv)
    handlers["elements"] = elements
    handlers["references.id"] = relationship_id_handler("refs_column")
    return handlers


def translate(filter_string, *, relationship_targets=(), resolver=None, handlers=None):
    search_variable = FakeVariable("structures")
    return translate_filter_ast(
        parse_optimade_filter(filter_string) if isinstance(filter_string, str) else filter_string,
        search_variable,
        "structures",
        FULLTYPES,
        handlers if handlers is not None else make_handlers(),
        ("_httk_",),
        False,
        relationship_targets=relationship_targets,
        related_property_resolver=resolver,
    )


class StubResolver:
    """Records (related_type, sub_ast) calls; returns preset ids (or per-call ids)."""

    def __init__(self, ids=("references-1", "references-2"), per_call=None):
        self.ids = ids
        self.per_call = list(per_call) if per_call is not None else None
        self.calls = []

    def __call__(self, related_type, sub_ast):
        self.calls.append((related_type, sub_ast))
        if self.per_call is not None:
            return self.per_call.pop(0)
        return self.ids


# ---------------------------------------------------------------------- plain translation


def test_number_comparison():
    expr, needs_post = translate("nelements=3")
    assert expr.tree == ("eq", ("column", "number_of_elements"), 3)
    assert needs_post is False


def test_inverted_constant_first_comparison():
    expr, _ = translate("3 < nelements")
    assert expr.tree == ("gt", ("column", "number_of_elements"), 3)


def test_string_comparison():
    expr, _ = translate('chemical_formula_descriptive = "GaTi"')
    assert expr.tree == ("eq", ("column", "formula"), "GaTi")


def test_id_maps_to_dunder_id():
    expr, _ = translate('id = "abc"')
    assert expr.tree == ("eq", ("column", "__id"), "abc")


def test_stringmatching_contains_uses_like_with_escapes():
    expr, _ = translate('chemical_formula_descriptive CONTAINS "Ga_x"')
    assert expr.tree == ("like", ("column", "formula"), r"%Ga\_x%")


def test_stringmatching_starts_and_ends():
    starts, _ = translate('chemical_formula_descriptive STARTS WITH "Ga"')
    assert starts.tree == ("like", ("column", "formula"), "Ga%")
    ends, _ = translate('chemical_formula_descriptive ENDS WITH "Ga"')
    assert ends.tree == ("like", ("column", "formula"), "%Ga")


def test_has_all_becomes_conjunction_of_has_any():
    expr, needs_post = translate('elements HAS ALL "Ga","Ti"')
    assert expr.tree == (
        "AND",
        ("has_any", ("column", "formula_symbols"), ("Ga",)),
        ("has_any", ("column", "formula_symbols"), ("Ti",)),
    )
    assert needs_post is False


def test_has_any():
    expr, needs_post = translate('elements HAS ANY "Ga","Ti"')
    assert expr.tree == ("has_any", ("column", "formula_symbols"), ("Ga", "Ti"))
    assert needs_post is False


def test_has_only_needs_post_filter():
    expr, needs_post = translate('elements HAS ONLY "Ga","Ti"')
    assert expr.tree == ("has_only", ("column", "formula_symbols"), ("Ga", "Ti"))
    assert needs_post is True


def test_not_has_all_uses_inverted_set_ops():
    expr, needs_post = translate('NOT elements HAS ALL "Ga"')
    assert expr.tree == ("NOT", ("has_inv_any", ("column", "formula_symbols"), ("Ga",)))
    assert needs_post is True


def test_length():
    expr, _ = translate("elements LENGTH 2")
    assert expr.tree == ("eq", ("column", "number_of_elements"), 2)


def test_is_known_on_always_known_property_is_true():
    expr, _ = translate("nelements IS KNOWN")
    assert expr.tree == TRUE_TREE


def test_is_unknown_on_always_known_property_is_false():
    expr, _ = translate("nelements IS UNKNOWN")
    assert expr.tree == FALSE_TREE


def test_and_or_nesting():
    expr, _ = translate("nelements=1 AND (nelements=2 OR nelements=3)")
    assert expr.tree == (
        "AND",
        ("eq", ("column", "number_of_elements"), 1),
        (
            "OR",
            ("eq", ("column", "number_of_elements"), 2),
            ("eq", ("column", "number_of_elements"), 3),
        ),
    )


def test_not_comparison():
    expr, _ = translate("NOT nelements=3")
    assert expr.tree == ("NOT", ("eq", ("column", "number_of_elements"), 3))


# ---------------------------------------------------------------------- error categories


def test_unknown_nonprefixed_property_matches_nothing():
    expr, needs_post = translate("bananas = 3")
    assert expr.tree == FALSE_TREE
    assert needs_post is False


def test_unknown_prefixed_property_raises_unrecognized_property():
    with pytest.raises(FilterTranslationError) as excinfo:
        translate("_httk_bananas = 3")
    assert excinfo.value.category == "unrecognized-property"


def test_type_mismatch_category():
    with pytest.raises(FilterTranslationError) as excinfo:
        translate('nelements = "three"')
    assert excinfo.value.category == "type-mismatch"


def test_format_value_scalar_for_list_is_type_mismatch():
    with pytest.raises(FilterTranslationError) as excinfo:
        format_value("list of string", ("String", "Si"))
    assert excinfo.value.category == "type-mismatch"


def test_dict_property_not_implemented():
    with pytest.raises(FilterTranslationError) as excinfo:
        translate('blob = "x"')
    assert excinfo.value.category == "not-implemented"


def test_identifier_vs_identifier_not_implemented():
    with pytest.raises(FilterTranslationError) as excinfo:
        translate("nelements = nsites")
    assert excinfo.value.category == "not-implemented"


def test_has_all_with_operator_not_implemented():
    with pytest.raises(FilterTranslationError) as excinfo:
        translate('elements HAS ALL > "Ga","Ti"')
    assert excinfo.value.category == "not-implemented"


def test_has_with_operator_is_internal():
    with pytest.raises(FilterTranslationError) as excinfo:
        translate("elements HAS < 3")
    assert excinfo.value.category == "internal"


def test_boolean_with_ordering_operator_not_implemented():
    with pytest.raises(FilterTranslationError) as excinfo:
        translate("bananas > TRUE")
    assert excinfo.value.category == "not-implemented"


def test_property_without_handler_not_implemented():
    handlers = make_handlers()
    del handlers["nsites"]
    with pytest.raises(FilterTranslationError) as excinfo:
        translate("nsites = 3", handlers=handlers)
    assert excinfo.value.category == "not-implemented"


# ---------------------------------------------------------------------- relationship .id fast path


def test_relationship_id_has_translates_through_handler():
    expr, needs_post = translate(
        'references.id HAS "references-1"',
        relationship_targets=("references",),
    )
    assert expr.tree == ("has_any", ("column", "refs_column"), ("references-1",))
    assert needs_post is False


def test_not_relationship_id_has_uses_inverse_set_op():
    expr, needs_post = translate(
        'NOT references.id HAS "references-1"',
        relationship_targets=("references",),
    )
    assert expr.tree == ("NOT", ("has_inv_any", ("column", "refs_column"), ("references-1",)))
    assert needs_post is True


def test_relationship_id_has_without_handler_not_implemented():
    handlers = make_handlers()
    del handlers["references.id"]
    with pytest.raises(FilterTranslationError) as excinfo:
        translate('references.id HAS "references-1"', relationship_targets=("references",), handlers=handlers)
    assert excinfo.value.category == "not-implemented"


def test_relationship_id_handler_directly():
    table = relationship_id_handler("refs")
    variable = FakeVariable("t")
    expr, needs_post = table["HAS"]("references.id", ("=", "="), ["a", "b"], variable, "HAS_ANY", False)
    assert expr.tree == ("has_any", ("column", "refs"), ("a", "b"))
    assert needs_post is False
    expr, needs_post = table["HAS"]("references.id", ("=",), ["a"], variable, "HAS_ANY", True)
    assert expr.tree == ("has_inv_any", ("column", "refs"), ("a",))
    assert needs_post is True


# ---------------------------------------------------------------------- the two-phase semi-join


def test_resolver_receives_stripped_comparison_sub_ast():
    resolver = StubResolver()
    expr, needs_post = translate(
        "references.year >= 2000", relationship_targets=("references",), resolver=resolver
    )
    assert resolver.calls == [("references", (">=", ("Identifier", "year"), ("Number", "2000")))]
    assert expr.tree == ("has_any", ("column", "refs_column"), ("references-1", "references-2"))
    assert needs_post is False


def test_resolver_constant_first_comparison_is_swapped_before_stripping():
    # The core parser flattens dotted identifiers on the constant-first side
    # (`2000 <= references.year` parses to a plain 'references' identifier), so
    # exercise the swap path on a hand-built node.
    resolver = StubResolver()
    node = ("<=", ("Number", "2000"), ("Identifier", "references", "year"))
    translate(node, relationship_targets=("references",), resolver=resolver)
    assert resolver.calls == [("references", (">=", ("Identifier", "year"), ("Number", "2000")))]


def test_resolver_receives_stripped_id_comparison():
    resolver = StubResolver(ids=("references-2",))
    expr, _ = translate('references.id != "references-1"', relationship_targets=("references",), resolver=resolver)
    assert resolver.calls == [("references", ("!=", ("Identifier", "id"), ("String", "references-1")))]
    assert expr.tree == ("has_any", ("column", "refs_column"), ("references-2",))


def test_resolver_receives_stripped_stringmatching_sub_ast():
    for filter_string, node in [
        ('references.doi CONTAINS "10.1"', "CONTAINS"),
        ('references.doi STARTS WITH "10."', "STARTS"),
        ('references.doi ENDS WITH "/a"', "ENDS"),
    ]:
        resolver = StubResolver()
        translate(filter_string, relationship_targets=("references",), resolver=resolver)
        expected_value = {"CONTAINS": "10.1", "STARTS": "10.", "ENDS": "/a"}[node]
        assert resolver.calls == [("references", (node, ("Identifier", "doi"), ("String", expected_value)))]


def test_resolver_receives_stripped_known_unknown_sub_ast():
    resolver = StubResolver()
    translate("references.doi IS KNOWN", relationship_targets=("references",), resolver=resolver)
    assert resolver.calls == [("references", ("IS_KNOWN", ("Identifier", "doi")))]
    resolver = StubResolver()
    translate("references.doi IS UNKNOWN", relationship_targets=("references",), resolver=resolver)
    assert resolver.calls == [("references", ("IS_UNKNOWN", ("Identifier", "doi")))]


def test_resolver_receives_stripped_has_sub_ast():
    resolver = StubResolver()
    translate('references.authors HAS "who"', relationship_targets=("references",), resolver=resolver)
    assert resolver.calls == [
        ("references", ("HAS_ALL", ("=",), ("Identifier", "authors"), (("String", "who"),)))
    ]


def test_resolver_empty_ids_translate_to_false_without_post_filter():
    resolver = StubResolver(ids=())
    expr, needs_post = translate(
        'references.doi CONTAINS "nomatch"', relationship_targets=("references",), resolver=resolver
    )
    assert expr.tree == FALSE_TREE
    assert needs_post is False


def test_not_composes_through_the_semi_join_rewrite():
    # The resolver sees the sub-filter WITHOUT the surrounding NOT; inversion
    # applies to the id-set membership (the rewritten `<type>.id HAS ANY`).
    resolver = StubResolver(ids=("references-1",))
    expr, needs_post = translate(
        'NOT references.doi CONTAINS "10.1"', relationship_targets=("references",), resolver=resolver
    )
    assert resolver.calls == [("references", ("CONTAINS", ("Identifier", "doi"), ("String", "10.1")))]
    assert expr.tree == ("NOT", ("has_inv_any", ("column", "refs_column"), ("references-1",)))
    assert needs_post is True


def test_not_of_empty_resolver_result_is_not_of_false():
    resolver = StubResolver(ids=())
    expr, needs_post = translate(
        'NOT references.doi CONTAINS "nomatch"', relationship_targets=("references",), resolver=resolver
    )
    assert expr.tree == ("NOT", FALSE_TREE)
    assert needs_post is False


def test_per_node_independence():
    # Locked semantic: each dotted node resolves independently — some related
    # entry matches the doi condition AND some (possibly different) related
    # entry matches the year condition.
    resolver = StubResolver(per_call=[("references-1",), ("references-2",)])
    expr, _ = translate(
        'references.doi CONTAINS "10.1" AND references.year >= 2000',
        relationship_targets=("references",),
        resolver=resolver,
    )
    assert resolver.calls == [
        ("references", ("CONTAINS", ("Identifier", "doi"), ("String", "10.1"))),
        ("references", (">=", ("Identifier", "year"), ("Number", "2000"))),
    ]
    assert expr.tree == (
        "AND",
        ("has_any", ("column", "refs_column"), ("references-1",)),
        ("has_any", ("column", "refs_column"), ("references-2",)),
    )


def test_dotted_without_resolver_not_implemented():
    for filter_string in [
        'references.doi CONTAINS "10.1"',
        "references.year >= 2000",
        'references.id = "references-1"',
        "references.doi IS KNOWN",
        'references.authors HAS "who"',
    ]:
        with pytest.raises(FilterTranslationError) as excinfo:
            translate(filter_string, relationship_targets=("references",))
        assert excinfo.value.category == "not-implemented"


def test_nested_dotted_path_not_implemented():
    resolver = StubResolver()
    with pytest.raises(FilterTranslationError) as excinfo:
        translate("references.inner.x = 1", relationship_targets=("references",), resolver=resolver)
    assert excinfo.value.category == "not-implemented"
    assert resolver.calls == []


def test_dotted_length_not_implemented():
    resolver = StubResolver()
    with pytest.raises(FilterTranslationError) as excinfo:
        translate("references.authors LENGTH 2", relationship_targets=("references",), resolver=resolver)
    assert excinfo.value.category == "not-implemented"
    assert resolver.calls == []


def test_undeclared_dotted_prefix_is_an_unknown_property():
    # A dotted identifier whose first part is not a relationship target is an
    # ordinary (unknown, unprefixed) property: it matches nothing.
    expr, _ = translate('bananas.doi CONTAINS "10.1"')
    assert expr.tree == FALSE_TREE


# ---------------------------------------------------------------------- filter_searcher sugar


def test_filter_searcher_end_to_end_with_filter_string():
    rows = ["row-1", "row-2"]
    store = FakeStore(rows)
    searcher = filter_searcher(
        store,
        "structure-table",
        'nelements = 3 AND elements HAS ONLY "Ga","Ti"',
        entry_type="structures",
        property_fulltypes=FULLTYPES,
        columns=COLUMNS,
        recognized_prefixes=("_httk_",),
    )
    assert isinstance(searcher, FakeSearcher)
    assert searcher.variables[0].target == "structure-table"
    assert searcher.outputs[0][1] == "structures"
    expected = (
        "AND",
        ("eq", ("column", "number_of_elements"), 3),
        ("has_only", ("column", "formula_symbols"), ("Ga", "Ti")),
    )
    assert [expression.tree for expression in searcher.expressions] == [expected]
    # HAS ONLY needs the post-filter position as well.
    assert [expression.tree for expression in searcher.all_expressions] == [expected]
    assert [item[0][0] for item in searcher] == rows


def test_filter_searcher_accepts_parsed_ast_and_default_columns():
    store = FakeStore()
    searcher = filter_searcher(
        store,
        "structure-table",
        parse_optimade_filter('nelements = 3'),
        entry_type="structures",
        property_fulltypes={"nelements": "integer"},
    )
    # Default columns: identity map over property_fulltypes.
    assert [expression.tree for expression in searcher.expressions] == [
        ("eq", ("column", "nelements"), 3)
    ]
    assert searcher.all_expressions == []
