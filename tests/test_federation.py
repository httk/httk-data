"""Federated source bindings and phase-two query-plan construction tests."""

from collections.abc import Mapping
from datetime import UTC, datetime
from decimal import Decimal
from fractions import Fraction
from typing import Any

import pytest

from httk.data import FederatedSourceError, FederatedStore, UnsupportedQueryError


class ProbeSearcher:
    def __init__(self, store: "ProbeStore") -> None:
        self._store = store

    def variable(self, target: object) -> object:
        self._store.bound_targets.append(target)
        if target in self._store.unsupported_targets:
            raise UnsupportedQueryError("unsupported test target")
        return object()


class ProbeStore:
    def __init__(self, unsupported_targets: tuple[object, ...] = ()) -> None:
        self.searcher_calls = 0
        self.bound_targets: list[object] = []
        self.unsupported_targets = unsupported_targets
        self.close_calls = 0

    def searcher(self) -> ProbeSearcher:
        self.searcher_calls += 1
        return ProbeSearcher(self)

    def close(self) -> None:
        self.close_calls += 1


@pytest.mark.parametrize(
    "sources",
    ({}, {"one": ProbeStore()}, {"": ProbeStore(), "two": ProbeStore()}, {1: ProbeStore(), "two": ProbeStore()}),
)
def test_invalid_sources_fail_deterministically(sources: Mapping[object, ProbeStore]) -> None:
    with pytest.raises((TypeError, ValueError)):
        FederatedStore(sources)  # type: ignore[arg-type]


def test_sources_are_ordered_frozen_and_do_not_query_during_construction() -> None:
    first = ProbeStore()
    second = ProbeStore()
    sources = {"second": second, "first": first}
    store = FederatedStore(sources)
    sources["later"] = ProbeStore()

    assert store.source_names == ("second", "first")
    assert first.searcher_calls == second.searcher_calls == 0
    with pytest.raises(TypeError):
        store._sources["later"] = ProbeStore()  # type: ignore[index]


def test_targets_are_ordered_frozen_and_retain_exact_objects() -> None:
    store = FederatedStore({"first": ProbeStore(), "second": ProbeStore(), "third": ProbeStore()})
    second_target = object()
    first_target = object()
    targets = {"second": second_target, "first": first_target}

    target = store.target("records", targets)
    targets["third"] = object()

    assert target.name == "records"
    assert tuple(target.targets) == ("first", "second")
    assert target.targets["first"] is first_target
    assert target.targets["second"] is second_target
    with pytest.raises(TypeError):
        target.targets["first"] = object()  # type: ignore[index]
    with pytest.raises((AttributeError, TypeError)):
        target.name = "other"  # type: ignore[misc]


def test_invalid_targets_fail_deterministically() -> None:
    store = FederatedStore({"first": ProbeStore(), "second": ProbeStore()})
    for name, targets in (("", {"first": object()}), ("records", {}), ("records", {"other": object()})):
        with pytest.raises((TypeError, ValueError)):
            store.target(name, targets)


def test_direct_target_binds_the_same_target_to_every_source() -> None:
    first = ProbeStore()
    second = ProbeStore()
    store = FederatedStore({"first": first, "second": second})
    target = object()

    variable = store.searcher().variable(target)

    assert isinstance(variable, object)
    assert first.bound_targets == [target]
    assert second.bound_targets == [target]
    assert first.bound_targets[0] is second.bound_targets[0] is target


def test_explicit_target_binds_only_its_ordered_source_subset() -> None:
    first = ProbeStore()
    second = ProbeStore()
    third = ProbeStore()
    store = FederatedStore({"first": first, "second": second, "third": third})
    second_target = object()
    first_target = object()

    store.searcher().variable(store.target("records", {"second": second_target, "first": first_target}))

    assert first.bound_targets == [first_target]
    assert second.bound_targets == [second_target]
    assert third.bound_targets == []


def test_foreign_target_and_second_root_are_unsupported() -> None:
    first = FederatedStore({"first": ProbeStore(), "second": ProbeStore()})
    second = FederatedStore({"first": ProbeStore(), "second": ProbeStore()})
    foreign = first.target("records", {"first": object()})
    searcher = second.searcher()

    with pytest.raises(UnsupportedQueryError, match="another federation"):
        searcher.variable(foreign)
    searcher.variable(object())
    with pytest.raises(UnsupportedQueryError, match="second root"):
        searcher.variable(object())


def test_source_unsupported_target_binding_keeps_neutral_category_and_context() -> None:
    unsupported = object()
    first = ProbeStore()
    second = ProbeStore((unsupported,))
    store = FederatedStore({"first": first, "second": second})

    with pytest.raises(UnsupportedQueryError, match="second") as excinfo:
        store.searcher().variable(unsupported)

    assert isinstance(excinfo.value, FederatedSourceError)
    assert excinfo.value.source == "second"
    assert excinfo.value.operation == "target binding"


def test_federation_never_propagates_close_to_borrowed_sources() -> None:
    first = ProbeStore()
    second = ProbeStore()
    store = FederatedStore({"first": first, "second": second})

    assert not hasattr(store, "close")
    assert first.close_calls == second.close_calls == 0


# ------------------------------------------------------------------- phase two


class QueryProbeExpression:
    def __init__(self, searcher: "QueryProbeSearcher", tree: tuple[Any, ...]) -> None:
        self.searcher = searcher
        self.tree = tree

    def _other(self, other: object) -> "QueryProbeExpression":
        if not isinstance(other, QueryProbeExpression) or other.searcher is not self.searcher:
            raise UnsupportedQueryError("expression belongs to another child searcher")
        return other

    def __and__(self, other: object) -> "QueryProbeExpression":
        right = self._other(other)
        self.searcher.calls.append(("AND", self.tree, right.tree))
        return QueryProbeExpression(self.searcher, ("AND", self.tree, right.tree))

    def __or__(self, other: object) -> "QueryProbeExpression":
        right = self._other(other)
        self.searcher.calls.append(("OR", self.tree, right.tree))
        return QueryProbeExpression(self.searcher, ("OR", self.tree, right.tree))

    def __invert__(self) -> "QueryProbeExpression":
        self.searcher.calls.append(("NOT", self.tree))
        return QueryProbeExpression(self.searcher, ("NOT", self.tree))


class QueryProbeField:
    def __init__(self, searcher: "QueryProbeSearcher", path: tuple[str, ...]) -> None:
        self.searcher = searcher
        self.path = path

    def _predicate(self, operation: str, *arguments: object) -> QueryProbeExpression:
        if (self.path, operation) in self.searcher.unsupported:
            raise UnsupportedQueryError(f"unsupported {operation} on {self.path!r}")
        self.searcher.calls.append(("predicate", self.path, operation, arguments))
        return QueryProbeExpression(self.searcher, (operation, self.path, arguments))

    def __eq__(self, value: object) -> QueryProbeExpression:  # type: ignore[override]
        return self._predicate("__eq__", value)

    def __ne__(self, value: object) -> QueryProbeExpression:  # type: ignore[override]
        return self._predicate("__ne__", value)

    def __lt__(self, value: object) -> QueryProbeExpression:
        return self._predicate("__lt__", value)

    def __le__(self, value: object) -> QueryProbeExpression:
        return self._predicate("__le__", value)

    def __gt__(self, value: object) -> QueryProbeExpression:
        return self._predicate("__gt__", value)

    def __ge__(self, value: object) -> QueryProbeExpression:
        return self._predicate("__ge__", value)

    def contains(self, value: str) -> QueryProbeExpression:
        return self._predicate("contains", value)

    def startswith(self, value: str) -> QueryProbeExpression:
        return self._predicate("startswith", value)

    def endswith(self, value: str) -> QueryProbeExpression:
        return self._predicate("endswith", value)

    def has(self, value: object) -> QueryProbeExpression:
        return self._predicate("has", value)

    def has_any(self, *values: object) -> QueryProbeExpression:
        return self._predicate("has_any", *values)

    def has_only(self, *values: object) -> QueryProbeExpression:
        return self._predicate("has_only", *values)

    def is_in(self, *values: object) -> QueryProbeExpression:
        return self._predicate("is_in", *values)

    def __getattr__(self, name: str) -> "QueryProbeField":
        if name.startswith("_"):
            raise AttributeError(name)
        return QueryProbeField(self.searcher, (*self.path, name))


class QueryProbeVariable:
    def __init__(self, searcher: "QueryProbeSearcher") -> None:
        self.searcher = searcher

    def always_true(self) -> QueryProbeExpression:
        self.searcher.calls.append(("constant", True))
        return QueryProbeExpression(self.searcher, ("constant", True))

    def always_false(self) -> QueryProbeExpression:
        self.searcher.calls.append(("constant", False))
        return QueryProbeExpression(self.searcher, ("constant", False))

    def __getattr__(self, name: str) -> QueryProbeField:
        if name.startswith("_"):
            raise AttributeError(name)
        return QueryProbeField(self.searcher, (name,))


class QueryProbeSearcher:
    def __init__(self, unsupported: set[tuple[tuple[str, ...], str]]) -> None:
        self.unsupported = unsupported
        self.calls: list[tuple[Any, ...]] = []
        self.outputs: list[tuple[object, str]] = []
        self.added: list[QueryProbeExpression] = []

    def variable(self, target: object) -> QueryProbeVariable:
        self.calls.append(("variable", target))
        return QueryProbeVariable(self)

    def output(self, value: object, name: str) -> None:
        if any(existing_name == name for _existing_value, existing_name in self.outputs):
            raise UnsupportedQueryError(f"duplicate child output name: {name!r}")
        self.outputs.append((value, name))

    def add(self, expression: object) -> None:
        if not isinstance(expression, QueryProbeExpression) or expression.searcher is not self:
            raise UnsupportedQueryError("foreign child expression")
        self.added.append(expression)


class QueryProbeStore:
    def __init__(self, unsupported: set[tuple[tuple[str, ...], str]] | None = None) -> None:
        self.unsupported = unsupported or set()
        self.searchers: list[QueryProbeSearcher] = []
        self.execution_calls = 0

    def searcher(self) -> QueryProbeSearcher:
        searcher = QueryProbeSearcher(self.unsupported)
        self.searchers.append(searcher)
        return searcher


def _query_searcher() -> tuple[Any, QueryProbeStore, QueryProbeStore]:
    first = QueryProbeStore()
    second = QueryProbeStore()
    return FederatedStore({"first": first, "second": second}).searcher(), first, second


@pytest.mark.parametrize(
    ("operation", "value"),
    (
        ("__eq__", Fraction(2, 3)),
        ("__eq__", None),
        ("__ne__", Decimal("1.25")),
        ("__ne__", None),
        ("__lt__", datetime(2026, 7, 30, tzinfo=UTC)),
        ("__le__", Fraction(4, 5)),
        ("__gt__", Fraction(7, 5)),
        ("__ge__", Decimal("9.5")),
    ),
)
def test_literal_comparisons_preserve_exact_identity(operation: str, value: object) -> None:
    searcher, first, second = _query_searcher()
    variable = searcher.variable("records")

    expression = getattr(variable.measurement, operation)(value)

    assert expression._ast.arguments[0] is value
    for store in (first, second):
        predicate = next(call for call in store.searchers[0].calls if call[0] == "predicate")
        assert predicate[2] == operation
        assert predicate[3][0] is value


def test_string_set_membership_and_nested_field_paths_are_validated() -> None:
    searcher, first, second = _query_searcher()
    variable = searcher.variable("records")
    fraction = Fraction(3, 7)
    decimal = Decimal("2.50")

    expressions = (
        variable.name.contains("x%_"),
        variable.name.startswith("pre"),
        variable.name.endswith("post"),
        variable.tags.has(fraction),
        variable.tags.has_any(fraction, decimal),
        variable.tags.has_only(decimal, fraction),
        variable.ref.identifier.is_in(fraction, decimal),
    )

    assert expressions[3]._ast.arguments[0] is fraction
    assert expressions[4]._ast.arguments[0] is fraction
    assert expressions[4]._ast.arguments[1] is decimal
    assert expressions[6]._ast.path == ("ref", "identifier")
    for store in (first, second):
        calls = [call for call in store.searchers[0].calls if call[0] == "predicate"]
        assert [call[2] for call in calls] == [
            "contains",
            "startswith",
            "endswith",
            "has",
            "has_any",
            "has_only",
            "is_in",
        ]
        assert calls[4][3][0] is fraction
        assert calls[4][3][1] is decimal


def test_constants_booleans_and_add_replay_each_child_own_expressions() -> None:
    searcher, first, second = _query_searcher()
    variable = searcher.variable("records")
    expression = (variable.always_true() & (variable.name == "kept")) | ~variable.always_false()

    searcher.add(expression)

    assert type(expression._ast).__name__ == "_Or"
    for store in (first, second):
        child = store.searchers[0]
        assert ("constant", True) in child.calls
        assert ("constant", False) in child.calls
        assert any(call[0] == "AND" for call in child.calls)
        assert any(call[0] == "OR" for call in child.calls)
        assert any(call[0] == "NOT" for call in child.calls)
        assert len(child.added) == 1
        assert child.added[0].searcher is child


def test_capability_mismatch_has_source_context_before_execution() -> None:
    first = QueryProbeStore()
    second = QueryProbeStore({(("name",), "contains")})
    searcher = FederatedStore({"first": first, "second": second}).searcher()
    variable = searcher.variable("records")

    with pytest.raises(UnsupportedQueryError, match="second") as excinfo:
        variable.name.contains("missing")

    assert isinstance(excinfo.value, FederatedSourceError)
    assert excinfo.value.source == "second"
    assert first.execution_calls == second.execution_calls == 0


def test_fields_expressions_and_outputs_cannot_cross_federated_searchers() -> None:
    first_searcher, _first, _second = _query_searcher()
    second_searcher, _third, _fourth = _query_searcher()
    first = first_searcher.variable("records")
    second = second_searcher.variable("records")
    expression = first.name == "first"
    foreign_expression = second.name == "second"

    with pytest.raises(UnsupportedQueryError, match="field-to-field"):
        assert first.name == second.name
    with pytest.raises(UnsupportedQueryError, match="different searchers"):
        expression & foreign_expression
    with pytest.raises(UnsupportedQueryError, match="this searcher only"):
        first_searcher.add(foreign_expression)
    for value in (second, second.name, second_searcher.origin):
        with pytest.raises(UnsupportedQueryError, match="this federated searcher"):
            first_searcher.output(value, "foreign")
        with pytest.raises(UnsupportedQueryError, match="this federated searcher"):
            first_searcher._plan({"foreign": value})


def test_output_and_results_planning_record_scalars_and_origin_without_execution() -> None:
    searcher, first, second = _query_searcher()
    variable = searcher.variable("records")
    searcher.output(variable, "record")
    searcher.output(variable.ref.identifier, "identifier")
    searcher.output(searcher.origin, "origin")

    plan = searcher._plan()

    assert tuple(type(output).__name__ for output in plan.outputs) == (
        "_RecordOutput",
        "_FieldOutput",
        "_OriginOutput",
    )
    assert plan.outputs[1].path == ("ref", "identifier")
    assert tuple(source.source for source in plan.sources) == ("first", "second")
    for store in (first, second):
        names = [name for child in store.searchers for _value, name in child.outputs]
        assert names == ["record", "identifier"]
    with pytest.raises(NotImplementedError, match="phase three"):
        searcher.results(record=variable, identifier=variable.ref.identifier, origin=searcher.origin)


def test_results_projection_validation_is_idempotent_and_replaces_retained_outputs() -> None:
    searcher, first, second = _query_searcher()
    variable = searcher.variable("records")
    searcher.output(variable.name, "retained")

    for _ in range(2):
        with pytest.raises(NotImplementedError, match="phase three"):
            searcher.results(record=variable, origin=searcher.origin)

    for store in (first, second):
        assert all(len({name for _value, name in child.outputs}) == len(child.outputs) for child in store.searchers)


def test_results_planning_requires_at_least_one_output() -> None:
    searcher, _first, _second = _query_searcher()
    searcher.variable("records")

    with pytest.raises(ValueError, match="no outputs"):
        searcher.results()


def test_duplicate_output_names_foreign_results_and_global_sort_are_rejected() -> None:
    searcher, _first, _second = _query_searcher()
    variable = searcher.variable("records")
    searcher.output(variable, "record")

    with pytest.raises(ValueError, match="duplicate output name"):
        searcher.output(variable.name, "record")
    with pytest.raises(UnsupportedQueryError, match="add_sort"):
        searcher.add_sort(variable.name)
    other_searcher, _third, _fourth = _query_searcher()
    other = other_searcher.variable("records")
    with pytest.raises(UnsupportedQueryError, match="this federated searcher"):
        searcher.results(record=other)
