"""Federated source bindings and phase-two query-plan construction tests."""

from collections.abc import Mapping
from datetime import UTC, datetime
from decimal import Decimal
from fractions import Fraction
from typing import Any

import pytest

from httk.store import (
    CountUnavailableError,
    FederatedSourceError,
    FederatedStore,
    FederatedTarget,
    MultipleResultsError,
    NoResultError,
    SearchResult,
    UnsupportedQueryError,
)


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


def test_direct_target_construction_copies_freezes_and_orders_its_mapping() -> None:
    store = FederatedStore({"first": ProbeStore(), "second": ProbeStore(), "third": ProbeStore()})
    first_target = object()
    second_target = object()
    targets = {"second": second_target, "first": first_target}

    target = FederatedTarget("records", targets, store)
    targets["third"] = object()

    assert tuple(target.targets) == ("first", "second")
    assert target.targets["first"] is first_target
    assert target.targets["second"] is second_target
    with pytest.raises(TypeError):
        target.targets["third"] = object()  # type: ignore[index]


@pytest.mark.parametrize(
    ("name", "targets", "owner"),
    (
        ("", {"first": object()}, None),
        ("records", {}, None),
        ("records", {"other": object()}, None),
        ("records", {"first": object()}, object()),
        ("records", (), None),
    ),
)
def test_direct_target_construction_rejects_invalid_invariants(
    name: str,
    targets: object,
    owner: object,
) -> None:
    store = FederatedStore({"first": ProbeStore(), "second": ProbeStore()})
    if owner is None:
        owner = store

    with pytest.raises((TypeError, ValueError)):
        FederatedTarget(name, targets, owner)  # type: ignore[arg-type]


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


def test_child_type_error_during_target_binding_has_source_context_and_cause() -> None:
    failure = TypeError("child rejected target construction")

    class FailingSearcher:
        def variable(self, target: object) -> object:
            raise failure

    class FailingStore:
        def searcher(self) -> FailingSearcher:
            return FailingSearcher()

    store = FederatedStore({"first": ProbeStore(), "second": FailingStore()})

    with pytest.raises(FederatedSourceError, match="second") as excinfo:
        store.searcher().variable(object())

    assert not isinstance(excinfo.value, UnsupportedQueryError)
    assert excinfo.value.source == "second"
    assert excinfo.value.operation == "target binding"
    assert excinfo.value.__cause__ is failure


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


def test_concrete_child_field_is_rejected_without_dynamic_instance_lookup() -> None:
    searcher, _first, _second = _query_searcher()
    variable = searcher.variable("records")
    raw_searcher = QueryProbeSearcher(set())
    raw_field = raw_searcher.variable("raw-records").name

    with pytest.raises(UnsupportedQueryError, match="field-to-field"):
        assert variable.name == raw_field

    assert raw_searcher.calls == [("variable", "raw-records")]


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
    result = searcher.results(record=variable, identifier=variable.ref.identifier, origin=searcher.origin)
    assert result.names == ("record", "identifier", "origin")


def test_results_projection_validation_is_idempotent_and_replaces_retained_outputs() -> None:
    searcher, first, second = _query_searcher()
    variable = searcher.variable("records")
    searcher.output(variable.name, "retained")

    for _ in range(2):
        assert searcher.results(record=variable, origin=searcher.origin).names == ("record", "origin")

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


# ----------------------------------------------------------------- phase three


class ExecutionSearcher(QueryProbeSearcher):
    def __init__(self, store: "ExecutionStore", *, runtime_failure: bool) -> None:
        super().__init__(store.unsupported)
        self._execution_store = store
        self._runtime_failure = runtime_failure
        self.limit: int | None = None
        self.counted = False
        self.executed = False

    def add(self, expression: object) -> None:
        if self._runtime_failure and self._execution_store.failure == "expression replay":
            raise self._execution_store.error
        super().add(expression)

    def output(self, value: object, name: str) -> None:
        if self._runtime_failure and self._execution_store.failure == "output declaration":
            raise self._execution_store.error
        super().output(value, name)

    def set_limit(self, limit: int) -> None:
        if self._runtime_failure and self._execution_store.failure == "limit pushdown":
            raise self._execution_store.error
        self.limit = limit

    def count(self) -> int:
        self.counted = True
        self._execution_store.count_calls += 1
        if self._runtime_failure and self._execution_store.failure == "count unavailable":
            raise self._execution_store.error
        if self._runtime_failure and self._execution_store.failure == "count":
            raise self._execution_store.error
        return sum(
            all(self._matches(expression, row) for expression in self.added) for row in self._execution_store.rows
        )

    @staticmethod
    def _value(row: dict[str, object], path: tuple[str, ...]) -> object:
        value: object = row
        for name in path:
            assert isinstance(value, dict)
            value = value[name]
        return value

    def _matches(self, expression: QueryProbeExpression, row: dict[str, object]) -> bool:
        tree = expression.tree
        if tree[0] == "constant":
            return tree[1]
        if tree[0] == "AND":
            return self._matches(QueryProbeExpression(self, tree[1]), row) and self._matches(
                QueryProbeExpression(self, tree[2]), row
            )
        if tree[0] == "OR":
            return self._matches(QueryProbeExpression(self, tree[1]), row) or self._matches(
                QueryProbeExpression(self, tree[2]), row
            )
        if tree[0] == "NOT":
            return not self._matches(QueryProbeExpression(self, tree[1]), row)
        operation, path, arguments = tree
        value = self._value(row, path)
        if operation == "__eq__":
            return value == arguments[0]
        raise AssertionError(f"test fake does not evaluate {operation}")

    def __iter__(self) -> Any:
        self.executed = True
        self._execution_store.execution_calls += 1
        if self._runtime_failure and self._execution_store.failure == "iteration":
            raise self._execution_store.error
        rows = [
            row
            for row in self._execution_store.rows
            if all(self._matches(expression, row) for expression in self.added)
        ]
        if self.limit is not None:
            rows = rows[: self.limit]
        names = tuple(name for _value, name in self.outputs)

        def results() -> Any:
            for index, row in enumerate(rows):
                if self._runtime_failure and self._execution_store.failure == "iteration-after-prefix" and index == 1:
                    raise self._execution_store.error
                values: list[object] = []
                for output, _name in self.outputs:
                    if isinstance(output, QueryProbeVariable):
                        values.append(row)
                    else:
                        assert isinstance(output, QueryProbeField)
                        values.append(self._value(row, output.path))
                yield SearchResult(tuple(values), names)

        return results()


class ExecutionStore(QueryProbeStore):
    def __init__(
        self,
        rows: list[dict[str, object]],
        *,
        failure: str | None = None,
        error: Exception | None = None,
    ) -> None:
        super().__init__()
        self.rows = rows
        self.failure = failure
        self.error = error or RuntimeError("test child failure")
        self.count_calls = 0

    def searcher(self) -> ExecutionSearcher:
        runtime_failure = len(self.searchers) >= 2
        if runtime_failure and self.failure == "searcher construction":
            raise self.error
        searcher = ExecutionSearcher(self, runtime_failure=runtime_failure)
        self.searchers.append(searcher)
        return searcher


def _execution_searcher(
    first_rows: list[dict[str, object]] | None = None,
    second_rows: list[dict[str, object]] | None = None,
) -> tuple[Any, ExecutionStore, ExecutionStore]:
    first = ExecutionStore(first_rows or [])
    second = ExecutionStore(second_rows or [])
    return FederatedStore({"first": first, "second": second}).searcher(), first, second


def test_execution_is_source_major_preserves_records_and_replays_fresh_filters() -> None:
    first_record = {"id": "same", "value": "first"}
    second_record = {"id": "same", "value": "second"}
    searcher, first, second = _execution_searcher([first_record], [second_record])
    variable = searcher.variable("records")
    searcher.add(variable.id == "same")
    result = searcher.results(record=variable, value=variable.value, origin=searcher.origin)

    rows = list(result)

    assert [(row.origin, row.value) for row in rows] == [("first", "first"), ("second", "second")]
    assert rows[0].record is first_record
    assert rows[1].record is second_record
    for store in (first, second):
        executed = [child for child in store.searchers if child.executed]
        assert len(executed) == 1
        assert executed[0].added[0].searcher is executed[0]
        assert executed[0].calls[0] == ("variable", "records")


def test_global_offset_limit_and_early_stop_are_owned_by_the_coordinator() -> None:
    first_rows = [{"id": str(index), "value": f"first-{index}"} for index in range(2)]
    second_rows = [{"id": str(index), "value": f"second-{index}"} for index in range(2)]
    searcher, first, second = _execution_searcher(first_rows, second_rows)
    variable = searcher.variable("records")
    searcher.add_offset(1)
    searcher.set_limit(2)

    assert [result.values for result in searcher.results(value=variable.value, origin=searcher.origin)] == [
        ("first-1", "first"),
        ("second-0", "second"),
    ]
    assert [child.limit for child in first.searchers if child.executed] == [3]
    assert [child.limit for child in second.searchers if child.executed] == [3]

    stopped, first_stopped, second_stopped = _execution_searcher(first_rows, second_rows)
    stopped_variable = stopped.variable("records")
    stopped.set_limit(1)
    assert [row.value for row in stopped.results(value=stopped_variable.value)] == ["first-0"]
    assert first_stopped.execution_calls == 1
    assert second_stopped.execution_calls == 0


def test_limit_zero_contacts_no_child_and_origin_only_uses_hidden_record_output() -> None:
    searcher, first, second = _execution_searcher([{"id": "1", "value": "a"}], [{"id": "2", "value": "b"}])
    searcher.variable("records")
    searcher.set_limit(0)
    assert [row for row in searcher.results(origin=searcher.origin)] == []
    assert first.execution_calls == second.execution_calls == 0

    unbounded, first, second = _execution_searcher([{"id": "1", "value": "a"}], [{"id": "2", "value": "b"}])
    unbounded.variable("records")
    assert list(unbounded.results(origin=unbounded.origin).scalars()) == ["first", "second"]
    for store in (first, second):
        child = next(child for child in store.searchers if child.executed)
        assert [name for _value, name in child.outputs] == ["__httk_federated_hidden_record__"]


def test_direct_searcher_iteration_yields_search_results() -> None:
    searcher, _first, _second = _execution_searcher([{"id": "1", "value": "a"}], [{"id": "2", "value": "b"}])
    variable = searcher.variable("records")
    searcher.output(variable.value, "value")

    results = list(searcher)

    assert all(isinstance(result, SearchResult) for result in results)
    assert [result.values for result in results] == [("a",), ("b",)]


def test_result_plans_are_frozen_reiterable_and_support_direct_iteration_helpers_and_slices() -> None:
    rows = [{"id": str(index), "value": str(index)} for index in range(3)]
    searcher, _first, _second = _execution_searcher(rows, [{"id": "3", "value": "3"}])
    variable = searcher.variable("records")
    result = searcher.results(value=variable.value)
    searcher.add(variable.value == "not-in-frozen-plan")

    assert [row.value for row in result] == ["0", "1", "2", "3"]
    assert [row.value for row in result] == ["0", "1", "2", "3"]
    assert [row.value for row in result[1:3]] == ["1", "2"]
    assert result.first().value == "0"
    with pytest.raises(MultipleResultsError):
        result.one()
    assert list(result.scalars()) == ["0", "1", "2", "3"]
    assert list(result.column("value")) == ["0", "1", "2", "3"]
    with pytest.raises(NotImplementedError):
        result.cursor()
    assert len(result) == 4
    with pytest.raises(NoResultError):
        result[4:].one()
    fresh = searcher.results(value=variable.value)
    assert [row.values for row in fresh] == []


@pytest.mark.parametrize("value", (True, 1.5, "1"))
def test_limit_and_offset_validate_like_neutral_backends(value: object) -> None:
    searcher, _first, _second = _execution_searcher()
    searcher.variable("records")
    with pytest.raises(TypeError):
        searcher.set_limit(value)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        searcher.add_offset(value)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        searcher.add_offset(-1)
    with pytest.raises(ValueError):
        searcher.results(origin=searcher.origin)[True:]


@pytest.mark.parametrize("step", (True, False, -1, 0, 2, 1.0, "1"))
def test_result_slices_reject_non_unit_integer_steps(step: object) -> None:
    searcher, _first, _second = _execution_searcher()
    searcher.variable("records")
    result = searcher.results(origin=searcher.origin)

    with pytest.raises(ValueError, match="unit step"):
        result[slice(None, None, step)]  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "failure",
    ("searcher construction", "expression replay", "output declaration", "limit pushdown", "iteration"),
)
def test_execution_failures_are_source_attributed_and_keep_unsupported_category(failure: str) -> None:
    error: Exception = (
        UnsupportedQueryError("late unsupported") if failure == "iteration" else RuntimeError("late failure")
    )
    first = ExecutionStore([])
    second = ExecutionStore([], failure=failure, error=error)
    searcher = FederatedStore({"first": first, "second": second}).searcher()
    variable = searcher.variable("records")
    if failure == "expression replay":
        searcher.add(variable.id == "x")
    searcher.set_limit(1)
    with pytest.raises(FederatedSourceError, match="second") as excinfo:
        [row for row in searcher.results(value=variable.value)]
    assert excinfo.value.source == "second"
    assert excinfo.value.operation == failure
    if failure == "iteration":
        assert isinstance(excinfo.value, UnsupportedQueryError)


def test_later_iteration_failure_is_not_silent_after_a_yielded_prefix() -> None:
    first = ExecutionStore([{"id": "1", "value": "first"}])
    second = ExecutionStore(
        [{"id": "2", "value": "second"}, {"id": "3", "value": "third"}],
        failure="iteration-after-prefix",
    )
    searcher = FederatedStore({"first": first, "second": second}).searcher()
    variable = searcher.variable("records")
    iterator = iter(searcher.results(value=variable.value))
    assert next(iterator).value == "first"
    assert next(iterator).value == "second"
    with pytest.raises(FederatedSourceError, match="second"):
        next(iterator)


# ------------------------------------------------------------------ phase four


def test_exact_counts_sum_filters_cache_and_are_unpaged() -> None:
    searcher, first, second = _execution_searcher(
        [{"id": "1", "value": "kept"}, {"id": "2", "value": "discarded"}],
        [{"id": "3", "value": "kept"}, {"id": "4", "value": "kept"}],
    )
    variable = searcher.variable("records")
    searcher.add(variable.value == "kept")
    searcher.add_offset(1)
    searcher.set_limit(1)
    result = searcher.results(value=variable.value)

    assert searcher.count() == 3
    assert len(result) == 1
    assert len(result[1:]) == 0
    assert searcher.count() == 3
    assert first.count_calls == second.count_calls == 1
    counted = [child for store in (first, second) for child in store.searchers if child.counted]
    assert len(counted) == 2
    assert all(not child.outputs for child in counted)


def test_count_cache_detaches_after_mutation_without_changing_frozen_result_plan() -> None:
    searcher, first, second = _execution_searcher(
        [{"id": "1", "value": "kept"}, {"id": "2", "value": "discarded"}],
        [{"id": "3", "value": "kept"}],
    )
    variable = searcher.variable("records")
    frozen = searcher.results(value=variable.value)

    assert len(frozen) == 3
    searcher.add(variable.value == "kept")
    assert len(frozen) == 3
    assert searcher.count() == 2
    assert [row.value for row in searcher.results(value=variable.value)] == ["kept", "kept"]
    assert first.count_calls == second.count_calls == 2


def test_count_unavailable_has_source_context_and_preserves_cause_without_partial_cache() -> None:
    unavailable = CountUnavailableError("child cannot count exactly")
    first = ExecutionStore([{"id": "1", "value": "first"}])
    second = ExecutionStore([], failure="count unavailable", error=unavailable)
    searcher = FederatedStore({"first": first, "second": second}).searcher()
    variable = searcher.variable("records")
    result = searcher.results(value=variable.value)

    for _ in range(2):
        with pytest.raises(CountUnavailableError, match="second") as excinfo:
            len(result)
        assert isinstance(excinfo.value, FederatedSourceError)
        assert excinfo.value.source == "second"
        assert excinfo.value.operation == "count"
        assert excinfo.value.__cause__ is unavailable
    assert first.count_calls == second.count_calls == 2


def test_list_ignores_unavailable_length_hint_and_streams_until_early_limit() -> None:
    unavailable = CountUnavailableError("later child cannot count exactly")
    first = ExecutionStore([{"id": "1", "value": "first"}])
    second = ExecutionStore(
        [{"id": "2", "value": "second"}],
        failure="count unavailable",
        error=unavailable,
    )
    searcher = FederatedStore({"first": first, "second": second}).searcher()
    variable = searcher.variable("records")
    searcher.set_limit(1)
    result = searcher.results(value=variable.value)

    assert [row.value for row in list(result)] == ["first"]
    assert first.count_calls == second.count_calls == 1
    assert first.execution_calls == 1
    assert second.execution_calls == 0

    with pytest.raises(CountUnavailableError, match="second") as excinfo:
        len(result)
    assert isinstance(excinfo.value, FederatedSourceError)
    assert isinstance(excinfo.value, TypeError)
    assert excinfo.value.__cause__ is unavailable


@pytest.mark.parametrize("error", (RuntimeError("child count failed"), UnsupportedQueryError("late unsupported count")))
def test_count_failures_are_source_attributed_and_keep_neutral_categories(error: Exception) -> None:
    first = ExecutionStore([])
    second = ExecutionStore([], failure="count", error=error)
    searcher = FederatedStore({"first": first, "second": second}).searcher()
    variable = searcher.variable("records")
    searcher.results(value=variable.value)

    with pytest.raises(FederatedSourceError, match="second") as excinfo:
        searcher.count()
    assert excinfo.value.source == "second"
    assert excinfo.value.operation == "count"
    assert excinfo.value.__cause__ is error
    assert isinstance(excinfo.value, UnsupportedQueryError) is isinstance(error, UnsupportedQueryError)


def test_result_len_applies_global_offset_limit_and_zero_limit_skips_counts() -> None:
    rows = [{"id": str(index), "value": str(index)} for index in range(2)]
    searcher, first, second = _execution_searcher(rows, rows)
    variable = searcher.variable("records")
    searcher.add_offset(1)
    searcher.set_limit(2)
    result = searcher.results(value=variable.value)

    assert len(result) == 2
    assert len(result[1:]) == 1
    assert len(result[5:]) == 0
    assert first.count_calls == second.count_calls == 1

    zero, first_zero, second_zero = _execution_searcher(rows, rows)
    zero_variable = zero.variable("records")
    zero.set_limit(0)
    assert len(zero.results(value=zero_variable.value)) == 0
    assert first_zero.count_calls == second_zero.count_calls == 0
