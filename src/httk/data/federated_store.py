"""Frozen query planning and sequential execution for a federated store.

Federated expressions are deliberately represented by a private, backend-neutral
AST. Child searcher expressions are used only while validating that each
participating source accepts an operation; every execution replays that AST
into fresh child searchers.
"""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import cast

from .query import (
    CountUnavailableError,
    MultipleResultsError,
    NoResultError,
    ResultRow,
    Searcher,
    SearchExpression,
    SearchField,
    SearchResult,
    SearchVariable,
    Store,
    UnsupportedQueryError,
)

__all__ = [
    "FederatedExpression",
    "FederatedField",
    "FederatedResultColumn",
    "FederatedResultSet",
    "FederatedSearcher",
    "FederatedSourceError",
    "FederatedStore",
    "FederatedStoreError",
    "FederatedTarget",
    "FederatedVariable",
]


class FederatedStoreError(RuntimeError):
    """A federation-level store operation failed."""


class FederatedSourceError(FederatedStoreError):
    """A named source rejected or failed a federated operation."""

    def __init__(self, source: str, operation: str) -> None:
        self.source = source
        self.operation = operation
        super().__init__(f"federated source {source!r} failed during {operation}")


class _FederatedUnsupportedQueryError(FederatedSourceError, UnsupportedQueryError):
    """Add source context without losing the neutral unsupported category."""


class _FederatedCountUnavailableError(FederatedSourceError, CountUnavailableError, TypeError):
    """Retain count/source categories while allowing optional length hints to fail."""


@dataclass(frozen=True, slots=True)
class _Constant:
    value: bool


@dataclass(frozen=True, slots=True)
class _Predicate:
    path: tuple[str, ...]
    operation: str
    arguments: tuple[object, ...]


@dataclass(frozen=True, slots=True)
class _And:
    left: "_FederatedAst"
    right: "_FederatedAst"


@dataclass(frozen=True, slots=True)
class _Or:
    left: "_FederatedAst"
    right: "_FederatedAst"


@dataclass(frozen=True, slots=True)
class _Not:
    expression: "_FederatedAst"


type _FederatedAst = _Constant | _Predicate | _And | _Or | _Not


@dataclass(frozen=True, slots=True)
class _RecordOutput:
    name: str


@dataclass(frozen=True, slots=True)
class _FieldOutput:
    name: str
    path: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _OriginOutput:
    name: str


type _FederatedOutput = _RecordOutput | _FieldOutput | _OriginOutput


@dataclass(frozen=True, slots=True)
class _FederatedSourcePlan:
    source: str
    target: object


@dataclass(slots=True)
class _FederatedCountCache:
    """One successful exact total shared by equivalent frozen plans."""

    total: int | None = None


@dataclass(frozen=True, slots=True)
class _FederatedPlan:
    """The immutable query description executed by a federated result plan."""

    sources: tuple[_FederatedSourcePlan, ...]
    expressions: tuple[_FederatedAst, ...]
    outputs: tuple[_FederatedOutput, ...]
    offset: int
    limit: int | None
    count_cache: _FederatedCountCache


def _child_field(variable: SearchVariable, path: tuple[str, ...]) -> SearchField:
    field: SearchField | SearchVariable = variable
    for name in path:
        field = cast(SearchField, getattr(field, name))
    return cast(SearchField, field)


def _replay_ast(ast: _FederatedAst, variable: SearchVariable) -> SearchExpression:
    if isinstance(ast, _Constant):
        return variable.always_true() if ast.value else variable.always_false()
    if isinstance(ast, _Predicate):
        operation = getattr(_child_field(variable, ast.path), ast.operation)
        return cast(SearchExpression, operation(*ast.arguments))
    if isinstance(ast, _And):
        return _replay_ast(ast.left, variable) & _replay_ast(ast.right, variable)
    if isinstance(ast, _Or):
        return _replay_ast(ast.left, variable) | _replay_ast(ast.right, variable)
    if isinstance(ast, _Not):
        return ~_replay_ast(ast.expression, variable)
    raise AssertionError(f"unknown federated AST node: {type(ast).__name__}")


def _source_error(source: str, operation: str, exc: Exception) -> FederatedSourceError:
    if isinstance(exc, (UnsupportedQueryError, AttributeError)):
        return _FederatedUnsupportedQueryError(source, operation)
    return FederatedSourceError(source, operation)


_SEARCH_FIELD_SURFACE = (
    "__getattr__",
    "__eq__",
    "__ne__",
    "__lt__",
    "__le__",
    "__gt__",
    "__ge__",
    "contains",
    "startswith",
    "endswith",
    "has",
    "has_any",
    "has_only",
    "is_in",
)


def _type_declares_callable(cls: type, name: str) -> bool:
    """Inspect a class hierarchy without invoking an instance's dynamic lookup."""

    for base in type.__getattribute__(cls, "__mro__"):
        namespace = type.__getattribute__(base, "__dict__")
        if name not in namespace:
            continue
        member = namespace[name]
        if isinstance(member, classmethod | staticmethod):
            member = member.__func__
        return callable(member)
    return False


def _is_search_field_object(value: object) -> bool:
    """Whether ``value`` structurally implements the complete neutral field API."""

    cls = type(value)
    return all(_type_declares_callable(cls, name) for name in _SEARCH_FIELD_SURFACE)


def _count_plan(store: "FederatedStore", plan: _FederatedPlan) -> int:
    """Return an exact unpaged total, requesting only child exact counts.

    The cache is populated only once every source succeeds, so a failed count
    can never turn a prefix total into an apparently exact result.
    """

    if plan.count_cache.total is not None:
        return plan.count_cache.total

    total = 0
    for source_plan in plan.sources:
        source = source_plan.source
        try:
            child_searcher = store._sources[source].searcher()
            child_variable = cast(SearchVariable, child_searcher.variable(source_plan.target))
        except Exception as exc:
            raise _source_error(source, "count searcher construction", exc) from exc
        try:
            for ast in plan.expressions:
                child_searcher.add(_replay_ast(ast, child_variable))
        except Exception as exc:
            raise _source_error(source, "count expression replay", exc) from exc
        try:
            total += child_searcher.count()
        except CountUnavailableError as exc:
            raise _FederatedCountUnavailableError(source, "count") from exc
        except Exception as exc:
            raise _source_error(source, "count", exc) from exc

    plan.count_cache.total = total
    return total


def _execute_plan(
    store: "FederatedStore", plan: _FederatedPlan, *, maximum: int | None = None
) -> Iterator[SearchResult]:
    """Execute one frozen plan, with coordinator-owned global paging.

    Every invocation constructs fresh child searchers.  ``maximum`` is used by
    ``first()`` and ``one()``; it never changes the frozen plan itself.
    """

    effective_limit = plan.limit
    if maximum is not None:
        effective_limit = maximum if effective_limit is None else min(effective_limit, maximum)
    if effective_limit == 0:
        return

    child_outputs = tuple(output for output in plan.outputs if not isinstance(output, _OriginOutput))
    hidden_output = not child_outputs
    child_limit = None if effective_limit is None else plan.offset + effective_limit
    skipped = 0
    yielded = 0

    for source_plan in plan.sources:
        if effective_limit is not None and yielded >= effective_limit:
            return
        source = source_plan.source
        try:
            child_searcher = store._sources[source].searcher()
            child_variable = cast(SearchVariable, child_searcher.variable(source_plan.target))
        except Exception as exc:
            raise _source_error(source, "searcher construction", exc) from exc
        try:
            for ast in plan.expressions:
                child_searcher.add(_replay_ast(ast, child_variable))
        except Exception as exc:
            raise _source_error(source, "expression replay", exc) from exc
        try:
            for output in child_outputs:
                if isinstance(output, _RecordOutput):
                    child_searcher.output(child_variable, output.name)
                else:
                    assert isinstance(output, _FieldOutput)
                    child_searcher.output(_child_field(child_variable, output.path), output.name)
            if hidden_output:
                child_searcher.output(child_variable, "__httk_federated_hidden_record__")
        except Exception as exc:
            raise _source_error(source, "output declaration", exc) from exc
        if child_limit is not None:
            try:
                child_searcher.set_limit(child_limit)
            except Exception as exc:
                raise _source_error(source, "limit pushdown", exc) from exc
        try:
            child_results = iter(child_searcher)
        except Exception as exc:
            raise _source_error(source, "iteration", exc) from exc
        while True:
            try:
                child_result = next(child_results)
            except StopIteration:
                break
            except Exception as exc:
                raise _source_error(source, "iteration", exc) from exc
            try:
                child_values = child_result.values
                if len(child_values) != len(child_outputs) + hidden_output:
                    raise ValueError("child result has a different number of outputs than its declared projection")
                if skipped < plan.offset:
                    skipped += 1
                    continue
                values: list[object] = []
                child_index = 0
                for planned_output in plan.outputs:
                    if isinstance(planned_output, _OriginOutput):
                        values.append(source)
                    else:
                        values.append(child_values[child_index])
                        child_index += 1
                yield SearchResult(tuple(values), tuple(output.name for output in plan.outputs))
                yielded += 1
                if effective_limit is not None and yielded >= effective_limit:
                    return
            except Exception as exc:
                raise _source_error(source, "iteration", exc) from exc


class FederatedResultColumn:
    """A lightweight lazy scalar projection from a federated result set."""

    def __init__(self, result: "FederatedResultSet", index: int) -> None:
        self._result = result
        self._index = index
        self.name = result.names[index]

    def __iter__(self) -> Iterator[object]:
        return (row[self._index] for row in self._result)


class FederatedResultSet:
    """A frozen, lazy, re-iterable federated result plan."""

    def __init__(self, store: "FederatedStore", plan: object) -> None:
        self._store = store
        self._plan = cast(_FederatedPlan, plan)
        self._names = tuple(output.name for output in self._plan.outputs)

    @property
    def names(self) -> tuple[str, ...]:
        return self._names

    def __iter__(self) -> Iterator[ResultRow]:
        return (ResultRow(result.values, result.names) for result in _execute_plan(self._store, self._plan))

    def __len__(self) -> int:
        if self._plan.limit == 0:
            return 0
        available = max(_count_plan(self._store, self._plan) - self._plan.offset, 0)
        return available if self._plan.limit is None else min(available, self._plan.limit)

    def __getitem__(self, item: slice) -> "FederatedResultSet":
        if not isinstance(item, slice):
            raise TypeError("federated result sets support slicing only")
        if item.step is not None and (not isinstance(item.step, int) or isinstance(item.step, bool) or item.step != 1):
            raise ValueError("federated result slices require a unit step")
        start = 0 if item.start is None else item.start
        stop = item.stop
        if (
            not isinstance(start, int)
            or isinstance(start, bool)
            or start < 0
            or (stop is not None and (not isinstance(stop, int) or isinstance(stop, bool) or stop < 0))
        ):
            raise ValueError("federated result slice bounds must be nonnegative integers")
        remaining = None if self._plan.limit is None else max(self._plan.limit - start, 0)
        slice_limit = None if stop is None else max(0, stop - start)
        if slice_limit is not None:
            remaining = slice_limit if remaining is None else min(remaining, slice_limit)
        return FederatedResultSet(
            self._store,
            _FederatedPlan(
                self._plan.sources,
                self._plan.expressions,
                self._plan.outputs,
                self._plan.offset + start,
                remaining,
                self._plan.count_cache,
            ),
        )

    def first(self) -> ResultRow | None:
        result = next(_execute_plan(self._store, self._plan, maximum=1), None)
        return None if result is None else ResultRow(result.values, result.names)

    def one(self) -> ResultRow:
        results = _execute_plan(self._store, self._plan, maximum=2)
        first = next(results, None)
        if first is None:
            raise NoResultError("expected exactly one result, found none")
        if next(results, None) is not None:
            raise MultipleResultsError("expected exactly one result, found more than one")
        return ResultRow(first.values, first.names)

    def scalars(self, name: str | None = None) -> Iterator[object]:
        if name is None:
            if len(self.names) != 1:
                raise ValueError(f"scalars() without a name requires exactly one output; declared: {self.names}")
            name = self.names[0]
        if name not in self.names:
            raise KeyError(f"unknown output {name!r}; declared: {self.names}")
        index = self.names.index(name)
        return (row[index] for row in self)

    def column(self, name: str) -> FederatedResultColumn:
        scalar_names = tuple(output.name for output in self._plan.outputs if not isinstance(output, _RecordOutput))
        if name not in self.names:
            raise KeyError(f"unknown column {name!r}; declared scalar projections: {scalar_names}")
        index = self.names.index(name)
        if isinstance(self._plan.outputs[index], _RecordOutput):
            raise TypeError(f"column {name!r} is an object output; declared scalar projections: {scalar_names}")
        return FederatedResultColumn(self, index)

    def cursor(self) -> Iterator[ResultRow]:
        raise NotImplementedError("federated cursors are not implemented")


@dataclass(frozen=True, slots=True)
class FederatedTarget:
    """One logical target with exact concrete targets for named sources."""

    name: str
    targets: Mapping[str, object]
    _owner: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("federated target names must be nonempty strings")
        if not isinstance(self.targets, Mapping):
            raise TypeError("federated targets must be a mapping")
        if not isinstance(self._owner, FederatedStore):
            raise TypeError("a federated target owner must be a FederatedStore")
        if not self.targets:
            raise ValueError("a federated target requires at least one source target")
        unknown = tuple(source for source in self.targets if source not in self._owner._sources)
        if unknown:
            raise ValueError(f"unknown federation source in target {self.name!r}: {unknown[0]!r}")
        ordered = {source: self.targets[source] for source in self._owner.source_names if source in self.targets}
        object.__setattr__(self, "targets", MappingProxyType(ordered))


class FederatedStore:
    """A read-only ordered collection of borrowed child stores."""

    def __init__(self, sources: Mapping[str, Store]) -> None:
        if not isinstance(sources, Mapping):
            raise TypeError("federation sources must be a mapping")
        if len(sources) < 2:
            raise ValueError("a federation requires at least two sources")
        copied: dict[str, Store] = {}
        for name, store in sources.items():
            if not isinstance(name, str) or not name:
                raise ValueError("federation source names must be nonempty strings")
            copied[name] = store
        self._sources = MappingProxyType(copied)
        self._source_names = tuple(copied)

    @property
    def source_names(self) -> tuple[str, ...]:
        """The immutable source names in constructor iteration order."""

        return self._source_names

    def target(self, name: str, targets: Mapping[str, object]) -> FederatedTarget:
        """Create an immutable target mapping for an intentional source subset."""

        return FederatedTarget(name, targets, self)

    def searcher(self) -> "FederatedSearcher":
        """Create an unbound federated searcher without touching child stores."""

        return FederatedSearcher(self)


class FederatedVariable:
    """The one root variable supported by a federated query."""

    __slots__ = ("_searcher", "_targets", "_variables")

    def __init__(
        self,
        searcher: "FederatedSearcher",
        variables: Mapping[str, SearchVariable],
        targets: Mapping[str, object],
    ) -> None:
        self._searcher = searcher
        self._variables = MappingProxyType(dict(variables))
        self._targets = MappingProxyType(dict(targets))

    def always_true(self) -> "FederatedExpression":
        return self._searcher._expression(_Constant(True), "always_true")

    def always_false(self) -> "FederatedExpression":
        return self._searcher._expression(_Constant(False), "always_false")

    def __getattr__(self, name: str) -> "FederatedField":
        if name.startswith("_"):
            raise AttributeError(name)
        return FederatedField(self, (name,))


class FederatedField:
    """A backend-neutral path from a federated root variable."""

    __slots__ = ("_path", "_variable")

    def __init__(self, variable: FederatedVariable, path: tuple[str, ...]) -> None:
        self._variable = variable
        self._path = path

    def _predicate(self, operation: str, *arguments: object) -> "FederatedExpression":
        if any(
            isinstance(argument, (FederatedField, FederatedExpression)) or _is_search_field_object(argument)
            for argument in arguments
        ):
            raise UnsupportedQueryError("federated queries do not support field-to-field comparisons")
        return self._variable._searcher._expression(_Predicate(self._path, operation, arguments), operation)

    def __eq__(self, value: object) -> "FederatedExpression":  # type: ignore[override]
        return self._predicate("__eq__", value)

    def __ne__(self, value: object) -> "FederatedExpression":  # type: ignore[override]
        return self._predicate("__ne__", value)

    def __lt__(self, value: object) -> "FederatedExpression":
        return self._predicate("__lt__", value)

    def __le__(self, value: object) -> "FederatedExpression":
        return self._predicate("__le__", value)

    def __gt__(self, value: object) -> "FederatedExpression":
        return self._predicate("__gt__", value)

    def __ge__(self, value: object) -> "FederatedExpression":
        return self._predicate("__ge__", value)

    def contains(self, text: str) -> "FederatedExpression":
        return self._predicate("contains", text)

    def startswith(self, prefix: str) -> "FederatedExpression":
        return self._predicate("startswith", prefix)

    def endswith(self, suffix: str) -> "FederatedExpression":
        return self._predicate("endswith", suffix)

    def has(self, value: object) -> "FederatedExpression":
        return self._predicate("has", value)

    def has_any(self, *values: object) -> "FederatedExpression":
        return self._predicate("has_any", *values)

    def has_only(self, *values: object) -> "FederatedExpression":
        return self._predicate("has_only", *values)

    def is_in(self, *values: object) -> "FederatedExpression":
        return self._predicate("is_in", *values)

    def __getattr__(self, name: str) -> "FederatedField":
        if name.startswith("_"):
            raise AttributeError(name)
        return FederatedField(self._variable, (*self._path, name))


class FederatedExpression:
    """A federated expression backed only by the neutral private AST."""

    __slots__ = ("_ast", "_searcher")

    def __init__(self, searcher: "FederatedSearcher", ast: object) -> None:
        self._searcher = searcher
        self._ast = cast(_FederatedAst, ast)

    def _other(self, other: object) -> "FederatedExpression":
        if not isinstance(other, FederatedExpression) or other._searcher is not self._searcher:
            raise UnsupportedQueryError("cannot combine federated expressions from different searchers")
        return other

    def __and__(self, other: object) -> "FederatedExpression":
        right = self._other(other)
        return self._searcher._expression(_And(self._ast, right._ast), "AND")

    def __or__(self, other: object) -> "FederatedExpression":
        right = self._other(other)
        return self._searcher._expression(_Or(self._ast, right._ast), "OR")

    def __invert__(self) -> "FederatedExpression":
        return self._searcher._expression(_Not(self._ast), "NOT")


class _FederatedOrigin:
    """Opaque marker for a source-name output in a federated result plan."""

    __slots__ = ()


class FederatedSearcher:
    """Build and validate one portable, single-root federated query."""

    __slots__ = (
        "_count_cache",
        "_expressions",
        "_limit",
        "_outputs",
        "_prototypes",
        "_store",
        "_variable",
        "offset",
        "origin",
    )

    def __init__(self, store: FederatedStore) -> None:
        self._store = store
        self._variable: FederatedVariable | None = None
        self._prototypes: Mapping[str, Searcher] = MappingProxyType({})
        self._expressions: list[_FederatedAst] = []
        self._outputs: list[_FederatedOutput] = []
        self._count_cache = _FederatedCountCache()
        self._limit: int | None = None
        self.offset = 0
        self.origin = _FederatedOrigin()

    def variable(self, target: object) -> FederatedVariable:
        """Bind one shared or explicit target against child searcher prototypes."""

        if self._variable is not None:
            raise UnsupportedQueryError("federated queries support one root variable; a second root was requested")
        if isinstance(target, FederatedTarget):
            if target._owner is not self._store:
                raise UnsupportedQueryError("a FederatedTarget from another federation or stale ownership was supplied")
            source_targets = target.targets
        else:
            source_targets = MappingProxyType({source: target for source in self._store.source_names})

        variables: dict[str, SearchVariable] = {}
        prototypes: dict[str, Searcher] = {}
        for source in self._store.source_names:
            if source not in source_targets:
                continue
            try:
                child_searcher = self._store._sources[source].searcher()
                variables[source] = cast(SearchVariable, child_searcher.variable(source_targets[source]))
                prototypes[source] = child_searcher
            except Exception as exc:
                raise _source_error(source, "target binding", exc) from exc
        self._prototypes = MappingProxyType(prototypes)
        self._variable = FederatedVariable(self, variables, source_targets)
        return self._variable

    def _require_variable(self) -> FederatedVariable:
        if self._variable is None:
            raise ValueError("this federated searcher has no query variable; call variable() first")
        return self._variable

    def _validate(self, ast: _FederatedAst, operation: str) -> None:
        variable = self._require_variable()
        for source, child_variable in variable._variables.items():
            try:
                _replay_ast(ast, child_variable)
            except Exception as exc:
                raise _source_error(source, operation, exc) from exc

    def _expression(self, ast: _FederatedAst, operation: str) -> FederatedExpression:
        self._validate(ast, operation)
        return FederatedExpression(self, ast)

    def add(self, expression: object) -> None:
        """Validate and retain a portable condition for the future frozen plan."""

        variable = self._require_variable()
        if not isinstance(expression, FederatedExpression) or expression._searcher is not self:
            raise UnsupportedQueryError("federated queries accept expressions from this searcher only")
        for source, child_variable in variable._variables.items():
            try:
                self._prototypes[source].add(_replay_ast(expression._ast, child_variable))
            except Exception as exc:
                raise _source_error(source, "add", exc) from exc
        self._expressions.append(expression._ast)
        # Existing result plans retain their old cache; the mutable searcher is
        # now a different unpaged query and must obtain a fresh exact total.
        self._count_cache = _FederatedCountCache()

    def _output(self, value: object, name: str, *, retain: bool) -> _FederatedOutput:
        variable = self._require_variable()
        if not isinstance(name, str) or not name:
            raise ValueError("output name must be a nonempty string")
        outputs = self._outputs if retain else ()
        if any(output.name == name for output in outputs):
            raise ValueError(f"duplicate output name: {name!r}")
        if value is self.origin:
            return _OriginOutput(name)
        field_path: tuple[str, ...] | None
        if value is variable:
            output: _FederatedOutput = _RecordOutput(name)
            field_path = None
        elif isinstance(value, FederatedField) and value._variable is variable:
            output = _FieldOutput(name, value._path)
            field_path = value._path
        else:
            raise UnsupportedQueryError("outputs must belong to this federated searcher or be its origin sentinel")
        # Output validation uses disposable child searchers.  The long-lived
        # prototypes validate expressions and may retain child output state, so
        # reusing them here would make repeated results() planning spuriously
        # fail on a child's duplicate-output guard.
        for source, target in variable._targets.items():
            try:
                child_searcher = self._store._sources[source].searcher()
                child_variable = child_searcher.variable(target)
                child_value = (
                    child_variable
                    if field_path is None
                    else _child_field(cast(SearchVariable, child_variable), field_path)
                )
                child_searcher.output(child_value, name)
            except Exception as exc:
                raise _source_error(source, "output", exc) from exc
        return output

    def output(self, value: object, name: str) -> None:
        """Declare a record, scalar field, or origin output for a future plan."""

        self._outputs.append(self._output(value, name, retain=True))

    def add_sort(self, field: object, descending: bool = False) -> None:
        """Reject global sorting until a portable sort-semantics contract exists."""

        raise UnsupportedQueryError("federated queries do not support ordinary add_sort()")

    def _plan(self, outputs: Mapping[str, object] | None = None, *, require_outputs: bool = True) -> _FederatedPlan:
        """Freeze validated declarations into the private phase-three input."""

        variable = self._require_variable()
        planned_outputs = (
            tuple(self._output(value, name, retain=False) for name, value in outputs.items())
            if outputs
            else tuple(self._outputs)
        )
        if require_outputs and not planned_outputs:
            raise ValueError("this federated result plan has no outputs; call output() or pass results() projections")
        sources = tuple(_FederatedSourcePlan(source, target) for source, target in variable._targets.items())
        return _FederatedPlan(
            sources,
            tuple(self._expressions),
            planned_outputs,
            self.offset,
            self._limit,
            self._count_cache,
        )

    def count(self) -> int:
        """Return the exact unpaged count of the current filtered union."""

        return _count_plan(self._store, self._plan(require_outputs=False))

    def set_limit(self, limit: int) -> None:
        """Set the global output limit; a negative value clears it."""

        if not isinstance(limit, int) or isinstance(limit, bool):
            raise TypeError("limit must be an integer")
        self._limit = None if limit < 0 else limit

    def add_offset(self, offset: int) -> None:
        """Add a global source-union offset."""

        if not isinstance(offset, int) or isinstance(offset, bool):
            raise TypeError("offset must be an integer")
        if offset < 0:
            raise ValueError("offset must be nonnegative")
        self.offset += offset

    def __iter__(self) -> Iterator[SearchResult]:
        """Execute the retained-output plan directly as ``SearchResult`` values."""

        return _execute_plan(self._store, self._plan())

    def results(self, **outputs: object) -> FederatedResultSet:
        """Freeze a projection plan into a lazy, re-iterable result set."""

        return FederatedResultSet(self._store, self._plan(outputs or None))
