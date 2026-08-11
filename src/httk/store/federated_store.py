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
    """Report that a federation-level store operation failed."""


class FederatedSourceError(FederatedStoreError):
    """Report that a named source rejected or failed a federated operation.

    :param source: The source name that failed.
    :param operation: The federation operation being performed.
    """

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
    as_of: object
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


def _child_searcher(store: Store, as_of: object) -> Searcher:
    return store.searcher() if as_of is None else store.searcher(as_of=as_of)


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
            child_searcher = _child_searcher(store._sources[source], plan.as_of)
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
            child_searcher = _child_searcher(store._sources[source], plan.as_of)
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
    """Expose one scalar projection lazily from a federated result set.

    :param result: The result set supplying rows.
    :param index: The zero-based projection index.
    """

    def __init__(self, result: "FederatedResultSet", index: int) -> None:
        self._result = result
        self._index = index
        self.name = result.names[index]

    def __iter__(self) -> Iterator[object]:
        return (row[self._index] for row in self._result)


class FederatedResultSet:
    """Represent a frozen, lazy, re-iterable federated result plan.

    Results execute source-major in federation source order, preserve duplicate
    rows, and remain read-only views over the borrowed stores.

    :param store: The federation whose sources execute the plan.
    :param plan: The validated frozen federation plan.
    """

    def __init__(self, store: "FederatedStore", plan: object) -> None:
        self._store = store
        self._plan = cast(_FederatedPlan, plan)
        self._names = tuple(output.name for output in self._plan.outputs)

    @property
    def names(self) -> tuple[str, ...]:
        """Return the declared projection names.

        :return: The result projection names in declaration order.
        """
        return self._names

    def __iter__(self) -> Iterator[ResultRow]:
        """Iterate through the source-major union as named result rows.

        :return: An iterator over the result rows.
        """
        return (ResultRow(result.values, result.names) for result in _execute_plan(self._store, self._plan))

    def __len__(self) -> int:
        """Return the exact unpaged union count after the result slice.

        :return: The exact number of rows available to this result plan.
        """
        if self._plan.limit == 0:
            return 0
        available = max(_count_plan(self._store, self._plan) - self._plan.offset, 0)
        return available if self._plan.limit is None else min(available, self._plan.limit)

    def __getitem__(self, item: slice) -> "FederatedResultSet":
        """Return a unit-step, nonnegative slice of this result plan.

        :param item: The nonnegative unit-step slice to apply.
        :return: A new frozen result plan with the adjusted offset and limit.
        :raises TypeError: If ``item`` is not a slice.
        :raises ValueError: If the slice has invalid bounds or a non-unit step.
        """
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
                self._plan.as_of,
                self._plan.count_cache,
            ),
        )

    def first(self) -> ResultRow | None:
        """Return the first result row, or ``None`` when no row matches.

        :return: The first matching row, or ``None``.
        """
        result = next(_execute_plan(self._store, self._plan, maximum=1), None)
        return None if result is None else ResultRow(result.values, result.names)

    def one(self) -> ResultRow:
        """Return the only result row.

        :return: The sole matching result row.
        :raises httk.store.query.protocols.NoResultError: If no row matches.
        :raises httk.store.query.protocols.MultipleResultsError: If more than one row matches.
        """
        results = _execute_plan(self._store, self._plan, maximum=2)
        first = next(results, None)
        if first is None:
            raise NoResultError("expected exactly one result, found none")
        if next(results, None) is not None:
            raise MultipleResultsError("expected exactly one result, found more than one")
        return ResultRow(first.values, first.names)

    def scalars(self, name: str | None = None) -> Iterator[object]:
        """Iterate over one named projection.

        :param name: The projection name, required when more than one output exists.
        :return: An iterator over the selected projection values.
        :raises ValueError: If no name is supplied for multiple outputs.
        :raises KeyError: If ``name`` is not a declared output.
        """
        if name is None:
            if len(self.names) != 1:
                raise ValueError(f"scalars() without a name requires exactly one output; declared: {self.names}")
            name = self.names[0]
        if name not in self.names:
            raise KeyError(f"unknown output {name!r}; declared: {self.names}")
        index = self.names.index(name)
        return (row[index] for row in self)

    def column(self, name: str) -> FederatedResultColumn:
        """Return a lazy scalar column by projection name.

        :param name: The scalar projection name.
        :return: A lazy column view over the projection.
        :raises KeyError: If ``name`` is not a declared output.
        :raises TypeError: If ``name`` identifies an object output.
        """
        scalar_names = tuple(output.name for output in self._plan.outputs if not isinstance(output, _RecordOutput))
        if name not in self.names:
            raise KeyError(f"unknown column {name!r}; declared scalar projections: {scalar_names}")
        index = self.names.index(name)
        if isinstance(self._plan.outputs[index], _RecordOutput):
            raise TypeError(f"column {name!r} is an object output; declared scalar projections: {scalar_names}")
        return FederatedResultColumn(self, index)

    def cursor(self) -> Iterator[ResultRow]:
        """Reject cursor access because federation cursors are unsupported.

        :return: Never returns.
        :raises NotImplementedError: Always, because federated cursors are not implemented.
        """
        raise NotImplementedError("federated cursors are not implemented")


@dataclass(frozen=True, slots=True)
class FederatedTarget:
    """Bind one logical target to exact concrete targets for named sources.

    :param name: The nonempty logical target name.
    :param targets: Concrete targets keyed by federation source name.
    :param _owner: The federation that owns this target binding.
    :raises TypeError: If ``targets`` is not a mapping or ``_owner`` is not a federation.
    :raises ValueError: If the name, source set, or source names are invalid.
    """

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
    """Fan out read-only queries over an ordered collection of borrowed stores.

    The union is source-major, lazy, and non-deduplicating. Queries require the
    strict common query surface accepted by every participating source, and
    counts are exact sums of the unpaged source counts. This live borrowed-store
    view is distinct from the persisted registry in
    :mod:`httk.store.db.stored_federation`.

    :param sources: Child stores keyed by stable federation source name.
    :raises TypeError: If ``sources`` is not a mapping.
    :raises ValueError: If fewer than two sources or an invalid source name is supplied.
    """

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
        """Return the immutable source names in constructor iteration order.

        :return: The source names in constructor order.
        """

        return self._source_names

    def target(self, name: str, targets: Mapping[str, object]) -> FederatedTarget:
        """Create an immutable target mapping for an intentional source subset.

        :param name: The logical target name.
        :param targets: Concrete targets keyed by federation source name.
        :return: The validated target binding.
        :raises TypeError: If ``targets`` is not a mapping.
        :raises ValueError: If a target name or source name is invalid.
        """

        return FederatedTarget(name, targets, self)

    def searcher(self, *, as_of: object = None) -> "FederatedSearcher":
        """Create an unbound federated searcher without touching child stores.

        :param as_of: Optional historic cutoff forwarded to every child store.
            A child that cannot honor it raises and the federation surfaces that
            child failure; this is a user-facing store API.
        :return: A new mutable query builder.
        """

        return FederatedSearcher(self, as_of=as_of)


class FederatedVariable:
    """Represent the one root variable supported by a federated query.

    :param searcher: The owning federated searcher.
    :param variables: Child variables keyed by source name.
    :param targets: Concrete child targets keyed by source name.
    """

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
        """Build an expression that matches every federated row.

        :return: A federated expression matching every row.
        """
        return self._searcher._expression(_Constant(True), "always_true")

    def always_false(self) -> "FederatedExpression":
        """Build an expression that matches no federated row.

        :return: A federated expression matching no row.
        """
        return self._searcher._expression(_Constant(False), "always_false")

    def __getattr__(self, name: str) -> "FederatedField":
        if name.startswith("_"):
            raise AttributeError(name)
        return FederatedField(self, (name,))


class FederatedField:
    """Represent a backend-neutral path from a federated root variable.

    :param variable: The federated root variable owning the path.
    :param path: The field path relative to that variable.
    """

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
        """Match literal values containing ``text``.

        :param text: The literal substring to find.
        :return: The resulting federated expression.
        """
        return self._predicate("contains", text)

    def startswith(self, prefix: str) -> "FederatedExpression":
        """Match literal values beginning with ``prefix``.

        :param prefix: The literal prefix to find.
        :return: The resulting federated expression.
        """
        return self._predicate("startswith", prefix)

    def endswith(self, suffix: str) -> "FederatedExpression":
        """Match literal values ending with ``suffix``.

        :param suffix: The literal suffix to find.
        :return: The resulting federated expression.
        """
        return self._predicate("endswith", suffix)

    def has(self, value: object) -> "FederatedExpression":
        """Match a list field containing ``value``.

        :param value: The list member to match.
        :return: The resulting federated expression.
        """
        return self._predicate("has", value)

    def has_any(self, *values: object) -> "FederatedExpression":
        """Match a list field containing any of ``values``.

        :param \\*values: The list members to match.
        :return: The resulting federated expression.
        """
        return self._predicate("has_any", *values)

    def has_only(self, *values: object) -> "FederatedExpression":
        """Match a list field containing no values outside ``values``.

        :param \\*values: The complete allowed list-member set.
        :return: The resulting federated expression.
        """
        return self._predicate("has_only", *values)

    def is_in(self, *values: object) -> "FederatedExpression":
        """Match a scalar field whose value is one of ``values``.

        :param \\*values: The accepted field values.
        :return: The resulting federated expression.
        """
        return self._predicate("is_in", *values)

    def __getattr__(self, name: str) -> "FederatedField":
        if name.startswith("_"):
            raise AttributeError(name)
        return FederatedField(self._variable, (*self._path, name))


class FederatedExpression:
    """Represent a federated expression backed by the neutral private AST.

    :param searcher: The owning federated searcher.
    :param ast: The validated private expression tree.
    """

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
    """Build and validate one portable, single-root federated query.

    :param store: The federation whose child stores provide the query surface.
    :param as_of: Optional historic cutoff forwarded to child stores.
    """

    __slots__ = (
        "_as_of",
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

    def __init__(self, store: FederatedStore, *, as_of: object = None) -> None:
        self._store = store
        self._as_of = as_of
        self._variable: FederatedVariable | None = None
        self._prototypes: Mapping[str, Searcher] = MappingProxyType({})
        self._expressions: list[_FederatedAst] = []
        self._outputs: list[_FederatedOutput] = []
        self._count_cache = _FederatedCountCache()
        self._limit: int | None = None
        self.offset = 0
        self.origin = _FederatedOrigin()

    def variable(self, target: object) -> FederatedVariable:
        """Bind one shared or explicit target against child searcher prototypes.

        :param target: A shared child target or a source-specific target binding.
        :return: The federated root variable.
        :raises httk.store.query.protocols.UnsupportedQueryError: If a second root or foreign target is supplied.
        :raises FederatedSourceError: If a source rejects target binding.
        """

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
                child_searcher = _child_searcher(self._store._sources[source], self._as_of)
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
        """Validate and retain a portable condition for the future frozen plan.

        :param expression: An expression produced by this searcher.
        :return: None.
        :raises httk.store.query.protocols.UnsupportedQueryError: If the expression belongs to another searcher.
        :raises FederatedSourceError: If a source rejects the expression.
        """

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
                child_searcher = _child_searcher(self._store._sources[source], self._as_of)
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
        """Declare a record, scalar field, or origin output for a future plan.

        :param value: The root variable, field, or ``origin`` sentinel to project.
        :param name: The nonempty output name.
        :return: None.
        :raises ValueError: If ``name`` is empty or already declared.
        :raises httk.store.query.protocols.UnsupportedQueryError: If ``value`` is not owned by this searcher.
        :raises FederatedSourceError: If a source rejects the output.
        """

        self._outputs.append(self._output(value, name, retain=True))

    def add_sort(self, field: object, descending: bool = False) -> None:
        """Reject global sorting until a portable sort-semantics contract exists.

        :param field: The requested sort field.
        :param descending: Whether the requested order is descending.
        :raises httk.store.query.protocols.UnsupportedQueryError: Always, because global federation sorting
            has no portable contract.
        """

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
            self._as_of,
            self._count_cache,
        )

    def count(self) -> int:
        """Return the exact unpaged count of the current filtered union.

        :return: The exact sum of matching rows across participating sources.
        :raises httk.store.query.protocols.CountUnavailableError: If a source cannot provide an exact count.
        :raises FederatedSourceError: If a source fails while counting.
        """

        return _count_plan(self._store, self._plan(require_outputs=False))

    def set_limit(self, limit: int) -> None:
        """Set the global output limit; a negative value clears it.

        :param limit: The nonnegative limit, or a negative value to clear it.
        :return: None.
        :raises TypeError: If ``limit`` is not an integer.
        """

        if not isinstance(limit, int) or isinstance(limit, bool):
            raise TypeError("limit must be an integer")
        self._limit = None if limit < 0 else limit

    def add_offset(self, offset: int) -> None:
        """Add a global source-union offset.

        :param offset: The nonnegative number of union rows to skip.
        :return: None.
        :raises TypeError: If ``offset`` is not an integer.
        :raises ValueError: If ``offset`` is negative.
        """

        if not isinstance(offset, int) or isinstance(offset, bool):
            raise TypeError("offset must be an integer")
        if offset < 0:
            raise ValueError("offset must be nonnegative")
        self.offset += offset

    def __iter__(self) -> Iterator[SearchResult]:
        """Execute the retained-output plan directly as ``SearchResult`` values."""

        return _execute_plan(self._store, self._plan())

    def results(self, **outputs: object) -> FederatedResultSet:
        """Freeze a projection plan into a lazy, re-iterable result set.

        :param \\*\\*outputs: Optional output names mapped to root variables or fields.
        :return: The lazy frozen result set.
        :raises ValueError: If no outputs are declared or an output name is invalid.
        :raises httk.store.query.protocols.UnsupportedQueryError: If an output does not belong to this searcher.
        :raises FederatedSourceError: If a source rejects an output.
        """

        return FederatedResultSet(self._store, self._plan(outputs or None))
