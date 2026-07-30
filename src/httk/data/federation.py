"""Frozen source bindings and query planning for a federated store.

Federated expressions are deliberately represented by a private, backend-neutral
AST.  Child searcher expressions are used only while validating that each
participating source accepts an operation; they never escape their source.
Execution of a frozen federated plan arrives in phase three.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, cast

from .query import Searcher, SearchExpression, SearchField, SearchVariable, Store, UnsupportedQueryError

__all__ = [
    "FederatedSourceError",
    "FederatedStore",
    "FederatedStoreError",
    "FederatedTarget",
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


@dataclass(frozen=True, slots=True)
class _FederatedPlan:
    """The immutable query description that phase three will execute."""

    sources: tuple[_FederatedSourcePlan, ...]
    expressions: tuple[_FederatedAst, ...]
    outputs: tuple[_FederatedOutput, ...]


@dataclass(frozen=True, slots=True)
class FederatedTarget:
    """One logical target with exact concrete targets for named sources."""

    name: str
    targets: Mapping[str, object]
    _owner: object = field(repr=False, compare=False)


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

        if not isinstance(name, str) or not name:
            raise ValueError("federated target names must be nonempty strings")
        if not isinstance(targets, Mapping):
            raise TypeError("federated targets must be a mapping")
        if not targets:
            raise ValueError("a federated target requires at least one source target")
        unknown = tuple(source for source in targets if source not in self._sources)
        if unknown:
            raise ValueError(f"unknown federation source in target {name!r}: {unknown[0]!r}")
        ordered = {source: targets[source] for source in self._source_names if source in targets}
        return FederatedTarget(name, MappingProxyType(ordered), self)

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
        if any(isinstance(argument, (FederatedField, FederatedExpression)) for argument in arguments):
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

    def __init__(self, searcher: "FederatedSearcher", ast: _FederatedAst) -> None:
        self._searcher = searcher
        self._ast = ast

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

    __slots__ = ("_expressions", "_outputs", "_prototypes", "_store", "_variable", "origin")

    def __init__(self, store: FederatedStore) -> None:
        self._store = store
        self._variable: FederatedVariable | None = None
        self._prototypes: Mapping[str, Searcher] = MappingProxyType({})
        self._expressions: list[_FederatedAst] = []
        self._outputs: list[_FederatedOutput] = []
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
            except (UnsupportedQueryError, AttributeError) as exc:
                raise _FederatedUnsupportedQueryError(source, "target binding") from exc
        self._prototypes = MappingProxyType(prototypes)
        self._variable = FederatedVariable(self, variables, source_targets)
        return self._variable

    def _require_variable(self) -> FederatedVariable:
        if self._variable is None:
            raise ValueError("this federated searcher has no query variable; call variable() first")
        return self._variable

    @staticmethod
    def _field(variable: SearchVariable, path: tuple[str, ...]) -> SearchField:
        field: SearchField | SearchVariable = variable
        for name in path:
            field = cast(SearchField, getattr(field, name))
        return cast(SearchField, field)

    def _replay(self, ast: _FederatedAst, variable: SearchVariable) -> SearchExpression:
        if isinstance(ast, _Constant):
            return variable.always_true() if ast.value else variable.always_false()
        if isinstance(ast, _Predicate):
            operation = getattr(self._field(variable, ast.path), ast.operation)
            return cast(SearchExpression, operation(*ast.arguments))
        if isinstance(ast, _And):
            return self._replay(ast.left, variable) & self._replay(ast.right, variable)
        if isinstance(ast, _Or):
            return self._replay(ast.left, variable) | self._replay(ast.right, variable)
        if isinstance(ast, _Not):
            return ~self._replay(ast.expression, variable)
        raise AssertionError(f"unknown federated AST node: {type(ast).__name__}")

    def _validate(self, ast: _FederatedAst, operation: str) -> None:
        variable = self._require_variable()
        for source, child_variable in variable._variables.items():
            try:
                self._replay(ast, child_variable)
            except (UnsupportedQueryError, AttributeError) as exc:
                raise _FederatedUnsupportedQueryError(source, operation) from exc

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
                self._prototypes[source].add(self._replay(expression._ast, child_variable))
            except (UnsupportedQueryError, AttributeError) as exc:
                raise _FederatedUnsupportedQueryError(source, "add") from exc
        self._expressions.append(expression._ast)

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
                    else self._field(cast(SearchVariable, child_variable), field_path)
                )
                child_searcher.output(child_value, name)
            except (UnsupportedQueryError, AttributeError) as exc:
                raise _FederatedUnsupportedQueryError(source, "output") from exc
        return output

    def output(self, value: object, name: str) -> None:
        """Declare a record, scalar field, or origin output for a future plan."""

        self._outputs.append(self._output(value, name, retain=True))

    def add_sort(self, field: object, descending: bool = False) -> None:
        """Reject global sorting until a portable sort-semantics contract exists."""

        raise UnsupportedQueryError("federated queries do not support ordinary add_sort()")

    def _plan(self, outputs: Mapping[str, object] | None = None) -> _FederatedPlan:
        """Freeze validated declarations into the private phase-three input."""

        variable = self._require_variable()
        planned_outputs = (
            tuple(self._output(value, name, retain=False) for name, value in outputs.items())
            if outputs
            else tuple(self._outputs)
        )
        if not planned_outputs:
            raise ValueError("this federated result plan has no outputs; call output() or pass results() projections")
        sources = tuple(_FederatedSourcePlan(source, target) for source, target in variable._targets.items())
        return _FederatedPlan(sources, tuple(self._expressions), planned_outputs)

    def results(self, **outputs: object) -> Any:
        """Validate a frozen projection plan; phase three will execute it."""

        self._plan(outputs or None)
        raise NotImplementedError("federated result execution is not implemented until phase three")
