"""Frozen source and target bindings for a future federated query store."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from .query import Store, UnsupportedQueryError

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
class FederatedTarget:
    """One logical target with exact concrete targets for named sources."""

    name: str
    targets: Mapping[str, object]
    _owner: object = field(repr=False, compare=False)


class FederatedStore:
    """A read-only ordered collection of borrowed child stores.

    This phase only binds targets against child searcher prototypes; it does
    not construct or execute federated queries.
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
    """A root variable whose child variables were prototype-bound successfully."""

    __slots__ = ("_searcher", "_variables")

    def __init__(self, searcher: "FederatedSearcher", variables: Mapping[str, object]) -> None:
        self._searcher = searcher
        self._variables = MappingProxyType(dict(variables))


class FederatedSearcher:
    """A phase-one searcher that validates one root target across its sources."""

    __slots__ = ("_store", "_variable")

    def __init__(self, store: FederatedStore) -> None:
        self._store = store
        self._variable: FederatedVariable | None = None

    def variable(self, target: object) -> FederatedVariable:
        """Bind one shared or explicit target against child searcher prototypes."""

        if self._variable is not None:
            raise UnsupportedQueryError("federated queries support one root variable; a second root was requested")
        if isinstance(target, FederatedTarget):
            if target._owner is not self._store:
                raise UnsupportedQueryError("a FederatedTarget from another federation or stale ownership was supplied")
            source_targets = target.targets
        else:
            source_targets = {source: target for source in self._store.source_names}

        variables: dict[str, object] = {}
        for source in self._store.source_names:
            if source not in source_targets:
                continue
            try:
                child_searcher = self._store._sources[source].searcher()
                variables[source] = child_searcher.variable(source_targets[source])
            except UnsupportedQueryError as exc:
                raise _FederatedUnsupportedQueryError(source, "target binding") from exc
        self._variable = FederatedVariable(self, variables)
        return self._variable
