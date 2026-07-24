"""Typed protocols for the store/searcher query contract.

These protocols define the backend-agnostic query interface shared by httk
data stores: the database layer in httk-data implements them over SQL, and
serving modules (such as *httk-optimade*, whose in-memory store also conforms)
program against them. They mirror the query interface of the httk v1 database
layer (``httk.db`` ``FilteredCollection`` searchers), so lightweight fakes can
stand in for a real store in tests.
"""

from typing import Any, Iterator, Protocol

__all__ = [
    "SearchExpression",
    "SearchColumn",
    "SearchVariable",
    "Searcher",
    "Store",
]


class SearchExpression(Protocol):
    def __and__(self, other: "SearchExpression") -> "SearchExpression": ...

    def __or__(self, other: "SearchExpression") -> "SearchExpression": ...

    def __invert__(self) -> "SearchExpression": ...


class SearchColumn(Protocol):
    """A queryable column of a search variable.

    In addition to the methods below, columns support the rich comparison
    operators (``==``, ``!=``, ``<``, ``<=``, ``>``, ``>=``) and
    ``startswith``/``endswith``, returning :class:`SearchExpression`. The
    handlers invoke those via ``getattr(column, '__eq__')(value)`` since the
    comparison dunders cannot be typed as expression-returning.
    """

    def has_any(self, *values: Any) -> SearchExpression: ...

    def has_inv_any(self, *values: Any) -> SearchExpression: ...

    def has_only(self, *values: Any) -> SearchExpression: ...

    def has_inv_only(self, *values: Any) -> SearchExpression: ...

    def like(self, pattern: str) -> SearchExpression: ...


class SearchVariable(Protocol):
    """A query variable bound to a target table/type; attribute access yields columns."""

    def __getattr__(self, name: str) -> SearchColumn: ...


class Searcher(Protocol):
    """A single query under construction, and its results once iterated.

    Iteration yields items where ``item[0][0]`` is the matched row object.
    The expressions received by ``add``/``add_all`` are always ones produced
    by this same backend's search variables, so implementations may type them
    as their own expression class.
    """

    offset: int

    def variable(self, target: Any) -> Any: ...

    def output(self, variable: Any, name: str) -> None: ...

    def add(self, expression: Any) -> None: ...

    def add_all(self, expression: Any) -> None: ...

    def count(self) -> int: ...

    def set_limit(self, limit: int) -> None: ...

    def add_offset(self, offset: int) -> None: ...

    def add_sort(self, column: Any, descending: bool) -> None: ...

    def __iter__(self) -> Iterator[Any]: ...


class Store(Protocol):
    def searcher(self) -> Searcher: ...
