"""Typed protocols for the store/searcher query contract.

These protocols define the backend-agnostic query interface shared by httk
data stores: the database layer in httk-data implements them over SQL, and
serving modules (such as *httk-optimade*, whose in-memory store also conforms)
program against them. They mirror the query interface of the httk v1 database
layer (``httk.db`` ``FilteredCollection`` searchers), so lightweight fakes can
stand in for a real store in tests.
"""

from typing import Any, Iterator, NamedTuple, Protocol

__all__ = [
    "SearchExpression",
    "SearchField",
    "SearchVariable",
    "SearchResult",
    "Searcher",
    "Store",
]


class SearchExpression(Protocol):
    def __and__(self, other: "SearchExpression") -> "SearchExpression": ...

    def __or__(self, other: "SearchExpression") -> "SearchExpression": ...

    def __invert__(self) -> "SearchExpression": ...


class SearchField(Protocol):
    """A queryable field of a search variable.

    In addition to the methods below, fields support the rich comparison
    operators (``==``, ``!=``, ``<``, ``<=``, ``>``, ``>=``), returning
    :class:`SearchExpression`. The handlers invoke those via
    ``getattr(field, '__eq__')(value)`` since the comparison dunders cannot be
    typed as expression-returning.

    The three string-matching methods take **literal** text: no wildcard or
    pattern syntax whatsoever crosses this contract, so ``%`` and ``_`` (and
    any other metacharacter) match themselves. A backend is therefore free to
    implement them with SQL ``LIKE`` over an escaped pattern, with a regular
    expression, or with a full-text index — the choice is invisible here.
    """

    def has_any(self, *values: Any) -> SearchExpression: ...

    def has_only(self, *values: Any) -> SearchExpression: ...

    def contains(self, text: str) -> SearchExpression:
        """Match values containing ``text`` as a literal substring."""
        ...

    def startswith(self, prefix: str) -> SearchExpression:
        """Match values beginning with the literal ``prefix``."""
        ...

    def endswith(self, suffix: str) -> SearchExpression:
        """Match values ending with the literal ``suffix``."""
        ...


class SearchVariable(Protocol):
    """A query variable bound to a target type; attribute access yields fields.

    ``always_true``/``always_false`` are reserved names: they are real methods
    of the variable, never stored fields resolved through ``__getattr__``.
    They exist so a translation layer can express a constant truth value
    without inventing a probe field — the old ``field == field`` trick was
    also NULL-unsound, since it yields NULL (not true) for a NULL field.
    """

    def always_true(self) -> SearchExpression:
        """An expression that matches every row."""
        ...

    def always_false(self) -> SearchExpression:
        """An expression that matches no row."""
        ...

    def __getattr__(self, name: str) -> SearchField: ...


class SearchResult(NamedTuple):
    """One match: the declared output values, and the names they were declared under.

    ``values`` holds one entry per :meth:`Searcher.output` call in declaration
    order; it is a tuple, so ``values, names = result`` and ``result[0][0]``
    both work.
    """

    values: tuple[Any, ...]
    names: tuple[str, ...]


class Searcher(Protocol):
    """A single query under construction, and its results once iterated.

    Iteration yields one :class:`SearchResult` per match, so ``item[0][0]`` is
    the first declared output of the match (typically the matched row object).
    The expressions received by ``add`` are always ones produced by this same
    backend's search variables, so implementations may type them as their own
    expression class; a backend that needs a second (post-filter) evaluation
    position decides that from the expression itself, not from the caller.
    """

    offset: int

    def variable(self, target: Any) -> Any: ...

    def output(self, variable: Any, name: str) -> None: ...

    def add(self, expression: Any) -> None: ...

    def count(self) -> int: ...

    def set_limit(self, limit: int) -> None: ...

    def add_offset(self, offset: int) -> None: ...

    def add_sort(self, field: Any, descending: bool) -> None: ...

    def __iter__(self) -> Iterator[SearchResult]: ...


class Store(Protocol):
    def searcher(self) -> Searcher: ...
