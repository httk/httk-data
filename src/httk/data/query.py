"""Typed protocols for the store/searcher query contract.

These protocols define the backend-agnostic query interface shared by httk
data stores: the database layer in httk-data implements them over SQL, and
serving modules (such as *httk-optimade*, whose in-memory store also conforms)
program against them. They mirror the query interface of the httk v1 database
layer (``httk.db`` ``FilteredCollection`` searchers), so lightweight fakes can
stand in for a real store in tests.
"""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple, Protocol, Self

__all__ = [
    "ContinuationToken",
    "CountUnavailableError",
    "MultipleResultsError",
    "NoResultError",
    "PageOrder",
    "PageableResultSetLike",
    "PaginationCursorError",
    "ResultPage",
    "ResultRow",
    "ResultRowLike",
    "ResultSetLike",
    "SearchExpression",
    "SearchField",
    "SearchResult",
    "SearchVariable",
    "Searcher",
    "Store",
    "UnsupportedQueryError",
]


class UnsupportedQueryError(ValueError):
    """A valid query operation is outside a store's supported query profile."""


class CountUnavailableError(RuntimeError):
    """A store cannot provide the exact count required by a query operation."""


class NoResultError(LookupError):
    """A result-set :meth:`one` operation found no matching result."""


class MultipleResultsError(LookupError):
    """A result-set :meth:`one` operation found more than one matching result."""


class PaginationCursorError(ValueError):
    """A continuation cursor is malformed, expired, or belongs to another result plan."""


@dataclass(frozen=True, slots=True)
class PageOrder:
    """One named scalar result projection used to order a continuation page.

    ``name`` identifies the name supplied to ``results()`` (or
    :meth:`Searcher.output`), never a backend column object.  The result-set
    implementation validates that it is a root scalar projection before it
    generates SQL.
    """

    name: str
    descending: bool = False
    nulls: Literal["first", "last"] = "last"

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("PageOrder.name must be a non-empty string")
        if not isinstance(self.descending, bool):
            raise TypeError("PageOrder.descending must be bool")
        if self.nulls not in {"first", "last"}:
            raise ValueError("PageOrder.nulls must be 'first' or 'last'")


class ContinuationToken(str):
    """An opaque URL-safe continuation value.

    It is a ``str`` subclass so normal JSON serializers preserve it as a
    scalar value.  Applications should pass a token returned by a page back
    unchanged; data backends validate its version, structure, and result-plan
    fingerprint before using any decoded value as a bound parameter.
    """

    def __new__(cls, value: str) -> Self:
        if not isinstance(value, str):
            raise TypeError(f"ContinuationToken requires str, got {type(value).__name__}")
        return str.__new__(cls, value)


@dataclass(frozen=True, slots=True)
class ResultPage:
    """An immutable continuation-page result.

    ``rows`` is always a tuple.  Returned rows are ordinary persistent result
    rows, not the expiring proxies produced by ``SqlResultSet.cursor()``.
    ``total`` is populated only when the caller explicitly asks for it.
    """

    rows: tuple["ResultRowLike", ...]
    next: ContinuationToken | None
    previous: ContinuationToken | None
    total: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.rows, tuple):
            object.__setattr__(self, "rows", tuple(self.rows))
        if self.total is not None and (
            isinstance(self.total, bool) or not isinstance(self.total, int) or self.total < 0
        ):
            raise ValueError("ResultPage.total must be a non-negative integer or None")


class PageableResultSetLike(Protocol):
    """Optional continuation-page capability of a frozen result set.

    This deliberately extends neither :class:`ResultSetLike` nor
    :class:`Searcher`: stores that do not support seek pagination remain fully
    conforming to the required portable contracts.
    """

    def page(
        self,
        *,
        size: int,
        order_by: Iterable[PageOrder],
        cursor: ContinuationToken | None = None,
        include_total: bool = False,
    ) -> ResultPage: ...


class ResultRow:
    """One named result row, usable by position, name, or attribute."""

    __slots__ = ("_guard", "_names", "_resolver", "_values")

    def __init__(
        self,
        values: tuple[Any, ...],
        names: tuple[str, ...],
        resolver: Any = None,
        guard: Any = None,
    ) -> None:
        self._values = values
        self._names = names
        self._resolver = resolver
        self._guard = guard

    @property
    def names(self) -> tuple[str, ...]:
        self._check()
        return self._names

    @property
    def values(self) -> tuple[Any, ...]:
        self._check()
        return tuple(self[index] for index in range(len(self._values)))

    def _check(self) -> None:
        if self._guard is not None:
            self._guard()

    def _value(self, index: int) -> Any:
        if self._guard is not None:
            self._guard()
        value = self._resolver(index, self._values[index]) if self._resolver is not None else self._values[index]
        activate = getattr(value, "_activate", None)
        if activate is not None:
            activate()
        return value

    def __getitem__(self, key: int | str) -> Any:
        if isinstance(key, str):
            try:
                key = self._names.index(key)
            except ValueError:
                raise KeyError(key) from None
        if not isinstance(key, int):
            raise TypeError(f"result row indices must be integers or strings, got {type(key).__name__}")
        return self._value(key)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None

    def __iter__(self) -> Iterator[Any]:
        return iter(self.values)

    def __len__(self) -> int:
        self._check()
        return len(self._values)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ResultRow):
            return NotImplemented
        return self.values == other.values

    def __repr__(self) -> str:
        return f"ResultRow({dict(zip(self.names, self.values, strict=True))!r})"

    def __copy__(self) -> "ResultRow":
        if self._guard is not None:
            raise TypeError("expired cursor rows cannot be copied")
        return type(self)(self._values, self._names, self._resolver, self._guard)

    def __deepcopy__(self, memo: dict[int, Any]) -> "ResultRow":
        if self._guard is not None:
            raise TypeError("expired cursor rows cannot be copied")
        return type(self)(self._values, self._names, self._resolver, self._guard)

    def __reduce_ex__(self, protocol: Any) -> Any:
        if self._guard is not None:
            raise TypeError("expired cursor rows cannot be pickled")
        return type(self), (self._values, self._names, self._resolver, self._guard)


class ResultRowLike(Protocol):
    @property
    def names(self) -> tuple[str, ...]: ...

    def __getitem__(self, key: int | str) -> Any: ...


class ResultSetLike(Protocol):
    def __iter__(self) -> Iterator[ResultRowLike]: ...

    def __len__(self) -> int: ...

    def first(self) -> ResultRowLike | None: ...

    def one(self) -> ResultRowLike: ...

    def scalars(self, name: str | None = None) -> Iterator[Any]: ...

    # column() is an optional backend capability; SQL stores expose it.


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

    def has(self, value: Any) -> SearchExpression:
        """Match a list field containing ``value``."""
        ...

    def has_any(self, *values: Any) -> SearchExpression: ...

    def has_only(self, *values: Any) -> SearchExpression: ...

    def is_in(self, *values: Any) -> SearchExpression:
        """Match a root scalar field whose value is one of ``values``.

        ``None`` is an explicit member: it matches a null field value, and its
        negation excludes nulls rather than inheriting SQL's three-valued
        ``NOT IN (..., NULL)`` behavior.

        Backends define the corresponding semantics for child or set fields;
        for example, a backend may use the existing ``has_only``-style
        all-values reading for a child field.
        """
        ...

    def contains(self, text: str) -> SearchExpression:
        """Match values containing ``text`` as a literal substring."""
        ...

    def startswith(self, prefix: str) -> SearchExpression:
        """Match values beginning with the literal ``prefix``."""
        ...

    def endswith(self, suffix: str) -> SearchExpression:
        """Match values ending with the literal ``suffix``."""
        ...

    def __getattr__(self, name: str) -> "SearchField":
        """The field ``name`` of the record this field refers to.

        A field may hold a *reference* to another stored record, and attribute
        access chains into that record's own fields — ``variable.ref.doi`` —
        to any depth. Whether a given field is a reference is a property of the
        stored class, not of this contract, so the chain is only resolved when
        the query is built: a backend raises :class:`AttributeError` for a name
        that is not a field of the referenced record, and for chaining off a
        field that refers to nothing.
        """
        ...


class SearchVariable(Protocol):
    """A query variable bound to a target type; attribute access yields fields.

    ``always_true``/``always_false`` are reserved names: they are real methods
    of the variable, never stored fields resolved through ``__getattr__``.
    They exist so a translation layer can express a constant truth value
    without inventing a probe field. A ``field == field`` probe is NULL-unsound,
    since it yields NULL (not true) for a NULL field.
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

    def results(self, **outputs: Any) -> ResultSetLike: ...


class Store(Protocol):
    def searcher(self) -> Searcher: ...
