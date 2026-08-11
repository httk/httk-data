"""Lazy, named result rows for :class:`~httk.store.db.searcher.SqlSearcher`."""

import copy
import dataclasses
from array import array
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any, Final, Literal, cast

import sqlalchemy
from httk.core import FracScalar, FracVector

from httk.store.db.mapping import SID_COLUMN
from httk.store.db.rows import RowHydrator
from httk.store.db.schema import FieldSpec, resolve_schema
from httk.store.db.searcher import SqlSearcher, _Output
from httk.store.query import (
    ContinuationToken,
    MultipleResultsError,
    NoResultError,
    PageOrder,
    ResultPage,
    ResultRow,
    UnsupportedQueryError,
)
from httk.store.query.paging_tokens import (
    _decode_continuation,
    _DecodedContinuation,
    _encode_continuation,
    _plan_fingerprint,
)

__all__ = [
    "ExpiredCursorRowError",
    "MultipleResultsError",
    "NoResultError",
    "ResultColumn",
    "ResultRow",
    "SqlResultSet",
]

_CHUNK = 500
_PAGE_SIZE_MAX: Final = 10_000
_PAGE_ORDER_MAX: Final = 32


@dataclass(frozen=True, slots=True)
class _PageKey:
    """A validated root scalar projection and its continuation ordering."""

    order: PageOrder
    output: _Output


class ExpiredCursorRowError(RuntimeError):
    """A cursor proxy was used after the cursor advanced."""


class _Generation:
    def __init__(self) -> None:
        self.value = 0


class _CursorProxy:
    __httk_cursor_proxy__ = True
    __hash__ = None  # type: ignore[assignment]  # pyright: ignore[reportIncompatibleMethodOverride]

    def __init__(self, hydrator: RowHydrator, generation: _Generation) -> None:
        object.__setattr__(self, "_hydrator", hydrator)
        object.__setattr__(self, "_generation", generation)
        object.__setattr__(self, "_sid", None)
        object.__setattr__(self, "_bound_generation", -1)
        object.__setattr__(self, "_access_generation", -1)

    def _bind(self, sid: int) -> None:
        object.__setattr__(self, "_sid", int(sid))
        object.__setattr__(self, "_bound_generation", self._generation.value)
        object.__setattr__(self, "_access_generation", -1)

    def _activate(self) -> None:
        self._check_bound()
        object.__setattr__(self, "_access_generation", self._generation.value)

    def _check_bound(self) -> int:
        if self._sid is None or self._bound_generation != self._generation.value:
            raise ExpiredCursorRowError("cursor row expired; use it before advancing the cursor")
        return self._sid

    def _check(self) -> int:
        if self._access_generation != self._generation.value:
            raise ExpiredCursorRowError("cursor row expired; use it before advancing the cursor")
        return self._check_bound()

    def _field(self, name: str) -> Any:
        sid = self._check()
        return getattr(self._hydrator.row(sid), name)

    def __getattribute__(self, name: str) -> Any:
        if not name.startswith("_"):
            fields = object.__getattribute__(self, "_cursor_fields")
            if name in fields:
                return object.__getattribute__(self, "_field")(name)
        return object.__getattribute__(self, name)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return self._field(name)

    def __repr__(self) -> str:
        self._check()
        return f"{type(self).__name__}(sid={self._sid})"

    def __eq__(self, other: object) -> bool:
        self._check()
        return NotImplemented

    def __copy__(self) -> Any:
        raise TypeError("cursor rows cannot be copied; they expire when the cursor advances")

    def __deepcopy__(self, memo: dict[int, Any]) -> Any:
        raise TypeError("cursor rows cannot be copied; they expire when the cursor advances")

    def __reduce__(self) -> Any:
        raise TypeError("cursor rows cannot be pickled or saved")

    def __reduce_ex__(self, protocol: Any) -> Any:
        raise TypeError("cursor rows cannot be pickled or saved")

    def __getstate__(self) -> Any:
        raise TypeError("cursor rows cannot be pickled or saved")


def _proxy_class(cls: type) -> type:
    return type(
        f"{cls.__name__}CursorProxy",
        (_CursorProxy, cls),
        {
            "__module__": cls.__module__,
            "_cursor_fields": frozenset(field.name for field in dataclasses.fields(cls)),
        },
    )


class ResultColumn:
    """A declared scalar projection, with exact and float presentations.

    :param result: The result set containing the projection.
    :param index: The projection's position in the result set.
    """

    def __init__(self, result: "SqlResultSet", index: int) -> None:
        self._result = result
        self._index = index
        self.name = result.names[index]

    def __len__(self) -> int:
        return len(self._result)

    def __iter__(self) -> Iterator[Any]:
        self._result._ensure()
        for position in self._result._positions:
            yield self._result._value_at(position, self._index)

    def floats(self) -> Iterator[Any]:
        """Yield the query-domain values for this projection.

        :yield: A projection value in the query-domain representation.
        """
        self._result._ensure()
        for position in self._result._positions:
            yield self._result._float_at(position, self._index)

    def to_fracvector(self) -> FracVector:
        """Convert this rational projection to a fraction vector.

        :return: The projection values as a fraction vector.
        :raises TypeError: If the projection is not a supported rational column.
        """
        output = self._result._plan._outputs[self._index]
        rational = output.codec is not None and output.codec.name in {"fraction", "fracscalar"}
        rational = rational or (
            output.spec is not None and output.spec.role == "scalar" and output.spec.python_type is int
        )
        if not rational:
            raise TypeError(f"column {self.name!r} is not rational-valued; surds and other codecs are unsupported")
        values = [value.to_fraction() if isinstance(value, FracScalar) else value for value in self]
        return FracVector(values)


class SqlResultSet:
    """A frozen, lazy result plan.

    :param searcher: The search whose query state is frozen.
    :param outputs: Optional output names mapped to query variables or columns.
    """

    def __init__(self, searcher: SqlSearcher, outputs: dict[str, Any] | None = None) -> None:
        self._plan = copy.copy(searcher)
        self._plan._variables = list(searcher._variables)
        self._plan._where = list(searcher._where)
        self._plan._having = list(searcher._having)
        self._plan._sorts = list(searcher._sorts)
        self._plan._outputs = []
        if outputs:
            for name, value in outputs.items():
                self._plan.output(value, name)
        else:
            self._plan._outputs = list(searcher._outputs)
        for output in self._plan._outputs:
            if output.target is None and output.spec is not None and output.spec.role == "child":
                raise TypeError(
                    f"projection {output.name!r} for {output.spec.field!r} is a variable-length child field; "
                    "child projections are not supported by results()"
                )
        self._names = tuple(output.name for output in self._plan._outputs)
        self._projection_extras = tuple(self._extra_columns(output) for output in self._plan._outputs)
        self._extra_offsets: tuple[int, ...] = tuple(
            sum(len(extra) for extra in self._projection_extras[:index]) for index in range(len(self._names))
        )
        self._hidden_start = len(self._names) + sum(len(extra) for extra in self._projection_extras)
        self._root = self
        self._positions: tuple[int, ...] = ()
        self._rows: tuple[tuple[Any, ...], ...] | None = None
        self._hydrators: dict[int, RowHydrator] = {}
        self._object_index: array | None = None
        self._proxies: dict[int, _CursorProxy] = {}
        self._proxy_classes: dict[int, type] = {}

    @property
    def names(self) -> tuple[str, ...]:
        """Return the names of the declared projections in order.

        :return: The declared projection names.
        """
        return self._names

    def _state(self) -> "SqlResultSet":
        return cast(SqlResultSet, self._root)

    def _columns(self) -> tuple[list[sqlalchemy.ColumnElement[Any]], int]:
        plan = self._plan
        columns = [output.element for output in plan._outputs]
        for extras in self._projection_extras:
            columns.extend(extras)
        hidden = len(columns)
        columns.extend(
            cast(sqlalchemy.ColumnElement[Any], variable._alias.c[SID_COLUMN]) for variable in plan._variables
        )
        return columns, hidden

    @staticmethod
    def _extra_columns(output: _Output) -> tuple[sqlalchemy.ColumnElement[Any], ...]:
        if output.target is not None or output.spec is None or output.variable is None:
            return ()
        if output.spec.role == "encoded":
            return tuple(
                output.variable._alias.c[column.name]
                for column in output.spec.columns
                if column.name != output.element.name
            )
        if output.spec.role == "fixed_array" and output.exact_element is not None:
            return (output.exact_element,)
        return ()

    def _statement(self, *, limit: int | None = None) -> sqlalchemy.Select[Any]:
        columns, _hidden = self._columns()
        group_columns = [cast(sqlalchemy.ColumnElement[Any], v._alias.c[SID_COLUMN]) for v in self._plan._variables]
        group_columns += [
            output.element for output in self._plan._outputs if output.target is None and not output.from_child
        ]
        group_columns += [extra for extras in self._projection_extras for extra in extras]
        statement = self._plan._base_select(columns, group_columns)
        for column, descending in self._plan._sorts:
            statement = statement.order_by(column._element.desc() if descending else column._element.asc())
        if limit is None:
            limit = self._plan._limit
        elif self._plan._limit is not None:
            limit = min(limit, self._plan._limit)
        if limit is not None:
            statement = statement.limit(limit)
        if self._plan.offset > 0:
            statement = statement.offset(self._plan.offset)
        return statement

    def _execute(self, *, limit: int | None = None) -> tuple[tuple[Any, ...], ...]:
        if not self._plan._outputs:
            raise ValueError("this search has no outputs; declare outputs or pass them to results()")
        if self._plan._vacuous:
            return ()
        with self._plan._store._read_connection() as connection:
            statement = self._statement(limit=limit)
            rows = connection.execute(statement).fetchall()
        if self._plan._store._database.engine.dialect.name == "clickhousedb":
            from httk.store.db.clickhouse import normalize_clickhouse_value

            columns, _hidden = self._columns()
            return tuple(
                tuple(normalize_clickhouse_value(value, column.type) for value, column in zip(row, columns))
                for row in rows
            )
        return tuple(tuple(row) for row in rows)

    def _prepare(self, rows: tuple[tuple[Any, ...], ...]) -> None:
        self._rows = rows
        self._positions = tuple(range(len(rows)))
        self._hydrators = {}
        object_indices = [index for index, output in enumerate(self._plan._outputs) if output.target is not None]
        for index in object_indices:
            target = cast(type, self._plan._outputs[index].target)
            sids = [int(row[index]) for row in rows if row[index] is not None]
            self._hydrators[index] = RowHydrator(self._plan._store, target, array("q", sids))

    def _ensure(self) -> None:
        state = self._state()
        if state._rows is not None:
            return
        rows = state._execute()
        state._prepare(rows)
        object_indices = [index for index, output in enumerate(state._plan._outputs) if output.target is not None]
        if len(object_indices) == 1:
            # Keep the Phase-1 compact sole-object index explicit; rows still retain
            # scalar query values needed by ResultRow without another outer query.
            index = object_indices[0]
            state._object_index = array("q", (int(row[index]) for row in rows if row[index] is not None))

    def __len__(self) -> int:
        self._ensure()
        return len(self._positions)

    def __getitem__(self, item: int | slice) -> "ResultRow | SqlResultSet":
        self._ensure()
        state = self._state()
        if isinstance(item, slice):
            view = object.__new__(SqlResultSet)
            view._plan = state._plan
            view._names = state._names
            view._projection_extras = state._projection_extras
            view._extra_offsets = state._extra_offsets
            view._hidden_start = state._hidden_start
            view._root = state
            view._rows = state._rows
            view._positions = tuple(self._positions[item])
            view._hydrators = state._hydrators
            view._proxies = {}
            view._proxy_classes = state._proxy_classes
            view._object_index = state._object_index
            return view
        position = self._positions[item]
        return state._row_at(position)

    def _row_at(self, position: int, *, values: tuple[Any, ...] | None = None, guard: Any = None) -> ResultRow:
        state = self._state()
        assert state._rows is not None
        raw = state._rows[position] if values is None else values
        return ResultRow(
            raw[: len(state._names)], state._names, lambda index, _value: state._value_at(position, index), guard
        )

    def _value_at(self, position: int, index: int) -> Any:
        state = self._state()
        output = state._plan._outputs[index]
        assert state._rows is not None
        value = state._rows[position][index]
        if output.target is not None:
            if value is None:
                return None
            return state._hydrators[index].row(int(value))
        if output.spec is None or output.spec.role == "scalar":
            return output.presentation_converter(value) if output.presentation_converter is not None else value
        offset = len(state._names) + state._extra_offsets[index]
        extras = state._projection_extras[index]
        if output.spec.role == "fixed_array":
            text = state._rows[position][offset]
            return None if text is None else _decode_fixed(text, output.spec)
        assert output.codec is not None
        query_index = next(
            index for index, (suffix, _kind) in enumerate(output.codec.columns) if suffix == output.codec.query_suffix
        )
        parts: list[Any] = [None] * len(output.codec.columns)
        parts[query_index] = value
        extra_by_name = {
            extra.name: state._rows[position][offset + extra_index] for extra_index, extra in enumerate(extras)
        }
        for codec_index, (suffix, _kind) in enumerate(output.codec.columns):
            if codec_index == query_index:
                continue
            parts[codec_index] = extra_by_name[output.spec.field + suffix]
        return None if all(part is None for part in parts) else output.codec.decode(tuple(parts))

    def _float_at(self, position: int, index: int) -> Any:
        state = self._state()
        if state._rows is None:
            return None
        value = state._rows[position][index]
        output = state._plan._outputs[index]
        return output.presentation_converter(value) if output.presentation_converter is not None else value

    def __iter__(self) -> Iterator[ResultRow]:
        self._ensure()
        state = self._state()
        for position in self._positions:
            yield state._row_at(position)

    def first(self) -> ResultRow | None:
        """Return the first result, or ``None`` when no result exists.

        :return: The first result row, if present.
        """
        if self._root is not self:
            return None if not self._positions else self._state()._row_at(self._positions[0])
        rows = self._execute(limit=1)
        if not rows:
            return None
        result = SqlResultSet(self._plan)
        result._prepare(rows)
        return result._row_at(0)

    def one(self) -> ResultRow:
        """Return the only result.

        :return: The sole result row.
        :raises ~httk.store.query.NoResultError: If no result exists.
        :raises ~httk.store.query.MultipleResultsError: If more than one result exists.
        """
        if self._root is not self:
            if not self._positions:
                raise NoResultError("expected exactly one result, found none")
            if len(self._positions) > 1:
                raise MultipleResultsError("expected exactly one result, found more than one")
            return self._state()._row_at(next(iter(self._positions)))
        rows = self._execute(limit=2)
        if not rows:
            raise NoResultError("expected exactly one result, found none")
        if len(rows) > 1:
            raise MultipleResultsError("expected exactly one result, found more than one")
        result = SqlResultSet(self._plan)
        result._prepare(rows)
        return result._row_at(0)

    def scalars(self, name: str | None = None) -> Iterator[Any]:
        """Yield one projection from each result row.

        :param name: The projection name, required when several projections are declared.
        :return: An iterator over the selected projection values.
        :raises ValueError: If ``name`` is omitted while several projections are declared.
        :raises KeyError: If ``name`` is not a declared projection.
        """
        if name is None:
            if len(self.names) != 1:
                raise ValueError(f"scalars() without a name requires exactly one output; declared: {self.names}")
            name = self.names[0]
        index = self.names.index(name) if name in self.names else -1
        if index < 0:
            raise KeyError(f"unknown output {name!r}; declared: {self.names}")
        return (row[index] for row in self)

    def column(self, name: str) -> ResultColumn:
        """Return a scalar projection by name.

        :param name: The declared scalar projection name.
        :return: The selected result column.
        :raises KeyError: If ``name`` is not declared.
        :raises TypeError: If ``name`` names an object projection.
        """
        scalar_names = tuple(output.name for output in self._plan._outputs if output.target is None)
        if name not in self.names:
            raise KeyError(f"unknown column {name!r}; declared scalar projections: {scalar_names}")
        index = self.names.index(name)
        if self._plan._outputs[index].target is not None:
            raise TypeError(f"column {name!r} is an object output; declared scalar projections: {scalar_names}")
        return ResultColumn(self, index)

    def page(
        self,
        *,
        size: int,
        order_by: Iterable[PageOrder],
        cursor: ContinuationToken | None = None,
        include_total: bool = False,
    ) -> ResultPage:
        """Fetch one bounded, live keyset page of this frozen SQL result plan.

        Paging is intentionally separate from :meth:`cursor`: this method
        returns independent rows whose page-local hydration state remains alive
        through their result-row resolvers, while ``cursor()`` returns reusable
        proxies that expire on advance.  It performs one ``LIMIT size + 1``
        match query and never materializes this result set's ordinary match
        cache.  A total is deliberately opt-in because it requires the normal
        exact count query.

        :param size: The maximum number of rows in the page.
        :param order_by: The declared scalar projections used for keyset ordering.
        :param cursor: The continuation token identifying the page position.
        :param include_total: Whether to include the exact filtered total.
        :return: The requested result page.
        :raises ~httk.store.query.UnsupportedQueryError: If the query or ordering is not pageable.
        """
        self._validate_page_size(size)
        if not isinstance(include_total, bool):
            raise TypeError("include_total must be bool")
        keys = self._page_keys(order_by)
        fingerprint = self._page_fingerprint(keys)
        decoded: _DecodedContinuation | None = None
        if cursor is not None:
            decoded = _decode_continuation(cursor, fingerprint=fingerprint, anchors=len(keys))
        if self._plan._vacuous:
            return ResultPage((), None, None, total=0 if include_total else None)

        statement, raw_width = self._page_statement(keys, decoded, size)
        with self._plan._store._read_connection() as connection:
            fetched = tuple(tuple(row) for row in connection.execute(statement).fetchall())
        if self._plan._store._database.engine.dialect.name == "clickhousedb":
            from httk.store.db.clickhouse import normalize_clickhouse_value

            fetched = tuple(
                tuple(
                    normalize_clickhouse_value(value, column.type)
                    for value, column in zip(row, statement.selected_columns)
                )
                for row in fetched
            )

        more_in_fetch_direction = len(fetched) > size
        visible = fetched[:size]
        if decoded is not None and decoded.direction == "backward":
            visible = tuple(reversed(visible))
        rows = self._page_rows(visible, raw_width)
        next_token, previous_token = self._page_tokens(
            visible,
            raw_width,
            keys,
            fingerprint,
            decoded,
            more_in_fetch_direction,
        )
        total = self._plan.count() if include_total else None
        return ResultPage(rows, next_token, previous_token, total)

    @staticmethod
    def _validate_page_size(size: int) -> None:
        if isinstance(size, bool) or not isinstance(size, int):
            raise TypeError("page size must be an integer (bool is not accepted)")
        if not 1 <= size <= _PAGE_SIZE_MAX:
            raise ValueError(f"page size must be between 1 and {_PAGE_SIZE_MAX}")

    def _page_keys(self, order_by: Iterable[PageOrder]) -> tuple[_PageKey, ...]:
        """Validate the intentionally small SQL seek-pagination profile."""
        if len(self._plan._variables) != 1:
            raise UnsupportedQueryError("paging requires exactly one root query variable")
        if self._plan._sorts:
            raise UnsupportedQueryError("paging does not compose with add_sort(); pass PageOrder values instead")
        if self._plan.offset != 0:
            raise UnsupportedQueryError("paging does not compose with a nonzero query offset")
        if self._plan._limit is not None:
            raise UnsupportedQueryError("paging does not compose with a query limit")
        if not self._plan._outputs:
            raise ValueError("this search has no outputs; declare outputs or pass them to results()")
        if isinstance(order_by, (str, bytes)):
            raise TypeError("order_by must be an iterable of PageOrder values")
        try:
            iterator = iter(order_by)
        except TypeError as error:
            raise TypeError("order_by must be an iterable of PageOrder values") from error
        requested: list[PageOrder] = []
        for order in iterator:
            if len(requested) >= _PAGE_ORDER_MAX:
                raise UnsupportedQueryError(f"paging supports at most {_PAGE_ORDER_MAX} order keys")
            requested.append(order)
        root = self._plan._variables[0]
        keys: list[_PageKey] = []
        seen: set[str] = set()
        for order in requested:
            if not isinstance(order, PageOrder):
                raise TypeError(f"order_by entries must be PageOrder values, got {type(order).__name__}")
            if order.name in seen:
                raise UnsupportedQueryError(f"duplicate paging order name {order.name!r}")
            seen.add(order.name)
            matching = [
                (index, output) for index, output in enumerate(self._plan._outputs) if output.name == order.name
            ]
            if not matching:
                raise UnsupportedQueryError(f"paging order {order.name!r} is not a declared result projection")
            if len(matching) > 1:
                raise UnsupportedQueryError(f"paging order {order.name!r} names duplicate result projections")
            _index, output = matching[0]
            if output.target is not None:
                raise UnsupportedQueryError(f"paging order {order.name!r} is an object projection, not a scalar")
            if output.from_child or output.variable is not root:
                raise UnsupportedQueryError(
                    f"paging order {order.name!r} must be a scalar projection of the root query variable"
                )
            if output.spec is not None and output.spec.role not in {"scalar", "encoded"}:
                raise UnsupportedQueryError(f"paging order {order.name!r} has an unsupported stored representation")
            keys.append(_PageKey(order, output))
        return tuple(keys)

    def _page_inner(self, keys: tuple[_PageKey, ...]) -> tuple[sqlalchemy.Subquery, tuple[str, ...], tuple[str, ...]]:
        """Build the grouped one-match-per-root inner relation.

        Child-list filters make the searcher grouped.  The inner query retains
        that grouping/HAVING logic and exposes one root row with all requested
        reconstruction columns, normalized order columns, and the root sid.
        The outer query can then seek safely without changing child-filter
        semantics or multiplying a root into several page rows.
        """
        root = self._plan._variables[0]
        root_sid = cast(sqlalchemy.ColumnElement[Any], root._alias.c[SID_COLUMN])
        columns: list[sqlalchemy.ColumnElement[Any]] = []
        raw_names: list[str] = []
        for index, output in enumerate(self._plan._outputs):
            name = f"_httk_page_value_{index}"
            columns.append(output.element.label(name))
            raw_names.append(name)
        for index, extras in enumerate(self._projection_extras):
            for extra_index, extra in enumerate(extras):
                name = f"_httk_page_extra_{index}_{extra_index}"
                columns.append(extra.label(name))
                raw_names.append(name)
        sid_name = "_httk_page_sid"
        columns.append(root_sid.label(sid_name))
        raw_names.append(sid_name)
        key_names: list[str] = []
        for index, key in enumerate(keys):
            name = f"_httk_page_key_{index}"
            columns.append(key.output.element.label(name))
            key_names.append(name)

        group_columns = [root_sid]
        group_columns += [
            output.element for output in self._plan._outputs if output.target is None and not output.from_child
        ]
        group_columns += [extra for extras in self._projection_extras for extra in extras]
        group_columns += [key.output.element for key in keys]
        return (
            self._plan._base_select(columns, group_columns).subquery("_httk_page_matches"),
            tuple(raw_names),
            tuple(key_names),
        )

    def _page_statement(
        self,
        keys: tuple[_PageKey, ...],
        cursor: _DecodedContinuation | None,
        size: int,
    ) -> tuple[sqlalchemy.Select[Any], int]:
        inner, raw_names, key_names = self._page_inner(keys)
        raw_columns = [cast(sqlalchemy.ColumnElement[Any], inner.c[name]) for name in raw_names]
        key_columns = [cast(sqlalchemy.ColumnElement[Any], inner.c[name]) for name in key_names]
        sid = raw_columns[-1]
        statement = sqlalchemy.select(*raw_columns, *key_columns).select_from(inner)
        backward = cursor is not None and cursor.direction == "backward"
        if cursor is not None:
            statement = statement.where(self._page_seek_predicate(key_columns, sid, keys, cursor, before=backward))
        for key, column in zip(keys, key_columns, strict=True):
            rank = self._page_null_rank(column, key.order)
            statement = statement.order_by(rank.desc() if backward else rank.asc())
            descending = key.order.descending != backward
            statement = statement.order_by(column.desc() if descending else column.asc())
        statement = statement.order_by(sid.desc() if backward else sid.asc()).limit(size + 1)
        return statement, len(raw_columns)

    def _page_null_rank(self, column: sqlalchemy.ColumnElement[Any], order: PageOrder) -> sqlalchemy.ColumnElement[Any]:
        if self._plan._store._database.engine.dialect.name == "clickhousedb":
            from httk.store.db.clickhouse import null_order_rank

            return null_order_rank(column, order.nulls, dialect_name="clickhousedb")
        null_rank = 0 if order.nulls == "first" else 1
        value_rank = 1 - null_rank
        return sqlalchemy.case((column.is_(None), null_rank), else_=value_rank)

    def _page_seek_predicate(
        self,
        columns: list[sqlalchemy.ColumnElement[Any]],
        sid: sqlalchemy.ColumnElement[Any],
        keys: tuple[_PageKey, ...],
        cursor: _DecodedContinuation,
        *,
        before: bool,
    ) -> sqlalchemy.ColumnElement[bool]:
        """Return the lexicographic strict-before/after predicate for an anchor."""
        prefix: list[sqlalchemy.ColumnElement[bool]] = []
        choices: list[sqlalchemy.ColumnElement[bool]] = []
        for key, column, anchor in zip(keys, columns, cursor.anchors, strict=True):
            rank = self._page_null_rank(column, key.order)
            anchor_rank = 0 if (anchor is None) == (key.order.nulls == "first") else 1
            rank_compare = rank < anchor_rank if before else rank > anchor_rank
            comparisons: list[sqlalchemy.ColumnElement[bool]] = [rank_compare]
            if anchor is not None:
                # The null-rank is always ascending in public order, while
                # the value component follows PageOrder.descending.  Moving
                # before/after the anchor therefore reverses a descending
                # value comparison but never its null-rank comparison.
                value_compare = column > anchor if key.order.descending == before else column < anchor
                comparisons.append(sqlalchemy.and_(rank == anchor_rank, value_compare))
            choices.append(sqlalchemy.and_(*prefix, sqlalchemy.or_(*comparisons)))
            prefix.append(rank == anchor_rank)
            prefix.append(column.is_(None) if anchor is None else column == anchor)
        sid_compare = sid < cursor.sid if before else sid > cursor.sid
        choices.append(sqlalchemy.and_(*prefix, sid_compare))
        return sqlalchemy.or_(*choices)

    def _page_rows(self, raw_rows: tuple[tuple[Any, ...], ...], raw_width: int) -> tuple[ResultRow, ...]:
        """Attach page-scoped hydrators without touching this result's cache."""
        if not raw_rows:
            return ()
        result = SqlResultSet(self._plan)
        result._prepare(tuple(row[:raw_width] for row in raw_rows))
        return tuple(result._row_at(index) for index in range(len(raw_rows)))

    def _page_tokens(
        self,
        rows: tuple[tuple[Any, ...], ...],
        raw_width: int,
        keys: tuple[_PageKey, ...],
        fingerprint: str,
        cursor: _DecodedContinuation | None,
        more_in_fetch_direction: bool,
    ) -> tuple[ContinuationToken | None, ContinuationToken | None]:
        if not rows:
            return None, None

        def token(row: tuple[Any, ...], direction: "Literal['forward', 'backward']") -> ContinuationToken:
            try:
                return _encode_continuation(
                    direction=direction,
                    anchors=tuple(row[raw_width : raw_width + len(keys)]),
                    sid=int(row[self._hidden_start]),
                    fingerprint=fingerprint,
                )
            except (TypeError, ValueError) as error:
                raise UnsupportedQueryError(
                    "paging order values cannot be represented in a continuation cursor"
                ) from error

        first, last = rows[0], rows[-1]
        if cursor is None:
            return (token(last, "forward") if more_in_fetch_direction else None, None)
        if cursor.direction == "forward":
            return (
                token(last, "forward") if more_in_fetch_direction else None,
                token(first, "backward"),
            )
        return (
            token(last, "forward"),
            token(first, "backward") if more_in_fetch_direction else None,
        )

    def _page_fingerprint(self, keys: tuple[_PageKey, ...]) -> str:
        """Hash a stable description of query structure without exposing SQL in a token."""
        inner, _raw_names, _key_names = self._page_inner(keys)
        dialect = self._plan._store._database.engine.dialect
        compiled = sqlalchemy.select(inner).compile(dialect=dialect)
        root = self._plan._variables[0]
        payload = {
            "version": 1,
            "dialect": {
                "name": dialect.name,
                "driver": dialect.driver,
                "class": f"{type(dialect).__module__}.{type(dialect).__qualname__}",
            },
            "schema": self._page_schema(root._schema),
            "outputs": [
                {
                    "name": output.name,
                    "target": None
                    if output.target is None
                    else f"{output.target.__module__}.{output.target.__qualname__}",
                    "root": output.variable is root,
                    "child": output.from_child,
                    "field": None if output.spec is None else output.spec.field,
                    "role": None if output.spec is None else output.spec.role,
                }
                for output in self._plan._outputs
            ],
            "order": [
                {"name": key.order.name, "descending": key.order.descending, "nulls": key.order.nulls} for key in keys
            ],
            "statement": compiled.string,
            "parameters": compiled.params,
        }
        return _plan_fingerprint(payload)

    @staticmethod
    def _page_schema(schema: Any, seen: set[type] | None = None) -> dict[str, Any]:
        """A structural schema description, including reachable reference schemas."""
        seen = set() if seen is None else seen
        cls = schema.cls
        identity = f"{cls.__module__}.{cls.__qualname__}"
        if cls in seen:
            return {"class": identity, "cycle": True}
        seen.add(cls)
        return {
            "class": identity,
            "table": schema.table_name,
            "dedup": schema.dedup,
            "indexes": schema.composite_indexes,
            "fields": [
                {
                    "field": field.field,
                    "role": field.role,
                    "columns": [
                        {
                            "name": column.name,
                            "kind": column.kind,
                            "nullable": column.nullable,
                            "indexed": column.indexed,
                            "unique": column.unique,
                        }
                        for column in field.columns
                    ],
                    "codec": field.codec_name,
                    "shape": None if field.shape is None else (field.shape.rows, field.shape.cols),
                    "target": None
                    if field.target is None
                    else SqlResultSet._page_schema(resolve_schema(field.target), seen),
                }
                for field in schema.fields
            ],
        }

    def cursor(self) -> Iterator[ResultRow]:
        """Yield cursor rows whose object proxies expire when iteration advances.

        :return: An iterator over live cursor result rows.
        """
        self._ensure()
        state = self._state()
        generation = _Generation()
        proxy_indices = [index for index, output in enumerate(state._plan._outputs) if output.target is not None]
        proxies = {
            index: _proxy_class(cast(type, state._plan._outputs[index].target))(state._hydrators[index], generation)
            for index in proxy_indices
        }

        def rows() -> Iterator[ResultRow]:
            assert state._rows is not None
            for position in self._positions:
                generation.value += 1
                raw = state._rows[position]
                values = list(raw[: len(state._names)])
                for index in proxy_indices:
                    proxy = proxies[index]
                    if raw[index] is None:
                        values[index] = None
                    else:
                        proxy._bind(int(raw[index]))
                        values[index] = proxy
                for index in range(len(values)):
                    if index not in proxy_indices:
                        values[index] = state._value_at(position, index)
                current = generation.value
                yield ResultRow(
                    tuple(values),
                    state._names,
                    guard=lambda current=current: _cursor_guard(generation, current),
                )

        return rows()


def _cursor_guard(generation: _Generation, current: int) -> None:
    if generation.value != current:
        raise ExpiredCursorRowError("cursor row expired; use it before advancing the cursor")


def _decode_fixed(text: str, spec: FieldSpec) -> FracVector:
    from httk.store.db.codecs import decode_fracvector_exact

    assert spec.shape is not None
    return decode_fracvector_exact(text, spec.shape.rows, spec.shape.cols)
