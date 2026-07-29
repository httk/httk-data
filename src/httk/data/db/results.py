"""Lazy, named result rows for :class:`~httk.data.db.searcher.SqlSearcher`."""

import copy
import dataclasses
from array import array
from collections.abc import Iterator
from typing import Any, cast

import sqlalchemy
from httk.core import FracScalar, FracVector

from httk.data.db.mapping import SID_COLUMN
from httk.data.db.rows import RowHydrator
from httk.data.db.schema import FieldSpec
from httk.data.db.searcher import SqlSearcher, _Output
from httk.data.query import ResultRow

__all__ = [
    "ExpiredCursorRowError",
    "MultipleResultsError",
    "NoResultError",
    "ResultColumn",
    "ResultRow",
    "SqlResultSet",
]

_CHUNK = 500


class NoResultError(LookupError):
    """Raised by :meth:`SqlResultSet.one` when there are no matches."""


class MultipleResultsError(LookupError):
    """Raised by :meth:`SqlResultSet.one` when there is more than one match."""


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
    """A declared scalar projection, with exact and float presentations."""

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
        self._result._ensure()
        for position in self._result._positions:
            yield self._result._float_at(position, self._index)

    def to_fracvector(self) -> FracVector:
        output = self._result._plan._outputs[self._index]
        rational = output.codec is not None and output.codec.name in {"fraction", "fracscalar"}
        rational = rational or (output.spec is not None and output.spec.role == "scalar" and output.spec.python_type is int)
        if not rational:
            raise TypeError(f"column {self.name!r} is not rational-valued; surds and other codecs are unsupported")
        values = [value.to_fraction() if isinstance(value, FracScalar) else value for value in self]
        return FracVector.create(values)


class SqlResultSet:
    """A frozen, lazy result plan."""

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
        return self._names

    def _state(self) -> "SqlResultSet":
        return cast(SqlResultSet, self._root)

    def _columns(self) -> tuple[list[sqlalchemy.ColumnElement[Any]], int]:
        plan = self._plan
        columns = [output.element for output in plan._outputs]
        for extras in self._projection_extras:
            columns.extend(extras)
        hidden = len(columns)
        columns.extend(cast(sqlalchemy.ColumnElement[Any], variable._alias.c[SID_COLUMN]) for variable in plan._variables)
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
        group_columns += [output.element for output in self._plan._outputs if output.target is None and not output.from_child]
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
        with self._plan._store._read_connection() as connection:
            return tuple(tuple(row) for row in connection.execute(self._statement(limit=limit)).fetchall())

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
        return ResultRow(raw[: len(state._names)], state._names, lambda index, _value: state._value_at(position, index), guard)

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
            return value
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
        extra_by_name = {extra.name: state._rows[position][offset + extra_index] for extra_index, extra in enumerate(extras)}
        for codec_index, (suffix, _kind) in enumerate(output.codec.columns):
            if codec_index == query_index:
                continue
            parts[codec_index] = extra_by_name[output.spec.field + suffix]
        return None if all(part is None for part in parts) else output.codec.decode(tuple(parts))

    def _float_at(self, position: int, index: int) -> Any:
        state = self._state()
        return state._rows[position][index] if state._rows is not None else None

    def __iter__(self) -> Iterator[ResultRow]:
        self._ensure()
        state = self._state()
        for position in self._positions:
            yield state._row_at(position)

    def first(self) -> ResultRow | None:
        if self._root is not self:
            return None if not self._positions else self._state()._row_at(self._positions[0])
        rows = self._execute(limit=1)
        if not rows:
            return None
        result = SqlResultSet(self._plan)
        result._prepare(rows)
        return result._row_at(0)

    def one(self) -> ResultRow:
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
        if name is None:
            if len(self.names) != 1:
                raise ValueError(f"scalars() without a name requires exactly one output; declared: {self.names}")
            name = self.names[0]
        index = self.names.index(name) if name in self.names else -1
        if index < 0:
            raise KeyError(f"unknown output {name!r}; declared: {self.names}")
        return (row[index] for row in self)

    def column(self, name: str) -> ResultColumn:
        scalar_names = tuple(
            output.name for output in self._plan._outputs if output.target is None
        )
        if name not in self.names:
            raise KeyError(f"unknown column {name!r}; declared scalar projections: {scalar_names}")
        index = self.names.index(name)
        if self._plan._outputs[index].target is not None:
            raise TypeError(f"column {name!r} is an object output; declared scalar projections: {scalar_names}")
        return ResultColumn(self, index)

    def cursor(self) -> Iterator[ResultRow]:
        self._ensure()
        state = self._state()
        generation = _Generation()
        proxy_indices = [index for index, output in enumerate(state._plan._outputs) if output.target is not None]
        proxies = {
            index: _proxy_class(cast(type, state._plan._outputs[index].target))(
                state._hydrators[index], generation
            )
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
    from httk.data.db.codecs import decode_fracvector_exact

    assert spec.shape is not None
    return decode_fracvector_exact(text, spec.shape.rows, spec.shape.cols)
