"""Batched storage rows with lazy, exact field reconstruction."""

import dataclasses
import functools
import typing
import weakref
from array import array
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import sqlalchemy
from httk.core import FracVector

from httk.data.db.codecs import codec_named, decode_fracvector_exact
from httk.data.db.mapping import SID_COLUMN
from httk.data.db.schema import FieldSpec, SchemaError, TableSchema, resolve_schema

if typing.TYPE_CHECKING:
    from httk.data.db.store import SqlStore

__all__ = ["RowHydrator", "StaleResultError", "decode_field", "row_class"]

_CHUNK = 500
_ROW_CHUNK = "_httk_row_chunk_6f4a"
_ROW_SID = "_httk_row_sid_6f4a"
_ROW_STORE = "_httk_row_store_6f4a"
_ROW_BASE = "_httk_row_base_6f4a"


class StaleResultError(RuntimeError):
    """A search result sid disappeared before its lazy row was hydrated."""


class _Context:
    def __init__(self) -> None:
        self.rows: weakref.WeakValueDictionary[tuple[type, int], Any] = weakref.WeakValueDictionary()
        self.hydrators: list[RowHydrator] = []
        self.in_progress: set[tuple[type, int]] = set()

    def find(self, cls: type, sid: int) -> "RowHydrator | None":
        return next(
            (hydrator for hydrator in self.hydrators if hydrator._cls is cls and sid in hydrator._positions), None
        )


class _Chunk:
    def __init__(self, hydrator: "RowHydrator", index: int, sids: tuple[int, ...]) -> None:
        self.hydrator = hydrator
        self.index = index
        self.sids = sids
        self.parent_rows: dict[int, tuple[Any, ...]] = {}
        self.columns: dict[str, int] = {}
        self.children: dict[str, dict[int, list[tuple[Any, ...]]]] = {}
        self.child_columns: dict[str, dict[str, int]] = {}
        self.references: dict[str, dict[int, RowHydrator]] = {}

        store = hydrator._store
        schema = hydrator._schema
        store._ensure_tables(None, (hydrator._cls,))
        table = store._table(schema.table_name)
        with store._read_connection() as connection:
            result = connection.execute(sqlalchemy.select(table).where(table.c[SID_COLUMN].in_(sids))).fetchall()
        self.columns = {column.name: index for index, column in enumerate(table.columns)}
        for row in result:
            self.parent_rows[int(row[self.columns[SID_COLUMN]])] = tuple(row)
        for sid in sids:
            if sid not in self.parent_rows:
                raise StaleResultError(f"{schema.cls.__name__} sid {sid} is no longer present")

    def _child_rows(self, spec: FieldSpec) -> tuple[dict[int, list[tuple[Any, ...]]], dict[str, int]]:
        found = self.children.get(spec.field)
        if found is not None:
            return found, self.child_columns[spec.field]
        assert spec.child is not None
        table = self.hydrator._store._table(spec.child.table_name)
        parent_column = f"{self.hydrator._schema.table_name}_sid"
        index_column = f"{spec.field}_index"
        with self.hydrator._store._read_connection() as connection:
            result = connection.execute(
                sqlalchemy.select(table)
                .where(table.c[parent_column].in_(self.sids))
                .order_by(table.c[parent_column], table.c[index_column])
            ).fetchall()
        columns = {column.name: index for index, column in enumerate(table.columns)}
        grouped: dict[int, list[tuple[Any, ...]]] = defaultdict(list)
        for row in result:
            grouped[int(row[columns[parent_column]])].append(tuple(row))
        self.children[spec.field] = grouped
        self.child_columns[spec.field] = columns
        return grouped, columns

    def value(self, sid: int, spec: FieldSpec, *, eager: bool = False) -> Any:
        row = self.parent_rows[sid]
        if spec.role == "scalar":
            return row[self.columns[spec.columns[0].name]]
        if spec.role == "encoded":
            assert spec.codec_name is not None
            parts = tuple(row[self.columns[column.name]] for column in spec.columns)
            return None if all(part is None for part in parts) else codec_named(spec.codec_name).decode(parts)
        if spec.role == "fixed_array":
            exact = row[self.columns[f"{spec.field}_exact"]]
            if exact is None:
                return None
            assert spec.shape is not None
            # The float columns are query-only; exact text is the round-trip source.
            return decode_fracvector_exact(exact, spec.shape.rows, spec.shape.cols)
        if spec.role == "reference":
            target_sid = row[self.columns[spec.columns[0].name]]
            if target_sid is None:
                return None
            assert spec.target is not None
            references = self.references.get(spec.field)
            if references is None:
                target_sids = [
                    int(parent[self.columns[spec.columns[0].name]])
                    for parent in self.parent_rows.values()
                    if parent[self.columns[spec.columns[0].name]] is not None
                ]
                references = self._target_map(spec.target, target_sids)
                self.references[spec.field] = references
            target = references[int(target_sid)]
            return target.materialize(int(target_sid)) if eager else target.row(int(target_sid))
        return self._child_value(sid, spec, eager=eager)

    def _target_map(self, cls: type, sids: Sequence[int]) -> dict[int, "RowHydrator"]:
        missing = [
            sid for sid in dict.fromkeys(int(sid) for sid in sids) if self.hydrator._context.find(cls, sid) is None
        ]
        if missing:
            RowHydrator(self.hydrator._store, cls, missing, context=self.hydrator._context)
        result = {int(sid): self.hydrator._context.find(cls, int(sid)) for sid in dict.fromkeys(sids)}
        assert all(hydrator is not None for hydrator in result.values())
        return typing.cast(dict[int, RowHydrator], result)

    def _child_value(self, sid: int, spec: FieldSpec, *, eager: bool) -> Any:
        grouped, columns = self._child_rows(spec)
        entries = grouped.get(sid, [])
        assert spec.child is not None
        if spec.shape is not None:
            rows = [
                decode_fracvector_exact(entry[columns[f"{spec.field}_exact"]], 1, spec.shape.cols).to_fractions()[0]
                for entry in entries
            ]
            return FracVector.create(rows)
        if spec.target is not None:
            target_sids = [int(entry[columns[spec.child.element_columns[0].name]]) for entry in entries]
            all_target_sids = [
                int(entry[columns[spec.child.element_columns[0].name]])
                for child_entries in grouped.values()
                for entry in child_entries
            ]
            targets = self._target_map(spec.target, all_target_sids) if all_target_sids else {}
            elements = [
                targets[target_sid].materialize(target_sid) if eager else targets[target_sid].row(target_sid)
                for target_sid in target_sids
            ]
        elif spec.codec_name is not None:
            codec = codec_named(spec.codec_name)
            elements = [
                codec.decode(tuple(entry[columns[column.name]] for column in spec.child.element_columns))
                for entry in entries
            ]
        else:
            elements = [entry[columns[spec.child.element_columns[0].name]] for entry in entries]
        return tuple(elements) if typing.get_origin(spec.python_type) is tuple else elements


class RowHydrator:
    """Hydrate a sid sequence in 500-row batches, without touching field values yet."""

    def __init__(
        self,
        store: "SqlStore",
        schema_or_cls: TableSchema | type,
        sids: Sequence[int],
        *,
        context: _Context | None = None,
    ) -> None:
        self._store = store
        self._schema = schema_or_cls if isinstance(schema_or_cls, TableSchema) else resolve_schema(schema_or_cls)
        self._cls = self._schema.cls
        self._sids = array("q", (int(sid) for sid in sids))
        self._positions = {sid: index for index, sid in enumerate(self._sids)}
        self._chunks: weakref.WeakValueDictionary[int, _Chunk] = weakref.WeakValueDictionary()
        self._context = context or _Context()
        self._context.hydrators.append(self)

    def _chunk_for(self, sid: int) -> tuple[int, _Chunk]:
        try:
            position = self._positions[int(sid)]
        except KeyError:
            raise KeyError((self._cls, sid)) from None
        index = position // _CHUNK
        chunk = self._chunks.get(index)
        if chunk is None:
            start = index * _CHUNK
            chunk = _Chunk(self, index, tuple(self._sids[start : start + _CHUNK]))
            self._chunks[index] = chunk
        return index, chunk

    def row(self, sid: int) -> Any:
        sid = int(sid)
        existing = self._context.rows.get((self._cls, sid))
        if existing is not None:
            return existing
        _index, chunk = self._chunk_for(sid)
        instance: Any = object.__new__(row_class(self._cls))
        object.__setattr__(instance, _ROW_STORE, self._store)
        object.__setattr__(instance, _ROW_SID, sid)
        object.__setattr__(instance, _ROW_CHUNK, chunk)
        self._context.rows[(self._cls, sid)] = instance
        return instance

    def materialize(self, sid: int) -> Any:
        sid = int(sid)
        key = (self._cls, sid)
        cached = self._store._instances.get(key)
        if cached is not None:
            self._context.rows[key] = cached
            return cached
        existing = self._context.rows.get((self._cls, sid))
        if existing is not None and type(existing) is self._cls:
            return existing
        if key in self._context.in_progress:
            raise SchemaError(f"cyclic eager hydration of {self._cls.__name__} sid {sid}")
        row = self.row(sid)
        self._context.in_progress.add(key)
        try:
            values = {
                spec.field: row._httk_decode(spec, eager=True) for spec in self._schema.fields if not spec.derived
            }
            instance = self._cls(**values)
            self._context.rows[key] = instance
            self._store._remember(self._cls, sid, instance)
            return instance
        finally:
            self._context.in_progress.discard(key)


def decode_field(store: "SqlStore", schema: TableSchema, spec: FieldSpec, sid: int, row: Any) -> Any:
    """Decode one pinned parent row; child and reference fields use the row's chunk."""
    if spec.role == "scalar":
        return row[spec.columns[0].name]
    if spec.role == "encoded":
        assert spec.codec_name is not None
        parts = tuple(row[column.name] for column in spec.columns)
        return None if all(part is None for part in parts) else codec_named(spec.codec_name).decode(parts)
    if spec.role == "fixed_array":
        exact = row[f"{spec.field}_exact"]
        if exact is None:
            return None
        assert spec.shape is not None
        return decode_fracvector_exact(exact, spec.shape.rows, spec.shape.cols)
    raise TypeError(f"{spec.role} fields need their RowHydrator chunk")


class _Field:
    def __init__(self, spec: FieldSpec) -> None:
        self.spec = spec

    def __get__(self, instance: Any, owner: type | None = None) -> Any:
        if instance is None:
            return self
        cached = instance.__dict__.get(self.spec.field, _MISSING)
        if cached is not _MISSING:
            return cached
        value = instance._httk_decode(self.spec)
        # Frozen base dataclasses reject normal assignment; object.__setattr__ is
        # the required cache write and does not invoke the inherited frozen setter.
        object.__setattr__(instance, self.spec.field, value)
        return value


class _DefaultField:
    def __init__(self, field: dataclasses.Field[Any]) -> None:
        self.field = field

    def __get__(self, instance: Any, owner: type | None = None) -> Any:
        if instance is None:
            return self
        cached = instance.__dict__.get(self.field.name, _MISSING)
        if cached is not _MISSING:
            return cached
        if self.field.default_factory is not dataclasses.MISSING:
            value = self.field.default_factory()
        elif self.field.default is not dataclasses.MISSING:
            value = self.field.default
        else:
            raise AttributeError(f"{self.field.name!r} has no default")
        object.__setattr__(instance, self.field.name, value)
        return value


_MISSING = object()


def _row_decode(self: Any, spec: FieldSpec, *, eager: bool = False) -> Any:
    chunk = self.__dict__.get(_ROW_CHUNK)
    if chunk is None:
        return self.__dict__[spec.field]
    return chunk.value(self.__dict__[_ROW_SID], spec, eager=eager)


@functools.cache
def row_class(cls: type) -> type:
    """Return the cached lazy subclass for a frozen storable dataclass."""
    if "__slots__" in cls.__dict__:
        raise SchemaError(f"{cls.__name__}: lazy storage rows do not support slots dataclasses")
    resolve_schema(cls)
    params: Any = cls.__dict__["__dataclass_params__"]
    for name in ("__eq__", "__hash__"):
        method = cls.__dict__.get(name)
        custom_eq = name == "__eq__" and not params.eq
        if method is not None and (custom_eq or not _is_generated_dataclass_method(method)):
            raise SchemaError(f"{cls.__name__}: lazy rows do not support custom {name}")

    row_type: type
    dataclass_fields = dataclasses.fields(cls)
    compare_fields = tuple(field for field in dataclass_fields if field.compare)
    hash_fields = tuple(
        field for field in dataclass_fields if field.hash is True or (field.hash is None and field.compare)
    )
    repr_fields = tuple(field for field in dataclass_fields if field.repr)
    schema_fields = {spec.field for spec in resolve_schema(cls).fields}

    def eq(self: Any, other: Any) -> bool:
        if type(other) is not cls and type(other) is not row_type:
            return NotImplemented
        return tuple(getattr(self, field.name) for field in compare_fields) == tuple(
            getattr(other, field.name) for field in compare_fields
        )

    def ne(self: Any, other: Any) -> bool:
        result = eq(self, other)
        return NotImplemented if result is NotImplemented else not result

    def row_hash(self: Any) -> int:
        return hash(tuple(getattr(self, field.name) for field in hash_fields))

    def row_repr(self: Any) -> str:
        values = ", ".join(f"{field.name}={getattr(self, field.name)!r}" for field in repr_fields)
        return f"{cls.__qualname__}({values})"

    def sid(self: Any) -> int | None:
        return self.__dict__.get(_ROW_SID)

    attrs: dict[str, Any] = {
        "__module__": cls.__module__,
        "__httk_row_base__": cls,
        _ROW_BASE: cls,
        "__eq__": eq,
        "__ne__": ne,
        "__hash__": row_hash,
        "__repr__": row_repr,
        "sid": property(sid),
        "_httk_decode": _row_decode,
        "__copy__": lambda self: _reject_copy("copy.copy"),
        "__deepcopy__": lambda self, memo: _reject_copy("copy.deepcopy"),
        "__reduce_ex__": lambda self, protocol: _reject_copy("pickle"),
    }
    for spec in resolve_schema(cls).fields:
        if not spec.derived:
            attrs[spec.field] = _Field(spec)
    for field in dataclass_fields:
        if field.name not in schema_fields:
            attrs[field.name] = _DefaultField(field)
    row_type = type(f"{cls.__name__}Row", (cls,), attrs)
    return row_type


def _is_generated_dataclass_method(method: Any) -> bool:
    code = getattr(method, "__code__", None)
    return code is not None and code.co_filename == "<string>"


def _reject_copy(operation: str) -> Any:
    raise TypeError(f"lazy storage rows do not support {operation}; materialize with store.fetch first")


def lazy_row_identity(obj: Any) -> tuple[Any, int] | None:
    """Return a lazy row's owning store and sid, for the store reverse lookup."""
    if _ROW_STORE not in getattr(obj, "__dict__", {}):
        return None
    return obj.__dict__[_ROW_STORE], int(obj.__dict__[_ROW_SID])


def is_lazy_row(obj: Any) -> bool:
    return lazy_row_identity(obj) is not None
