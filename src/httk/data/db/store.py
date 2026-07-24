"""The SQL store: save and fetch storable frozen dataclasses through a :class:`~httk.data.db.engine.Database`.

:class:`SqlStore` is the object-level storage API on top of the schema IR
(:mod:`httk.data.db.schema`), the value codecs (:mod:`httk.data.db.codecs`),
the content identity (:mod:`httk.data.db.identity`), and the SQLAlchemy table
mapping (:mod:`httk.data.db.mapping`):

- :meth:`SqlStore.save` writes an instance (recursing into referenced and
  child-element storables) and returns its integer ``sid``, deduplicating per
  the class's :attr:`~httk.core.StorageInfo.dedup` policy;
- :meth:`SqlStore.fetch` reconstructs the instance stored under a ``sid`` —
  exactly, via the ``*_exact`` companion columns for rationals — with an
  identity guarantee: while an instance is alive, fetching its sid again
  returns the very same object;
- :meth:`SqlStore.transaction` scopes several operations into one database
  transaction (commit on exit, roll back on exception); outside of it every
  operation autocommits;
- :meth:`SqlStore.referring` finds join-objects (tags, references) pointing at
  a stored instance, replacing v1's implicit codependent-data machinery;
- :meth:`SqlStore.searcher` starts a query through the search DSL
  (:mod:`httk.data.db.searcher`), implementing the :mod:`httk.data.query`
  protocols.

Deduplication semantics (ported from v1): under ``"content_id"`` an equal
instance maps to the existing row (children are not re-inserted); under
``"by_value"`` a row matching **all parent-table columns** is reused — child
table contents are *not* part of the match, mirroring v1 which matched key
columns only; under ``"none"`` every save inserts a new row.

Two small, documented liberties: an optional child-table field saved as
``None`` comes back as an empty container (the relational layout cannot tell
the two apart), and the identity/sid caches are best-effort — instances that
cannot be weak-referenced or hashed (e.g. records holding lists) are simply
not cached, which affects only :meth:`SqlStore.sid_of` and the same-object
guarantee, never storage correctness.
"""

import contextlib
import threading
import typing
import weakref
from collections.abc import Iterable, Iterator
from typing import Any, cast

import sqlalchemy
from httk.core import FracVector, Shape

from httk.data.db.codecs import (
    codec_named,
    decode_fracvector_exact,
    encode_fracvector_exact,
    encode_fracvector_floats,
)
from httk.data.db.engine import Database
from httk.data.db.identity import content_id
from httk.data.db.mapping import CONTENT_ID_COLUMN, SID_COLUMN, table_for
from httk.data.db.schema import FieldSpec, SchemaError, TableSchema, resolve_schema
from httk.data.db.searcher import SqlSearcher

__all__ = [
    "SqlStore",
]


class SqlStore:
    """Object storage for storable frozen dataclasses in a relational :class:`~httk.data.db.engine.Database`.

    Tables are created on demand (first save/fetch of a class, or explicitly
    via :meth:`ensure_tables`) unless ``create_tables=False``, in which case the
    schema is expected to exist already.
    """

    def __init__(self, database: Database, *, create_tables: bool = True) -> None:
        self._database = database
        self._create_tables = create_tables
        self._metadata = sqlalchemy.MetaData()
        self._instances: weakref.WeakValueDictionary[tuple[type, int], Any] = weakref.WeakValueDictionary()
        self._sids: weakref.WeakKeyDictionary[Any, int] = weakref.WeakKeyDictionary()
        self._local = threading.local()

    # ------------------------------------------------------------------ tables and transactions

    def ensure_tables(self, *classes: type) -> None:
        """Resolve each class's schema and create its tables (and those it references, transitively).

        Existing tables are left alone. Saving or fetching calls this
        implicitly, but calling it up front is useful to separate DDL from a
        data transaction (SQLite rolls back tables created inside a rolled-back
        transaction along with the data).
        """
        self._ensure_tables(self._current_connection(), classes)

    def transaction(self) -> contextlib.AbstractContextManager[None]:
        """Scope the operations of a ``with`` block into one database transaction.

        Every :meth:`save`/:meth:`fetch` (on this thread) inside the block runs
        on the same open connection; the transaction commits when the block
        exits normally and rolls back if it raises. On a rollback the identity
        caches are flushed, since they may name rows that no longer exist.
        Nesting is flat: an inner ``transaction()`` block simply joins the
        outer transaction. Outside any transaction block, each operation runs
        (and autocommits) on its own.
        """
        return self._transaction_scope()

    @contextlib.contextmanager
    def _transaction_scope(self) -> Iterator[None]:
        stack = self._connection_stack()
        if stack:
            yield
            return
        try:
            with self._database.engine.begin() as connection:
                stack.append(connection)
                try:
                    yield
                finally:
                    stack.pop()
        except BaseException:
            self._instances.clear()
            self._sids.clear()
            raise

    def _connection_stack(self) -> list[sqlalchemy.Connection]:
        stack = getattr(self._local, "stack", None)
        if stack is None:
            stack = []
            self._local.stack = stack
        return cast(list[sqlalchemy.Connection], stack)

    def _current_connection(self) -> sqlalchemy.Connection | None:
        stack = self._connection_stack()
        return stack[-1] if stack else None

    @contextlib.contextmanager
    def _write_connection(self) -> Iterator[sqlalchemy.Connection]:
        current = self._current_connection()
        if current is not None:
            yield current
            return
        with self._database.engine.begin() as connection:
            yield connection

    @contextlib.contextmanager
    def _read_connection(self) -> Iterator[sqlalchemy.Connection]:
        current = self._current_connection()
        if current is not None:
            yield current
            return
        with self._database.engine.connect() as connection:
            yield connection

    def _ensure_tables(self, connection: sqlalchemy.Connection | None, classes: Iterable[type]) -> None:
        before = len(self._metadata.tables)
        for cls in classes:
            table_for(resolve_schema(cls), self._metadata)
        if not self._create_tables or len(self._metadata.tables) == before:
            return
        if connection is not None:
            self._metadata.create_all(connection, checkfirst=True)
        else:
            self._metadata.create_all(self._database.engine, checkfirst=True)

    def _table(self, name: str) -> sqlalchemy.Table:
        return self._metadata.tables[name]

    # ------------------------------------------------------------------ saving

    def save(self, obj: Any) -> int:
        """Store ``obj`` (deduplicating per its class's policy) and return its integer sid.

        Referenced storables and storable child elements are saved recursively
        first; when deduplication finds an existing row, that row's sid is
        returned and no children are re-inserted.
        """
        with self._write_connection() as connection:
            return self._save(connection, obj)

    def _save(self, connection: sqlalchemy.Connection, obj: Any) -> int:
        cls = type(obj)
        schema = resolve_schema(cls)
        self._ensure_tables(connection, (cls,))
        table = self._table(schema.table_name)

        key: str | None = None
        if schema.dedup == "content_id":
            key = content_id(obj)
            found = connection.execute(
                sqlalchemy.select(table.c[SID_COLUMN]).where(table.c[CONTENT_ID_COLUMN] == key)
            ).first()
            if found is not None:
                sid = int(found[0])
                self._remember(cls, sid, obj)
                return sid

        values = self._parent_row(connection, schema, obj)

        if schema.dedup == "by_value":
            # v1 semantics: a by_value match compares the parent table's stored
            # columns only; child-table contents are not part of the match.
            conditions = [
                table.c[name].is_(None) if value is None else table.c[name] == value for name, value in values.items()
            ]
            statement = sqlalchemy.select(table.c[SID_COLUMN])
            if conditions:
                statement = statement.where(*conditions)
            found = connection.execute(statement.limit(1)).first()
            if found is not None:
                sid = int(found[0])
                self._remember(cls, sid, obj)
                return sid

        if key is not None:
            values[CONTENT_ID_COLUMN] = key
        insert = sqlalchemy.insert(table).values(values) if values else sqlalchemy.insert(table)
        result = connection.execute(insert)
        sid = int(cast(Any, result.inserted_primary_key)[0])
        for spec in schema.fields:
            if spec.role == "child":
                self._insert_child_rows(connection, schema, spec, sid, getattr(obj, spec.field))
        self._remember(cls, sid, obj)
        return sid

    def _parent_row(self, connection: sqlalchemy.Connection, schema: TableSchema, obj: Any) -> dict[str, Any]:
        """Encode the parent-table column values of ``obj``, saving referenced storables recursively."""
        values: dict[str, Any] = {}
        for spec in schema.fields:
            if spec.role == "child":
                continue
            value = getattr(obj, spec.field)
            if value is None:
                for column in spec.columns:
                    values[column.name] = None
            elif spec.role == "scalar":
                values[spec.columns[0].name] = value
            elif spec.role == "encoded":
                assert spec.codec_name is not None
                encoded = codec_named(spec.codec_name).encode(value)
                for column, part in zip(spec.columns, encoded, strict=True):
                    values[column.name] = part
            elif spec.role == "fixed_array":
                assert spec.shape is not None
                tensor = _as_fixed_tensor(schema, spec, spec.shape, value)
                for i, part in enumerate(encode_fracvector_floats(tensor)):
                    values[f"{spec.field}_{i}"] = part
                values[f"{spec.field}_exact"] = encode_fracvector_exact(tensor)
            else:  # reference
                values[spec.columns[0].name] = self._save(connection, value)
        return values

    def _insert_child_rows(
        self, connection: sqlalchemy.Connection, schema: TableSchema, spec: FieldSpec, sid: int, value: Any
    ) -> None:
        assert spec.child is not None
        table = self._table(spec.child.table_name)
        parent_column = f"{schema.table_name}_sid"
        index_column = f"{spec.field}_index"
        rows: list[dict[str, Any]] = []
        if spec.shape is not None:
            for position, row_tensor in enumerate(_tensor_rows(schema, spec, spec.shape, value)):
                row: dict[str, Any] = {parent_column: sid, index_column: position}
                for i, part in enumerate(encode_fracvector_floats(row_tensor)):
                    row[f"{spec.field}_{i}"] = part
                row[f"{spec.field}_exact"] = encode_fracvector_exact(row_tensor)
                rows.append(row)
        else:
            codec = codec_named(spec.codec_name) if spec.codec_name is not None else None
            for position, element in enumerate(value if value is not None else ()):
                row = {parent_column: sid, index_column: position}
                if spec.target is not None:
                    row[spec.child.element_columns[0].name] = self._save(connection, element)
                elif codec is not None:
                    for column, part in zip(spec.child.element_columns, codec.encode(element), strict=True):
                        row[column.name] = part
                else:
                    row[spec.child.element_columns[0].name] = element
                rows.append(row)
        if rows:
            connection.execute(sqlalchemy.insert(table), rows)

    # ------------------------------------------------------------------ fetching

    def fetch[T](self, cls: type[T], sid: int) -> T:
        """Reconstruct the ``cls`` instance stored under ``sid``.

        While a previously fetched (or saved) instance for this ``(class,
        sid)`` is alive, the very same object is returned. Raises
        :class:`KeyError` (carrying the class and sid) when no such row exists.
        """
        with self._read_connection() as connection:
            return cast(T, self._fetch(connection, cls, sid))

    def fetch_by_content_id[T](self, cls: type[T], key: str) -> T | None:
        """The ``cls`` instance whose content identity is ``key``, or None if not stored.

        Only classes with the ``"content_id"`` dedup policy carry a content
        identity column; :class:`~httk.data.db.schema.SchemaError` is raised
        for any other class.
        """
        schema = resolve_schema(cls)
        if schema.dedup != "content_id":
            raise SchemaError(
                f"{cls.__name__} has dedup policy {schema.dedup!r}; only classes with the "
                f"'content_id' policy have a content identity column"
            )
        with self._read_connection() as connection:
            self._ensure_tables(connection, (cls,))
            table = self._table(schema.table_name)
            found = connection.execute(
                sqlalchemy.select(table.c[SID_COLUMN]).where(table.c[CONTENT_ID_COLUMN] == key)
            ).first()
            if found is None:
                return None
            return cast(T, self._fetch(connection, cls, int(found[0])))

    def sid_of(self, obj: Any) -> int | None:
        """The sid under which ``obj`` was saved or fetched through this store, else None.

        This consults the in-memory reverse cache only — it never probes the
        database — so it knows exactly the instances that passed through
        :meth:`save` or :meth:`fetch` (and, being weak and equality-keyed, is
        best-effort for instances that cannot be hashed or weak-referenced).
        """
        try:
            return self._sids.get(obj)
        except TypeError:
            return None

    def searcher(self) -> SqlSearcher:
        """A new :class:`~httk.data.db.searcher.SqlSearcher` querying this store.

        The searcher runs on this store's read path — inside an open
        :meth:`transaction` block it sees uncommitted writes — and
        reconstructs matched objects through :meth:`fetch`, so the identity
        cache applies.
        """
        return SqlSearcher(self)

    def referring(self, cls: type, *, field: str, to: Any) -> list[Any]:
        """All stored ``cls`` instances whose reference field ``field`` points at ``to``.

        ``field`` must be a reference field of ``cls`` targeting ``to``'s class
        (:class:`~httk.data.db.schema.SchemaError` otherwise), and ``to`` must
        be known to this store — saved or fetched through it — else
        :class:`ValueError` is raised. Results are ordered by sid.
        """
        schema = resolve_schema(cls)
        spec = schema.field(field)
        if spec.role != "reference":
            raise SchemaError(f"{cls.__name__}.{field} is not a reference field (its role is {spec.role!r})")
        assert spec.target is not None
        if not isinstance(to, spec.target):
            raise SchemaError(f"{cls.__name__}.{field} references {spec.target.__name__}, not {type(to).__name__}")
        sid = self.sid_of(to)
        if sid is None:
            raise ValueError(f"the {type(to).__name__} instance has not been stored or fetched through this store")
        with self._read_connection() as connection:
            self._ensure_tables(connection, (cls,))
            table = self._table(schema.table_name)
            found = connection.execute(
                sqlalchemy.select(table.c[SID_COLUMN])
                .where(table.c[spec.columns[0].name] == sid)
                .order_by(table.c[SID_COLUMN])
            ).all()
            return [self._fetch(connection, cls, int(row[0])) for row in found]

    def _fetch(self, connection: sqlalchemy.Connection, cls: type, sid: int) -> Any:
        sid = int(sid)
        cached = self._instances.get((cls, sid))
        if cached is not None:
            return cached
        schema = resolve_schema(cls)
        self._ensure_tables(connection, (cls,))
        table = self._table(schema.table_name)
        row = connection.execute(sqlalchemy.select(table).where(table.c[SID_COLUMN] == sid)).mappings().first()
        if row is None:
            raise KeyError(cls, sid)
        values: dict[str, Any] = {}
        for spec in schema.fields:
            if spec.derived:
                continue
            values[spec.field] = self._decode_field(connection, schema, spec, sid, row)
        instance = cls(**values)
        self._remember(cls, sid, instance)
        return instance

    def _decode_field(
        self,
        connection: sqlalchemy.Connection,
        schema: TableSchema,
        spec: FieldSpec,
        sid: int,
        row: sqlalchemy.RowMapping,
    ) -> Any:
        if spec.role == "scalar":
            return row[spec.columns[0].name]
        if spec.role == "encoded":
            assert spec.codec_name is not None
            parts = tuple(row[column.name] for column in spec.columns)
            if all(part is None for part in parts):
                return None
            return codec_named(spec.codec_name).decode(parts)
        if spec.role == "fixed_array":
            assert spec.shape is not None
            exact = row[f"{spec.field}_exact"]
            if exact is None:
                return None
            # The float companion columns exist for querying only; the exact
            # text is the round-trip source of truth.
            return decode_fracvector_exact(exact, spec.shape.rows, spec.shape.cols)
        if spec.role == "reference":
            target_sid = row[spec.columns[0].name]
            if target_sid is None:
                return None
            assert spec.target is not None
            return self._fetch(connection, spec.target, int(target_sid))
        return self._fetch_child(connection, schema, spec, sid)

    def _fetch_child(self, connection: sqlalchemy.Connection, schema: TableSchema, spec: FieldSpec, sid: int) -> Any:
        assert spec.child is not None
        table = self._table(spec.child.table_name)
        parent_column = f"{schema.table_name}_sid"
        index_column = f"{spec.field}_index"
        result = (
            connection.execute(
                sqlalchemy.select(table).where(table.c[parent_column] == sid).order_by(table.c[index_column])
            )
            .mappings()
            .all()
        )
        if spec.shape is not None:
            fractions_rows = [
                decode_fracvector_exact(entry[f"{spec.field}_exact"], 1, spec.shape.cols).to_fractions()[0]
                for entry in result
            ]
            return FracVector.create(fractions_rows)
        elements: list[Any]
        if spec.target is not None:
            elements = [
                self._fetch(connection, spec.target, int(entry[spec.child.element_columns[0].name])) for entry in result
            ]
        elif spec.codec_name is not None:
            codec = codec_named(spec.codec_name)
            elements = [
                codec.decode(tuple(entry[column.name] for column in spec.child.element_columns)) for entry in result
            ]
        else:
            elements = [entry[spec.child.element_columns[0].name] for entry in result]
        if typing.get_origin(spec.python_type) is tuple:
            return tuple(elements)
        return elements

    # ------------------------------------------------------------------ identity caches

    def _remember(self, cls: type, sid: int, obj: Any) -> None:
        try:
            self._instances[(cls, sid)] = obj
        except TypeError:
            return  # Not weak-referenceable; identity caching is best-effort.
        try:
            self._sids[obj] = sid
        except TypeError:
            pass  # Not hashable (e.g. holds a list); the reverse cache is best-effort.


def _as_fixed_tensor(schema: TableSchema, spec: FieldSpec, shape: Shape, value: Any) -> FracVector:
    """Normalize a fixed-shape field value to a ``(rows, cols)`` FracVector, validating its shape."""
    tensor = FracVector.use(value)
    dim = tensor.dim
    if dim == (shape.rows, shape.cols):
        return tensor
    if shape.rows == 1 and dim == (shape.cols,):
        return FracVector((tensor.noms,), tensor.denom)
    raise ValueError(
        f"{schema.cls.__name__}.{spec.field}: expected a FracVector of shape "
        f"({shape.rows}, {shape.cols}), got {dim}"
    )


def _tensor_rows(schema: TableSchema, spec: FieldSpec, shape: Shape, value: Any) -> list[FracVector]:
    """The rows of a variable-rows (``Shape(0, c)``) field value, each as a ``(c,)`` FracVector."""
    if value is None:
        return []
    tensor = FracVector.use(value)
    dim = tensor.dim
    if dim == () or dim == (0,):
        return []
    if len(dim) != 2 or dim[1] != shape.cols:
        raise ValueError(
            f"{schema.cls.__name__}.{spec.field}: expected a FracVector with {shape.cols} columns per row, "
            f"got shape {dim}"
        )
    rows = cast(tuple[tuple[int, ...], ...], tensor.noms)  # dim was validated two-dimensional above
    return [FracVector(noms_row, tensor.denom) for noms_row in rows]
