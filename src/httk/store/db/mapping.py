"""Schema-to-SQL mapping: build SQLAlchemy Core tables from resolved :class:`~httk.store.db.schema.TableSchema` IR.

:func:`table_for` turns one resolved schema into a :class:`sqlalchemy.Table`
registered in a :class:`sqlalchemy.MetaData` (idempotently: an already-built
table is returned as-is), recursing into referenced and child-element storable
classes so that the complete logical layout is present in the same metadata.
:func:`sqlalchemy_metadata` is the convenience wrapper that maps a batch of
schemas into one fresh metadata.

The relational layout produced here is exactly the one the schema IR
documents, plus the store-managed columns:

- every parent table gets an ``sid`` integer primary key (autoincrementing,
  with an attached ``<table>_sid_seq`` sequence for dialects such as DuckDB
  that need one; SQLite ignores it) and — only under the ``"content_id"``
  dedup policy — a unique-indexed ``content_id`` text column;
- every child table gets a ``<parent table>_sid`` integer sid column (NOT NULL,
  indexed) and a ``<field>_index`` integer ordering column ahead of its element
  columns; logical references are defined by :mod:`httk.store.db.graph`.

Index names are deterministic and table-scoped — ``ix_<table>_<column>`` for
plain indexes, ``uq_<table>_<column>`` for unique ones, columns joined by
underscores for composites — truncated with a stable hash suffix when they
would exceed common identifier-length limits.
"""

import hashlib
from collections.abc import Iterable, Sequence
from typing import Any, Final, Literal

import sqlalchemy
from sqlalchemy import text

from httk.store.db.codecs import ScalarKind
from httk.store.db.schema import (
    ChildTableSpec,
    ColumnSpec,
    FieldSpec,
    TableSchema,
    resolve_schema,
)

__all__ = [
    "CONTENT_ID_COLUMN",
    "DISPATCH_CONTENT_ID_COLUMN",
    "REPLACED_BY_COLUMN",
    "ROLE_COLUMN",
    "SID_COLUMN",
    "TS_END_COLUMN",
    "TS_START_COLUMN",
    "backing_dispatch_column_name",
    "dispatch_table_for",
    "entry_dispatch_table_name",
    "sqlalchemy_metadata",
    "table_for",
]

SID_COLUMN: Final = "sid"
"""The store-managed integer primary-key column present on every table."""

CONTENT_ID_COLUMN: Final = "content_id"
"""The store-managed content-identity column of tables with the ``"content_id"`` dedup policy."""

ROLE_COLUMN: Final = "_httk_role"
"""The permanentization role of a parent record — ``0`` dependency, ``1`` main."""

TS_START_COLUMN: Final = "ts_start"
"""The store-managed integer creation timestamp on every parent record table."""

TS_END_COLUMN: Final = "ts_end"
"""The store-managed end-of-life timestamp on versioned family tables (NULL = current)."""

REPLACED_BY_COLUMN: Final = "replaced_by_sid"
"""The store-managed successor sid on versioned family tables (fsck-only consumer, no index)."""

type _TimestampMode = Literal["off", "creation", "versioned"]

DISPATCH_CONTENT_ID_COLUMN: Final = "content_id"
"""The content identity primary key of an entry-family dispatch table."""

_MAX_IDENTIFIER_LENGTH: Final = 63

_TYPE_FOR_KIND: Final[dict[ScalarKind, type[sqlalchemy.types.TypeEngine[Any]]]] = {
    "int": sqlalchemy.Integer,
    # Double (a Float subclass), not Float: a Python float is a C double, and on
    # dialects that distinguish the two (DuckDB renders Float as a 4-byte FLOAT)
    # plain Float would silently round query companions on dialects such as
    # DuckDB. Exact float reconstruction uses the codec's text companion.
    "float": sqlalchemy.Double,
    "str": sqlalchemy.Text,
    "bool": sqlalchemy.Boolean,
    "bytes": sqlalchemy.LargeBinary,
}


def sqlalchemy_metadata(
    schemas: Iterable[TableSchema],
    *,
    timestamps: _TimestampMode = "creation",
    versioned_tables: frozenset[str] = frozenset(),
    supports_partial_unique_indexes: bool = True,
) -> sqlalchemy.MetaData:
    """A fresh :class:`sqlalchemy.MetaData` holding the tables of ``schemas`` (recursively)."""
    metadata = sqlalchemy.MetaData()
    for schema in schemas:
        table_for(
            schema,
            metadata,
            timestamps=timestamps,
            versioned_tables=versioned_tables,
            supports_partial_unique_indexes=supports_partial_unique_indexes,
        )
    return metadata


def _stable_identifier(prefix: str, value: str) -> str:
    """Return a portable, readable identifier with a collision-resistant suffix."""
    safe = "".join(character if character.isalnum() else "_" for character in value.lower()).strip("_") or "entry"
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:8]
    body = f"{prefix}_{safe}_{digest}"
    if len(body) <= _MAX_IDENTIFIER_LENGTH:
        return body
    return f"{body[: _MAX_IDENTIFIER_LENGTH - 9]}_{digest}"


def entry_dispatch_table_name(family_name: str) -> str:
    """The deterministic reserved table name for one registered entry family."""
    return _stable_identifier("_httk_entry_dispatch", family_name)


def backing_dispatch_column_name(backing_name: str) -> str:
    """The deterministic nullable foreign-key column for one backing in a dispatch table."""
    return f"{_stable_identifier('backing', backing_name)}_sid"


def dispatch_table_for(
    family_name: str,
    backings: Sequence[tuple[str, TableSchema]],
    metadata: sqlalchemy.MetaData,
) -> sqlalchemy.Table:
    """Build the one-of-many dispatch table for an entry family.

    A single-backing family has no dispatch table and must not call this
    helper. The primary key is the backing record's canonical content id;
    every nullable backing sid is unique on its own, and the named check
    constraint makes precisely one of them non-null.
    """
    if len(backings) < 2:
        raise ValueError("an entry dispatch table requires at least two backings")
    name = entry_dispatch_table_name(family_name)
    existing = metadata.tables.get(name)
    if existing is not None:
        return existing
    columns: list[Any] = [
        sqlalchemy.Column(DISPATCH_CONTENT_ID_COLUMN, sqlalchemy.Text, primary_key=True, nullable=False),
    ]
    column_names: list[str] = []
    for backing_name, schema in backings:
        column_name = backing_dispatch_column_name(backing_name)
        if column_name in column_names:
            raise ValueError(
                f"entry family {family_name!r} has colliding dispatch columns for backing {backing_name!r}"
            )
        column_names.append(column_name)
        columns.append(
            sqlalchemy.Column(
                column_name,
                sqlalchemy.Integer,
                nullable=True,
                unique=True,
            )
        )
    terms = " + ".join(f"CASE WHEN {column_name} IS NOT NULL THEN 1 ELSE 0 END" for column_name in column_names)
    columns.append(sqlalchemy.CheckConstraint(f"({terms}) = 1", name=_index_name("ck", name, ("exactly_one",))))
    return sqlalchemy.Table(name, metadata, *columns)


def table_for(
    schema: TableSchema,
    metadata: sqlalchemy.MetaData,
    *,
    timestamps: _TimestampMode = "creation",
    versioned_tables: frozenset[str] = frozenset(),
    supports_partial_unique_indexes: bool = True,
) -> sqlalchemy.Table:
    """The :class:`sqlalchemy.Table` of ``schema`` within ``metadata``, building it on first use.

    Building is idempotent per metadata — if the table is already registered it
    is returned unchanged — and recursive: the child tables of the schema and
    the tables of every referenced storable class (reference fields and
    storable child elements alike) are built into the same metadata, so the
    complete logical layout is available to the storage algorithms.
    """
    existing = metadata.tables.get(schema.table_name)
    if existing is not None:
        return existing
    table = _build_parent_table(
        schema,
        metadata,
        timestamps=timestamps,
        versioned_tables=versioned_tables,
        supports_partial_unique_indexes=supports_partial_unique_indexes,
    )
    for spec in schema.fields:
        if spec.child is not None:
            _build_child_table(schema, spec, spec.child, metadata, reverse_index=timestamps == "versioned")
    for target in schema.referenced_classes():
        table_for(
            resolve_schema(target),
            metadata,
            timestamps=timestamps,
            versioned_tables=versioned_tables,
            supports_partial_unique_indexes=supports_partial_unique_indexes,
        )
    return table


def _index_name(prefix: str, table_name: str, columns: Sequence[str]) -> str:
    """A deterministic, table-scoped index name, hash-truncated if absurdly long."""
    name = f"{prefix}_{table_name}_{'_'.join(columns)}"
    if len(name) > _MAX_IDENTIFIER_LENGTH:
        digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:8]
        name = f"{name[: _MAX_IDENTIFIER_LENGTH - 9]}_{digest}"
    return name


def _column(spec: ColumnSpec) -> sqlalchemy.Column[Any]:
    return sqlalchemy.Column(spec.name, _TYPE_FOR_KIND[spec.kind](), nullable=spec.nullable)


def _column_index(
    table_name: str,
    spec: ColumnSpec,
    *,
    versioned_family: bool = False,
    supports_partial_unique_indexes: bool = True,
) -> sqlalchemy.Index | None:
    if spec.unique:
        if versioned_family:
            if supports_partial_unique_indexes:
                # Uniqueness is among current rows only; the partial predicate
                # excludes superseded (ts_end IS NOT NULL) history.
                return sqlalchemy.Index(
                    _index_name("uq", table_name, (spec.name,)),
                    spec.name,
                    unique=True,
                    sqlite_where=text(f"{TS_END_COLUMN} IS NULL"),
                    postgresql_where=text(f"{TS_END_COLUMN} IS NULL"),
                )
            # No partial-index support (DuckDB): a plain lookup index instead;
            # unique-among-current is enforced in the save transaction.
            return sqlalchemy.Index(_index_name("ix", table_name, (spec.name,)), spec.name)
        return sqlalchemy.Index(_index_name("uq", table_name, (spec.name,)), spec.name, unique=True)
    if spec.indexed:
        return sqlalchemy.Index(_index_name("ix", table_name, (spec.name,)), spec.name)
    return None


def _build_parent_table(
    schema: TableSchema,
    metadata: sqlalchemy.MetaData,
    *,
    timestamps: _TimestampMode = "creation",
    versioned_tables: frozenset[str] = frozenset(),
    supports_partial_unique_indexes: bool = True,
) -> sqlalchemy.Table:
    name = schema.table_name
    versioned = timestamps == "versioned" and name in versioned_tables
    items: list[Any] = [
        sqlalchemy.Column(
            SID_COLUMN,
            sqlalchemy.Integer,
            sqlalchemy.Sequence(f"{name}_sid_seq"),
            primary_key=True,
            autoincrement=True,
        )
    ]
    # This is storage bookkeeping, deliberately not part of a schema's value
    # identity, canonical content encoding, or hydrated entry surface.
    items.append(sqlalchemy.Column(ROLE_COLUMN, sqlalchemy.SmallInteger, nullable=False))
    items.append(
        sqlalchemy.CheckConstraint(f"{ROLE_COLUMN} IN (0, 1)", name=_index_name("ck", name, (ROLE_COLUMN, "valid")))
    )
    if timestamps != "off":
        items.append(sqlalchemy.Column(TS_START_COLUMN, sqlalchemy.BigInteger, nullable=False))
        items.append(sqlalchemy.Index(_index_name("ix", name, (TS_START_COLUMN,)), TS_START_COLUMN))
    if versioned:
        items.append(sqlalchemy.Column(TS_END_COLUMN, sqlalchemy.BigInteger, nullable=True))
        items.append(sqlalchemy.Index(_index_name("ix", name, (TS_END_COLUMN,)), TS_END_COLUMN))
        # No index: replaced_by_sid is consulted only by fsck, never a hot path.
        items.append(sqlalchemy.Column(REPLACED_BY_COLUMN, sqlalchemy.Integer, nullable=True))
    if schema.dedup == "content_id":
        # The content-id unique index stays GLOBAL: dedup identity is store-wide,
        # spanning both current and superseded rows.
        items.append(sqlalchemy.Column(CONTENT_ID_COLUMN, sqlalchemy.Text, nullable=False))
        items.append(sqlalchemy.Index(_index_name("uq", name, (CONTENT_ID_COLUMN,)), CONTENT_ID_COLUMN, unique=True))
    reverse_index = timestamps == "versioned"
    for spec in schema.fields:
        if spec.role == "child":
            if spec.optional:
                items.append(sqlalchemy.Column(f"{spec.field}_present", sqlalchemy.Boolean, nullable=False))
            continue
        for column_spec in spec.columns:
            items.append(_column(column_spec))
            index = _column_index(
                name,
                column_spec,
                versioned_family=versioned,
                supports_partial_unique_indexes=supports_partial_unique_indexes,
            )
            if index is not None:
                items.append(index)
            elif reverse_index and spec.role == "reference":
                # ponytail: superset indexing over all reference columns; narrow
                # to ownership-path columns if index bloat matters.
                items.append(sqlalchemy.Index(_index_name("ix", name, (column_spec.name,)), column_spec.name))
    for columns in schema.composite_indexes:
        items.append(sqlalchemy.Index(_index_name("ix", name, columns), *columns))
    return sqlalchemy.Table(name, metadata, *items)


def _build_child_table(
    schema: TableSchema,
    spec: FieldSpec,
    child: ChildTableSpec,
    metadata: sqlalchemy.MetaData,
    *,
    reverse_index: bool = False,
) -> sqlalchemy.Table:
    existing = metadata.tables.get(child.table_name)
    if existing is not None:
        return existing
    parent_sid = f"{schema.table_name}_sid"
    items: list[Any] = [
        sqlalchemy.Column(
            parent_sid,
            sqlalchemy.Integer,
            nullable=False,
        ),
        sqlalchemy.Index(_index_name("ix", child.table_name, (parent_sid,)), parent_sid),
        sqlalchemy.Column(f"{spec.field}_index", sqlalchemy.Integer, nullable=False),
    ]
    for column_spec in child.element_columns:
        items.append(_column(column_spec))
        # ponytail: superset indexing over all reference columns; narrow to
        # ownership-path columns if index bloat matters.
        if reverse_index and child.target is not None and column_spec.name.endswith("_sid"):
            items.append(sqlalchemy.Index(_index_name("ix", child.table_name, (column_spec.name,)), column_spec.name))
    return sqlalchemy.Table(child.table_name, metadata, *items)
