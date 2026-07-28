"""Schema-to-SQL mapping: build SQLAlchemy Core tables from resolved :class:`~httk.data.db.schema.TableSchema` IR.

:func:`table_for` turns one resolved schema into a :class:`sqlalchemy.Table`
registered in a :class:`sqlalchemy.MetaData` (idempotently: an already-built
table is returned as-is), recursing into referenced and child-element storable
classes so that every foreign-key target exists in the same metadata.
:func:`sqlalchemy_metadata` is the convenience wrapper that maps a batch of
schemas into one fresh metadata.

The relational layout produced here is exactly the one the schema IR
documents, plus the store-managed columns:

- every parent table gets an ``sid`` integer primary key (autoincrementing,
  with an attached ``<table>_sid_seq`` sequence for dialects such as DuckDB
  that need one; SQLite ignores it) and — only under the ``"content_id"``
  dedup policy — a unique-indexed ``content_id`` text column;
- every child table gets a ``<parent table>_sid`` integer foreign key (NOT
  NULL, indexed) and a ``<field>_index`` integer ordering column ahead of its
  element columns; storable elements become a ``<field>_sid`` foreign key to
  the element class's table.

Index names are deterministic and table-scoped — ``ix_<table>_<column>`` for
plain indexes, ``uq_<table>_<column>`` for unique ones, columns joined by
underscores for composites — truncated with a stable hash suffix when they
would exceed common identifier-length limits.
"""

import hashlib
from collections.abc import Iterable, Sequence
from typing import Any, Final

import sqlalchemy

from httk.data.db.codecs import ScalarKind
from httk.data.db.schema import (
    ChildTableSpec,
    ColumnSpec,
    FieldSpec,
    TableSchema,
    resolve_schema,
)

__all__ = [
    "CONTENT_ID_COLUMN",
    "SID_COLUMN",
    "sqlalchemy_metadata",
    "table_for",
]

SID_COLUMN: Final = "sid"
"""The store-managed integer primary-key column present on every table."""

CONTENT_ID_COLUMN: Final = "content_id"
"""The store-managed content-identity column of tables with the ``"content_id"`` dedup policy."""

_MAX_IDENTIFIER_LENGTH: Final = 63

_TYPE_FOR_KIND: Final[dict[ScalarKind, type[sqlalchemy.types.TypeEngine[Any]]]] = {
    "int": sqlalchemy.Integer,
    # Double (a Float subclass), not Float: a Python float is a C double, and on
    # dialects that distinguish the two (DuckDB renders Float as a 4-byte FLOAT)
    # plain Float would silently round scalar float fields, which have no exact
    # companion column to recover from. SQLite stores REAL either way.
    "float": sqlalchemy.Double,
    "str": sqlalchemy.Text,
    "bool": sqlalchemy.Boolean,
    "bytes": sqlalchemy.LargeBinary,
}


def sqlalchemy_metadata(schemas: Iterable[TableSchema]) -> sqlalchemy.MetaData:
    """A fresh :class:`sqlalchemy.MetaData` holding the tables of ``schemas`` (recursively)."""
    metadata = sqlalchemy.MetaData()
    for schema in schemas:
        table_for(schema, metadata)
    return metadata


def table_for(schema: TableSchema, metadata: sqlalchemy.MetaData) -> sqlalchemy.Table:
    """The :class:`sqlalchemy.Table` of ``schema`` within ``metadata``, building it on first use.

    Building is idempotent per metadata — if the table is already registered it
    is returned unchanged — and recursive: the child tables of the schema and
    the tables of every referenced storable class (reference fields and
    storable child elements alike) are built into the same metadata, so all
    foreign keys resolve.
    """
    existing = metadata.tables.get(schema.table_name)
    if existing is not None:
        return existing
    table = _build_parent_table(schema, metadata)
    for spec in schema.fields:
        if spec.child is not None:
            _build_child_table(schema, spec, spec.child, metadata)
    for target in schema.referenced_classes():
        table_for(resolve_schema(target), metadata)
    return table


def _index_name(prefix: str, table_name: str, columns: Sequence[str]) -> str:
    """A deterministic, table-scoped index name, hash-truncated if absurdly long."""
    name = f"{prefix}_{table_name}_{'_'.join(columns)}"
    if len(name) > _MAX_IDENTIFIER_LENGTH:
        digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:8]
        name = f"{name[: _MAX_IDENTIFIER_LENGTH - 9]}_{digest}"
    return name


def _column(spec: ColumnSpec, foreign_key: str | None = None) -> sqlalchemy.Column[Any]:
    args: list[Any] = []
    if foreign_key is not None:
        args.append(sqlalchemy.ForeignKey(foreign_key))
    return sqlalchemy.Column(spec.name, _TYPE_FOR_KIND[spec.kind](), *args, nullable=spec.nullable)


def _column_index(table_name: str, spec: ColumnSpec) -> sqlalchemy.Index | None:
    if spec.unique:
        return sqlalchemy.Index(_index_name("uq", table_name, (spec.name,)), spec.name, unique=True)
    if spec.indexed:
        return sqlalchemy.Index(_index_name("ix", table_name, (spec.name,)), spec.name)
    return None


def _build_parent_table(schema: TableSchema, metadata: sqlalchemy.MetaData) -> sqlalchemy.Table:
    name = schema.table_name
    items: list[Any] = [
        sqlalchemy.Column(
            SID_COLUMN,
            sqlalchemy.Integer,
            sqlalchemy.Sequence(f"{name}_sid_seq"),
            primary_key=True,
            autoincrement=True,
        )
    ]
    if schema.dedup == "content_id":
        items.append(sqlalchemy.Column(CONTENT_ID_COLUMN, sqlalchemy.Text, nullable=False))
        items.append(sqlalchemy.Index(_index_name("uq", name, (CONTENT_ID_COLUMN,)), CONTENT_ID_COLUMN, unique=True))
    for spec in schema.fields:
        if spec.role == "child":
            continue
        foreign_key = _reference_target(spec)
        for column_spec in spec.columns:
            items.append(_column(column_spec, foreign_key))
            index = _column_index(name, column_spec)
            if index is not None:
                items.append(index)
    for columns in schema.composite_indexes:
        items.append(sqlalchemy.Index(_index_name("ix", name, columns), *columns))
    return sqlalchemy.Table(name, metadata, *items)


def _reference_target(spec: FieldSpec) -> str | None:
    if spec.role != "reference":
        return None
    assert spec.target is not None
    return f"{resolve_schema(spec.target).table_name}.{SID_COLUMN}"


def _build_child_table(
    schema: TableSchema, spec: FieldSpec, child: ChildTableSpec, metadata: sqlalchemy.MetaData
) -> sqlalchemy.Table:
    existing = metadata.tables.get(child.table_name)
    if existing is not None:
        return existing
    parent_sid = f"{schema.table_name}_sid"
    items: list[Any] = [
        sqlalchemy.Column(
            parent_sid,
            sqlalchemy.Integer,
            sqlalchemy.ForeignKey(f"{schema.table_name}.{SID_COLUMN}"),
            nullable=False,
        ),
        sqlalchemy.Index(_index_name("ix", child.table_name, (parent_sid,)), parent_sid),
        sqlalchemy.Column(f"{spec.field}_index", sqlalchemy.Integer, nullable=False),
    ]
    element_foreign_key = None
    if child.target is not None:
        element_foreign_key = f"{resolve_schema(child.target).table_name}.{SID_COLUMN}"
    for column_spec in child.element_columns:
        items.append(_column(column_spec, element_foreign_key))
    return sqlalchemy.Table(child.table_name, metadata, *items)
