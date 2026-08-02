"""Versioned physical layout for :class:`httk.data.db.store.SqlStore`.

The SQL object mapper deliberately permits ordinary frozen dataclasses to be
introduced on demand.  A persisted store, however, also needs a small stable
contract for the entry families it serves.  This module owns that contract:
the protocol marker, its registry-name declaration, and deterministic dispatch
tables.
"""

import dataclasses
import json
from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

import sqlalchemy
from httk.core import (
    entry_family_info,
    entry_record_info,
    known_entry_families,
    known_entry_records,
    resolve_entry_family,
    resolve_entry_record,
)

from httk.data.db.mapping import dispatch_table_for, entry_dispatch_table_name, table_for
from httk.data.db.schema import resolve_schema

__all__ = [
    "METADATA_TABLE_NAME",
    "STORAGE_PROTOCOL_VERSION",
    "EntryFamilyLayout",
    "StorageLayout",
    "StorageLayoutUpgradeRequiredError",
    "actual_schema_objects",
    "actual_table_names",
    "declaration_json",
    "expected_metadata",
    "metadata_table_for",
    "normalize_entry_records",
    "read_store_metadata",
]

STORAGE_PROTOCOL_VERSION: Final = "v2.0.3"
# One bump covers this stamp-trust/DDL-on-write cycle's layout changes.
"""The persisted SqlStore layout protocol implemented by this package."""

METADATA_TABLE_NAME: Final = "_httk_store_metadata"
"""Reserved key/value table holding the store protocol and entry declaration."""

_METADATA_PROTOCOL_KEY: Final = "protocol"
_METADATA_DECLARATION_KEY: Final = "entry_declaration"
_RESERVED_PREFIX: Final = "_httk_"


class StorageLayoutUpgradeRequiredError(RuntimeError):
    """A database does not exactly implement the current persisted store layout.

    ``diff`` is immutable and JSON-shaped.  Its top-level keys are stable
    categories (currently ``protocol``, ``declaration`` and ``schema``), so a
    caller can present a precise upgrade diagnostic without parsing
    the human-readable exception message.
    """

    def __init__(self, diff: Mapping[str, object]) -> None:
        frozen = _freeze_mapping(diff)
        self.diff: Mapping[str, object] = frozen
        categories = ", ".join(frozen) or "unknown layout difference"
        super().__init__(f"SqlStore layout upgrade is required ({categories})")


@dataclasses.dataclass(frozen=True)
class EntryFamilyLayout:
    """One immutable configured entry family and its concrete records."""

    name: str
    family: type
    record_names: tuple[str, ...]
    records: tuple[type, ...]
    dispatch_table_name: str | None


@dataclasses.dataclass(frozen=True)
class StorageLayout:
    """The immutable normalized entry declaration of an initialized store."""

    protocol_version: str
    families: tuple[EntryFamilyLayout, ...]

    @property
    def entry_records(self) -> Mapping[type, tuple[type, ...]]:
        """Configured family classes mapped to their ordered concrete record classes."""
        return MappingProxyType({family.family: family.records for family in self.families})

    @property
    def declaration(self) -> Mapping[str, tuple[str, ...]]:
        """Configured stable family names mapped to their ordered stable record names."""
        return MappingProxyType({family.name: family.record_names for family in self.families})


def normalize_entry_records(entry_records: Mapping[type, type | tuple[type, ...]]) -> StorageLayout:
    """Validate an explicit class declaration and replace it with stable registry names.

    Registry aliases are rejected rather than selected arbitrarily: a
    persistent declaration must have exactly one stable spelling for every
    supplied class.
    """
    if not isinstance(entry_records, Mapping):
        raise TypeError("entry_records must be a mapping from entry-family classes to record classes")
    entries: list[EntryFamilyLayout] = []
    for family, supplied_records in entry_records.items():
        if not isinstance(family, type):
            raise TypeError("entry_records keys must be entry-family classes")
        family_name = _registered_family_name(family)
        records: tuple[type, ...]
        if isinstance(supplied_records, type):
            records = (supplied_records,)
        elif isinstance(supplied_records, tuple):
            records = supplied_records
        else:
            raise TypeError(f"entry_records[{family.__name__}] must be a record class or a tuple of record classes")
        if not records:
            raise ValueError(f"entry_records[{family.__name__}] cannot be an empty tuple")
        if any(not isinstance(record, type) for record in records):
            raise TypeError(f"entry_records[{family.__name__}] contains a non-class record")
        if len(set(records)) != len(records):
            raise ValueError(f"entry_records[{family.__name__}] repeats a record class")
        record_names: list[str] = []
        for record in records:
            record_name = _registered_record_name(record)
            _, registered_family_name, _ = entry_record_info(record_name)
            if registered_family_name is None:
                raise ValueError(
                    f"entry record {record_name!r} has no registered family and cannot be used in a family store"
                )
            if registered_family_name != family_name:
                raise ValueError(
                    f"entry record {record.__name__} belongs to registered family {registered_family_name!r}, "
                    f"not {family_name!r}"
                )
            schema = resolve_schema(record)
            if schema.dedup != "content_id":
                raise ValueError(
                    f"configured entry record {record.__name__} must use dedup='content_id', got {schema.dedup!r}"
                )
            record_names.append(record_name)
        entries.append(
            EntryFamilyLayout(
                name=family_name,
                family=family,
                record_names=tuple(record_names),
                records=records,
                dispatch_table_name=entry_dispatch_table_name(family_name) if len(records) > 1 else None,
            )
        )
    entries.sort(key=lambda entry: entry.name)
    if len({entry.name for entry in entries}) != len(entries):
        raise ValueError("entry_records contains multiple registered aliases for the same family")
    layout = StorageLayout(STORAGE_PROTOCOL_VERSION, tuple(entries))
    _validate_physical_names(layout)
    return layout


def declaration_json(layout: StorageLayout) -> str:
    """Serialize a normalized declaration in its exact deterministic persisted form."""
    document = {
        "families": [{"records": list(family.record_names), "family": family.name} for family in layout.families],
        "format": 1,
    }
    return json.dumps(document, sort_keys=True, separators=(",", ":"))


def _layout_from_declaration(value: str) -> StorageLayout:
    try:
        document = json.loads(value)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError("stored entry declaration is not valid JSON") from error
    if not isinstance(document, dict) or set(document) != {"families", "format"} or document["format"] != 1:
        raise ValueError("stored entry declaration does not use format 1")
    families = document["families"]
    if not isinstance(families, list):
        raise ValueError("stored entry declaration families must be a list")
    supplied: dict[type, tuple[type, ...]] = {}
    previous = ""
    for item in families:
        if not isinstance(item, dict) or set(item) != {"records", "family"}:
            raise ValueError("stored entry declaration family entry is malformed")
        family_name = item["family"]
        record_names = item["records"]
        if not isinstance(family_name, str) or not isinstance(record_names, list) or not record_names:
            raise ValueError("stored entry declaration has an invalid family or record list")
        if family_name <= previous:
            raise ValueError("stored entry declaration families are not deterministically ordered")
        previous = family_name
        family = resolve_entry_family(family_name)
        resolved_records: list[type] = []
        for record_name in record_names:
            if not isinstance(record_name, str):
                raise ValueError("stored entry declaration record names must be strings")
            _, declared_family, _ = entry_record_info(record_name)
            if declared_family is None:
                raise ValueError(
                    f"entry record {record_name!r} has no registered family and cannot be used in a family store"
                )
            if declared_family != family_name:
                raise ValueError(
                    f"stored entry record {record_name!r} is registered for {declared_family!r}, not {family_name!r}"
                )
            resolved_records.append(resolve_entry_record(record_name))
        supplied[family] = tuple(resolved_records)
    layout = normalize_entry_records(supplied)
    if declaration_json(layout) != value:
        raise ValueError("stored entry declaration is not in its canonical deterministic encoding")
    return layout


def expected_metadata(layout: StorageLayout) -> sqlalchemy.MetaData:
    """Return SQLAlchemy metadata for all protocol-owned tables of ``layout``."""
    metadata = sqlalchemy.MetaData()
    metadata_table_for(metadata)
    for family in layout.families:
        schemas = tuple(resolve_schema(record) for record in family.records)
        for schema in schemas:
            table_for(schema, metadata)
        if len(schemas) > 1:
            dispatch_table_for(family.name, tuple(zip(family.record_names, schemas, strict=True)), metadata)
    return metadata


def metadata_table_for(metadata: sqlalchemy.MetaData) -> sqlalchemy.Table:
    """Return the reserved protocol key/value table in ``metadata``."""
    existing = metadata.tables.get(METADATA_TABLE_NAME)
    if existing is not None:
        return existing
    return sqlalchemy.Table(
        METADATA_TABLE_NAME,
        metadata,
        sqlalchemy.Column("key", sqlalchemy.Text, primary_key=True, nullable=False),
        sqlalchemy.Column("value", sqlalchemy.Text, nullable=False),
    )


def actual_schema_objects(connection: sqlalchemy.Connection) -> Mapping[str, frozenset[str]]:
    """Return application schema-object names mapped to their stable object kinds.

    The DuckDB SQLAlchemy inspector presently routes column inspection through
    a PostgreSQL catalogue relation DuckDB does not expose, so the whole
    layout path intentionally uses the dialect catalogues directly.
    """
    if connection.dialect.name == "sqlite":
        rows = connection.execute(
            sqlalchemy.text(
                "SELECT name, type FROM sqlite_master WHERE type IN ('table', 'view') AND name NOT LIKE 'sqlite_%'"
            )
        )
    elif connection.dialect.name == "duckdb":
        rows = connection.execute(
            sqlalchemy.text(
                "SELECT table_name, lower(table_type) FROM information_schema.tables "
                "WHERE table_schema = current_schema() "
                "UNION ALL "
                "SELECT sequence_name, 'sequence' FROM duckdb_sequences() WHERE schema_name = current_schema()"
            )
        )
    else:
        raise ValueError(f"SqlStore layout validation does not support dialect {connection.dialect.name!r}")
    result: dict[str, set[str]] = {}
    for name, kind in rows:
        result.setdefault(str(name), set()).add(str(kind).lower().replace("base ", ""))
    return MappingProxyType({name: frozenset(kinds) for name, kinds in result.items()})


def actual_table_names(connection: sqlalchemy.Connection) -> frozenset[str]:
    """Return application base-table names without SQLAlchemy reflection."""
    return frozenset(name for name, kinds in actual_schema_objects(connection).items() if "table" in kinds)


def read_store_metadata(connection: sqlalchemy.Connection) -> Mapping[str, str] | None:
    """Read the marker values, or return ``None`` when no metadata table exists."""
    if METADATA_TABLE_NAME not in actual_table_names(connection):
        return None
    table = metadata_table_for(sqlalchemy.MetaData())
    rows = connection.execute(sqlalchemy.select(table.c.key, table.c.value)).all()
    result: dict[str, str] = {}
    for key, value in rows:
        if not isinstance(key, str) or not isinstance(value, str) or key in result:
            raise ValueError("stored metadata rows are malformed")
        result[key] = value
    return MappingProxyType(result)


def _registered_family_name(family: type) -> str:
    matches: list[str] = []
    for name in known_entry_families():
        entry_family_info(name)
        if resolve_entry_family(name) is family:
            matches.append(name)
    if len(matches) != 1:
        found = ", ".join(matches) or "none"
        raise ValueError(f"entry family {family.__name__} must resolve to exactly one registered name (found {found})")
    return matches[0]


def _registered_record_name(record: type) -> str:
    matches: list[str] = []
    for name in known_entry_records():
        entry_record_info(name)
        if resolve_entry_record(name) is record:
            matches.append(name)
    if len(matches) != 1:
        found = ", ".join(matches) or "none"
        raise ValueError(f"entry record {record.__name__} must resolve to exactly one registered name (found {found})")
    return matches[0]


def _validate_physical_names(layout: StorageLayout) -> None:
    owners: dict[str, type] = {}
    visited: set[type] = set()

    def visit(record: type) -> None:
        if record in visited:
            return
        visited.add(record)
        schema = resolve_schema(record)
        names = [schema.table_name]
        names.extend(spec.child.table_name for spec in schema.fields if spec.child is not None)
        for name in names:
            if name.startswith(_RESERVED_PREFIX):
                raise ValueError(f"record {record.__name__} claims reserved SqlStore table name {name!r}")
            previous = owners.get(name)
            if previous is not None and previous is not record:
                raise ValueError(
                    f"records {previous.__name__} and {record.__name__} collide on physical table name {name!r}"
                )
            owners[name] = record
        for target in schema.referenced_classes():
            visit(target)

    for family in layout.families:
        for record in family.records:
            visit(record)
        if family.dispatch_table_name is not None:
            if family.dispatch_table_name in owners:
                raise ValueError(
                    f"entry family {family.name!r} dispatch table collides with record table {family.dispatch_table_name!r}"
                )
            owners[family.dispatch_table_name] = family.family


def _freeze_mapping(value: Mapping[str, object]) -> Mapping[str, object]:
    def freeze(item: object) -> object:
        if isinstance(item, Mapping):
            return MappingProxyType({str(key): freeze(member) for key, member in item.items()})
        if isinstance(item, list):
            return tuple(freeze(member) for member in item)
        if isinstance(item, tuple):
            return tuple(freeze(member) for member in item)
        if isinstance(item, set | frozenset):
            return tuple(sorted((freeze(member) for member in item), key=repr))
        return item

    return MappingProxyType({str(key): freeze(member) for key, member in value.items()})
