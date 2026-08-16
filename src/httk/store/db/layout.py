"""Versioned physical layout for :class:`httk.store.db.store.SqlStore`."""

import dataclasses
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Final, Literal

import sqlalchemy

from httk.store.db.mapping import dispatch_table_for, entry_dispatch_table_name, table_for
from httk.store.db.schema import resolve_schema
from httk.store.storage_layout import (
    EntryFamilyDeclaration,
    EntryFamilyLayout,
    StorageLayout,
    StorageLayoutUpgradeRequiredError,
    _merge_storage_layouts,
    declaration_json,
)
from httk.store.storage_layout import (
    normalize_entry_families as _normalize_entry_families,
)
from httk.store.storage_layout import (
    normalize_entry_records as _normalize_entry_records,
)

__all__ = [
    "METADATA_TABLE_NAME",
    "STORAGE_PROTOCOL_VERSION",
    "WRITE_PROFILE_VOCABULARY",
    "BackendFacts",
    "EntryFamilyLayout",
    "StorageLayout",
    "StorageLayoutUpgradeRequiredError",
    "StoreUnderConstructionError",
    "actual_schema_objects",
    "actual_table_names",
    "backend_facts_for_dialect",
    "declaration_json",
    "expected_metadata",
    "metadata_table_for",
    "normalize_entry_declaration",
    "normalize_entry_families",
    "normalize_entry_records",
    "read_store_metadata",
]

STORAGE_PROTOCOL_VERSION: Final = "v3.0.0"
# This bump renames the store-managed ``store_timestamp`` column to ``ts_start``
# and records the timestamp mode (off/creation/versioned) in the persisted
# marker, so v2 layouts and markers no longer parse.
"""The persisted SqlStore layout protocol implemented by this package."""

METADATA_TABLE_NAME: Final = "_httk_store_metadata"
"""Reserved key/value table holding the store protocol and entry declaration."""

_METADATA_PROTOCOL_KEY: Final = "protocol"
_METADATA_DECLARATION_KEY: Final = "entry_declaration"
_RESERVED_PREFIX: Final = "_httk_"
WRITE_PROFILE_VOCABULARY: Final = frozenset({"transactional", "degraded", "bulk-fenced"})


class StoreUnderConstructionError(RuntimeError):
    """A new open found an interrupted empty-store bulk ingest.

    Crash window for new SQLite/DuckDB opens: before the marker commits the
    old clean state remains accepted; after the marker and through ingest,
    finalize, or before marker clear the store is rejected; after clear it is
    accepted again.  The marker is intentionally not a resume protocol.

    ClickHouse marker residue is fail-closed: the default recovery is to drop
    the database and re-ingest. Clearing the marker is valid only after an
    operator has restored and verified the declared empty-store invariant.
    """


@dataclasses.dataclass(frozen=True)
class BackendFacts:
    """Dialect capabilities used by the SQL storage protocol."""

    transactional_ddl: bool
    transactional_dml: bool
    supports_sequences: bool
    atomic_upsert: bool
    serial_stage_format: Literal["sqlite", "duckdb-attach", "parquet"]
    parallel_shard_format: Literal["sqlite", "parquet"]
    supports_deferred_finalize: bool
    supports_degraded: bool
    write_profiles: tuple[str, ...]
    metadata_backend: Literal["table", "keepermap"]
    supports_incremental_save: bool
    system_catalog: Literal["sqlite", "duckdb", "clickhouse", "postgresql"]
    stage_load: Literal["attach", "duckdb-views", "client-stream"]
    finalize_map_maintenance: Literal["update", "swap"]
    supports_adhoc_indexes: bool


_BACKEND_FACTS: Final[dict[str, BackendFacts]] = {
    "sqlite": BackendFacts(
        transactional_ddl=False,
        transactional_dml=True,
        supports_sequences=False,
        atomic_upsert=True,
        serial_stage_format="sqlite",
        parallel_shard_format="sqlite",
        supports_deferred_finalize=True,
        supports_degraded=True,
        write_profiles=("transactional", "degraded"),
        metadata_backend="table",
        supports_incremental_save=True,
        system_catalog="sqlite",
        stage_load="attach",
        finalize_map_maintenance="update",
        supports_adhoc_indexes=True,
    ),
    "duckdb": BackendFacts(
        transactional_ddl=True,
        transactional_dml=True,
        supports_sequences=True,
        atomic_upsert=True,
        serial_stage_format="duckdb-attach",
        parallel_shard_format="parquet",
        supports_deferred_finalize=True,
        supports_degraded=False,
        write_profiles=("transactional",),
        metadata_backend="table",
        supports_incremental_save=True,
        system_catalog="duckdb",
        stage_load="duckdb-views",
        finalize_map_maintenance="update",
        supports_adhoc_indexes=True,
    ),
    "postgresql": BackendFacts(
        transactional_ddl=True,
        transactional_dml=True,
        supports_sequences=True,
        atomic_upsert=True,
        serial_stage_format="sqlite",
        parallel_shard_format="sqlite",
        supports_deferred_finalize=True,
        supports_degraded=False,
        write_profiles=("transactional",),
        metadata_backend="table",
        supports_incremental_save=True,
        system_catalog="postgresql",
        stage_load="attach",
        finalize_map_maintenance="update",
        supports_adhoc_indexes=True,
    ),
    "clickhousedb": BackendFacts(
        transactional_ddl=False,
        transactional_dml=False,
        supports_sequences=False,
        atomic_upsert=False,
        serial_stage_format="parquet",
        parallel_shard_format="parquet",
        supports_deferred_finalize=True,
        supports_degraded=False,
        write_profiles=("bulk-fenced",),
        metadata_backend="keepermap",
        supports_incremental_save=False,
        system_catalog="clickhouse",
        stage_load="client-stream",
        finalize_map_maintenance="swap",
        supports_adhoc_indexes=False,
    ),
}


def backend_facts_for_dialect(dialect_name: str) -> BackendFacts:
    """Resolve the hardcoded protocol facts for one supported dialect."""
    try:
        return _BACKEND_FACTS[dialect_name]
    except KeyError as error:
        raise ValueError(f"SqlStore layout validation does not support dialect {dialect_name!r}") from error


def normalize_entry_records(entry_records: Mapping[type, type | tuple[type, ...]]) -> StorageLayout:
    """Normalize a declaration and apply SQL physical-name validation."""
    layout = _normalize_entry_records(entry_records)
    _validate_physical_names(layout)
    return layout


def normalize_entry_families(entry_families: Sequence[EntryFamilyDeclaration]) -> StorageLayout:
    """Normalize application-owned declarations and apply SQL physical-name validation."""
    layout = _normalize_entry_families(entry_families)
    _validate_physical_names(layout)
    return layout


def normalize_entry_declaration(
    entry_records: Mapping[type, type | tuple[type, ...]] | None,
    entry_families: Sequence[EntryFamilyDeclaration] | None,
) -> StorageLayout | None:
    """Merge registered and application-owned declarations and validate SQL names."""
    layouts = []
    if entry_records is not None:
        layouts.append(_normalize_entry_records(entry_records))
    if entry_families is not None:
        layouts.append(_normalize_entry_families(entry_families))
    if not layouts:
        return None
    layout = _merge_storage_layouts(*layouts)
    _validate_physical_names(layout)
    return layout


def _layout_from_declaration(value: str) -> StorageLayout:
    """Parse a declaration and apply SQL physical-name validation."""
    from httk.store.storage_layout import _layout_from_declaration as parse_declaration

    layout = parse_declaration(value)
    _validate_physical_names(layout)
    return layout


def expected_metadata(layout: StorageLayout, *, store_timestamps: bool = True) -> sqlalchemy.MetaData:
    """Return SQLAlchemy metadata for all protocol-owned tables of ``layout``."""
    metadata = sqlalchemy.MetaData()
    metadata_table_for(metadata)
    for family in layout.families:
        schemas = tuple(resolve_schema(record) for record in family.records)
        for schema in schemas:
            table_for(schema, metadata, store_timestamps=store_timestamps)
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
        info={"httk_metadata": True},
    )


def actual_schema_objects(connection: sqlalchemy.Connection) -> Mapping[str, frozenset[str]]:
    """Return application schema-object names mapped to their stable object kinds.

    The DuckDB SQLAlchemy inspector presently routes column inspection through
    a PostgreSQL catalogue relation DuckDB does not expose, so the whole
    layout path intentionally uses the dialect catalogues directly.
    """
    facts = backend_facts_for_dialect(connection.dialect.name)
    if facts.system_catalog == "sqlite":
        rows = connection.execute(
            sqlalchemy.text(
                "SELECT name, type FROM sqlite_master WHERE type IN ('table', 'view') AND name NOT LIKE 'sqlite_%'"
            )
        )
    elif facts.system_catalog == "duckdb":
        rows = connection.execute(
            sqlalchemy.text(
                "SELECT table_name, lower(table_type) FROM information_schema.tables "
                "WHERE table_catalog = current_database() AND table_schema = current_schema() "
                "UNION ALL "
                "SELECT sequence_name, 'sequence' FROM duckdb_sequences() "
                "WHERE database_name = current_database() AND schema_name = current_schema()"
            )
        )
    elif facts.system_catalog == "postgresql":
        rows = connection.execute(
            sqlalchemy.text(
                "SELECT table_name, lower(table_type) FROM information_schema.tables "
                "WHERE table_catalog = current_database() AND table_schema = current_schema() "
                "UNION ALL "
                "SELECT sequence_name, 'sequence' FROM information_schema.sequences "
                "WHERE sequence_catalog = current_database() AND sequence_schema = current_schema() "
                "UNION ALL "
                # Materialized views are absent from information_schema.tables; a
                # stray one must still count as a non-empty schema object.
                "SELECT matviewname, 'view' FROM pg_matviews WHERE schemaname = current_schema()"
            )
        )
    else:
        from httk.store.db.clickhouse import actual_schema_objects as clickhouse_schema_objects

        return clickhouse_schema_objects(connection)
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
        dispatch_name = entry_dispatch_table_name(family.name) if len(family.records) > 1 else None
        if dispatch_name is not None:
            if dispatch_name in owners:
                raise ValueError(
                    f"entry family {family.name!r} dispatch table collides with record table {dispatch_name!r}"
                )
            owners[dispatch_name] = family.family
