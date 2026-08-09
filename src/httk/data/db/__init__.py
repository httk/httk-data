"""The SQL storage layer of httk-data: store frozen dataclasses in relational databases.

This subpackage turns plain frozen dataclasses — declared storable with the
stdlib-only marker vocabulary in httk-core (``Indexed``, ``Unique``, ``Skip``,
``Shape``, ``StorageInfo``, ``stored_property``) — into relational storage.
The pure-Python foundation lives here:

- :mod:`httk.data.db.schema` — :func:`resolve_schema` reads a storable class
  into a :class:`~httk.data.db.schema.TableSchema`, the single source of truth for DDL, inserts,
  selects, and reconstruction;
- :mod:`httk.data.db.codecs` — the :class:`ValueCodec` registry with exact,
  round-trippable encodings for rationals, surds, and datetimes;
- :mod:`httk.core.storage` — ``canonical_form`` and ``content_id``,
  the content identity used for deduplication.

These modules import cleanly without sqlalchemy. The SQL layer proper builds
on them and requires the ``httk-data[db]`` extra (sqlalchemy):

- :class:`~httk.data.db.engine.Database` — the engine wrapper naming where
  data lives (``Database.sqlite(...)``, ``Database.duckdb(...)``);
- :class:`~httk.data.db.store.SqlStore` — save/fetch/dedup/transactions for
  storable instances, on top of the schema-to-table mapping in
  :mod:`httk.data.db.mapping`;
- :class:`~httk.data.db.searcher.SqlSearcher` (from
  :meth:`~httk.data.db.store.SqlStore.searcher`) — the query DSL implementing
  the :mod:`httk.data.query` search protocols, with
  :class:`~httk.data.db.searcher.SqlVariable`,
  :class:`~httk.data.db.searcher.SqlColumn` and
  :class:`~httk.data.db.searcher.SqlExpression`;
- :class:`~httk.data.db.entry_provider.StoreEntryProvider` — the bridge that
  serves stored classes through the neutral :class:`~httk.core.EntryProvider`
  contract (e.g. as an OPTIMADE API via *httk-serve*);
- :func:`~httk.data.db.optimade.optimade_filter_searcher` — OPTIMADE-filter
  querying over storable classes, tying the generic filter translation in
  :mod:`httk.data.query.optimade_filters` to the SQL layer.

The sqlalchemy-backed names are imported lazily on first attribute access, so
``import httk.data.db`` keeps working without sqlalchemy; touching them
without sqlalchemy installed raises :class:`ImportError` naming the extra.
"""

import importlib
from typing import Any

from ..query import MultipleResultsError, NoResultError, ResultRow
from .codecs import ValueCodec, register_value_codec
from .schema import SchemaError, register_schema_override, resolve_schema

__all__ = [
    "STORAGE_PROTOCOL_VERSION",  # pyright: ignore[reportUnsupportedDunderAll]  (provided lazily via __getattr__)
    "BackendFacts",  # pyright: ignore[reportUnsupportedDunderAll]
    "BulkIngest",  # pyright: ignore[reportUnsupportedDunderAll]  (provided lazily via __getattr__)
    "Database",  # pyright: ignore[reportUnsupportedDunderAll]  (provided lazily via __getattr__)
    "DuplicateEntryIdError",  # pyright: ignore[reportUnsupportedDunderAll]
    "EntryDispatchIntegrityError",  # pyright: ignore[reportUnsupportedDunderAll]
    "EntryMetadataConflictError",  # pyright: ignore[reportUnsupportedDunderAll]
    "ExpiredCursorRowError",  # pyright: ignore[reportUnsupportedDunderAll]
    "FsckSummary",  # pyright: ignore[reportUnsupportedDunderAll]
    "MultipleResultsError",
    "NoResultError",
    "ResultColumn",  # pyright: ignore[reportUnsupportedDunderAll]
    "ResultRow",  # pyright: ignore[reportUnsupportedDunderAll]
    "SchemaError",
    "SqlResultSet",  # pyright: ignore[reportUnsupportedDunderAll]
    "SqlSearcher",  # pyright: ignore[reportUnsupportedDunderAll]  (provided lazily via __getattr__)
    "SqlStore",  # pyright: ignore[reportUnsupportedDunderAll]  (provided lazily via __getattr__)
    "StaleResultError",  # pyright: ignore[reportUnsupportedDunderAll]  (provided lazily via __getattr__)
    "StorageLayoutUpgradeRequiredError",  # pyright: ignore[reportUnsupportedDunderAll]
    "StoreEntryProvider",  # pyright: ignore[reportUnsupportedDunderAll]  (provided lazily via __getattr__)
    "StoreUnderConstructionError",  # pyright: ignore[reportUnsupportedDunderAll]
    "StoredEntryFederation",  # pyright: ignore[reportUnsupportedDunderAll]
    "StoredEntrySource",  # pyright: ignore[reportUnsupportedDunderAll]
    "StoredPropertySqlConfigurationError",  # pyright: ignore[reportUnsupportedDunderAll]
    "ValueCodec",
    "optimade_filter_searcher",  # pyright: ignore[reportUnsupportedDunderAll]  (provided lazily via __getattr__)
    "register_schema_override",
    "register_value_codec",
    "resolve_schema",
    "stored_property_sql_plan",  # pyright: ignore[reportUnsupportedDunderAll]
]

_SQL_EXPORTS = {
    "STORAGE_PROTOCOL_VERSION": ".layout",
    "BackendFacts": ".layout",
    "BulkIngest": ".bulk",
    "Database": ".engine",
    "EntryDispatchIntegrityError": ".store",
    "EntryMetadataConflictError": ".store",
    "SqlStore": ".store",
    "SqlSearcher": ".searcher",
    "StoreEntryProvider": ".entry_provider",
    "StoredEntryFederation": ".stored_federation",
    "StoredEntrySource": ".stored_federation",
    "DuplicateEntryIdError": ".stored_federation",
    "StoredPropertySqlConfigurationError": ".stored_properties",
    "StorageLayoutUpgradeRequiredError": ".layout",
    "StoreUnderConstructionError": ".layout",
    "optimade_filter_searcher": ".optimade",
    "stored_property_sql_plan": ".stored_properties",
    "StaleResultError": ".rows",
    "ExpiredCursorRowError": ".results",
    "FsckSummary": ".fsck",
    "ResultColumn": ".results",
    "SqlResultSet": ".results",
}


def __getattr__(name: str) -> Any:
    """Import a SQLAlchemy-backed export lazily.

    :param name: The module attribute to import.
    :return: The requested SQL-layer export.
    :raises AttributeError: If ``name`` is not an exported database attribute.
    :raises ImportError: If SQLAlchemy is unavailable for the requested export.
    """
    module_name = _SQL_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        module = importlib.import_module(module_name, __name__)
    except ModuleNotFoundError as error:
        if error.name is not None and error.name.partition(".")[0] == "sqlalchemy":
            raise ImportError(
                f"{__name__}.{name} needs sqlalchemy; install the 'httk-data[db]' extra to use the SQL layer"
            ) from error
        raise
    return getattr(module, name)
