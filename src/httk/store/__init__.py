"""Provide httk-store's data-management capability layer for httk v2.

Built on the stdlib-only *contracts and models* in *httk-core*, httk-store
supplies *capabilities*:

- in-memory :class:`~httk.core.EntryProvider` implementations for the standard
  OPTIMADE entry types (:class:`ReferenceEntryProvider`,
  :class:`FileEntryProvider`, :class:`CalculationEntryProvider`), serving
  httk-core's record models through the neutral provider contract; and
- **property-definition validation** (:func:`validate_property`,
  :func:`validate_record`, :class:`PropertyValidationError`) built on
  ``jsonschema`` (Draft 2020-12), checking record values against their OPTIMADE
  property definitions fully offline; and
- the **store/searcher query protocols** (:mod:`httk.store.query`) — the
  backend-agnostic query contract implemented by httk data stores and consumed
  by serving modules; and
- the **federated store** (:mod:`httk.store.federated_store`) — ordered, immutable
  source and target bindings plus lazy sequential union query execution; and
- the **generic OPTIMADE filter translation** (:mod:`httk.store.query.optimade_filters`)
  — turning filter syntax trees parsed by
  :func:`httk.core.optimade.parse_optimade_filter` into search expressions over the
  query protocols (the machinery in :mod:`httk.store.query.optimade_filters`, including
  :func:`~httk.store.query.optimade_filters.filter_searcher`), with
  neutral :class:`~httk.store.query.optimade_filters.FilterTranslationError` categories; and

- the **database storage layer** (:mod:`httk.store.backend.sql`, requiring the
  ``httk-store[db]`` extra) — relational storage and querying of plain frozen
  dataclasses (:class:`~httk.store.backend.sql.store.SqlStore` over SQLite or DuckDB),
  served through the provider contract by
  :class:`~httk.store.backend.sql.entry_provider.StoreEntryProvider`.

The providers self-register (under ``httk.registry.entries.store``, as
``store-references``/``store-files``/``store-calculations``/``store-db-store``)
when ``httk.core`` discovers the module, so a serving module (such as
*httk-serve*) can find them through the registry.

.. py:class:: StandardEntryProvider
   :canonical: httk.store.entry_providers.StandardEntryProvider
"""

import importlib
from typing import Any

from .entry_providers import (
    CalculationEntryProvider,
    DataRecordEntryProvider,
    FileEntryProvider,
    ReferenceEntryProvider,
    RunEntryProvider,
    product_relationships,
)
from .export import export_dataset
from .federated_store import (
    FederatedResultSet,
    FederatedSearcher,
    FederatedSourceError,
    FederatedStore,
    FederatedStoreError,
    FederatedTarget,
)
from .query import (
    ContinuationToken,
    CountUnavailableError,
    MultipleResultsError,
    NoResultError,
    PageableResultSetLike,
    PageOrder,
    PaginationCursorError,
    PortableQueryCapabilities,
    ResultPage,
    ResultRow,
    ResultRowLike,
    ResultSetLike,
    Searcher,
    SearchExpression,
    SearchField,
    SearchResult,
    SearchVariable,
    Store,
    UnsupportedQueryError,
    portable_query_capabilities,
    portable_query_fields,
)
from .query.optimade_filters import (
    FilterTranslationCategory,
    FilterTranslationError,
    filter_searcher,
)
from .storage_layout import EntryFamilyDeclaration, EntryLayoutBindingError, EntryRecordDeclaration
from .store_common import EntryStore
from .validation import PropertyValidationError, validate_property, validate_record

__all__ = [
    "Backend",  # pyright: ignore[reportUnsupportedDunderAll]  (provided lazily via __getattr__)
    "CalculationEntryProvider",
    "ContinuationToken",
    "CountUnavailableError",
    "DataRecordEntryProvider",
    "EntryFamilyDeclaration",
    "EntryLayoutBindingError",
    "EntryRecordDeclaration",
    "EntryStore",
    "FederatedResultSet",
    "FederatedSearcher",
    "FederatedSourceError",
    "FederatedStore",
    "FederatedStoreError",
    "FederatedTarget",
    "FileEntryProvider",
    "FilterTranslationCategory",
    "FilterTranslationError",
    "MongoStore",  # pyright: ignore[reportUnsupportedDunderAll]  (provided lazily via __getattr__)
    "MultipleResultsError",
    "NoResultError",
    "PageOrder",
    "PageableResultSetLike",
    "PaginationCursorError",
    "PortableQueryCapabilities",
    "PropertyValidationError",
    "ReferenceEntryProvider",
    "ResultPage",
    "ResultRow",
    "ResultRowLike",
    "ResultSetLike",
    "RunEntryProvider",
    "SearchExpression",
    "SearchField",
    "SearchResult",
    "SearchVariable",
    "Searcher",
    "SqlStore",  # pyright: ignore[reportUnsupportedDunderAll]  (provided lazily via __getattr__)
    "Store",
    "UnsupportedQueryError",
    "export_dataset",
    "filter_searcher",
    "portable_query_capabilities",
    "portable_query_fields",
    "product_relationships",
    "validate_property",
    "validate_record",
]

# The storage-backend engines are re-exported lazily: importing ``httk.store``
# must stay free of both ``sqlalchemy`` and ``pymongo``, so ``Backend`` and
# ``SqlStore`` load the SQL layer, and ``MongoStore`` the MongoDB layer, only on
# first attribute access.
_LAZY_EXPORTS = {
    "Backend": "httk.store.backend.sql.engine",
    "SqlStore": "httk.store.backend.sql.store",
    "MongoStore": "httk.store.backend.mongo",
}


def __getattr__(name: str) -> Any:
    """Import a storage-backend engine lazily on first access.

    :param name: The module attribute to import.
    :return: The requested storage-backend export.
    :raises AttributeError: If ``name`` is not a lazily re-exported attribute.
    """
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(module_name), name)
