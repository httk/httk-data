"""httk-data: the data-management capability layer for httk v2.

Built on the stdlib-only *contracts and models* in *httk-core*, httk-data
supplies *capabilities*:

- in-memory :class:`~httk.core.EntryProvider` implementations for the standard
  OPTIMADE entry types (:class:`ReferenceEntryProvider`,
  :class:`FileEntryProvider`, :class:`CalculationEntryProvider`), serving
  httk-core's record models through the neutral provider contract; and
- **property-definition validation** (:func:`validate_property`,
  :func:`validate_record`, :class:`PropertyValidationError`) built on
  ``jsonschema`` (Draft 2020-12), checking record values against their OPTIMADE
  property definitions fully offline; and
- the **store/searcher query protocols** (:mod:`httk.data.query`) — the
  backend-agnostic query contract implemented by httk data stores and consumed
  by serving modules; and
- the **federated store** (:mod:`httk.data.federated_store`) — ordered, immutable
  source and target bindings plus lazy sequential union query execution; and
- the **generic OPTIMADE filter translation** (:mod:`httk.data.query.optimade_filters`)
  — turning filter syntax trees parsed by
  :func:`httk.core.optimade.parse_optimade_filter` into search expressions over the
  query protocols (the machinery in :mod:`httk.data.query.optimade_filters`, including
  :func:`~httk.data.query.optimade_filters.filter_searcher`), with
  neutral :class:`~httk.data.query.optimade_filters.FilterTranslationError` categories; and

- the **database storage layer** (:mod:`httk.data.db`, requiring the
  ``httk-data[db]`` extra) — relational storage and querying of plain frozen
  dataclasses (:class:`~httk.data.db.store.SqlStore` over SQLite or DuckDB),
  served through the provider contract by
  :class:`~httk.data.db.entry_provider.StoreEntryProvider`.

The providers self-register (under ``httk.registry.entries.data``, as
``data-references``/``data-files``/``data-calculations``/``data-db-store``)
when ``httk.core`` discovers the module, so a serving module (such as
*httk-serve*) can find them through the registry.

.. py:class:: StandardEntryProvider
   :canonical: httk.data.entry_providers.StandardEntryProvider
"""

from .entry_providers import (
    CalculationEntryProvider,
    FileEntryProvider,
    ReferenceEntryProvider,
)
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
from .validation import PropertyValidationError, validate_property, validate_record

__all__ = [
    "CalculationEntryProvider",
    "ContinuationToken",
    "CountUnavailableError",
    "FederatedResultSet",
    "FederatedSearcher",
    "FederatedSourceError",
    "FederatedStore",
    "FederatedStoreError",
    "FederatedTarget",
    "FileEntryProvider",
    "FilterTranslationCategory",
    "FilterTranslationError",
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
    "SearchExpression",
    "SearchField",
    "SearchResult",
    "SearchVariable",
    "Searcher",
    "Store",
    "UnsupportedQueryError",
    "filter_searcher",
    "portable_query_capabilities",
    "portable_query_fields",
    "validate_property",
    "validate_record",
]
