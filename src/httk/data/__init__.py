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
- the **database storage layer** (:mod:`httk.data.db`, requiring the
  ``httk-data[db]`` extra) — relational storage and querying of plain frozen
  dataclasses (:class:`~httk.data.db.store.SqlStore` over SQLite or DuckDB),
  served through the provider contract by
  :class:`~httk.data.db.entry_provider.StoreEntryProvider`.

The providers self-register (under ``httk.handlers.data``, as
``data-references``/``data-files``/``data-calculations``/``data-db-store``)
when ``httk.core`` discovers the module, so a serving module (such as
*httk-optimade*) can find them through the registry.
"""

from .entry_providers import (
    CalculationEntryProvider,
    FileEntryProvider,
    ReferenceEntryProvider,
)
from .query import SearchColumn, Searcher, SearchExpression, SearchVariable, Store
from .validation import PropertyValidationError, validate_property, validate_record

__all__ = [
    "ReferenceEntryProvider",
    "FileEntryProvider",
    "CalculationEntryProvider",
    "PropertyValidationError",
    "validate_property",
    "validate_record",
    "SearchExpression",
    "SearchColumn",
    "SearchVariable",
    "Searcher",
    "Store",
]
