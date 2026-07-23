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
  property definitions fully offline.

The providers self-register (under ``httk.handlers.data``, as
``data-references``/``data-files``/``data-calculations``) when ``httk.core``
discovers the module, so a serving module (such as *httk-optimade*) can find
them through the registry. httk-data is also the intended future home of the
v1-style sqlite/database storage layer, which is not built yet.
"""

from .entry_providers import (
    CalculationEntryProvider,
    FileEntryProvider,
    ReferenceEntryProvider,
)
from .validation import PropertyValidationError, validate_property, validate_record

__all__ = [
    "ReferenceEntryProvider",
    "FileEntryProvider",
    "CalculationEntryProvider",
    "PropertyValidationError",
    "validate_property",
    "validate_record",
]
