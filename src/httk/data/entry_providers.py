"""In-memory :class:`~httk.core.EntryProvider` implementations for the standard entry types.

These providers map ``{id: record}`` mappings of the stdlib-only record models
defined in *httk-core* (:class:`~httk.core.Reference`, :class:`~httk.core.File`,
:class:`~httk.core.Calculation`) onto the neutral httk-core entry-provider
contract, so a serving module (such as *httk-optimade*) can expose them as
OPTIMADE ``references``/``files``/``calculations`` endpoints without either side
depending on the other. Each provider describes its entry type with the vendored
OPTIMADE standard definition loaded from httk-core via
:func:`~httk.core.standard_entry_type`.

The record *models* live in httk-core (contracts and models); these *providers*
live in httk-data (the capability layer built on those models), together with
property-definition validation. httk-data is also the intended future home of
the v1-style sqlite/database storage layer, which is not built yet.
"""

from collections.abc import Iterable, Mapping
from dataclasses import fields
from typing import Any

from httk.core import (
    Calculation,
    EntryProvider,
    EntryTypeDefinition,
    File,
    Reference,
    standard_entry_type,
)


def _provider_columns(record_type: type[Any]) -> dict[str, str]:
    """The served-property to record-column map for a standard entry type."""
    columns = {"id": "__id", "type": "type"}
    columns.update({field.name: field.name for field in fields(record_type)})
    return columns


def _provider_records(entry_type: str, record_type: type[Any], entries: Mapping[str, Any]) -> list[dict[str, Any]]:
    """JSON-able records for a standard entry type, one per stored instance."""
    field_names = [field.name for field in fields(record_type)]
    records: list[dict[str, Any]] = []
    for entry_id, record in entries.items():
        row: dict[str, Any] = {"__id": entry_id, "type": entry_type}
        for name in field_names:
            row[name] = getattr(record, name)
        records.append(row)
    return records


class ReferenceEntryProvider(EntryProvider):
    """Serves OPTIMADE ``references`` from a mapping of id to :class:`~httk.core.Reference`."""

    def __init__(self, entries: Mapping[str, Reference | Mapping[str, Any]]) -> None:
        self._entries: dict[str, Reference] = {str(key): Reference.create(value) for key, value in entries.items()}

    def entry_types(self) -> Mapping[str, EntryTypeDefinition]:
        return {"references": standard_entry_type("references")}

    def columns(self, entry_type: str) -> Mapping[str, str]:
        if entry_type != "references":
            raise KeyError("ReferenceEntryProvider serves only the 'references' entry type.")
        return _provider_columns(Reference)

    def records(self, entry_type: str) -> Iterable[Mapping[str, Any]]:
        if entry_type != "references":
            raise KeyError("ReferenceEntryProvider serves only the 'references' entry type.")
        return _provider_records("references", Reference, self._entries)


class FileEntryProvider(EntryProvider):
    """Serves OPTIMADE ``files`` from a mapping of id to :class:`~httk.core.File`."""

    def __init__(self, entries: Mapping[str, File | Mapping[str, Any]]) -> None:
        self._entries: dict[str, File] = {str(key): File.create(value) for key, value in entries.items()}

    def entry_types(self) -> Mapping[str, EntryTypeDefinition]:
        return {"files": standard_entry_type("files")}

    def columns(self, entry_type: str) -> Mapping[str, str]:
        if entry_type != "files":
            raise KeyError("FileEntryProvider serves only the 'files' entry type.")
        return _provider_columns(File)

    def records(self, entry_type: str) -> Iterable[Mapping[str, Any]]:
        if entry_type != "files":
            raise KeyError("FileEntryProvider serves only the 'files' entry type.")
        return _provider_records("files", File, self._entries)


class CalculationEntryProvider(EntryProvider):
    """Serves OPTIMADE ``calculations`` from a mapping of id to :class:`~httk.core.Calculation`."""

    def __init__(self, entries: Mapping[str, Calculation | Mapping[str, Any]]) -> None:
        self._entries: dict[str, Calculation] = {str(key): Calculation.create(value) for key, value in entries.items()}

    def entry_types(self) -> Mapping[str, EntryTypeDefinition]:
        return {"calculations": standard_entry_type("calculations")}

    def columns(self, entry_type: str) -> Mapping[str, str]:
        if entry_type != "calculations":
            raise KeyError("CalculationEntryProvider serves only the 'calculations' entry type.")
        return _provider_columns(Calculation)

    def records(self, entry_type: str) -> Iterable[Mapping[str, Any]]:
        if entry_type != "calculations":
            raise KeyError("CalculationEntryProvider serves only the 'calculations' entry type.")
        return _provider_records("calculations", Calculation, self._entries)
