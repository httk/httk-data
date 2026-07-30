"""In-memory :class:`~httk.core.EntryProvider` implementations for the standard entry types.

These providers map ``{id: record}`` mappings of the stdlib-only record models
defined in *httk-core* (:class:`~httk.core.Reference`, :class:`~httk.core.File`,
:class:`~httk.core.Calculation`) onto the neutral httk-core entry-provider
contract, so a serving module (such as *httk-serve*) can expose them as
OPTIMADE ``references``/``files``/``calculations`` endpoints without either side
depending on the other. Each provider describes its entry type with the vendored
OPTIMADE standard definition loaded from httk-core via
:func:`~httk.core.standard_entry_type`.

The record *models* live in httk-core (contracts and models); these *providers*
live in httk-data (the capability layer built on those models), together with
property-definition validation. The database storage layer in
:mod:`httk.data.db` complements them with a database-backed provider
(:class:`~httk.data.db.entry_provider.StoreEntryProvider`) serving stored
dataclasses the same way.
"""

import datetime
from collections.abc import Iterable, Mapping
from dataclasses import fields
from typing import Any

from httk.core import (
    Calculation,
    EntryProvider,
    EntryTypeDefinition,
    File,
    Reference,
    RelatedEntry,
    standard_entry_type,
)


def _normalized_relationships(
    relationships: Mapping[str, Iterable[RelatedEntry]] | None,
) -> dict[str, tuple[RelatedEntry, ...]]:
    """Normalize a caller-supplied relationships mapping to ``{str(id): tuple(entries)}``."""
    if relationships is None:
        return {}
    return {str(key): tuple(value) for key, value in relationships.items()}


def _provider_property_keys(record_type: type[Any]) -> dict[str, str]:
    """The served-property-name to record-key map for a standard entry type."""
    property_keys = {"id": "__id", "type": "type"}
    property_keys.update({field.name: field.name for field in fields(record_type)})
    return property_keys


def _json_value(value: Any) -> Any:
    """A record value as one of the JSON types the provider contract promises.

    The record models declare their sequence fields as tuples (immutable
    records), but :meth:`~httk.core.EntryProvider.records` is contracted to
    yield plain JSON-able values, and a JSON array is a ``list``. Passing a
    tuple through reaches a consumer that type-checks against the property
    definition — :func:`~httk.data.validation.validate_record` does — and is
    rejected as "not of type 'array'", even though it serializes fine.
    """
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, Mapping):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, datetime.datetime):
        return value.isoformat()
    return value


def _provider_records(entry_type: str, record_type: type[Any], entries: Mapping[str, Any]) -> list[dict[str, Any]]:
    """JSON-able records for a standard entry type, one per stored instance."""
    field_names = [field.name for field in fields(record_type)]
    records: list[dict[str, Any]] = []
    for entry_id, record in entries.items():
        row: dict[str, Any] = {"__id": entry_id, "type": entry_type}
        for name in field_names:
            row[name] = _json_value(getattr(record, name))
        records.append(row)
    return records


class ReferenceEntryProvider(EntryProvider):
    """Serves OPTIMADE ``references`` from a mapping of id to :class:`~httk.core.Reference`.

    ``relationships`` optionally maps a reference id to its related entries
    (:class:`~httk.core.RelatedEntry` values, served flat per id).
    """

    def __init__(
        self,
        entries: Mapping[str, Reference | Mapping[str, Any]],
        *,
        relationships: Mapping[str, Iterable[RelatedEntry]] | None = None,
    ) -> None:
        self._entries: dict[str, Reference] = {str(key): Reference.create(value) for key, value in entries.items()}
        self._relationships = _normalized_relationships(relationships)

    def entry_types(self) -> Mapping[str, EntryTypeDefinition]:
        return {"references": standard_entry_type("references")}

    def property_keys(self, entry_type: str) -> Mapping[str, str]:
        if entry_type != "references":
            raise KeyError("ReferenceEntryProvider serves only the 'references' entry type.")
        return _provider_property_keys(Reference)

    def records(self, entry_type: str) -> Iterable[Mapping[str, Any]]:
        if entry_type != "references":
            raise KeyError("ReferenceEntryProvider serves only the 'references' entry type.")
        return _provider_records("references", Reference, self._entries)

    def relationships(self, entry_type: str) -> Mapping[str, tuple[RelatedEntry, ...]]:
        if entry_type != "references":
            raise KeyError("ReferenceEntryProvider serves only the 'references' entry type.")
        return self._relationships


class FileEntryProvider(EntryProvider):
    """Serves OPTIMADE ``files`` from a mapping of id to :class:`~httk.core.File`.

    ``relationships`` optionally maps a file id to its related entries
    (:class:`~httk.core.RelatedEntry` values, served flat per id) — e.g. the
    calculations a file is ``input``/``output`` of.
    """

    def __init__(
        self,
        entries: Mapping[str, File | Mapping[str, Any]],
        *,
        relationships: Mapping[str, Iterable[RelatedEntry]] | None = None,
    ) -> None:
        self._entries: dict[str, File] = {str(key): File.create(value) for key, value in entries.items()}
        self._relationships = _normalized_relationships(relationships)

    def entry_types(self) -> Mapping[str, EntryTypeDefinition]:
        return {"files": standard_entry_type("files")}

    def property_keys(self, entry_type: str) -> Mapping[str, str]:
        if entry_type != "files":
            raise KeyError("FileEntryProvider serves only the 'files' entry type.")
        return _provider_property_keys(File)

    def records(self, entry_type: str) -> Iterable[Mapping[str, Any]]:
        if entry_type != "files":
            raise KeyError("FileEntryProvider serves only the 'files' entry type.")
        return _provider_records("files", File, self._entries)

    def relationships(self, entry_type: str) -> Mapping[str, tuple[RelatedEntry, ...]]:
        if entry_type != "files":
            raise KeyError("FileEntryProvider serves only the 'files' entry type.")
        return self._relationships


class CalculationEntryProvider(EntryProvider):
    """Serves OPTIMADE ``calculations`` from a mapping of id to :class:`~httk.core.Calculation`.

    ``relationships`` optionally maps a calculation id to its related entries
    (:class:`~httk.core.RelatedEntry` values, served flat per id) — e.g. its
    ``input``/``output`` files, expressed via the ``role`` metadata.
    """

    def __init__(
        self,
        entries: Mapping[str, Calculation | Mapping[str, Any]],
        *,
        relationships: Mapping[str, Iterable[RelatedEntry]] | None = None,
    ) -> None:
        self._entries: dict[str, Calculation] = {str(key): Calculation.create(value) for key, value in entries.items()}
        self._relationships = _normalized_relationships(relationships)

    def entry_types(self) -> Mapping[str, EntryTypeDefinition]:
        return {"calculations": standard_entry_type("calculations")}

    def property_keys(self, entry_type: str) -> Mapping[str, str]:
        if entry_type != "calculations":
            raise KeyError("CalculationEntryProvider serves only the 'calculations' entry type.")
        return _provider_property_keys(Calculation)

    def records(self, entry_type: str) -> Iterable[Mapping[str, Any]]:
        if entry_type != "calculations":
            raise KeyError("CalculationEntryProvider serves only the 'calculations' entry type.")
        return _provider_records("calculations", Calculation, self._entries)

    def relationships(self, entry_type: str) -> Mapping[str, tuple[RelatedEntry, ...]]:
        if entry_type != "calculations":
            raise KeyError("CalculationEntryProvider serves only the 'calculations' entry type.")
        return self._relationships
