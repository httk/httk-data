"""Serve standard entry types through in-memory :class:`~httk.core.EntryProvider` implementations.

These providers map ``{id: record}`` mappings of the stdlib-only record models
defined in *httk-core* (:class:`~httk.core.Reference`, :class:`~httk.core.File`,
:class:`~httk.core.Calculation`) onto the neutral httk-core entry-provider
contract, so a serving module (such as *httk-serve*) can expose them as
OPTIMADE ``references``/``files``/``calculations`` endpoints without either side
depending on the other. Each provider describes its entry type with the vendored
OPTIMADE standard definition loaded from httk-core via
:func:`~httk.core.standard_entry_type`.

The record *models* live in httk-core (contracts and models); these *providers*
live in httk-store (the capability layer built on those models), together with
property-definition validation. The database storage layer in
:mod:`httk.store.db` complements them with a database-backed provider
(:class:`~httk.store.db.entry_provider.StoreEntryProvider`) serving stored
dataclasses the same way.
"""

import datetime
from collections.abc import Iterable, Mapping
from dataclasses import fields
from functools import cache
from typing import Any

from httk.core import (
    Calculation,
    DataRecord,
    EntryProvider,
    EntryTypeDefinition,
    File,
    ProductLink,
    PropertyDefinition,
    Reference,
    RelatedEntry,
    Run,
    load_entry_type_definition,
    load_property_definition,
    standard_entry_type,
)
from httk.core.data_records import RECORDS_DEFINITION_ID
from httk.core.provenance import RUNS_DEFINITION_ID

from httk.store.query import ID_FIELD


def _normalized_relationships(
    relationships: Mapping[str, Iterable[RelatedEntry]] | None,
) -> dict[str, tuple[RelatedEntry, ...]]:
    """Normalize a caller-supplied relationships mapping to ``{str(id): tuple(entries)}``."""
    if relationships is None:
        return {}
    return {str(key): tuple(value) for key, value in relationships.items()}


def _provider_property_keys(record_type: type[Any]) -> dict[str, str]:
    """The served-property-name to record-key map for a standard entry type."""
    property_keys = {"id": ID_FIELD, "type": "type"}
    property_keys.update({field.name: field.name for field in fields(record_type)})
    return property_keys


def _json_value(value: Any) -> Any:
    """A record value as one of the JSON types the provider contract promises.

    The record models declare their sequence fields as tuples (immutable
    records), but :meth:`~httk.core.EntryProvider.records` is contracted to
    yield plain JSON-able values, and a JSON array is a ``list``. Passing a
    tuple through reaches a consumer that type-checks against the property
    definition — :func:`~httk.store.validation.validate_record` does — and is
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
        row: dict[str, Any] = {ID_FIELD: entry_id, "type": entry_type}
        for name in field_names:
            row[name] = _json_value(getattr(record, name))
        sha256 = getattr(record, "sha256", None)
        if entry_type == "files" and sha256 and row["checksums"] is None:
            row["checksums"] = {"sha256": sha256}
        records.append(row)
    return records


class StandardEntryProvider(EntryProvider):
    """Serve one standard entry type through the neutral provider contract.

    :param entries: The records keyed by their served identifiers.
    :param record_type: The core record class used to construct each entry.
    :param entry_type: The OPTIMADE entry type served by this provider.
    :param relationships: Optional related entries keyed by served identifier.
    """

    def __init__(
        self,
        entries: Mapping[str, Any],
        *,
        record_type: type[Any],
        entry_type: str,
        relationships: Mapping[str, Iterable[RelatedEntry]] | None,
    ) -> None:
        self._record_type = record_type
        self._entry_type = entry_type
        self._entries = {str(key): record_type.create(value) for key, value in entries.items()}
        self._relationships = _normalized_relationships(relationships)

    def _check_entry_type(self, entry_type: str) -> None:
        if entry_type != self._entry_type:
            raise KeyError(f"{type(self).__name__} serves only the '{self._entry_type}' entry type.")

    def entry_types(self) -> Mapping[str, EntryTypeDefinition]:
        """Return the one standard entry-type definition served by this provider.

        :return: The served entry-type definition.
        """
        return {self._entry_type: standard_entry_type(self._entry_type)}

    def property_keys(self, entry_type: str) -> Mapping[str, str]:
        """Return served property names mapped to record attribute names.

        :param entry_type: The entry type to inspect.
        :return: The served-property to record-key mapping.
        :raises KeyError: If ``entry_type`` is not this provider's entry type.
        """
        self._check_entry_type(entry_type)
        return _provider_property_keys(self._record_type)

    def records(self, entry_type: str) -> Iterable[Mapping[str, Any]]:
        """Return JSON-compatible records for the requested entry type.

        :param entry_type: The entry type to enumerate.
        :return: The provider's records in mapping iteration order.
        :raises KeyError: If ``entry_type`` is not this provider's entry type.
        """
        self._check_entry_type(entry_type)
        return _provider_records(self._entry_type, self._record_type, self._entries)

    def relationships(self, entry_type: str) -> Mapping[str, tuple[RelatedEntry, ...]]:
        """Return related entries keyed by served identifier.

        :param entry_type: The entry type to inspect.
        :return: The normalized relationship mapping.
        :raises KeyError: If ``entry_type`` is not this provider's entry type.
        """
        self._check_entry_type(entry_type)
        return self._relationships


class ReferenceEntryProvider(StandardEntryProvider):
    """Serves OPTIMADE ``references`` from a mapping of id to :class:`~httk.core.Reference`.

    ``relationships`` optionally maps a reference id to its related entries
    (:class:`~httk.core.RelatedEntry` values, served flat per id).

    :param entries: The references keyed by their served identifiers.
    :param relationships: Optional related entries keyed by reference identifier.
    """

    def __init__(
        self,
        entries: Mapping[str, Reference | Mapping[str, Any]],
        *,
        relationships: Mapping[str, Iterable[RelatedEntry]] | None = None,
    ) -> None:
        super().__init__(entries, record_type=Reference, entry_type="references", relationships=relationships)


class FileEntryProvider(StandardEntryProvider):
    """Serves OPTIMADE ``files`` from a mapping of id to :class:`~httk.core.File`.

    ``relationships`` optionally maps a file id to its related entries
    (:class:`~httk.core.RelatedEntry` values, served flat per id) — e.g. the
    calculations a file is ``input``/``output`` of.

    :param entries: The files keyed by their served identifiers.
    :param relationships: Optional related entries keyed by file identifier.
    """

    def __init__(
        self,
        entries: Mapping[str, File | Mapping[str, Any]],
        *,
        relationships: Mapping[str, Iterable[RelatedEntry]] | None = None,
    ) -> None:
        super().__init__(entries, record_type=File, entry_type="files", relationships=relationships)


class CalculationEntryProvider(StandardEntryProvider):
    """Serves OPTIMADE ``calculations`` from a mapping of id to :class:`~httk.core.Calculation`.

    ``relationships`` optionally maps a calculation id to its related entries
    (:class:`~httk.core.RelatedEntry` values, served flat per id) — e.g. its
    ``input``/``output`` files, expressed via the ``role`` metadata.

    :param entries: The calculations keyed by their served identifiers.
    :param relationships: Optional related entries keyed by calculation identifier.
    """

    def __init__(
        self,
        entries: Mapping[str, Calculation | Mapping[str, Any]],
        *,
        relationships: Mapping[str, Iterable[RelatedEntry]] | None = None,
    ) -> None:
        super().__init__(entries, record_type=Calculation, entry_type="calculations", relationships=relationships)


def _entry_definition(
    entry_type: str,
    definition_id: str,
    properties: Mapping[str, PropertyDefinition],
) -> EntryTypeDefinition:
    base = load_entry_type_definition(definition_id)
    return EntryTypeDefinition(
        entry_type,
        base.description,
        properties,
        definition_id=None,
        extends_id=definition_id,
    )


@cache
def _runs_definition() -> EntryTypeDefinition:
    base = load_entry_type_definition(RUNS_DEFINITION_ID)
    properties = {name: base.properties[name] for name in ("id", "type", "immutable_id", "last_modified")}
    workflow = base.properties["workflow_declaration_uri"]
    properties["_httk_workflow_declaration_uri"] = PropertyDefinition.from_optimade(
        "_httk_workflow_declaration_uri", workflow.as_optimade()
    )
    return _entry_definition("_httk_runs", RUNS_DEFINITION_ID, properties)


class RunEntryProvider(EntryProvider):
    """Serve core :class:`~httk.core.Run` records and their provenance edges.

    :param entries: The runs keyed by their served identifiers.
    """

    _entry_type = "_httk_runs"

    def __init__(self, entries: Mapping[str, Run | Mapping[str, Any]]) -> None:
        self._entries = {str(key): Run.create(value) for key, value in entries.items()}

    def _check_entry_type(self, entry_type: str) -> None:
        if entry_type != self._entry_type:
            raise KeyError(f"{type(self).__name__} serves only the '{self._entry_type}' entry type.")

    def entry_types(self) -> Mapping[str, EntryTypeDefinition]:
        """Return the vendored ``_httk_runs`` entry definition.

        :return: The served run entry-type definition.
        """
        return {self._entry_type: _runs_definition()}

    def property_keys(self, entry_type: str) -> Mapping[str, str]:
        """Return the served run-property to record-key mapping.

        :param entry_type: The entry type to inspect.
        :return: The served-property to record-key mapping.
        :raises KeyError: If ``entry_type`` is not ``_httk_runs``.
        """
        self._check_entry_type(entry_type)
        return {
            "id": ID_FIELD,
            "type": "type",
            "immutable_id": "immutable_id",
            "last_modified": "last_modified",
            "_httk_workflow_declaration_uri": "workflow_declaration_uri",
        }

    def records(self, entry_type: str) -> Iterable[Mapping[str, Any]]:
        """Return JSON-compatible run records.

        :param entry_type: The entry type to enumerate.
        :yield: Run records in input mapping order.
        :raises KeyError: If ``entry_type`` is not ``_httk_runs``.
        """
        self._check_entry_type(entry_type)
        for entry_id, run in self._entries.items():
            yield {
                ID_FIELD: entry_id,
                "type": self._entry_type,
                "immutable_id": run.immutable_id,
                "last_modified": _json_value(run.last_modified),
                "workflow_declaration_uri": run.workflow_declaration_uri,
            }

    def relationships(self, entry_type: str) -> Mapping[str, tuple[RelatedEntry, ...]]:
        """Return run provenance edges with role and edge-label metadata.

        :param entry_type: The entry type to inspect.
        :return: Relationships grouped by run identifier.
        :raises KeyError: If ``entry_type`` is not ``_httk_runs``.
        """
        self._check_entry_type(entry_type)
        return {
            entry_id: tuple(
                RelatedEntry(edge.entry_type, edge.entry_id, role=role, label=edge.label)
                for role, edges in (("input", run.inputs), ("artifact", run.artifacts), ("output", run.outputs))
                for edge in edges
            )
            for entry_id, run in self._entries.items()
        }


class DataRecordEntryProvider(EntryProvider):
    """Serve core :class:`~httk.core.DataRecord` values as provider properties.

    Definitions are resolved eagerly at construction. Every served property name
    must start with ``_``; absent record properties are emitted as JSON null.

    :param entries: The data records keyed by their served identifiers.
    :param definitions: Optional property definitions keyed by served property name.
    :param relationships: Optional related entries keyed by record identifier.
    :raises ValueError: If a property name, definition, or non-nullable property
        is inconsistent with the supplied records.
    """

    _entry_type = "_httk_records"

    def __init__(
        self,
        entries: Mapping[str, DataRecord | Mapping[str, Any]],
        *,
        definitions: Mapping[str, PropertyDefinition] | None = None,
        relationships: Mapping[str, Iterable[RelatedEntry]] | None = None,
    ) -> None:
        self._entries = {str(key): DataRecord.create(value) for key, value in entries.items()}
        self._relationships = _normalized_relationships(relationships)
        resolved = dict(definitions or {})
        for name in resolved:
            if not name.startswith("_"):
                raise ValueError(f"served property name {name!r} must start with '_'")
        for key, record in self._entries.items():
            definition = resolved.get(record.name)
            if definition is None:
                try:
                    definition = PropertyDefinition.from_optimade(
                        record.name, load_property_definition(record.definition_id).as_optimade()
                    )
                except Exception as exc:
                    raise ValueError(
                        f"record {key!r} name {record.name!r} has no registered definition "
                        f"for IRI {record.definition_id!r}"
                    ) from exc
                resolved[record.name] = definition
            if definition.definition_id and record.definition_id != definition.definition_id:
                raise ValueError(
                    f"record {key!r} name {record.name!r} has definition IRI {record.definition_id!r}, "
                    f"but resolved definition is {definition.definition_id!r}"
                )
        for name, definition in resolved.items():
            if not definition.nullable:
                missing = next((key for key, record in self._entries.items() if record.name != name), None)
                if missing is not None:
                    raise ValueError(
                        f"served property {name!r} is non-nullable, but record {missing!r} does not populate it"
                    )
        self._definitions = resolved
        base = load_entry_type_definition(RECORDS_DEFINITION_ID)
        properties = dict(base.properties)
        properties.update(resolved)
        self._definition = _entry_definition(self._entry_type, RECORDS_DEFINITION_ID, properties)

    def _check_entry_type(self, entry_type: str) -> None:
        if entry_type != self._entry_type:
            raise KeyError(f"{type(self).__name__} serves only the '{self._entry_type}' entry type.")

    def entry_types(self) -> Mapping[str, EntryTypeDefinition]:
        """Return the resolved ``_httk_records`` entry definition.

        :return: The served data-record entry-type definition.
        """
        return {self._entry_type: self._definition}

    def property_keys(self, entry_type: str) -> Mapping[str, str]:
        """Return served property names mapped to data-record keys.

        :param entry_type: The entry type to inspect.
        :return: The served-property to record-key mapping.
        :raises KeyError: If ``entry_type`` is not ``_httk_records``.
        """
        self._check_entry_type(entry_type)
        return {
            "id": ID_FIELD,
            "type": "type",
            "immutable_id": "immutable_id",
            "last_modified": "last_modified",
            **{name: name for name in self._definitions},
        }

    def records(self, entry_type: str) -> Iterable[Mapping[str, Any]]:
        """Return records with union-null values for unserved properties.

        :param entry_type: The entry type to enumerate.
        :yield: JSON-compatible records in input mapping order.
        :raises KeyError: If ``entry_type`` is not ``_httk_records``.
        """
        self._check_entry_type(entry_type)
        for entry_id, record in self._entries.items():
            yield {
                ID_FIELD: entry_id,
                "type": self._entry_type,
                "immutable_id": record.immutable_id,
                "last_modified": _json_value(record.last_modified),
                **{name: _json_value(record.value) if name == record.name else None for name in self._definitions},
            }

    def relationships(self, entry_type: str) -> Mapping[str, tuple[RelatedEntry, ...]]:
        """Return normalized data-record relationships by identifier.

        :param entry_type: The entry type to inspect.
        :return: The relationship mapping supplied at construction.
        :raises KeyError: If ``entry_type`` is not ``_httk_records``.
        """
        self._check_entry_type(entry_type)
        return self._relationships


def product_relationships(links: Iterable[ProductLink]) -> dict[str, dict[str, tuple[RelatedEntry, ...]]]:
    """Build source-side relationships for a provider's ``relationships=`` argument.

    Feed the inner mapping into the source-side provider's ``relationships=`` argument;
    per-edge ``workflow_declaration_uri`` is deliberately not served yet (relation-object
    serving is future work).

    :param links: The product links to group by source type and identifier.
    :return: Source-type mappings of source identifiers to related product entries.
    :raises ValueError: If one source has duplicate product labels.
    """
    result: dict[str, dict[str, list[RelatedEntry]]] = {}
    for link in links:
        source = result.setdefault(link.source_type, {}).setdefault(link.source_id, [])
        if any(entry.label == link.label for entry in source):
            raise ValueError(
                f"duplicate product label for source {link.source_type!r}/{link.source_id!r}: {link.label!r}"
            )
        source.append(RelatedEntry(link.target_type, link.target_id, role="product", label=link.label))
    return {
        source_type: {source_id: tuple(entries) for source_id, entries in sources.items()}
        for source_type, sources in result.items()
    }
