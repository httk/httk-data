"""Backend-neutral store declaration machinery shared by storage backends.

This module owns the logical entry-family declaration, its canonical JSON
encoding, and trust-on-reopen validation.  Physical names and backend-specific
layout validation belong to each storage backend.
"""

import dataclasses
import json
import sys
from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from httk.core.register import (
    entry_family_info,
    entry_record_info,
    known_entry_families,
    known_entry_records,
    resolve_entry_family,
    resolve_entry_record,
)

from httk.data.db.schema import resolve_schema

__all__ = [
    "DECLARATION_PROTOCOL_VERSION",
    "EntryFamilyLayout",
    "StorageLayout",
    "StorageLayoutUpgradeRequiredError",
    "declaration_json",
    "normalize_entry_records",
]

DECLARATION_PROTOCOL_VERSION: Final = "v2.1.0"
"""The current backend-neutral declaration protocol."""


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
            )
        )
    entries.sort(key=lambda entry: entry.name)
    if len({entry.name for entry in entries}) != len(entries):
        raise ValueError("entry_records contains multiple registered aliases for the same family")
    return StorageLayout(DECLARATION_PROTOCOL_VERSION, tuple(entries))


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


def _registered_family_name(family: type) -> str:
    matches = _registered_names_for(family, known_entry_families(), entry_family_info)
    if len(matches) != 1:
        found = ", ".join(matches) or "none"
        raise ValueError(f"entry family {family.__name__} must resolve to exactly one registered name (found {found})")
    return matches[0]


def _registered_record_name(record: type) -> str:
    matches = _registered_names_for(record, known_entry_records(), entry_record_info)
    if len(matches) != 1:
        found = ", ".join(matches) or "none"
        raise ValueError(f"entry record {record.__name__} must resolve to exactly one registered name (found {found})")
    return matches[0]


def _registered_names_for(record: type, names: list[str], info: object) -> list[str]:
    """Return registry names for ``record`` without importing unrelated lazy entries.

    A store declaration already has the concrete class in hand.  Resolving
    every registry reference just to find its stable name turns that harmless
    validation into a transitive import of every optional entry package.  Some
    such packages are deliberately heavyweight; more importantly, repeated
    store construction must not retain their import-time state.

    Registry references conventionally name the class's defining module.  A
    loaded alias remains supported by identity, while an unloaded unrelated
    entry is never imported merely for declaration validation.
    """
    get_info = info
    if not callable(get_info):  # pragma: no cover - defensive narrowing for the registry seam
        raise TypeError("registry info lookup must be callable")
    canonical = f"{record.__module__}:{record.__name__}"
    matches: list[str] = []
    for name in names:
        reference = get_info(name)[0]
        if reference == canonical:
            matches.append(name)
            continue
        module_name, separator, attribute = reference.partition(":")
        module = sys.modules.get(module_name) if separator else None
        if module is not None and getattr(module, attribute, None) is record:
            matches.append(name)
    return matches


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
