"""Backend-neutral store declaration machinery shared by storage backends.

This module owns the logical entry-family declaration, its canonical JSON
encoding, and trust-on-reopen validation.  Physical names and backend-specific
layout validation belong to each storage backend.
"""

import dataclasses
import json
import sys
import typing
from collections.abc import Callable, Mapping, Sequence
from types import MappingProxyType
from typing import Any, Final, cast

from httk.core.register import (
    entry_family_info,
    entry_record_info,
    known_entry_families,
    known_entry_records,
    resolve_entry_family,
    resolve_entry_record,
)
from httk.core.storage import storage_identity_name

from httk.store.backend.schema import ChildTableSpec, ColumnSpec, FieldSpec, TableSchema, resolve_schema

__all__ = [
    "ADDITIVE_UPGRADE_HINT",
    "DECLARATION_PROTOCOL_VERSION",
    "AdditiveUpgradePlan",
    "EntryFamilyDeclaration",
    "EntryFamilyLayout",
    "EntryLayoutBindingError",
    "EntryRecordDeclaration",
    "StorageLayout",
    "StorageLayoutUpgradeRequiredError",
    "classify_schema_upgrade",
    "declaration_json",
    "normalize_entry_families",
    "normalize_entry_records",
    "schema_fingerprint_diff",
    "schema_fingerprint_json",
]

DECLARATION_PROTOCOL_VERSION: Final = "v2.2.0"
"""The current backend-neutral declaration protocol."""

ADDITIVE_UPGRADE_HINT: Final = (
    "the schema difference is purely additive (new nullable columns / lazily created tables); "
    "reopen with upgrade=True to apply it"
)
"""The reopen hint appended when an additive-only schema mismatch is not applied."""


class StorageLayoutUpgradeRequiredError(RuntimeError):
    """A database does not exactly implement the current persisted store layout.

    ``diff`` is immutable and JSON-shaped.  Its top-level keys are stable
    categories (currently ``protocol``, ``declaration`` and ``schema``), so a
    caller can present a precise upgrade diagnostic without parsing
    the human-readable exception message.  The ``declaration`` category maps
    named aspect keys (``metadata_keys``, ``store_timestamps``,
    ``write_profile``, ``entry_declaration``) to their own diagnostics, so
    several independent declaration mismatches are reported together; the
    exception message names the mismatched aspects.

    :param diff: The immutable JSON-shaped category-keyed difference.
    :param hint: An optional remediation appended to the message and exposed as
        :attr:`hint` (e.g. that a purely additive schema change can be applied
        with ``upgrade=True``).
    """

    def __init__(self, diff: Mapping[str, object], *, hint: str | None = None) -> None:
        frozen = _freeze_mapping(diff)
        self.diff: Mapping[str, object] = frozen
        self.hint: str | None = hint
        categories = ", ".join(frozen) or "unknown layout difference"
        details = frozen.get("declaration")
        detail_names = f": {', '.join(details)}" if isinstance(details, Mapping) else ""
        hint_text = f"; {hint}" if hint else ""
        super().__init__(f"Store layout upgrade is required ({categories}{detail_names}){hint_text}")


class EntryLayoutBindingError(ValueError):
    """A persisted application-owned layout needs explicit Python class bindings."""


@dataclasses.dataclass(frozen=True)
class EntryRecordDeclaration:
    """Bind one stable store-local name to a concrete record class.

    :param name: Stable record identity persisted in the store declaration.
    :param record: Concrete frozen dataclass used for storage and hydration.
    :param definition_id: Optional entry-type definition IRI described by the record.
    """

    name: str
    record: type
    definition_id: str | None = None

    def __post_init__(self) -> None:
        _validate_name(self.name, label="entry record name")
        if not isinstance(self.record, type):
            raise TypeError("entry record must be a class")
        params = getattr(self.record, "__dataclass_params__", None)
        if not dataclasses.is_dataclass(self.record) or params is None or not params.frozen:
            raise TypeError("entry record must be a frozen dataclass class")
        _validate_optional_definition_id(self.definition_id)


@dataclasses.dataclass(frozen=True)
class EntryFamilyDeclaration:
    """Declare one application-owned entry family without global registration.

    Explicit declarations provide stable persistence identities directly to a
    store.  They are intended for application-private families which should
    not participate in plugin discovery.  The same declaration must be
    supplied whenever such a store is reopened.

    :param name: Stable family identity persisted in the store declaration.
    :param family: Logical entry-family class exposed through ``entry_layout``.
    :param records: Ordered concrete record declarations belonging to the family.
    :param definition_id: Optional entry-type definition IRI for the family.
    """

    name: str
    family: type
    records: tuple[EntryRecordDeclaration, ...]
    definition_id: str | None = None

    def __post_init__(self) -> None:
        _validate_name(self.name, label="entry family name")
        if not isinstance(self.family, type):
            raise TypeError("entry family must be a class")
        if not isinstance(self.records, tuple) or not self.records:
            raise ValueError("entry family records must be a nonempty tuple")
        if any(not isinstance(record, EntryRecordDeclaration) for record in self.records):
            raise TypeError("entry family records must contain EntryRecordDeclaration values")
        if len({record.name for record in self.records}) != len(self.records):
            raise ValueError("entry family repeats a record name")
        if len({record.record for record in self.records}) != len(self.records):
            raise ValueError("entry family repeats a record class")
        _validate_optional_definition_id(self.definition_id)


@dataclasses.dataclass(frozen=True)
class EntryFamilyLayout:
    """One immutable configured entry family and its concrete records."""

    name: str
    family: type
    definition_id: str | None
    record_names: tuple[str, ...]
    records: tuple[type, ...]
    record_definition_ids: tuple[str | None, ...]


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
    declarations: list[EntryFamilyDeclaration] = []
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
        record_declarations: list[EntryRecordDeclaration] = []
        for record in records:
            record_name = _registered_record_name(record)
            _, registered_family_name, definition_id = entry_record_info(record_name)
            if registered_family_name is None:
                raise ValueError(
                    f"entry record {record_name!r} has no registered family and cannot be used in a family store"
                )
            if registered_family_name != family_name:
                raise ValueError(
                    f"entry record {record.__name__} belongs to registered family {registered_family_name!r}, "
                    f"not {family_name!r}"
                )
            record_declarations.append(
                EntryRecordDeclaration(name=record_name, record=record, definition_id=definition_id)
            )
        _, family_definition_id = entry_family_info(family_name)
        declarations.append(
            EntryFamilyDeclaration(
                name=family_name,
                family=family,
                records=tuple(record_declarations),
                definition_id=family_definition_id,
            )
        )
    return _normalize_entry_families(declarations, explicit=False)


def normalize_entry_families(entry_families: Sequence[EntryFamilyDeclaration]) -> StorageLayout:
    """Validate application-owned entry declarations and build a store layout.

    Unlike :func:`normalize_entry_records`, this path does not require the
    family or record classes to be globally registered.  Stable names and
    optional definition identities are supplied by the application itself.

    :param entry_families: Explicit family declarations in any order.
    :return: The immutable normalized storage layout.
    :raises TypeError: If the declaration container or its members are invalid.
    :raises ValueError: If names, classes, definitions, or storage schemas conflict.
    """
    if not isinstance(entry_families, Sequence) or isinstance(entry_families, str | bytes):
        raise TypeError("entry_families must be a sequence of EntryFamilyDeclaration values")
    if any(not isinstance(item, EntryFamilyDeclaration) for item in entry_families):
        raise TypeError("entry_families must contain EntryFamilyDeclaration values")
    return _normalize_entry_families(entry_families, explicit=True)


def _normalize_entry_families(declarations: Sequence[EntryFamilyDeclaration], *, explicit: bool) -> StorageLayout:
    entries: list[EntryFamilyLayout] = []
    for declaration in declarations:
        if explicit:
            _reject_registry_conflicts(declaration)
        for record in declaration.records:
            schema = resolve_schema(record.record)
            if schema.dedup != "content_id":
                raise ValueError(
                    f"configured entry record {record.record.__name__} must use "
                    f"dedup='content_id', got {schema.dedup!r}"
                )
        entries.append(
            EntryFamilyLayout(
                name=declaration.name,
                family=declaration.family,
                definition_id=declaration.definition_id,
                record_names=tuple(record.name for record in declaration.records),
                records=tuple(record.record for record in declaration.records),
                record_definition_ids=tuple(record.definition_id for record in declaration.records),
            )
        )
    return _storage_layout(entries)


def _merge_storage_layouts(*layouts: StorageLayout) -> StorageLayout:
    return _storage_layout([family for layout in layouts for family in layout.families])


def _storage_layout(entries: list[EntryFamilyLayout]) -> StorageLayout:
    entries.sort(key=lambda entry: entry.name)
    if len({entry.name for entry in entries}) != len(entries):
        raise ValueError("entry declaration repeats a family name")
    if len({entry.family for entry in entries}) != len(entries):
        raise ValueError("entry declaration repeats a family class")
    record_names = [name for entry in entries for name in entry.record_names]
    if len(set(record_names)) != len(record_names):
        raise ValueError("entry declaration repeats a record name")
    records = [record for entry in entries for record in entry.records]
    if len(set(records)) != len(records):
        raise ValueError("entry declaration repeats a record class")
    return StorageLayout(DECLARATION_PROTOCOL_VERSION, tuple(entries))


def declaration_json(layout: StorageLayout) -> str:
    """Serialize a normalized declaration in its exact deterministic persisted form."""
    document = {
        "families": [
            {
                "definition_id": family.definition_id,
                "family": family.name,
                "records": [
                    {"definition_id": definition_id, "record": name}
                    for name, definition_id in zip(family.record_names, family.record_definition_ids, strict=True)
                ],
            }
            for family in layout.families
        ],
        "format": 2,
    }
    return json.dumps(document, sort_keys=True, separators=(",", ":"))


def _walk_closure(layout: StorageLayout, visit: Callable[[TableSchema], None]) -> None:
    """Invoke ``visit`` once per resolved schema across the declared closure.

    Each declared record class and the transitive closure of its referenced
    storable classes is resolved and passed to ``visit`` exactly once.

    :param layout: The normalized storage layout to walk.
    :param visit: The callback invoked with each distinct resolved schema.
    :return: None.
    """
    seen: set[type] = set()

    def descend(record: type) -> None:
        if record in seen:
            return
        seen.add(record)
        schema = resolve_schema(record)
        visit(schema)
        for target in schema.referenced_classes():
            descend(target)

    for family in layout.families:
        for record in family.records:
            descend(record)


def schema_fingerprint_json(layout: StorageLayout) -> str:
    """Serialize the resolved per-table schema of ``layout`` in deterministic form.

    The fingerprint covers every declared record class plus the transitive
    closure of referenced storable classes, resolved through
    :func:`~httk.store.backend.schema.resolve_schema`.  It captures what determines
    the on-disk layout, the stored value encoding, and the *content identity* of
    each table — the logical identity name, dedup policy, composite indexes,
    relationship links, and per-field roles, codecs, shapes, columns, child
    tables, identity participation, and list-vs-tuple container — so that
    reopening a store whose record classes changed is rejected up front.

    A code move or rename is safe only when the record pins an explicit
    :attr:`~httk.core.storage.StorageInfo.identity_name` (every shipped httk
    record does); without a pin the qualified class name *is* the content
    identity, so the move changes ``content_id`` and the store correctly
    refuses to open.  ``cls`` and ``python_type`` themselves are excluded — the
    identity name and resolved columns capture everything the store depends on.

    :param layout: The normalized storage layout to fingerprint.
    :return: A deterministic ``sort_keys`` JSON document ``{"tables": {...}}``.
    """
    # Duplicate physical table names across the closure are already rejected by
    # each backend's physical-name validation, which walks the identical
    # closure; keying by table_name here needs no second collision guard.
    schemas: dict[str, TableSchema] = {}

    def collect(schema: TableSchema) -> None:
        schemas.setdefault(schema.table_name, schema)

    _walk_closure(layout, collect)
    document = {"tables": {name: _table_fingerprint(schema) for name, schema in schemas.items()}}
    return json.dumps(document, sort_keys=True, separators=(",", ":"))


def schema_fingerprint_diff(stored: str | None, current: str) -> dict[str, object]:
    """Diff a stored fingerprint against the current one, per differing table.

    The persisted fingerprint shape is versioned by the store protocol, so this
    only needs to survive a corrupt stored value gracefully.

    :param stored: The persisted fingerprint JSON, or ``None`` when absent.
    :param current: The fingerprint recomputed from the persisted layout.
    :return: A mapping of table name to ``{"expected", "actual"}`` for each table
        that differs; ``{}`` when the two fingerprints are byte-equal.  A stored
        value that is not a parseable fingerprint yields a single
        ``"<fingerprint>"`` entry.
    """
    current_tables = json.loads(current)["tables"]
    try:
        stored_tables = json.loads(stored)["tables"] if stored is not None else None
    except (TypeError, KeyError, json.JSONDecodeError):
        stored_tables = None
    if not isinstance(stored_tables, dict):
        return {"<fingerprint>": {"expected": "schema fingerprint", "actual": stored}}
    diff: dict[str, object] = {}
    for name in sorted(set(stored_tables) | set(current_tables)):
        stored_table = stored_tables.get(name)
        current_table = current_tables.get(name)
        if stored_table != current_table:
            diff[name] = {"expected": stored_table, "actual": current_table}
    return diff


def _table_fingerprint(schema: TableSchema) -> dict[str, Any]:
    """Render the identity- and layout-determining attributes of one resolved table schema."""
    hints = typing.get_type_hints(schema.cls, include_extras=True)
    return {
        "identity_name": storage_identity_name(schema.cls),
        "dedup": schema.dedup,
        "composite_indexes": [list(index) for index in schema.composite_indexes],
        "links": [dataclasses.asdict(link) for link in schema.links],
        # Keyed by field name so a pure dataclass field reorder (no column,
        # value, or identity change) does not force a store rebuild.
        "fields": {spec.field: _field_fingerprint(spec, hints.get(spec.field)) for spec in schema.fields},
    }


def _field_fingerprint(spec: FieldSpec, annotation: Any) -> dict[str, Any]:
    # Local import: store_common imports EntryFamilyLayout from this module, so a
    # module-level import would form a cycle. The helper is a pure annotation
    # inspector; importing it lazily here is order-independent.
    from httk.store.store_common import _has_identity_skip

    origin = typing.get_origin(spec.python_type)
    return {
        "role": spec.role,
        "codec_name": spec.codec_name,
        "shape": None if spec.shape is None else [spec.shape.rows, spec.shape.cols],
        "optional": spec.optional,
        "derived": spec.derived,
        # list-vs-tuple and identity participation both change content_id.
        "container": origin.__name__ if origin in (list, tuple) else None,
        "identity_skipped": _has_identity_skip(annotation),
        "columns": [_column_fingerprint(column) for column in spec.columns],
        "child": None if spec.child is None else _child_fingerprint(spec.child),
        "target": None if spec.target is None else resolve_schema(spec.target).table_name,
        "related": None if spec.related is None else dataclasses.asdict(spec.related),
    }


def _child_fingerprint(child: ChildTableSpec) -> dict[str, Any]:
    return {
        "table_name": child.table_name,
        "element_columns": [_column_fingerprint(column) for column in child.element_columns],
        "target": None if child.target is None else resolve_schema(child.target).table_name,
    }


def _column_fingerprint(column: ColumnSpec) -> dict[str, Any]:
    return {
        "name": column.name,
        "kind": column.kind,
        "nullable": column.nullable,
        "indexed": column.indexed,
        "unique": column.unique,
    }


# The immutable value keys of one fingerprinted table besides its fields; an
# additive upgrade requires every one of these to stay byte-equal.
_TABLE_INVARIANT_KEYS: Final = ("identity_name", "dedup", "composite_indexes", "links")
# Every top-level key a fingerprinted table doc is allowed to carry; a table
# growing an unrecognized key can no longer be trusted as additive.
_KNOWN_TABLE_KEYS: Final = frozenset({*_TABLE_INVARIANT_KEYS, "fields"})


@dataclasses.dataclass(frozen=True)
class AdditiveUpgradePlan:
    """The nullable parent columns an additive fingerprint upgrade must add per table.

    :param added_columns: Physical table name mapped to the ordered
        :class:`~httk.store.backend.schema.ColumnSpec` values newly present in the
        current fingerprint.  New tables carry no entry (they are created whole);
        a table appears only when it already exists in the stored fingerprint
        and gained one or more nullable parent columns.
    """

    added_columns: Mapping[str, tuple[ColumnSpec, ...]]


def classify_schema_upgrade(stored: str | None, current: str) -> AdditiveUpgradePlan | str:
    """Classify a fingerprint mismatch as an additive upgrade plan or a rejection.

    The whole diff must be additive.  A table present only in the current
    fingerprint is a new table (additive; it is created whole).  A table present
    in both is additive only when its ``identity_name``, ``dedup``,
    ``composite_indexes`` and ``links`` are byte-equal, it carries no
    unrecognized top-level key, every stored field is present and byte-equal in
    the current fingerprint, and every added field is a non-child, non-derived,
    content-identity-excluded (``IdentitySkip``) field whose parent columns are
    all nullable.  Identity participation is required so a pre-existing row's
    ``content_id`` (and therefore dedup, dispatch, and federation identity) is
    unchanged by the upgrade.

    :param stored: The persisted fingerprint JSON, or ``None`` when absent.
    :param current: The fingerprint recomputed from the persisted layout.
    :return: An :class:`AdditiveUpgradePlan` when fully additive, otherwise a
        human-readable rejection reason naming the offending table/field/column.
    """
    current_tables = json.loads(current)["tables"]
    try:
        stored_tables = json.loads(stored)["tables"] if stored is not None else None
    except (TypeError, KeyError, json.JSONDecodeError):
        stored_tables = None
    if not isinstance(stored_tables, dict):
        return "stored schema fingerprint is not parseable"
    added: dict[str, tuple[ColumnSpec, ...]] = {}
    for name in sorted(set(stored_tables) | set(current_tables)):
        stored_table = stored_tables.get(name)
        current_table = current_tables.get(name)
        if stored_table == current_table:
            continue
        if stored_table is None:
            continue  # new table: additive, created whole by the upgrade
        if current_table is None:
            return f"table {name!r} was removed"
        if not _KNOWN_TABLE_KEYS >= set(stored_table) or not _KNOWN_TABLE_KEYS >= set(current_table):
            return f"table {name!r} has an unrecognized fingerprint key"
        for key in _TABLE_INVARIANT_KEYS:
            if stored_table.get(key) != current_table.get(key):
                return f"table {name!r} changed {key}"
        stored_fields = stored_table["fields"]
        current_fields = current_table["fields"]
        columns: list[ColumnSpec] = []
        for field, field_doc in stored_fields.items():
            if field not in current_fields:
                return f"table {name!r} dropped field {field!r}"
            if current_fields[field] != field_doc:
                return f"table {name!r} changed field {field!r}"
        for field, field_doc in current_fields.items():
            if field in stored_fields:
                continue
            reason, field_columns = _added_parent_columns(name, field, field_doc)
            if reason is not None:
                return reason
            columns.extend(field_columns)
        if columns:
            added[name] = tuple(columns)
    return AdditiveUpgradePlan(added)


def _added_parent_columns(table: str, field: str, field_doc: Mapping[str, Any]) -> tuple[str | None, list[ColumnSpec]]:
    """Return the nullable parent columns an added field contributes, or a rejection reason."""
    if field_doc["role"] == "child":
        # A child field reconstructs old rows to an empty/absent collection that
        # ignores the declared type and defaults; there is no safe backfill.
        return (
            f"table {table!r} adds child field {field!r}; child-field backfill is unsupported, rebuild the store",
            [],
        )
    if field_doc["derived"]:
        # Old rows hold NULL where the true computed value differs, so queries
        # would silently under-report until every row is rewritten.
        return (
            f"table {table!r} adds derived field {field!r}; stored-property backfill is unimplemented, rebuild the store",
            [],
        )
    if not field_doc["identity_skipped"]:
        reason = (
            f"table {table!r} adds field {field!r} that participates in content identity; "
            f"mark it IdentitySkip or rebuild the store"
        )
        return reason, []
    for column in field_doc["columns"]:
        if not column["nullable"]:
            return f"table {table!r} adds field {field!r} with non-nullable column {column['name']!r}", []
    return None, [
        ColumnSpec(
            name=column["name"],
            kind=column["kind"],
            nullable=column["nullable"],
            indexed=column["indexed"],
            unique=column["unique"],
        )
        for column in field_doc["columns"]
    ]


def _layout_from_declaration(value: str) -> StorageLayout:
    try:
        document = json.loads(value)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError("stored entry declaration is not valid JSON") from error
    if not isinstance(document, dict) or set(document) != {"families", "format"} or document["format"] != 2:
        raise ValueError("stored entry declaration does not use format 2")
    families = document["families"]
    if not isinstance(families, list):
        raise ValueError("stored entry declaration families must be a list")
    supplied: dict[type, tuple[type, ...]] = {}
    previous = ""
    for item in families:
        if not isinstance(item, dict) or set(item) != {"definition_id", "records", "family"}:
            raise ValueError("stored entry declaration family entry is malformed")
        family_name = item["family"]
        family_definition_id = item["definition_id"]
        record_names = item["records"]
        if not isinstance(family_name, str) or not isinstance(record_names, list) or not record_names:
            raise ValueError("stored entry declaration has an invalid family or record list")
        _validate_name(family_name, label="stored entry family name")
        _validate_optional_definition_id(family_definition_id)
        if family_name <= previous:
            raise ValueError("stored entry declaration families are not deterministically ordered")
        previous = family_name
        if family_name not in known_entry_families():
            raise EntryLayoutBindingError(
                f"stored entry family {family_name!r} is not registered; reopen the store with entry_families"
            )
        _, registered_family_definition_id = entry_family_info(family_name)
        if family_definition_id != registered_family_definition_id:
            raise ValueError(f"stored entry family {family_name!r} definition does not match its registration")
        family = resolve_entry_family(family_name)
        resolved_records: list[type] = []
        for record_item in record_names:
            if not isinstance(record_item, dict) or set(record_item) != {"definition_id", "record"}:
                raise ValueError("stored entry declaration record entry is malformed")
            record_name = record_item["record"]
            record_definition_id = record_item["definition_id"]
            if not isinstance(record_name, str):
                raise ValueError("stored entry declaration record names must be strings")
            _validate_name(record_name, label="stored entry record name")
            _validate_optional_definition_id(record_definition_id)
            if record_name not in known_entry_records():
                raise EntryLayoutBindingError(
                    f"stored entry record {record_name!r} is not registered; reopen the store with entry_families"
                )
            _, declared_family, registered_definition_id = entry_record_info(record_name)
            if declared_family is None:
                raise ValueError(
                    f"entry record {record_name!r} has no registered family and cannot be used in a family store"
                )
            if declared_family != family_name:
                raise ValueError(
                    f"stored entry record {record_name!r} is registered for {declared_family!r}, not {family_name!r}"
                )
            if record_definition_id != registered_definition_id:
                raise ValueError(f"stored entry record {record_name!r} definition does not match its registration")
            resolved_records.append(resolve_entry_record(record_name))
        supplied[family] = tuple(resolved_records)
    layout = normalize_entry_records(supplied)
    if declaration_json(layout) != value:
        raise ValueError("stored entry declaration is not in its canonical deterministic encoding")
    return layout


def _reject_registry_conflicts(declaration: EntryFamilyDeclaration) -> None:
    if declaration.name in known_entry_families():
        reference, definition_id = entry_family_info(declaration.name)
        matches = _reference_matches_class(reference, declaration.family)
        if not matches or definition_id != declaration.definition_id:
            raise ValueError(f"explicit entry family {declaration.name!r} conflicts with a global registration")
    for record in declaration.records:
        if record.name not in known_entry_records():
            continue
        reference, family_name, definition_id = entry_record_info(record.name)
        if (
            not _reference_matches_class(reference, record.record)
            or family_name != declaration.name
            or definition_id != record.definition_id
        ):
            raise ValueError(f"explicit entry record {record.name!r} conflicts with a global registration")


def _reference_matches_class(reference: str, cls: type) -> bool:
    canonical = f"{cls.__module__}:{cls.__name__}"
    if reference == canonical:
        return True
    module_name, separator, attribute = reference.partition(":")
    module = sys.modules.get(module_name) if separator else None
    return module is not None and getattr(module, attribute, None) is cls


def _validate_name(value: object, *, label: str) -> None:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{label} must be a nonempty string without surrounding whitespace")


def _validate_optional_definition_id(value: object) -> None:
    if value is not None:
        _validate_name(value, label="definition_id")


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
        reference = cast('tuple[str, ...]', get_info(name))[0]
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
