"""Pure MongoDB physical mapping derived from the schema intermediate form."""

import hashlib
from dataclasses import dataclass
from typing import Any, Final

from pymongo import ReturnDocument

from httk.store.db.schema import FieldSpec, TableSchema
from httk.store.storage_layout import EntryFamilyLayout

__all__ = [
    "COUNTERS_COLLECTION",
    "METADATA_COLLECTION",
    "DocumentFieldSpec",
    "IndexSpec",
    "collection_name_for",
    "counter_next",
    "dispatch_index_specs",
    "dispatch_validator_for",
    "document_fields_for",
    "entry_dispatch_table_name",
    "index_specs_for",
    "validator_for",
]

METADATA_COLLECTION: Final = "_httk_store_metadata"
COUNTERS_COLLECTION: Final = "_httk_counters"
_RESERVED_PREFIX: Final = "_httk_"
_MAX_IDENTIFIER_LENGTH: Final = 63


@dataclass(frozen=True)
class DocumentFieldSpec:
    """Describe one field's location and embedded value shape in ``f``.

    :param field: The logical schema field name.
    :param role: The schema field role.
    :param keys: Parent document keys used by non-child fields.
    :param element_keys: Keys used by one embedded child element.
    :param optional: Whether an absent key represents ``None``.
    :param shape: The schema shape marker, when present.
    """

    field: str
    role: str
    keys: tuple[str, ...]
    element_keys: tuple[str, ...] = ()
    optional: bool = False
    shape: Any = None

    @property
    def key(self) -> str:
        """Return the single parent key for this field.

        :return: The parent key.
        :raises ValueError: If the field has multiple or no parent keys.
        """
        if len(self.keys) != 1:
            raise ValueError(f"field {self.field!r} does not have one parent key")
        return self.keys[0]

    @property
    def columns(self) -> tuple[str, ...]:
        """Return the generated column names represented by this plan.

        :return: The physical field column names.
        """
        return self.keys if self.role != "child" else self.element_keys


@dataclass(frozen=True)
class IndexSpec:
    """Describe one MongoDB index without performing any I/O.

    :param keys: Ordered dotted field paths and ascending/descending directions.
    :param name: Deterministic index name.
    :param unique: Whether duplicate keys are rejected.
    :param partial_filter_expression: Optional MongoDB partial-index predicate.
    """

    keys: tuple[tuple[str, int], ...]
    name: str
    unique: bool = False
    partial_filter_expression: dict[str, Any] | None = None

    @property
    def key(self) -> tuple[tuple[str, int], ...]:
        """Return the ordered MongoDB key pattern.

        :return: The ordered key pattern.
        """
        return self.keys

    @property
    def partial_filter(self) -> dict[str, Any] | None:
        """Return the partial filter expression.

        :return: The partial filter, or ``None``.
        """
        return self.partial_filter_expression


def _stable_identifier(prefix: str, value: str) -> str:
    safe = "".join(character if character.isalnum() else "_" for character in value.lower()).strip("_") or "entry"
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:8]
    body = f"{prefix}_{safe}_{digest}"
    if len(body) <= _MAX_IDENTIFIER_LENGTH:
        return body
    return f"{body[: _MAX_IDENTIFIER_LENGTH - 9]}_{digest}"


# Kept byte-for-byte equivalent to httk.store.db.mapping.entry_dispatch_table_name.
def entry_dispatch_table_name(family_name: str) -> str:
    """Return the deterministic reserved dispatch collection name.

    :param family_name: Registered entry-family name.
    :return: The physical dispatch collection name.
    """
    return _stable_identifier("_httk_entry_dispatch", family_name)


def _index_name(prefix: str, collection_name: str, columns: tuple[str, ...]) -> str:
    name = f"{prefix}_{collection_name}_{'_'.join(columns)}"
    if len(name) > _MAX_IDENTIFIER_LENGTH:
        digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:8]
        name = f"{name[: _MAX_IDENTIFIER_LENGTH - 9]}_{digest}"
    return name


def collection_name_for(schema: TableSchema) -> str:
    """Return the collection name for a resolved schema.

    :param schema: Resolved storable schema.
    :return: The schema's physical collection name.
    :raises ValueError: If the name uses the reserved ``_httk_`` prefix.
    """
    name = schema.table_name
    if name.startswith(_RESERVED_PREFIX):
        raise ValueError(f"ordinary records may not claim reserved MongoStore collection name {name!r}")
    return name


def document_fields_for(schema: TableSchema) -> tuple[DocumentFieldSpec, ...]:
    """Derive the user-field document plan under the ``f`` subdocument.

    :param schema: Resolved storable schema.
    :return: One immutable document-field plan per stored schema field.
    """
    fields: list[DocumentFieldSpec] = []
    for spec in schema.fields:
        if spec.role == "child":
            assert spec.child is not None
            fields.append(
                DocumentFieldSpec(
                    field=spec.field,
                    role=spec.role,
                    keys=(spec.field,),
                    element_keys=tuple(column.name for column in spec.child.element_columns),
                    optional=spec.optional,
                    shape=spec.shape,
                )
            )
        else:
            fields.append(
                DocumentFieldSpec(
                    field=spec.field,
                    role=spec.role,
                    keys=tuple(column.name for column in spec.columns),
                    optional=spec.optional,
                    shape=spec.shape,
                )
            )
    return tuple(fields)


def _bson_type(kind: str) -> str | list[str]:
    types: dict[str, str | list[str]] = {
        "int": ["int", "long"],
        "float": "double",
        "str": "string",
        "bool": "bool",
        "bytes": "binData",
    }
    return types[kind]


def _partial_filter(path: str, kind: str) -> dict[str, Any]:
    return {path: {"$exists": True, "$type": _bson_type(kind)}}


def _make_index(
    prefix: str,
    collection_name: str,
    columns: tuple[str, ...],
    *,
    unique: bool = False,
    partial_filter_expression: dict[str, Any] | None = None,
) -> IndexSpec:
    return IndexSpec(
        keys=tuple((f"f.{column}", 1) for column in columns),
        name=_index_name(prefix, collection_name, columns),
        unique=unique,
        partial_filter_expression=partial_filter_expression,
    )


def index_specs_for(schema: TableSchema) -> list[IndexSpec]:
    """Derive all record-collection indexes from the schema IR.

    :param schema: Resolved storable schema.
    :return: Deterministically ordered index specifications.
    """
    collection = collection_name_for(schema)
    result: list[IndexSpec] = []
    if schema.dedup == "content_id":
        result.append(IndexSpec((("content_id", 1),), _index_name("uq", collection, ("content_id",)), True))
    result.append(
        IndexSpec(
            (("_httk_role", 1),),
            _index_name("ix", collection, ("_httk_role",)),
        )
    )
    for field in schema.fields:
        if field.role == "child":
            continue
        requested_unique = any(column.unique for column in field.columns)
        requested_indexed = any(column.indexed for column in field.columns)
        if not requested_unique and not requested_indexed:
            continue
        prefix = "uq" if requested_unique else "ix"
        for column in field.columns:
            partial = _partial_filter(f"f.{column.name}", column.kind) if requested_unique and column.nullable else None
            result.append(
                _make_index(
                    prefix,
                    collection,
                    (column.name,),
                    unique=requested_unique,
                    partial_filter_expression=partial,
                )
            )
    for columns in schema.composite_indexes:
        result.append(_make_index("ix", collection, tuple(columns)))
    return result


def _channel_dependencies(columns: tuple[Any, ...]) -> dict[str, list[str]]:
    exact = next((column.name for column in columns if column.name.endswith("_exact")), None)
    if exact is None:
        return {}
    return {column.name: [exact] for column in columns if column.name != exact}


def _field_validator(spec: FieldSpec) -> tuple[dict[str, Any], dict[str, list[str]], list[str]]:
    properties: dict[str, Any] = {}
    dependencies: dict[str, list[str]] = {}
    required: list[str] = []
    if spec.role == "child":
        assert spec.child is not None
        element_properties = {
            column.name: {"bsonType": _bson_type(column.kind)} for column in spec.child.element_columns
        }
        element_required = [column.name for column in spec.child.element_columns]
        element_dependencies = _channel_dependencies(spec.child.element_columns)
        item: dict[str, Any] = {
            "bsonType": "object",
            "properties": element_properties,
            "required": element_required,
            "additionalProperties": False,
        }
        if element_dependencies:
            item["dependencies"] = element_dependencies
        properties[spec.field] = {"bsonType": "array", "items": item}
        if not spec.optional:
            required.append(spec.field)
    else:
        for column in spec.columns:
            properties[column.name] = {"bsonType": _bson_type(column.kind)}
        if not spec.optional:
            required.extend(column.name for column in spec.columns)
    if spec.role in {"encoded", "fixed_array"}:
        dependencies.update(_channel_dependencies(spec.columns))
    return properties, dependencies, required


def validator_for(schema: TableSchema) -> dict[str, Any]:
    """Build the writer-owned ``$jsonSchema`` validator for a record collection.

    :param schema: Resolved storable schema.
    :return: A MongoDB collection validator command fragment.
    """
    collection_name_for(schema)
    properties: dict[str, Any] = {
        "_id": {"bsonType": ["int", "long"]},
        "_httk_role": {"enum": ["main", "dep"]},
        "f": {"bsonType": "object", "additionalProperties": True},
    }
    required = ["_id", "_httk_role", "f"]
    if schema.dedup == "content_id":
        properties["content_id"] = {"bsonType": "string", "pattern": "^[0-9a-fA-F]{64}$"}
        required.append("content_id")
    field_properties: dict[str, Any] = {}
    dependencies: dict[str, list[str]] = {}
    field_required: list[str] = []
    for spec in schema.fields:
        field_props, field_dependencies, required_fields = _field_validator(spec)
        field_properties.update(field_props)
        dependencies.update(field_dependencies)
        field_required.extend(required_fields)
    f_schema: dict[str, Any] = {
        "bsonType": "object",
        "properties": field_properties,
        "additionalProperties": True,
    }
    # MongoDB rejects ``required: []`` even though JSON Schema permits it.
    if field_required:
        f_schema["required"] = field_required
    properties["f"] = f_schema
    if dependencies:
        properties["f"]["dependencies"] = dependencies
    return {
        "$jsonSchema": {
            "bsonType": "object",
            "required": required,
            "properties": properties,
            "additionalProperties": False,
        }
    }


def _family_name_and_records(family: EntryFamilyLayout | type) -> tuple[str, tuple[str, ...]]:
    if isinstance(family, EntryFamilyLayout):
        return family.name, family.record_names
    raise TypeError("family must be an EntryFamilyLayout")


def dispatch_validator_for(family: EntryFamilyLayout) -> dict[str, Any]:
    """Build the validator for one entry-family dispatch collection.

    :param family: Normalized entry-family layout.
    :return: A MongoDB collection validator command fragment.
    """
    _, records = _family_name_and_records(family)
    return {
        "$jsonSchema": {
            "bsonType": "object",
            "required": ["_id", "record", "sid"],
            "properties": {
                "_id": {"bsonType": "string"},
                "record": {"enum": list(records)},
                "sid": {"bsonType": ["int", "long"]},
            },
            "additionalProperties": False,
        }
    }


def dispatch_index_specs(family: EntryFamilyLayout) -> list[IndexSpec]:
    """Derive the unique ``(record, sid)`` dispatch index.

    :param family: Normalized multi-record entry-family layout.
    :return: The dispatch collection's index specification.
    :raises ValueError: If the family has fewer than two backing records.
    """
    name, records = _family_name_and_records(family)
    if len(records) < 2:
        raise ValueError("a dispatch collection requires at least two backing records")
    collection = entry_dispatch_table_name(name)
    return [IndexSpec((("record", 1), ("sid", 1)), _index_name("uq", collection, ("record", "sid")), True)]


def counter_next(database: Any, collection_name: str, *, session: Any = None) -> int:
    """Atomically allocate the next integer sid from the counters collection.

    :param database: A PyMongo database handle.
    :param collection_name: Counter key, normally a record collection name.
    :param session: Optional active MongoDB transaction session.
    :return: The allocated monotonically increasing integer.
    """
    document = database[COUNTERS_COLLECTION].find_one_and_update(
        {"_id": collection_name},
        {"$inc": {"next": 1}},
        upsert=True,
        return_document=ReturnDocument.AFTER,
        session=session,
    )
    assert document is not None
    return int(document["next"])
