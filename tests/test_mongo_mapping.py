"""Serverless tests for the MongoDB schema-to-document mapping."""

from dataclasses import dataclass
from fractions import Fraction
from typing import Annotated, ClassVar

from httk.core import FracVector
from httk.core.register import register_entry_family, register_entry_record
from httk.core.storage import Indexed, Shape, StorageInfo, Unique

from httk.store.backend.sql.mapping import entry_dispatch_table_name as sql_dispatch_name
from httk.store.backend.sql.schema import resolve_schema
from httk.store.backend.mongo.mapping import (
    DocumentFieldSpec,
    collection_name_for,
    dispatch_index_specs,
    dispatch_validator_for,
    document_fields_for,
    entry_dispatch_table_name,
    index_specs_for,
    validator_for,
)
from httk.store.storage_layout import normalize_entry_records


class MongoMappingFamily:
    """Test family for mapping plans."""


@dataclass(frozen=True)
class MongoMappingRecord:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(indexes=(("name", "amount"),))

    name: Annotated[str, Indexed()]
    amount: Annotated[float | None, Unique()]
    cell: Annotated[FracVector, Shape(2, 2)]


@dataclass(frozen=True)
class MongoMappingOther:
    value: int


@dataclass(frozen=True)
class MongoMappingChildRecord:
    required: str
    values: list[Fraction]
    optional: str | None = None


register_entry_family(name="test-mongo-mapping-family", family=f"{__name__}:MongoMappingFamily")
register_entry_record(
    name="test-mongo-mapping-record", family="test-mongo-mapping-family", record=f"{__name__}:MongoMappingRecord"
)
register_entry_record(
    name="test-mongo-mapping-other", family="test-mongo-mapping-family", record=f"{__name__}:MongoMappingOther"
)


def test_collection_and_dispatch_names_match_sql_twin() -> None:
    """Mongo and SQL share the stable dispatch-name algorithm."""
    assert collection_name_for(resolve_schema(MongoMappingRecord)) == "mongo_mapping_record"
    for name in ["short", "Family with spaces", "x" * 200]:
        assert entry_dispatch_table_name(name) == sql_dispatch_name(name)


def test_document_plan_preserves_columns_and_embeds_children() -> None:
    """Generated columns stay authoritative under the document ``f`` key."""
    fields = document_fields_for(resolve_schema(MongoMappingRecord))
    assert all(isinstance(field, DocumentFieldSpec) for field in fields)
    assert next(field for field in fields if field.field == "cell").keys == (
        "cell_0",
        "cell_1",
        "cell_2",
        "cell_3",
        "cell_exact",
    )


def test_indexes_include_exact_channel_and_nullable_partial_unique() -> None:
    """Markers cover every generated column and nullable uniques are partial."""
    specs = index_specs_for(resolve_schema(MongoMappingRecord))
    names = {spec.name for spec in specs}
    assert "ix_mongo_mapping_record_name" in names
    assert "uq_mongo_mapping_record_amount" in names
    assert "uq_mongo_mapping_record_cell_exact" not in names
    partial = next(spec for spec in specs if spec.name == "uq_mongo_mapping_record_amount")
    assert partial.partial_filter_expression == {"f.amount": {"$exists": True, "$type": "double"}}


def test_validators_have_root_shape_exact_dependencies_and_dispatch_enum() -> None:
    """Validators constrain store-owned structure without leaf over-validation."""
    validator = validator_for(resolve_schema(MongoMappingRecord))["$jsonSchema"]
    assert validator["additionalProperties"] is False
    assert "f" in validator["required"]
    assert validator["properties"]["f"]["dependencies"]["cell_0"] == ["cell_exact"]
    family = normalize_entry_records({MongoMappingFamily: (MongoMappingRecord, MongoMappingOther)}).families[0]
    dispatch = dispatch_validator_for(family)["$jsonSchema"]
    assert dispatch["properties"]["record"]["enum"] == list(family.record_names)
    assert dispatch_index_specs(family)[0].unique is True


def test_validator_requires_non_optional_fields_and_child_channels() -> None:
    """Nested validators require parent fields and every embedded codec channel."""
    schema = resolve_schema(MongoMappingChildRecord)
    validator = validator_for(schema)["$jsonSchema"]
    field_validator = validator["properties"]["f"]
    assert set(field_validator["required"]) == {"required", "values"}
    item = field_validator["properties"]["values"]["items"]
    child_columns = schema.field("values").child.element_columns
    assert set(item["required"]) == {column.name for column in child_columns}
    exact = next(column.name for column in child_columns if column.name.endswith("_exact"))
    query = next(column.name for column in child_columns if column.name != exact)
    assert item["dependencies"][query] == [exact]


def test_timestamp_mapping_is_flagged_and_accepts_small_python_ints() -> None:
    """Timestamp validators accept PyMongo's int32 encoding for tiny test clocks."""
    schema = resolve_schema(MongoMappingRecord)
    enabled = validator_for(schema, store_timestamps=True)["$jsonSchema"]
    disabled = validator_for(schema, store_timestamps=False)["$jsonSchema"]
    assert enabled["properties"]["store_timestamp"]["bsonType"] == ["long", "int"]
    assert "store_timestamp" in enabled["required"]
    assert "store_timestamp" not in disabled["properties"]
    assert "store_timestamp" not in disabled["required"]
    assert any(spec.keys == (("store_timestamp", 1),) for spec in index_specs_for(schema))
    assert not any(spec.keys == (("store_timestamp", 1),) for spec in index_specs_for(schema, store_timestamps=False))
