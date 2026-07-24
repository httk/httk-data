"""Tests for the schema IR (httk.data.db.schema): one test per resolution rule."""

import datetime
from dataclasses import dataclass
from fractions import Fraction
from typing import Annotated, ClassVar

import pytest
from httk.core import (
    FracVector,
    Indexed,
    Related,
    RelationshipLink,
    Shape,
    Skip,
    StorageInfo,
    Unique,
    stored_property,
)

from httk.data.db import SchemaError, register_schema_override, resolve_schema, snake_case


@dataclass(frozen=True)
class ScalarRecord:
    count: int
    energy: float
    formula: str
    stable: bool
    payload: bytes


@dataclass(frozen=True)
class OptionalRecord:
    note: str | None
    weight: float | None = None


@dataclass(frozen=True)
class CodecRecord:
    ratio: Fraction
    created: datetime.datetime


@dataclass(frozen=True)
class FixedArrayRecord:
    cell: Annotated[FracVector, Shape(3, 3)]


@dataclass(frozen=True)
class VariableRowsRecord:
    coords: Annotated[FracVector, Shape(0, 3)]


@dataclass(frozen=True)
class ListScalarRecord:
    symbols: list[str]


@dataclass(frozen=True)
class TupleScalarRecord:
    tags: tuple[str, ...]


@dataclass(frozen=True)
class ListCodecRecord:
    ratios: list[Fraction]


@dataclass(frozen=True)
class TargetRecord:
    title: str


@dataclass(frozen=True)
class ListStorableRecord:
    references: list[TargetRecord]


@dataclass(frozen=True)
class ReferenceRecord:
    reference: TargetRecord
    optional_reference: TargetRecord | None = None


@dataclass(frozen=True)
class SkippedRecord:
    kept: int
    scratch: Annotated[str, Skip()] = ""


@dataclass(frozen=True)
class SkippedNoDefaultRecord:
    scratch: Annotated[str, Skip()]


@dataclass(frozen=True)
class MarkedRecord:
    formula: Annotated[str, Indexed()]
    fingerprint: Annotated[str, Unique()]
    ratio: Annotated[Fraction, Indexed()]


@dataclass(frozen=True)
class DerivedRecord:
    symbols: list[str]

    @stored_property
    def natoms(self) -> int:
        return len(self.symbols)


@dataclass(frozen=True)
class CompositeIndexRecord:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(
        indexes=(("spacegroup", "formula"), ("reference", "ratio"))
    )
    formula: str
    spacegroup: int
    ratio: Fraction
    reference: TargetRecord | None = None


@dataclass(frozen=True)
class NamedTableRecord:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(table_name="my_table", dedup="by_value")
    value: int


@dataclass(frozen=True)
class OverriddenRecord:
    value: int


@dataclass(frozen=True)
class ExplicitOverrideRecord:
    value: int


@dataclass(frozen=True)
class Node:
    name: str
    parent: "Node | None" = None


@dataclass(frozen=True)
class MutualA:
    partner: "MutualB | None" = None


@dataclass(frozen=True)
class MutualB:
    partner: "MutualA | None" = None


@dataclass(frozen=True)
class DictRecord:
    mapping: dict[str, int]


@dataclass(frozen=True)
class SidRecord:
    sid: int


@dataclass(frozen=True)
class ContentIdRecord:
    content_id: str


@dataclass(frozen=True)
class BadIndexRecord:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(indexes=(("no_such_field",),))
    value: int


@dataclass(frozen=True)
class BadChildIndexRecord:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(indexes=(("symbols",),))
    symbols: list[str]


@dataclass
class UnfrozenRecord:
    value: int


@dataclass(frozen=True)
class BareFracVectorRecord:
    cell: FracVector


def test_scalar_fields_become_one_column_each():
    schema = resolve_schema(ScalarRecord)
    assert schema.table_name == "scalar_record"
    kinds = {spec.field: (spec.columns[0].name, spec.columns[0].kind) for spec in schema.fields}
    assert kinds == {
        "count": ("count", "int"),
        "energy": ("energy", "float"),
        "formula": ("formula", "str"),
        "stable": ("stable", "bool"),
        "payload": ("payload", "bytes"),
    }
    assert all(spec.role == "scalar" for spec in schema.fields)


def test_bool_is_its_own_kind_not_int():
    schema = resolve_schema(ScalarRecord)
    assert schema.field("stable").columns[0].kind == "bool"


def test_optional_fields_are_nullable():
    schema = resolve_schema(OptionalRecord)
    for name in ("note", "weight"):
        spec = schema.field(name)
        assert spec.optional
        assert all(column.nullable for column in spec.columns)


def test_codec_fields_get_codec_columns():
    schema = resolve_schema(CodecRecord)
    ratio = schema.field("ratio")
    assert ratio.role == "encoded"
    assert ratio.codec_name == "fraction"
    assert [(column.name, column.kind) for column in ratio.columns] == [("ratio", "float"), ("ratio_exact", "str")]
    created = schema.field("created")
    assert created.codec_name == "datetime"
    assert [(column.name, column.kind) for column in created.columns] == [("created", "str")]


def test_fixed_shape_fracvector_flattens_with_exact_column():
    schema = resolve_schema(FixedArrayRecord)
    spec = schema.field("cell")
    assert spec.role == "fixed_array"
    assert spec.shape == Shape(3, 3)
    names = [column.name for column in spec.columns]
    assert names == [f"cell_{i}" for i in range(9)] + ["cell_exact"]
    assert [column.kind for column in spec.columns] == ["float"] * 9 + ["str"]


def test_variable_rows_fracvector_becomes_child_table():
    schema = resolve_schema(VariableRowsRecord)
    spec = schema.field("coords")
    assert spec.role == "child"
    assert spec.columns == ()
    assert spec.child is not None
    assert spec.child.table_name == "variable_rows_record_coords"
    assert [column.name for column in spec.child.element_columns] == [
        "coords_0",
        "coords_1",
        "coords_2",
        "coords_exact",
    ]
    assert spec.child.target is None


def test_list_of_scalars_becomes_child_table():
    spec = resolve_schema(ListScalarRecord).field("symbols")
    assert spec.role == "child"
    assert spec.child is not None
    assert spec.child.table_name == "list_scalar_record_symbols"
    assert [(column.name, column.kind) for column in spec.child.element_columns] == [("symbols", "str")]


def test_homogeneous_variable_tuple_treated_like_list():
    spec = resolve_schema(TupleScalarRecord).field("tags")
    assert spec.role == "child"
    assert spec.child is not None
    assert [(column.name, column.kind) for column in spec.child.element_columns] == [("tags", "str")]


def test_list_of_codec_values_gets_codec_element_columns():
    spec = resolve_schema(ListCodecRecord).field("ratios")
    assert spec.role == "child"
    assert spec.codec_name == "fraction"
    assert spec.child is not None
    assert [(column.name, column.kind) for column in spec.child.element_columns] == [
        ("ratios", "float"),
        ("ratios_exact", "str"),
    ]


def test_list_of_storables_becomes_child_table_of_foreign_keys():
    spec = resolve_schema(ListStorableRecord).field("references")
    assert spec.role == "child"
    assert spec.target is TargetRecord
    assert spec.child is not None
    assert [(column.name, column.kind) for column in spec.child.element_columns] == [("references_sid", "int")]


def test_reference_fields_become_sid_columns():
    schema = resolve_schema(ReferenceRecord)
    required = schema.field("reference")
    assert required.role == "reference"
    assert required.target is TargetRecord
    assert [(column.name, column.kind, column.nullable) for column in required.columns] == [
        ("reference_sid", "int", False)
    ]
    optional = schema.field("optional_reference")
    assert optional.optional
    assert optional.columns[0].nullable
    assert schema.referenced_classes() == (TargetRecord,)


def test_skip_omits_the_field():
    schema = resolve_schema(SkippedRecord)
    assert [spec.field for spec in schema.fields] == ["kept"]


def test_skip_without_default_is_an_error():
    with pytest.raises(SchemaError, match="SkippedNoDefaultRecord.*scratch.*default"):
        resolve_schema(SkippedNoDefaultRecord)


def test_indexed_and_unique_flags():
    schema = resolve_schema(MarkedRecord)
    assert schema.field("formula").columns[0].indexed
    assert not schema.field("formula").columns[0].unique
    assert schema.field("fingerprint").columns[0].unique
    ratio_columns = {column.name: column for column in schema.field("ratio").columns}
    assert ratio_columns["ratio"].indexed  # only the query column of an encoded field
    assert not ratio_columns["ratio_exact"].indexed


def test_stored_property_is_derived_with_type_from_return_annotation():
    schema = resolve_schema(DerivedRecord)
    spec = schema.field("natoms")
    assert spec.derived
    assert spec.role == "scalar"
    assert spec.python_type is int
    assert spec.columns[0].kind == "int"


def test_composite_indexes_resolve_to_column_names():
    schema = resolve_schema(CompositeIndexRecord)
    assert schema.composite_indexes == (("spacegroup", "formula"), ("reference_sid", "ratio"))


def test_table_name_and_dedup_from_storage_info():
    schema = resolve_schema(NamedTableRecord)
    assert schema.table_name == "my_table"
    assert schema.dedup == "by_value"


def test_default_dedup_is_content_id():
    assert resolve_schema(ScalarRecord).dedup == "content_id"


def test_snake_case_default_table_names():
    assert snake_case("StructureRecord") == "structure_record"
    assert snake_case("HTTPRecord") == "http_record"
    assert snake_case("Simple") == "simple"


def test_register_schema_override_wins_over_defaults():
    register_schema_override(OverriddenRecord, StorageInfo(table_name="external_name"))
    assert resolve_schema(OverriddenRecord).table_name == "external_name"


def test_explicit_override_argument_wins():
    default_schema = resolve_schema(ExplicitOverrideRecord)
    assert default_schema.table_name == "explicit_override_record"
    overridden = resolve_schema(ExplicitOverrideRecord, override=StorageInfo(table_name="elsewhere"))
    assert overridden.table_name == "elsewhere"
    # The two cache entries stay distinct.
    assert resolve_schema(ExplicitOverrideRecord) is default_schema


def test_self_reference_does_not_recurse_forever():
    schema = resolve_schema(Node)
    assert schema.field("parent").target is Node


def test_mutual_references_do_not_recurse_forever():
    schema = resolve_schema(MutualA)
    assert schema.field("partner").target is MutualB
    assert resolve_schema(MutualB).field("partner").target is MutualA


def test_caching_returns_the_same_object():
    assert resolve_schema(ScalarRecord) is resolve_schema(ScalarRecord)


def test_non_dataclass_is_an_error():
    with pytest.raises(SchemaError, match="not a dataclass"):
        resolve_schema(object)


def test_non_frozen_dataclass_is_an_error():
    with pytest.raises(SchemaError, match="UnfrozenRecord.*frozen"):
        resolve_schema(UnfrozenRecord)


def test_dict_field_is_an_error():
    with pytest.raises(SchemaError, match="DictRecord.*mapping"):
        resolve_schema(DictRecord)


def test_unknown_composite_index_field_is_an_error():
    with pytest.raises(SchemaError, match="BadIndexRecord.*no_such_field"):
        resolve_schema(BadIndexRecord)


def test_child_field_in_composite_index_is_an_error():
    with pytest.raises(SchemaError, match="BadChildIndexRecord.*symbols"):
        resolve_schema(BadChildIndexRecord)


def test_reserved_field_names_are_errors():
    with pytest.raises(SchemaError, match="SidRecord.sid.*reserved"):
        resolve_schema(SidRecord)
    with pytest.raises(SchemaError, match="ContentIdRecord.content_id.*reserved"):
        resolve_schema(ContentIdRecord)


def test_fracvector_without_shape_is_an_error():
    with pytest.raises(SchemaError, match="BareFracVectorRecord.cell.*Shape"):
        resolve_schema(BareFracVectorRecord)


# --------------------------------------------------------------------- Related markers and links


@dataclass(frozen=True)
class Author:
    name: str


@dataclass(frozen=True)
class RelatedMarkerRecord:
    author: Annotated[Author | None, Related(role="creator", description="Wrote it")] = None


@dataclass(frozen=True)
class RelatedChildRecord:
    authors: Annotated[tuple[Author, ...], Related(role="creator", serve=False)] = ()


@dataclass(frozen=True)
class RelatedOnScalarRecord:
    name: Annotated[str, Related(role="creator")] = ""


@dataclass(frozen=True)
class RelatedOnScalarListRecord:
    names: Annotated[list[str], Related()] = None  # type: ignore[assignment]


@dataclass(frozen=True)
class RelatedOnSkipRecord:
    author: Annotated[Author | None, Skip(), Related()] = None


def test_related_marker_resolved_on_reference_field():
    spec = resolve_schema(RelatedMarkerRecord).field("author")
    assert spec.role == "reference"
    assert spec.related == Related(role="creator", description="Wrote it")


def test_related_marker_resolved_on_child_of_storable_field():
    spec = resolve_schema(RelatedChildRecord).field("authors")
    assert spec.role == "child" and spec.target is Author
    assert spec.related is not None
    assert spec.related.serve is False


def test_related_marker_on_scalar_field_is_an_error():
    with pytest.raises(SchemaError, match="RelatedOnScalarRecord.name.*Related"):
        resolve_schema(RelatedOnScalarRecord)


def test_related_marker_on_scalar_list_field_is_an_error():
    with pytest.raises(SchemaError, match="RelatedOnScalarListRecord.names.*Related"):
        resolve_schema(RelatedOnScalarListRecord)


def test_related_marker_on_skipped_field_is_an_error():
    with pytest.raises(SchemaError, match="RelatedOnSkipRecord.author.*Skip"):
        resolve_schema(RelatedOnSkipRecord)


@dataclass(frozen=True)
class LinkedJoinRecord:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(
        links=(RelationshipLink("author", None, role="wrote", description="Inverse of author"),)
    )

    title: str
    author: Author | None = None


@dataclass(frozen=True)
class UnknownLinkFieldRecord:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(links=(RelationshipLink("nope", None),))

    author: Author | None = None


@dataclass(frozen=True)
class ScalarLinkEndpointRecord:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(links=(RelationshipLink("title", None),))

    title: str
    author: Author | None = None


@dataclass(frozen=True)
class ChildLinkEndpointRecord:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(links=(RelationshipLink(None, "authors"),))

    authors: tuple[Author, ...] = ()


@dataclass(frozen=True)
class DoubleDeclaredRecord:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(links=(RelationshipLink(None, "author"),))

    author: Annotated[Author | None, Related(role="creator")] = None


def test_valid_link_carried_on_schema():
    schema = resolve_schema(LinkedJoinRecord)
    assert schema.links == (RelationshipLink("author", None, role="wrote", description="Inverse of author"),)


def test_links_default_empty():
    assert resolve_schema(Author).links == ()


def test_link_with_unknown_field_is_an_error():
    with pytest.raises(SchemaError, match="UnknownLinkFieldRecord.*unknown field 'nope'"):
        resolve_schema(UnknownLinkFieldRecord)


def test_link_with_non_reference_endpoint_is_an_error():
    with pytest.raises(SchemaError, match="ScalarLinkEndpointRecord.*'title'.*'scalar'"):
        resolve_schema(ScalarLinkEndpointRecord)


def test_link_with_child_endpoint_is_an_error():
    with pytest.raises(SchemaError, match="ChildLinkEndpointRecord.*'authors'.*'child'"):
        resolve_schema(ChildLinkEndpointRecord)


def test_field_declared_via_both_forms_is_an_error():
    with pytest.raises(SchemaError, match="DoubleDeclaredRecord.author.*both.*Related.*RelationshipLink"):
        resolve_schema(DoubleDeclaredRecord)
