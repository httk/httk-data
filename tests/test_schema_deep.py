"""Deep checks for stored-field projections into served property definitions."""

import datetime
from dataclasses import dataclass
from fractions import Fraction
from typing import Annotated

from httk.core import FracScalar, FracVector, SurdScalar
from httk.core.schema_check import check_record_matches_definition
from httk.core.storage import Shape

from httk.store.backend.sql import resolve_schema
from httk.store.backend.sql.entry_provider import _fulltype_of, auto_definition


@dataclass(frozen=True)
class DeepCodecRecord:
    fraction: Fraction | None = None
    fracscalar: FracScalar | None = None
    surdscalar: SurdScalar | None = None
    created: datetime.datetime | None = None
    matrix: Annotated[FracVector, Shape(2, 2)] | None = None


def test_exact_codec_projections_and_shallow_checker_output() -> None:
    schema = resolve_schema(DeepCodecRecord)
    specs = {spec.field: spec for spec in schema.fields}
    assert {name: _fulltype_of(specs[name]) for name in specs} == {
        "fraction": "float",
        "fracscalar": "float",
        "surdscalar": "float",
        "created": "timestamp",
        "matrix": "list of list of float",
    }
    assert {name: specs[name].codec_name for name in ("fraction", "fracscalar", "surdscalar", "created")} == {
        "fraction": "fraction",
        "fracscalar": "fracscalar",
        "surdscalar": "surdscalar",
        "created": "datetime",
    }
    assert specs["matrix"].role == "fixed_array"
    assert specs["matrix"].shape == Shape(2, 2)

    definition = auto_definition("deep", schema, "_httk_")
    property_keys = {name: f"_httk_custom_{name}" for name in specs}
    assert {property_keys[name]: definition.properties[property_keys[name]].optimade_type for name in specs} == {
        "_httk_custom_fraction": "float",
        "_httk_custom_fracscalar": "float",
        "_httk_custom_surdscalar": "float",
        "_httk_custom_created": "timestamp",
        "_httk_custom_matrix": "list",
    }
    assert definition.properties["_httk_custom_matrix"].dimensions == {"names": ["rows", "cols"], "sizes": [2, 2]}

    mismatches = check_record_matches_definition(DeepCodecRecord, definition, property_keys=property_keys)
    assert mismatches == [
        (
            "field 'fracscalar' annotation httk.core.vectors.fracvector.FracScalar | None does not match property "
            "'_httk_custom_fracscalar' type 'float' (field -> property type shape)"
        ),
        (
            "field 'fraction' annotation fractions.Fraction | None does not match property '_httk_custom_fraction' type "
            "'float' (field -> property type shape)"
        ),
        (
            "field 'surdscalar' annotation httk.core.vectors.surdvector.SurdScalar | None does not match property "
            "'_httk_custom_surdscalar' type 'float' (field -> property type shape)"
        ),
    ]
    for field in ("fraction", "fracscalar", "surdscalar"):
        assert _fulltype_of(specs[field]) == "float"
        assert definition.properties[f"_httk_custom_{field}"].optimade_type == "float"
