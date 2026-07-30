"""Tests for OPTIMADE-definition-derived portable query fields."""

import pytest
from httk.core import EntryTypeDefinition, PropertyDefinition, load_entry_type_schema, standard_entry_type

from httk.data import portable_query_capabilities, portable_query_fields


@pytest.mark.parametrize("name", ("references", "files", "calculations"))
def test_standard_data_entry_types_have_only_the_portable_core_fields(name: str):
    assert portable_query_fields(standard_entry_type(name)) == (
        "id",
        "type",
        "immutable_id",
        "last_modified",
    )


def test_standard_structures_definition_has_the_expected_portable_fields():
    pytest.importorskip("httk.atomistic")
    structures = load_entry_type_schema("https://schemas.optimade.org/defs/v1.3/entrytypes/optimade/structures")
    assert portable_query_fields(structures) == (
        "id",
        "type",
        "immutable_id",
        "last_modified",
        "elements",
        "nelements",
        "elements_ratios",
        "chemical_formula_descriptive",
        "chemical_formula_reduced",
        "chemical_formula_anonymous",
        "nperiodic_dimensions",
        "nsites",
        "structure_features",
    )


def _property(name: str, fulltype: str, query_support: str) -> PropertyDefinition:
    document = PropertyDefinition.from_simple(name, description=name, fulltype=fulltype).as_optimade()
    document["x-optimade-requirements"] = {"query-support": query_support}
    return PropertyDefinition.from_optimade(name, document)


def test_structured_properties_are_excluded_unless_explicitly_included():
    entry_type = EntryTypeDefinition(
        "example",
        "example",
        {
            "flat": _property("flat", "list of string", "ALL MANDATORY"),
            "nested": _property("nested", "list of list of float", "all mandatory"),
            "mapping": _property("mapping", "dict", "all mandatory"),
            "disabled": _property("disabled", "string", "none"),
        },
    )

    assert portable_query_fields(entry_type) == ("flat",)
    assert portable_query_fields(entry_type, include=("mapping", "nested", "disabled")) == (
        "flat",
        "nested",
        "mapping",
        "disabled",
    )
    assert portable_query_fields(entry_type, include=("mapping",), exclude=("flat", "mapping")) == ()


@pytest.mark.parametrize(
    ("fulltype", "query_support", "operations"),
    [
        ("string", "all mandatory", {"equality", "ordering", "stringmatching"}),
        ("integer", "all mandatory", {"equality", "ordering"}),
        ("boolean", "all mandatory", {"equality"}),
        ("list of string", "all mandatory", {"set"}),
        ("string", "equality only", {"equality"}),
        ("string", "all optional", set()),
        ("string", "none", set()),
    ],
)
def test_definition_query_support_derives_exact_portable_operations(
    fulltype: str, query_support: str, operations: set[str]
) -> None:
    capabilities = portable_query_capabilities(_property("value", fulltype, query_support))
    assert capabilities.query_support == query_support
    assert capabilities.operations == operations


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"include": ("missing",)}, "unknown include"),
        ({"exclude": ("missing",)}, "unknown exclude"),
        ({"include": ("flat", "flat")}, "duplicate include"),
        ({"exclude": ("flat", "flat")}, "duplicate exclude"),
    ],
)
def test_include_and_exclude_names_are_checked(kwargs, message: str):
    entry_type = EntryTypeDefinition("example", "example", {"flat": _property("flat", "string", "all mandatory")})
    with pytest.raises(ValueError, match=message):
        portable_query_fields(entry_type, **kwargs)
