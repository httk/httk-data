"""Tests for httk-data's offline JSON-Schema property/record validation."""

import jsonschema
import pytest
from httk.core import PropertyDefinition, standard_entry_type

from httk.data import PropertyValidationError, validate_property, validate_record
from httk.data.validation import _validator_schema

REFERENCES = standard_entry_type("references")


# --- scalars ------------------------------------------------------------------


def test_scalar_valid_and_invalid_type() -> None:
    # The vendored 'references.year' is a (nullable) string property.
    year = REFERENCES.properties["year"]
    validate_property(year, "2021")  # ok: a string
    with pytest.raises(PropertyValidationError) as excinfo:
        validate_property(year, 2021)  # not a string
    assert excinfo.value.name == "year"


def test_generated_integer_scalar() -> None:
    count = PropertyDefinition.from_simple("_httk_count", description="A count.", fulltype="integer")
    validate_property(count, 5)
    with pytest.raises(PropertyValidationError):
        validate_property(count, "5")


# --- nullability --------------------------------------------------------------


def test_nullable_property_accepts_none() -> None:
    year = REFERENCES.properties["year"]
    assert year.nullable
    validate_property(year, None)  # nullable -> None is allowed


def test_non_nullable_property_rejects_none() -> None:
    # 'id' is required-response, hence non-nullable (type == ["string"]).
    id_def = REFERENCES.properties["id"]
    assert not id_def.nullable
    with pytest.raises(PropertyValidationError):
        validate_property(id_def, None)


# --- lists / dicts ------------------------------------------------------------


def test_list_of_list_structures_like() -> None:
    matrix = PropertyDefinition.from_simple(
        "_httk_matrix", description="A matrix.", fulltype="list of list of float"
    )
    validate_property(matrix, [[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(PropertyValidationError):
        validate_property(matrix, [[1.0, "x"]])  # inner element is not a number


def test_authors_list_of_dicts() -> None:
    authors = REFERENCES.properties["authors"]
    validate_property(authors, [{"name": "Ada Lovelace"}, {"name": "Alan Turing"}])
    with pytest.raises(PropertyValidationError):
        validate_property(authors, "Ada Lovelace")  # a string is not a list


# --- record-level -------------------------------------------------------------


def test_record_valid() -> None:
    validate_record(
        REFERENCES,
        {"id": "ref-1", "type": "references", "title": "A title", "year": "2021"},
    )


def test_record_unknown_property_rejected() -> None:
    with pytest.raises(PropertyValidationError) as excinfo:
        validate_record(REFERENCES, {"id": "ref-1", "type": "references", "sprocket": 3})
    message = str(excinfo.value)
    assert "sprocket" in message
    assert "references" in message


def test_record_requires_id_and_type() -> None:
    with pytest.raises(PropertyValidationError) as excinfo:
        validate_record(REFERENCES, {"type": "references"})
    assert excinfo.value.name == "id"
    with pytest.raises(PropertyValidationError) as excinfo:
        validate_record(REFERENCES, {"id": "ref-1"})
    assert excinfo.value.name == "type"


def test_record_bad_value_rejected() -> None:
    with pytest.raises(PropertyValidationError):
        validate_record(REFERENCES, {"id": "ref-1", "type": "references", "year": 2021})


def test_record_datetime_requires_rfc3339_offset() -> None:
    record = {"id": "ref-1", "type": "references", "last_modified": "2026-07-29T12:34:56+00:00"}
    validate_record(REFERENCES, record)

    for timestamp in ("2026-07-29T12:34:56", "2026-07-29"):
        with pytest.raises(PropertyValidationError) as excinfo:
            validate_record(REFERENCES, {**record, "last_modified": timestamp})
        assert "last_modified" in str(excinfo.value)


# --- error chaining & offline behavior ----------------------------------------


def test_chained_cause_present() -> None:
    year = REFERENCES.properties["year"]
    with pytest.raises(PropertyValidationError) as excinfo:
        validate_property(year, 2021)
    assert isinstance(excinfo.value.__cause__, jsonschema.exceptions.ValidationError)


def test_meta_schema_reference_removed_no_network() -> None:
    # A generated definition carries a '$schema' meta-schema URI ...
    generated = PropertyDefinition.from_simple("_httk_energy", description="E", fulltype="float")
    assert generated.as_optimade()["$schema"].startswith("https://")
    # ... but the schema handed to jsonschema has it stripped, so the validator
    # never treats that URI as a dialect to resolve or fetch.
    assert "$schema" not in _validator_schema(generated)
    validate_property(generated, 1.5)

    # Vendored definitions carry real '$id' URIs (and no '$ref'), so validating
    # against them is fully offline; this succeeding in the offline test env is
    # itself the proof that no network access is attempted.
    year = REFERENCES.properties["year"]
    assert year.as_optimade()["$id"].startswith("https://")
    assert "$schema" not in _validator_schema(year)
    validate_property(year, "2021")
