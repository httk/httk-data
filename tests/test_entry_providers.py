"""Tests for httk-store's in-memory entry providers and their registration."""

import datetime

import pytest
from httk.core import (
    Calculation,
    EntryTypeDefinition,
    File,
    FileRecord,
    Reference,
    RelatedEntry,
    known_entry_providers,
    standard_entry_type,
)
from httk.core._plugins import resolve_callable
from httk.core.register import entry_providers

from httk.store import CalculationEntryProvider, FileEntryProvider, ReferenceEntryProvider, validate_record

# --- provider round trips -----------------------------------------------------


def test_reference_provider_round_trip() -> None:
    provider = ReferenceEntryProvider({"ref-1": {"title": "T", "doi": "10.1/x"}})
    entry_types = provider.entry_types()
    assert set(entry_types) == {"references"}
    assert isinstance(entry_types["references"], EntryTypeDefinition)
    property_keys = provider.property_keys("references")
    assert property_keys["id"] == "__id"
    assert property_keys["type"] == "type"
    assert property_keys["title"] == "title"
    records = list(provider.records("references"))
    assert records[0]["__id"] == "ref-1"
    assert records[0]["type"] == "references"
    assert records[0]["title"] == "T"
    assert records[0]["url"] is None
    # Every served record key is present in every record:
    for record in records:
        for key in property_keys.values():
            assert key in record


def test_file_provider_records() -> None:
    provider = FileEntryProvider({"f-1": File(url="http://x/INCAR", name="INCAR", size=512)})
    record = next(iter(provider.records("files")))
    assert record["url"] == "http://x/INCAR"
    assert record["size"] == 512


def test_file_provider_serves_sha256_and_preserves_checksums() -> None:
    digest = "a" * 64
    provider = FileEntryProvider(
        {
            "record": FileRecord(url="http://x/data", name="data", sha256=digest),
            "plain": File(url="http://x/plain", name="plain", checksums={"md5": "b"}),
        }
    )

    rows = {row["__id"]: row for row in provider.records("files")}
    assert rows["record"]["checksums"] == {"sha256": digest}
    assert rows["plain"]["checksums"] == {"md5": "b"}


def test_file_provider_preserves_an_explicit_empty_checksums_mapping() -> None:
    provider = FileEntryProvider({"empty": FileRecord(url="http://x/e", name="e", checksums={}, sha256="c" * 64)})
    (row,) = provider.records("files")
    assert row["checksums"] == {}


def test_calculation_provider_property_keys_cover_id_type() -> None:
    provider = CalculationEntryProvider({"calc-1": Calculation()})
    property_keys = provider.property_keys("calculations")
    assert {"id", "type"} <= set(property_keys)


def test_provider_rejects_wrong_entry_type() -> None:
    provider = ReferenceEntryProvider({})
    with pytest.raises(KeyError):
        provider.property_keys("files")
    with pytest.raises(KeyError):
        provider.relationships("files")


def test_relationships_default_empty() -> None:
    assert ReferenceEntryProvider({"ref-1": {"title": "T"}}).relationships("references") == {}
    assert FileEntryProvider({}).relationships("files") == {}
    assert CalculationEntryProvider({}).relationships("calculations") == {}


def test_relationships_round_trip_with_roles() -> None:
    related = (
        RelatedEntry("files", "f-in", description="The input file", role="input"),
        RelatedEntry("files", "f-out", role="output"),
        RelatedEntry("structures", "s-1"),
    )
    provider = CalculationEntryProvider(
        {"calc-1": Calculation()},
        relationships={"calc-1": list(related)},  # any iterable normalizes to a tuple
    )
    assert provider.relationships("calculations") == {"calc-1": related}
    assert provider.relationships("calculations")["calc-1"][0].role == "input"


def test_relationships_keys_normalized_to_str() -> None:
    provider = ReferenceEntryProvider(
        {"1": {"title": "T"}},
        relationships={1: (RelatedEntry("structures", "s-1"),)},  # type: ignore[dict-item]
    )
    assert provider.relationships("references") == {"1": (RelatedEntry("structures", "s-1"),)}


# --- registry / discovery -----------------------------------------------------


def test_data_providers_registered() -> None:
    # With httk-store importable, importing httk.core discovers the entry tier
    # and registers the four providers under their "store-*" names.
    import httk.core  # noqa: F401  (imported for its discovery side effect)

    assert {
        "store-references",
        "store-files",
        "store-calculations",
        "store-db-store",
    } <= set(known_entry_providers())


def test_registered_factories_resolve_and_build() -> None:
    for name, entry_type in (
        ("store-references", "references"),
        ("store-files", "files"),
        ("store-calculations", "calculations"),
    ):
        factory = resolve_callable(entry_providers.require(name).handler)
        provider = factory({})
        assert list(provider.entry_types()) == [entry_type]


def test_records_yield_json_arrays_for_tuple_fields():
    """Tuple-declared fields are served as lists, per the JSON-able records contract.

    The record models declare their sequence fields as tuples, but a JSON array
    is a list: a tuple serializes fine yet fails a type check against the
    property definition, so the provider's own output would be rejected by
    validate_record.
    """
    reference = Reference(title="T", authors=({"name": "Ada Lovelace"},), editors=({"name": "Boole"},))
    provider = ReferenceEntryProvider({"r1": reference})
    (record,) = provider.records("references")

    assert isinstance(record["authors"], list)
    assert record["authors"] == [{"name": "Ada Lovelace"}]
    assert isinstance(record["editors"], list)

    served = {key: value for key, value in record.items() if key != "__id"} | {"id": "r1"}
    validate_record(standard_entry_type("references"), served)  # must not raise


def test_standard_provider_records_validate_against_served_definitions() -> None:
    timestamp = datetime.datetime(2026, 7, 29, 12, 34, 56, 123456, tzinfo=datetime.UTC)
    providers = (
        ("references", ReferenceEntryProvider({"r1": Reference(title="T", last_modified=timestamp)})),
        (
            "files",
            FileEntryProvider(
                {
                    "f1": File(
                        url="https://example.test/file",
                        name="file",
                        last_modified=timestamp,
                        url_stable_until=timestamp,
                        modification_timestamp=timestamp,
                        atime=timestamp,
                        ctime=timestamp,
                        mtime=timestamp,
                    )
                }
            ),
        ),
        ("calculations", CalculationEntryProvider({"c1": Calculation(last_modified=timestamp)})),
    )

    for entry_type, provider in providers:
        entry_definition = provider.entry_types()[entry_type]
        property_keys = provider.property_keys(entry_type)
        for row in provider.records(entry_type):
            served = {name: row[key] for name, key in property_keys.items()}
            validate_record(entry_definition, served)
            assert served["last_modified"] == timestamp.isoformat()
