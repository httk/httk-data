"""Tests for httk-data's in-memory entry providers and their registration."""

import pytest

from httk.core import Calculation, EntryTypeDefinition, File, known_entry_providers
from httk.core._plugins import resolve_callable
from httk.core.register import entry_providers
from httk.data import CalculationEntryProvider, FileEntryProvider, ReferenceEntryProvider

# --- provider round trips -----------------------------------------------------


def test_reference_provider_round_trip() -> None:
    provider = ReferenceEntryProvider({"ref-1": {"title": "T", "doi": "10.1/x"}})
    entry_types = provider.entry_types()
    assert set(entry_types) == {"references"}
    assert isinstance(entry_types["references"], EntryTypeDefinition)
    columns = provider.columns("references")
    assert columns["id"] == "__id"
    assert columns["type"] == "type"
    assert columns["title"] == "title"
    records = list(provider.records("references"))
    assert records[0]["__id"] == "ref-1"
    assert records[0]["type"] == "references"
    assert records[0]["title"] == "T"
    assert records[0]["url"] is None
    # Every served column key is present in every record:
    for record in records:
        for column in columns.values():
            assert column in record


def test_file_provider_records() -> None:
    provider = FileEntryProvider({"f-1": File(url="http://x/INCAR", name="INCAR", size=512)})
    record = list(provider.records("files"))[0]
    assert record["url"] == "http://x/INCAR"
    assert record["size"] == 512


def test_calculation_provider_columns_cover_id_type() -> None:
    provider = CalculationEntryProvider({"calc-1": Calculation()})
    columns = provider.columns("calculations")
    assert {"id", "type"} <= set(columns)


def test_provider_rejects_wrong_entry_type() -> None:
    provider = ReferenceEntryProvider({})
    with pytest.raises(KeyError):
        provider.columns("files")


# --- registry / discovery -----------------------------------------------------


def test_data_providers_registered() -> None:
    # With httk-data importable, importing httk.core discovers httk.handlers.data
    # and registers the three providers under their "data-*" names.
    import httk.core  # noqa: F401  (imported for its discovery side effect)

    assert {"data-references", "data-files", "data-calculations"} <= set(known_entry_providers())


def test_registered_factories_resolve_and_build() -> None:
    for name, entry_type in (
        ("data-references", "references"),
        ("data-files", "files"),
        ("data-calculations", "calculations"),
    ):
        factory = resolve_callable(entry_providers.require(name).handler)
        provider = factory({})
        assert list(provider.entry_types()) == [entry_type]
