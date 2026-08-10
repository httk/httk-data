"""Coverage for provenance providers and their SQL storage backings."""

import datetime
import json

import pytest
import sqlalchemy
from httk.core import (
    DataRecord,
    DataRecordEntry,
    ProductLink,
    PropertyDefinition,
    RelatedEntry,
    Run,
    RunEdge,
    RunEntry,
)

from httk.store import DataRecordEntryProvider, RunEntryProvider, product_relationships, validate_record
from httk.store.db import Database, EntryMetadataConflictError, SqlStore
from httk.store.db.schema import resolve_schema

UTC = datetime.UTC
ENERGY_ID = "https://schemas.example.org/properties/energy"
FORCE_ID = "https://schemas.example.org/properties/force"


def _definition(name: str, definition_id: str, *, required_response: bool = False) -> PropertyDefinition:
    return PropertyDefinition.from_simple(
        name,
        description=f"The {name} value.",
        fulltype="float",
        definition_id=definition_id,
        required_response=required_response,
    )


def _run() -> Run:
    return Run(
        inputs=(RunEdge("in-1", "_httk_records", "record-1"), RunEdge("in-2", "structures", "structure-1")),
        artifacts=(RunEdge("artifact", "files", "file-1"),),
        outputs=(RunEdge("out", "_httk_records", "record-2"),),
        immutable_id="run-immutable",
        last_modified=datetime.datetime(2026, 8, 1, 12, 0, tzinfo=UTC),
    )


def test_run_provider_serves_rows_and_relationships() -> None:
    run = _run()
    provider = RunEntryProvider({"run-key": run, "run-map": {"workflow_declaration_uri": None}})
    definition = provider.entry_types()['_httk_runs']
    assert set(definition.properties) == {
        "id",
        "type",
        "immutable_id",
        "last_modified",
        "_httk_workflow_declaration_uri",
    }
    assert "workflow_declaration_uri" not in definition.properties

    rows = list(provider.records("_httk_runs"))
    assert json.dumps(rows)
    assert rows[0]["__id"] == "run-key"
    assert rows[0]["workflow_declaration_uri"] is None
    assert provider.relationships("_httk_runs")["run-key"] == (
        RelatedEntry("_httk_records", "record-1", role="input", label="in-1"),
        RelatedEntry("structures", "structure-1", role="input", label="in-2"),
        RelatedEntry("files", "file-1", role="artifact", label="artifact"),
        RelatedEntry("_httk_records", "record-2", role="output", label="out"),
    )


def test_run_provider_rows_validate() -> None:
    provider = RunEntryProvider({"run-key": _run()})
    definition = provider.entry_types()['_httk_runs']
    keys = provider.property_keys("_httk_runs")
    for row in provider.records("_httk_runs"):
        validate_record(definition, {name: row[key] for name, key in keys.items()})


def test_data_record_provider_union_and_validation() -> None:
    definitions = {
        "_httk_energy": _definition("_httk_energy", ENERGY_ID),
        "_httk_force": _definition("_httk_force", FORCE_ID),
    }
    provider = DataRecordEntryProvider(
        {
            "energy-1": DataRecord.from_value(ENERGY_ID, "_httk_energy", 3.5),
            "force-1": DataRecord.from_value(FORCE_ID, "_httk_force", 2.0),
        },
        definitions=definitions,
    )
    keys = provider.property_keys("_httk_records")
    rows = list(provider.records("_httk_records"))
    assert rows[0]["_httk_energy"] == 3.5 and rows[0]["_httk_force"] is None
    assert rows[1]["_httk_energy"] is None and rows[1]["_httk_force"] == 2.0
    assert set(provider.entry_types()["_httk_records"].properties) == set(keys)
    for row in rows:
        validate_record(provider.entry_types()["_httk_records"], {name: row[key] for name, key in keys.items()})


def test_data_record_provider_fails_eagerly_on_definitions() -> None:
    record = DataRecord.from_value("https://unknown.example/energy", "_httk_energy", 1)
    with pytest.raises(ValueError, match="record 'r'.*_httk_energy.*https://unknown.example/energy"):
        DataRecordEntryProvider({"r": record})

    with pytest.raises(ValueError, match="record 'r'.*definition IRI"):
        DataRecordEntryProvider({"r": record}, definitions={"_httk_energy": _definition("_httk_energy", ENERGY_ID)})


def test_data_record_provider_rejects_non_nullable_union_nulls() -> None:
    energy = DataRecord.from_value(ENERGY_ID, "_httk_energy", 1.0)
    force = DataRecord.from_value(FORCE_ID, "_httk_force", 2.0)
    with pytest.raises(ValueError, match="_httk_energy.*force"):
        DataRecordEntryProvider(
            {"energy": energy, "force": force},
            definitions={
                "_httk_energy": _definition("_httk_energy", ENERGY_ID, required_response=True),
                "_httk_force": _definition("_httk_force", FORCE_ID),
            },
        )
    provider = DataRecordEntryProvider(
        {"energy-1": energy, "energy-2": DataRecord.from_value(ENERGY_ID, "_httk_energy", 3.0)},
        definitions={"_httk_energy": _definition("_httk_energy", ENERGY_ID, required_response=True)},
    )
    assert len(list(provider.records("_httk_records"))) == 2


def test_product_relationships_groups_in_input_order_and_checks_labels() -> None:
    links = [
        ProductLink("runs", "r-1", "records", "d-1", "energy"),
        ProductLink("runs", "r-1", "records", "d-2", "force"),
        ProductLink("runs", "r-2", "records", "d-3", "energy"),
    ]
    result = product_relationships(links)
    assert [entry.id for entry in result["runs"]["r-1"]] == ["d-1", "d-2"]
    assert result["runs"]["r-1"][0].role == "product"
    assert result["runs"]["r-2"][0].label == "energy"
    with pytest.raises(ValueError, match="runs.*r-1.*energy"):
        product_relationships([links[0], ProductLink("runs", "r-1", "records", "d-4", "energy")])


def test_data_record_provider_serves_product_relationships() -> None:
    records = {
        "record-1": DataRecord.from_value(ENERGY_ID, "_httk_energy", 1.0),
        "record-2": DataRecord.from_value(FORCE_ID, "_httk_force", 2.0),
    }
    links = product_relationships([ProductLink("_httk_records", "record-1", "_httk_records", "record-2", "derived")])
    provider = DataRecordEntryProvider(
        records,
        definitions={
            "_httk_energy": _definition("_httk_energy", ENERGY_ID),
            "_httk_force": _definition("_httk_force", FORCE_ID),
        },
        relationships=links["_httk_records"],
    )
    assert provider.relationships("_httk_records") == {
        "record-1": (RelatedEntry("_httk_records", "record-2", role="product", label="derived"),)
    }


def test_sql_store_round_trips_provenance_records_and_stored_number() -> None:
    with Database.sqlite() as database:
        store = SqlStore(database, entry_records={RunEntry: Run, DataRecordEntry: DataRecord})
        run = _run()
        store.save(run)
        fetched = store.fetch_entry(RunEntry, run.id)
        assert fetched == run
        assert fetched.immutable_id == run.immutable_id and fetched.last_modified == run.last_modified
        assert fetched.inputs == run.inputs

        conflicting = Run(
            inputs=run.inputs,
            artifacts=run.artifacts,
            outputs=run.outputs,
            immutable_id="different",
            last_modified=run.last_modified,
        )
        with pytest.raises(EntryMetadataConflictError):
            store.save(conflicting)

        record = DataRecord.from_value(ENERGY_ID, "_httk_energy", 3.5)
        store.save(record)
        assert store.fetch_entry(DataRecordEntry, record.id).value == 3.5
        table_name = resolve_schema(DataRecord).table_name
        columns = {column["name"] for column in sqlalchemy.inspect(database.engine).get_columns(table_name)}
        assert "value_number" in columns
        searcher = store.searcher()
        variable = searcher.variable(DataRecord)
        searcher.add(variable.value_number == 3.5)
        searcher.output(variable, "record")
        assert [row[0][0].value for row in searcher] == [3.5]


def test_product_link_storage_deduplicates_by_value() -> None:
    with Database.sqlite() as database:
        store = SqlStore(database, entry_records={})
        link = ProductLink("runs", "r-1", "records", "d-1", "energy")
        sid = store.save(link)
        assert store.save(link) == sid
        assert store.fetch(ProductLink, sid) == link


def test_run_provider_serves_through_optimade() -> None:
    pytest.importorskip("httk.serve.optimade")
    from httk.serve.optimade import adapter_from_providers, create_asgi_app
    from starlette.testclient import TestClient

    definition = _definition("_httk_energy", ENERGY_ID)
    record = DataRecord.from_value(ENERGY_ID, "_httk_energy", 3.5)
    product = DataRecord.from_value(FORCE_ID, "_httk_force", 2.0)
    run = Run(inputs=(RunEdge("labeled-input", "_httk_records", "record-1"),))
    product_links = product_relationships(
        [ProductLink("_httk_records", "record-1", "_httk_records", "record-2", "derived")]
    )
    adapter = adapter_from_providers(
        [
            RunEntryProvider({"run-1": run}),
            DataRecordEntryProvider(
                {"record-1": record, "record-2": product},
                definitions={"_httk_energy": definition, "_httk_force": _definition("_httk_force", FORCE_ID)},
                relationships=product_links["_httk_records"],
            ),
        ]
    )
    with TestClient(create_asgi_app(adapter, baseurl="http://testserver/")) as client:
        run_response = client.get("/_httk_runs")
        records_response = client.get("/_httk_records")
    assert run_response.status_code == records_response.status_code == 200
    run_relation = run_response.json()["data"][0]["relationships"]["_httk_records"]["data"][0]
    assert run_relation["meta"]["_httk_label"] == "labeled-input"
    record = next(item for item in records_response.json()["data"] if item["id"] == "record-1")
    product_relation = record["relationships"]["_httk_records"]["data"][0]
    assert product_relation["meta"] == {"role": "product", "_httk_label": "derived"}
