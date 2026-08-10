"""Exact SQL persistence for the core OPTIMADE source-resource models."""

from collections.abc import Mapping
from decimal import Decimal

import pytest
import sqlalchemy
from httk.core import Reference
from httk.core.optimade import (
    OptimadeDocument,
    OptimadeReference,
    OptimadeResource,
    OptimadeSchemaSnapshot,
    ReferenceView,
)

from httk.store.db import Database, SqlStore

PAGE_TEXT = '''{
  "jsonapi": {"version": "1.1"},
  "meta": {"provider": "example", "_unknown_meta": {"exact": 1.2300e+2}},
  "data": [
    {
      "id": "reference-1",
      "type": "references",
      "attributes": {
        "remote_immutable": "immutable-reference-1",
        "remote_modified": "2026-07-30T12:34:56+00:00",
        "remote_title": "Exact reference",
        "_vendor_measurement": 1.2300e+2,
        "_vendor_unknown": {"nested": ["x", 0.0100]}
      },
      "relationships": {
        "files": {"links": {"related": "https://example.invalid/files?token=kept-in-source"}, "data": []}
      },
      "_unknown_top_level": {"preserve": true}
    },
    {
      "id": "reference-2",
      "type": "references",
      "attributes": {
        "remote_immutable": "immutable-reference-2",
        "remote_modified": "2026-07-30T12:34:57+00:00",
        "remote_title": "Second reference",
        "_vendor_measurement": 9.900e-3
      },
      "relationships": {"files": {"data": []}}
    }
  ],
  "links": {"self": "https://example.invalid/v1/references"}
}'''

INFO_TEXT = '''{
  "data": {
    "id": "references",
    "type": "info",
    "properties": {
      "id_transport": {"$id": "https://schemas.optimade.org/defs/v1.2/properties/core/id"},
      "type_transport": {"$id": "https://schemas.optimade.org/defs/v1.2/properties/core/type"},
      "remote_immutable": {"$id": "https://schemas.optimade.org/defs/v1.2/properties/core/immutable_id"},
      "remote_modified": {"$id": "https://schemas.optimade.org/defs/v1.2/properties/core/last_modified"},
      "remote_title": {"$id": "https://schemas.optimade.org/defs/v1.2/properties/optimade/references/title"},
      "_vendor_measurement": {"$id": "https://example.invalid/definitions/measurement"}
    },
    "entry_types_by_format": {"json": ["references"]}
  },
  "meta": {"api_version": "1.2.0"}
}'''


@pytest.fixture(params=["sqlite", "duckdb"])
def database(request):
    """An in-memory database per supported dialect (DuckDB is optional)."""

    if request.param == "duckdb":
        pytest.importorskip("duckdb_engine")
        with Database.duckdb() as db:
            yield db
    else:
        with Database.sqlite() as db:
            yield db


def _resources() -> tuple[OptimadeResource, OptimadeResource]:
    page = OptimadeDocument(PAGE_TEXT, "https://example.invalid/v1/references?page_limit=2")
    info = OptimadeDocument(INFO_TEXT, "https://example.invalid/v1/info/references")
    snapshot = OptimadeSchemaSnapshot("references", info)
    return OptimadeResource(page, 0, snapshot), OptimadeResource(page, 1, snapshot)


def _count(database: Database, table_name: str) -> int:
    with database.engine.connect() as connection:
        return int(connection.execute(sqlalchemy.text(f"SELECT COUNT(*) FROM {table_name}")).scalar_one())


def _object_results(store: SqlStore, target: type, expression):
    searcher = store.searcher()
    variable = searcher.variable(target)
    searcher.add(expression(variable))
    return searcher.results(record=variable, identifier=variable.id, entry_type=variable.type)


def test_source_resources_round_trip_deduplicate_and_query_offline(database):
    first, second = _resources()
    store = SqlStore(database, entry_records={})
    first_sid = store.save(first)
    second_sid = store.save(second)

    # Equal immutable source values are content-deduplicated rather than
    # inserting duplicate page, info, snapshot, or resource rows.
    assert store.save(first) == first_sid
    assert store.save(_resources()[1]) == second_sid
    assert _count(database, "optimade_document") == 2  # one page + one /info document
    assert _count(database, "optimade_schema_snapshot") == 1
    assert _count(database, "optimade_resource") == 2

    fresh = SqlStore(database)
    fetched = fresh.fetch(OptimadeResource, first_sid)
    fetched_second = fresh.fetch(OptimadeResource, second_sid)
    assert isinstance(fetched, OptimadeResource)
    assert isinstance(fetched_second, OptimadeResource)
    assert fetched == first
    assert fetched_second == second
    assert fetched.document.text == PAGE_TEXT
    assert fetched.document.source_url == "https://example.invalid/v1/references?page_limit=2"
    assert fetched.schema.info_document.text == INFO_TEXT
    assert fetched.data_index == 0
    assert "1.2300e+2" in fetched.document.text
    attributes = fetched["attributes"]
    relationships = fetched["relationships"]
    extension = fetched["_unknown_top_level"]
    assert isinstance(attributes, Mapping)
    assert isinstance(relationships, Mapping)
    assert isinstance(extension, Mapping)
    measurement = attributes["_vendor_measurement"]
    unknown = attributes["_vendor_unknown"]
    files = relationships["files"]
    assert measurement == Decimal("123.00")
    assert isinstance(unknown, Mapping)
    nested = unknown["nested"]
    assert isinstance(nested, tuple)
    assert nested[1] == Decimal("0.0100")
    assert isinstance(files, Mapping)
    assert files["data"] == ()
    assert extension["preserve"] is True
    assert fetched.id == "reference-1"
    assert fetched.type == "references"

    # These stored properties are ordinary columns: no SQL JSON functions or
    # reparsing of the source document participates in filtering or projection.
    results = _object_results(
        fresh,
        OptimadeResource,
        lambda resource: (resource.id == "reference-1") & (resource.type == "references"),
    )
    assert len(results) == 1
    row = results.one()
    assert row.identifier == "reference-1"
    assert row.entry_type == "references"
    assert isinstance(row.record, OptimadeResource)
    assert row.record.id == "reference-1"
    assert row.record.document.text == PAGE_TEXT


def test_typed_reference_round_trip_query_and_view_offline(database):
    first, _second = _resources()
    backend = OptimadeReference(first)
    store = SqlStore(database, entry_records={})
    sid = store.save(backend)
    columns = set(store._table("optimade_reference").c.keys())
    assert {"id", "type", "immutable_id", "last_modified"} <= columns
    with database.engine.connect() as connection:
        materialized = connection.execute(
            sqlalchemy.text("SELECT id, type, immutable_id, last_modified FROM optimade_reference")
        ).one()
    assert tuple(materialized) == (
        "reference-1",
        "references",
        "immutable-reference-1",
        "2026-07-30T12:34:56+00:00",
    )

    fresh = SqlStore(database)
    fetched = fresh.fetch(OptimadeReference, sid)
    assert isinstance(fetched, OptimadeReference)
    assert fetched.resource == first
    assert fetched.resource.document.text == PAGE_TEXT
    assert fetched.id == "reference-1"
    assert fetched.type == "references"
    assert fetched.immutable_id == "immutable-reference-1"
    assert fetched.last_modified is not None
    assert fetched.last_modified.isoformat() == "2026-07-30T12:34:56+00:00"
    fetched_record = ReferenceView(fetched).record
    assert isinstance(fetched_record, Reference)
    assert fetched_record.title == "Exact reference"

    results = _object_results(
        fresh,
        OptimadeReference,
        lambda reference: (
            (reference.immutable_id == "immutable-reference-1") & (reference.last_modified == fetched.last_modified)
        ),
    )
    assert len(results) == 1
    row = results.one()
    assert row.identifier == "reference-1"
    assert row.entry_type == "references"
    assert isinstance(row.record, OptimadeReference)
    assert row.record.resource.document.text == PAGE_TEXT
    row_record = ReferenceView(row.record).record
    assert isinstance(row_record, Reference)
    assert row_record.title == "Exact reference"
