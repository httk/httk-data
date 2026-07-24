"""Tests for StoreEntryProvider (httk.data.db.entry_provider): definitions, records, relationships."""

import datetime
import json
from dataclasses import dataclass
from fractions import Fraction
from typing import Annotated

import pytest
from httk.core import EntryTypeDefinition, FracVector, PropertyDefinition, Shape, stored_property

from httk.data.db import Database, SqlStore, StoreEntryProvider
from httk.data.validation import validate_record


@dataclass(frozen=True)
class Writer:
    name: str
    born: int


@dataclass(frozen=True)
class Book:
    title: str
    pages: int
    price: Fraction
    in_print: bool
    cover: bytes
    published: datetime.datetime
    metric: Annotated[FracVector, Shape(2, 2)]
    samples: Annotated[FracVector, Shape(0, 2)]
    keywords: list[str]
    coauthors: list[Writer]
    author: Writer | None = None

    @stored_property
    def nkeywords(self) -> int:
        return len(self.keywords)


ADA = Writer("Ada", 1815)
BOOLE = Writer("Boole", 1815)
CARA = Writer("Cara", 1820)

BOOK_1 = Book(
    title="Analytical Engines",
    pages=350,
    price=Fraction(1, 3),
    in_print=True,
    cover=b"\x00\xff",
    published=datetime.datetime(2026, 7, 24, 12, 30, 0),
    metric=FracVector.create([[1, Fraction(1, 2)], [0, 1]]),
    samples=FracVector.create([[0, 0], [Fraction(1, 2), Fraction(1, 2)]]),
    keywords=["computing", "history"],
    coauthors=[BOOLE, CARA],
    author=ADA,
)
BOOK_2 = Book(
    title="Silence",
    pages=120,
    price=Fraction(-7, 5),
    in_print=False,
    cover=b"",
    published=datetime.datetime(2020, 1, 1, 0, 0, 0),
    metric=FracVector.create([[1, 0], [0, 1]]),
    samples=FracVector.create([]),
    keywords=[],
    coauthors=[],
    author=None,
)


@pytest.fixture()
def store():
    with Database.sqlite() as database:
        sql_store = SqlStore(database)
        with sql_store.transaction():
            # Writers first, in a known order, so their sids are 1, 2, 3.
            for writer in (ADA, BOOLE, CARA):
                sql_store.save(writer)
            sql_store.save(BOOK_1)  # book sid 1
            sql_store.save(BOOK_2)  # book sid 2
        yield sql_store


@pytest.fixture()
def provider(store):
    return StoreEntryProvider(store, {"books": Book, "writers": Writer})


def rows_by_id(provider, entry_type):
    return {row["__id"]: row for row in provider.records(entry_type)}


# --------------------------------------------------------------------- auto-generated definitions


def test_entry_types_serves_all_classes(provider):
    assert sorted(provider.entry_types()) == ["books", "writers"]


def test_auto_definition_properties_prefixed_and_core_present(provider):
    definition = provider.entry_types()["books"]
    assert isinstance(definition, EntryTypeDefinition)
    assert sorted(definition.properties) == [
        "_httk_in_print",
        "_httk_keywords",
        "_httk_metric",
        "_httk_nkeywords",
        "_httk_pages",
        "_httk_price",
        "_httk_published",
        "_httk_samples",
        "_httk_title",
        "id",
        "type",
    ]
    assert not definition.properties["id"].nullable
    assert not definition.properties["type"].nullable


def test_auto_definition_fulltype_mapping(provider):
    properties = provider.entry_types()["books"].properties
    assert properties["_httk_title"].optimade_type == "string"
    assert properties["_httk_pages"].optimade_type == "integer"
    assert properties["_httk_in_print"].optimade_type == "boolean"
    assert properties["_httk_price"].optimade_type == "float"  # rational served as float
    assert properties["_httk_published"].optimade_type == "timestamp"
    assert properties["_httk_nkeywords"].optimade_type == "integer"  # derived stored property
    keywords = properties["_httk_keywords"].as_optimade()
    assert keywords["x-optimade-type"] == "list"
    assert keywords["items"]["x-optimade-type"] == "string"
    metric = properties["_httk_metric"].as_optimade()
    assert metric["items"]["items"]["x-optimade-type"] == "float"
    assert metric["x-optimade-dimensions"]["sizes"] == [2, 2]
    samples = properties["_httk_samples"].as_optimade()
    assert samples["items"]["items"]["x-optimade-type"] == "float"


def test_bytes_reference_and_storable_children_not_served_as_properties(provider):
    properties = provider.entry_types()["books"].properties
    assert "_httk_cover" not in properties  # bytes: no OPTIMADE value representation
    assert "_httk_author" not in properties  # reference: surfaces through relationships()
    assert "_httk_coauthors" not in properties  # child of storables: relationships()


def test_custom_definition_id_under_httk_base(provider):
    definition = provider.entry_types()["books"].properties["_httk_title"]
    assert definition.definition_id.startswith("https://httk.org/")


def test_unregistered_prefix_raises(store):
    with pytest.raises(ValueError, match="_nope_"):
        StoreEntryProvider(store, {"books": Book}, prefix="_nope_")


def test_supplied_definitions_pass_through(store):
    generated = dict(StoreEntryProvider(store, {"books": Book, "writers": Writer}).entry_types())
    provider = StoreEntryProvider(store, {"books": Book, "writers": Writer}, definitions=generated)
    assert provider.entry_types()["books"] is generated["books"]
    assert provider.entry_types()["writers"] is generated["writers"]


def test_supplied_definition_must_describe_served_properties(store):
    incomplete = EntryTypeDefinition(
        "writers",
        "Writers.",
        {
            "id": PropertyDefinition.from_simple("id", description="id", required_response=True),
            "type": PropertyDefinition.from_simple("type", description="type", required_response=True),
        },
    )
    with pytest.raises(ValueError, match="_httk_born.*_httk_name|_httk_name"):
        StoreEntryProvider(store, {"writers": Writer}, definitions={"writers": incomplete})


def test_supplied_definition_for_unserved_entry_type_rejected(store):
    generated = dict(StoreEntryProvider(store, {"writers": Writer}).entry_types())
    with pytest.raises(ValueError, match="writers"):
        StoreEntryProvider(store, {"books": Book}, definitions=generated)


# --------------------------------------------------------------------- columns


def test_columns_id_type_and_identity_map(provider):
    columns = provider.columns("writers")
    assert columns == {"id": "__id", "type": "type", "_httk_name": "_httk_name", "_httk_born": "_httk_born"}
    book_columns = provider.columns("books")
    assert book_columns["id"] == "__id" and book_columns["type"] == "type"
    served = set(book_columns) - {"id", "type"}
    assert all(book_columns[name] == name for name in served)
    assert served == set(provider.entry_types()["books"].properties) - {"id", "type"}


def test_unknown_entry_type_raises_keyerror(provider):
    with pytest.raises(KeyError, match="books"):
        provider.columns("nope")
    with pytest.raises(KeyError):
        list(provider.records("nope"))
    with pytest.raises(KeyError):
        provider.relationships("nope")


# --------------------------------------------------------------------- records


def test_records_is_a_generator_of_json_able_rows(provider):
    records = provider.records("books")
    assert iter(records) is records  # a generator, not a materialized list
    rows = list(records)
    assert len(rows) == 2
    json.dumps(rows)  # every value must be JSON-able


def test_records_values(provider):
    row = rows_by_id(provider, "books")["books-1"]
    assert row["type"] == "books"
    assert row["_httk_title"] == "Analytical Engines"
    assert row["_httk_pages"] == 350
    assert row["_httk_in_print"] is True
    assert row["_httk_price"] == pytest.approx(float(Fraction(1, 3)))  # rational -> nearest float
    assert row["_httk_published"] == "2026-07-24T12:30:00"  # datetime -> ISO text
    assert row["_httk_metric"] == [[1.0, 0.5], [0.0, 1.0]]  # fixed tensor -> nested lists
    assert row["_httk_samples"] == [[0.0, 0.0], [0.5, 0.5]]  # variable rows -> list of lists
    assert row["_httk_keywords"] == ["computing", "history"]
    assert row["_httk_nkeywords"] == 2
    assert "_httk_cover" not in row and "_httk_author" not in row and "_httk_coauthors" not in row


def test_records_empty_containers(provider):
    row = rows_by_id(provider, "books")["books-2"]
    assert row["_httk_samples"] == []
    assert row["_httk_keywords"] == []
    assert row["_httk_nkeywords"] == 0
    assert row["_httk_in_print"] is False


def test_writer_records(provider):
    rows = rows_by_id(provider, "writers")
    assert set(rows) == {"writers-1", "writers-2", "writers-3"}
    assert rows["writers-1"]["_httk_name"] == "Ada"
    assert rows["writers-3"]["_httk_born"] == 1820


# --------------------------------------------------------------------- relationships


def test_relationships_across_served_classes(provider):
    related = provider.relationships("books")
    # book 1: the 'author' reference (Ada, sid 1) first, then the 'coauthors'
    # child rows in insertion order (Boole sid 2, Cara sid 3); book 2 has no
    # related entries and is omitted.
    assert related == {"books-1": {"writers": ("writers-1", "writers-2", "writers-3")}}
    assert provider.relationships("writers") == {}


def test_relationships_empty_when_target_class_not_served(store):
    provider = StoreEntryProvider(store, {"books": Book})
    assert provider.relationships("books") == {}
    assert "_httk_author" not in provider.entry_types()["books"].properties


def test_id_of_override_used_in_records_and_relationships(store):
    def id_of(entry_type, sid, obj):
        return f"{entry_type}/{getattr(obj, 'title', None) or obj.name}"

    provider = StoreEntryProvider(store, {"books": Book, "writers": Writer}, id_of=id_of)
    assert set(rows_by_id(provider, "books")) == {"books/Analytical Engines", "books/Silence"}
    related = provider.relationships("books")
    assert related == {"books/Analytical Engines": {"writers": ("writers/Ada", "writers/Boole", "writers/Cara")}}


# --------------------------------------------------------------------- validation


def test_every_record_validates_against_served_definition(provider):
    entry_types = provider.entry_types()
    for entry_type in entry_types:
        columns = provider.columns(entry_type)
        for row in provider.records(entry_type):
            validate_record(entry_types[entry_type], {name: row[column] for name, column in columns.items()})


# --------------------------------------------------------------------- OPTIMADE end to end


def test_optimade_adapter_end_to_end(provider):
    pytest.importorskip("httk.optimade")
    from httk.optimade import adapter_from_providers
    from httk.optimade.backend import execute_query
    from httk.optimade.filter import parse_optimade_filter

    adapter = adapter_from_providers([provider])
    assert set(adapter.schema.all_entries) == {"books", "writers"}

    results = list(
        execute_query(adapter, ["books"], ["id", "_httk_title"], [], 100, 0, parse_optimade_filter("_httk_pages > 200"))
    )
    assert [r.values["id"] for r in results] == ["books-1"]
    assert results[0].values["_httk_title"] == "Analytical Engines"

    results = list(
        execute_query(
            adapter, ["books"], ["id"], [], 100, 0, parse_optimade_filter('_httk_keywords HAS "history"')
        )
    )
    assert [r.values["id"] for r in results] == ["books-1"]

    results = list(
        execute_query(adapter, ["writers"], ["id"], [], 100, 0, parse_optimade_filter("_httk_born = 1820"))
    )
    assert [r.values["id"] for r in results] == ["writers-3"]
