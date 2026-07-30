"""Tests for StoreEntryProvider (httk.data.db.entry_provider): definitions, records, relationships."""

import contextlib
import datetime
import json
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Annotated, ClassVar

import pytest
import sqlalchemy
from httk.core import (
    EntryTypeDefinition,
    FracVector,
    PropertyDefinition,
    Related,
    RelatedEntry,
    RelationshipLink,
    Shape,
    StorageInfo,
    stored_property,
)

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
    published=datetime.datetime(2026, 7, 24, 12, 30, 0, tzinfo=datetime.UTC),
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
    published=datetime.datetime(2020, 1, 1, 0, 0, 0, tzinfo=datetime.UTC),
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


def test_custom_definition_id_under_httk_ad_hoc_base(provider):
    """A generated definition lands in httk's *ad-hoc* namespace, not the published one.

    ``schemas.httk.org/defs/`` is where definitions actually served from schemas.httk.org
    live; a definition synthesized here for a stored field is not published anywhere, and
    ``/ad-hoc/`` says so rather than implying it resolves.
    """
    definition = provider.entry_types()["books"].properties["_httk_title"]
    assert definition.definition_id.startswith("https://schemas.httk.org/ad-hoc/defs/properties/")


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


# ---------------------------------------------------------------- property keys


def test_property_keys_id_type_and_identity_map(provider):
    property_keys = provider.property_keys("writers")
    assert property_keys == {"id": "__id", "type": "type", "_httk_name": "_httk_name", "_httk_born": "_httk_born"}
    book_keys = provider.property_keys("books")
    assert book_keys["id"] == "__id" and book_keys["type"] == "type"
    served = set(book_keys) - {"id", "type"}
    assert all(book_keys[name] == name for name in served)
    assert served == set(provider.entry_types()["books"].properties) - {"id", "type"}


def test_unknown_entry_type_raises_keyerror(provider):
    with pytest.raises(KeyError, match="books"):
        provider.property_keys("nope")
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


def test_records_batches_child_field_reads(provider):
    statements: list[str] = []

    def count_select(_conn, _cursor, statement, _parameters, _context, _executemany):
        if statement.lstrip().upper().startswith("SELECT"):
            statements.append(statement)

    engine = provider._store._database.engine
    sqlalchemy.event.listen(engine, "before_cursor_execute", count_select)
    try:
        list(provider.records("books"))
    finally:
        sqlalchemy.event.remove(engine, "before_cursor_execute", count_select)
    # One outer match query, one parent chunk, and one batch per touched child field.
    assert len(statements) <= 1 + 1 + 2


def test_records_values(provider):
    row = rows_by_id(provider, "books")["books-1"]
    assert row["type"] == "books"
    assert row["_httk_title"] == "Analytical Engines"
    assert row["_httk_pages"] == 350
    assert row["_httk_in_print"] is True
    assert row["_httk_price"] == pytest.approx(float(Fraction(1, 3)))  # rational -> nearest float
    assert row["_httk_published"] == "2026-07-24T12:30:00+00:00"  # datetime -> ISO text
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
    # child rows in insertion order (Boole sid 2, Cara sid 3), served as one
    # flat tuple; book 2 has no related entries and is omitted.
    assert related == {
        "books-1": (
            RelatedEntry("writers", "writers-1"),
            RelatedEntry("writers", "writers-2"),
            RelatedEntry("writers", "writers-3"),
        )
    }
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
    assert related == {
        "books/Analytical Engines": (
            RelatedEntry("writers", "writers/Ada"),
            RelatedEntry("writers", "writers/Boole"),
            RelatedEntry("writers", "writers/Cara"),
        )
    }


# --------------------------------------------------------------------- relationship metadata and links


@dataclass(frozen=True)
class Person:
    name: str


@dataclass(frozen=True)
class Project:
    name: str
    lead: Annotated[Person | None, Related(role="lead", description="Project lead")] = None
    backup: Annotated[Person | None, Related(serve=False)] = None
    members: Annotated[list[Person], Related(role="member")] = field(default_factory=list)


@dataclass(frozen=True)
class Compound:
    formula: str


@dataclass(frozen=True)
class Citation:
    doi: str


@dataclass(frozen=True)
class CompoundCitation:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(
        dedup="by_value",
        links=(RelationshipLink("compound", "citation", role="citation", description="Cited by"),),
    )

    compound: Compound
    citation: Citation


@dataclass(frozen=True)
class CompoundTag:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(
        dedup="none",
        links=(
            RelationshipLink("compound", "citation", role="primary"),
            RelationshipLink("compound", "citation", role="secondary"),
        ),
    )

    compound: Compound
    citation: Citation


@dataclass(frozen=True)
class Simulation:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(links=(RelationshipLink("compound", None, role="output"),))

    label: str
    compound: Compound | None = None


@contextlib.contextmanager
def sqlite_store(*objects):
    with Database.sqlite() as database:
        sql_store = SqlStore(database)
        with sql_store.transaction():
            for obj in objects:
                sql_store.save(obj)
        yield sql_store


def test_related_marker_meta_and_serve_false():
    people = (Person("Ada"), Person("Boole"), Person("Cara"))
    project = Project("Engine", lead=people[0], backup=people[1], members=[people[1], people[2]])
    with sqlite_store(*people, project) as store:
        provider = StoreEntryProvider(store, {"people": Person, "projects": Project})
        # The suppressed 'backup' field neither becomes a property nor a relationship:
        assert "_httk_backup" not in provider.entry_types()["projects"].properties
        assert provider.relationships("projects") == {
            "projects-1": (
                RelatedEntry("people", "people-1", description="Project lead", role="lead"),
                RelatedEntry("people", "people-2", role="member"),
                RelatedEntry("people", "people-3", role="member"),
            )
        }


def test_join_class_links_via_link_classes():
    c1, c2 = Compound("NaCl"), Compound("SiO2")
    z1, z2 = Citation("10.1/a"), Citation("10.1/b")
    with sqlite_store(
        c1, c2, z1, z2, CompoundCitation(c1, z1), CompoundCitation(c1, z2), CompoundCitation(c2, z1)
    ) as store:
        provider = StoreEntryProvider(
            store, {"compounds": Compound, "citations": Citation}, link_classes=[CompoundCitation]
        )
        # The join class is not served as an entry type:
        assert sorted(provider.entry_types()) == ["citations", "compounds"]
        assert provider.relationships("compounds") == {
            "compounds-1": (
                RelatedEntry("citations", "citations-1", description="Cited by", role="citation"),
                RelatedEntry("citations", "citations-2", description="Cited by", role="citation"),
            ),
            "compounds-2": (RelatedEntry("citations", "citations-1", description="Cited by", role="citation"),),
        }
        # Links are indexed by FROM side only; the TO side gains nothing:
        assert provider.relationships("citations") == {}


def test_served_class_link_with_none_target_is_field_inverse():
    compound = Compound("NaCl")
    with sqlite_store(compound, Simulation("run-1", compound), Simulation("run-2", None)) as store:
        provider = StoreEntryProvider(store, {"compounds": Compound, "simulations": Simulation})
        # compounds gain the inverse relationship to the simulation entry itself;
        # the NULL-FK row (run-2) is skipped.
        assert provider.relationships("compounds") == {
            "compounds-1": (RelatedEntry("simulations", "simulations-1", role="output"),)
        }
        # The forward 'compound' reference field still serves as usual (no meta declared):
        assert provider.relationships("simulations") == {
            "simulations-1": (RelatedEntry("compounds", "compounds-1"),)
        }


def test_link_ordering_and_dedup():
    c1 = Compound("NaCl")
    z1 = Citation("10.1/a")
    tag = CompoundTag(c1, z1)
    # dedup="none": saving the same tag twice stores two rows; each row expresses
    # both declared links, so without dedup compounds-1 would get four entries.
    with sqlite_store(c1, z1, tag, CompoundTag(c1, z1)) as store:
        provider = StoreEntryProvider(
            store, {"compounds": Compound, "citations": Citation}, link_classes=[CompoundTag]
        )
        # Exact duplicates collapse (first occurrence wins); entries differing
        # only in meta are both kept, in link declaration order.
        assert provider.relationships("compounds") == {
            "compounds-1": (
                RelatedEntry("citations", "citations-1", role="primary"),
                RelatedEntry("citations", "citations-1", role="secondary"),
            )
        }


def test_link_with_unserved_endpoint_raises():
    with sqlite_store() as store:
        with pytest.raises(ValueError, match="Citation.*not served"):
            StoreEntryProvider(store, {"compounds": Compound}, link_classes=[CompoundCitation])
        with pytest.raises(ValueError, match="Compound.*not served"):
            StoreEntryProvider(store, {"citations": Citation}, link_classes=[CompoundCitation])
        # A served class declaring a link with an unserved endpoint is just as loud:
        with pytest.raises(ValueError, match="Compound.*not served"):
            StoreEntryProvider(store, {"simulations": Simulation})


def test_link_class_without_links_raises():
    with sqlite_store() as store, pytest.raises(ValueError, match="Person.*no relationship links"):
        StoreEntryProvider(store, {"compounds": Compound}, link_classes=[Person])


def test_custom_id_of_used_on_link_paths():
    c1 = Compound("NaCl")
    z1 = Citation("10.1/a")
    with sqlite_store(c1, z1, CompoundCitation(c1, z1)) as store:

        def id_of(entry_type, sid, obj):
            return f"{entry_type}/{obj.formula if entry_type == 'compounds' else obj.doi}"

        provider = StoreEntryProvider(
            store, {"compounds": Compound, "citations": Citation}, link_classes=[CompoundCitation], id_of=id_of
        )
        assert provider.relationships("compounds") == {
            "compounds/NaCl": (
                RelatedEntry("citations", "citations/10.1/a", description="Cited by", role="citation"),
            )
        }


# --------------------------------------------------------------------- validation


def test_every_record_validates_against_served_definition(provider):
    entry_types = provider.entry_types()
    for entry_type in entry_types:
        property_keys = provider.property_keys(entry_type)
        for row in provider.records(entry_type):
            validate_record(entry_types[entry_type], {name: row[key] for name, key in property_keys.items()})


# --------------------------------------------------------------------- OPTIMADE end to end


def test_optimade_adapter_end_to_end(provider):
    pytest.importorskip("httk.serve.optimade")
    from httk.serve.optimade import adapter_from_providers
    from httk.serve.optimade.backend import execute_query
    from httk.serve.optimade.filter import parse_optimade_filter

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
