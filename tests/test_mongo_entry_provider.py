"""Live Mongo semantic coverage for the Mongo entry provider."""

import datetime
import json
from dataclasses import dataclass
from fractions import Fraction
from typing import Annotated, ClassVar

import pytest
from httk.core import EntryTypeDefinition, FracVector, RelatedEntry
from httk.core.register import register_entry_family, register_entry_record
from httk.core.storage import Related, RelationshipLink, Shape, StorageInfo, stored_property

from httk.data.mongo import MongoStore, StoreEntryProvider
from httk.data.validation import validate_record


@dataclass(frozen=True)
class MongoWriter:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(dedup="content_id")

    name: str
    born: int


@dataclass(frozen=True)
class MongoBook:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(dedup="content_id")

    title: str
    pages: int
    price: Fraction
    in_print: bool
    cover: bytes
    published: datetime.datetime
    metric: Annotated[FracVector, Shape(2, 2)]
    samples: Annotated[FracVector, Shape(0, 2)]
    keywords: list[str]
    coauthors: list[MongoWriter]
    author: MongoWriter | None = None

    @stored_property
    def nkeywords(self) -> int:
        return len(self.keywords)


@dataclass(frozen=True)
class MongoPerson:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(dedup="content_id")
    name: str


@dataclass(frozen=True)
class MongoProject:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(dedup="content_id")
    name: str
    lead: Annotated[MongoPerson | None, Related(role="lead", description="Project lead")] = None
    backup: Annotated[MongoPerson | None, Related(serve=False)] = None
    members: Annotated[list[MongoPerson] | None, Related(role="member")] = None


@dataclass(frozen=True)
class MongoSimulation:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(
        dedup="content_id", links=(RelationshipLink("compound", None, role="output"),)
    )
    label: str
    compound: "MongoCompound | None" = None


@dataclass(frozen=True)
class MongoCompound:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(dedup="content_id")
    formula: str


@dataclass(frozen=True)
class MongoCitation:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(dedup="content_id")
    doi: str


@dataclass(frozen=True)
class MongoCompoundCitation:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(
        dedup="none", links=(RelationshipLink("compound", "citation", role="citation", description="Cited by"),)
    )
    compound: MongoCompound
    citation: MongoCitation


@dataclass(frozen=True)
class MongoCompoundTag:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(
        dedup="none",
        links=(
            RelationshipLink("compound", "citation", role="primary"),
            RelationshipLink("compound", "citation", role="secondary"),
        ),
    )
    compound: MongoCompound
    citation: MongoCitation


@dataclass(frozen=True)
class MongoBooks:
    type = "books"


class MongoWriters:
    type = "writers"


class MongoPeople:
    type = "people"


class MongoProjects:
    type = "projects"


class MongoSimulations:
    type = "simulations"


class MongoCompounds:
    type = "compounds"


class MongoCitations:
    type = "citations"


def _register(name: str, family: type, records: tuple[type, ...]) -> None:
    register_entry_family(name=f"mongo-entry-provider-{name}", family=f"{__name__}:{family.__name__}")
    for record in records:
        register_entry_record(
            name=f"mongo-entry-provider-{name}-{record.__name__.lower()}",
            family=f"mongo-entry-provider-{name}",
            record=f"{__name__}:{record.__name__}",
        )


_register("books", MongoBooks, (MongoBook,))
_register("writers", MongoWriters, (MongoWriter,))
_register("people", MongoPeople, (MongoPerson,))
_register("projects", MongoProjects, (MongoProject,))
_register("simulations", MongoSimulations, (MongoSimulation,))
_register("compounds", MongoCompounds, (MongoCompound,))
_register("citations", MongoCitations, (MongoCitation,))

ADA = MongoWriter("Ada", 1815)
BOOLE = MongoWriter("Boole", 1815)
CARA = MongoWriter("Cara", 1820)
BOOK_1 = MongoBook(
    "Analytical Engines",
    350,
    Fraction(1, 3),
    True,
    b"\x00\xff",
    datetime.datetime(2026, 7, 24, 12, 30, tzinfo=datetime.UTC),
    FracVector([[1, Fraction(1, 2)], [0, 1]]),
    FracVector([[0, 0], [Fraction(1, 2), Fraction(1, 2)]]),
    ["computing", "history"],
    [BOOLE, CARA],
    ADA,
)
BOOK_2 = MongoBook(
    "Silence",
    120,
    Fraction(-7, 5),
    False,
    b"",
    datetime.datetime(2020, 1, 1, tzinfo=datetime.UTC),
    FracVector([[1, 0], [0, 1]]),
    FracVector([]),
    [],
    [],
    None,
)


@pytest.fixture()
def store(mongo_test_database):
    value = MongoStore(
        mongo_test_database,
        entry_records={
            MongoBooks: MongoBook,
            MongoWriters: MongoWriter,
            MongoPeople: MongoPerson,
            MongoProjects: MongoProject,
            MongoSimulations: MongoSimulation,
            MongoCompounds: MongoCompound,
            MongoCitations: MongoCitation,
        },
    )
    for obj in (ADA, BOOLE, CARA, BOOK_1, BOOK_2):
        value.save(obj)
    return value


@pytest.fixture()
def provider(store):
    return StoreEntryProvider(store, {"books": MongoBook, "writers": MongoWriter})


def _rows(provider, entry_type):
    return list(provider.records(entry_type))


def test_entry_types_and_auto_definition(provider):
    assert sorted(provider.entry_types()) == ["books", "writers"]
    properties = provider.entry_types()["books"].properties
    assert sorted(properties) == [
        "_httk_custom_in_print",
        "_httk_custom_keywords",
        "_httk_custom_metric",
        "_httk_custom_nkeywords",
        "_httk_custom_pages",
        "_httk_custom_price",
        "_httk_custom_published",
        "_httk_custom_samples",
        "_httk_custom_title",
        "id",
        "type",
    ]
    assert properties["_httk_custom_price"].optimade_type == "float"
    assert properties["_httk_custom_metric"].as_optimade()["x-optimade-dimensions"]["sizes"] == [2, 2]
    assert properties["_httk_custom_title"].definition_id.startswith("https://schemas.httk.org/ad-hoc/defs/properties/")
    assert "_httk_custom_cover" not in properties
    assert "_httk_custom_author" not in properties
    assert "_httk_custom_coauthors" not in properties
    assert properties["id"].nullable is False


def test_unregistered_prefix_and_definition_for_unserved_type(store):
    with pytest.raises(ValueError, match="_nope_"):
        StoreEntryProvider(store, {"books": MongoBook}, prefix="_nope_")
    generated = dict(StoreEntryProvider(store, {"writers": MongoWriter}).entry_types())
    with pytest.raises(ValueError, match="writers"):
        StoreEntryProvider(store, {"books": MongoBook}, definitions=generated)


def test_unconfigured_backing_is_rejected(mongo_test_database):
    store = MongoStore(mongo_test_database, entry_records={})
    with pytest.raises(ValueError, match="MongoWriter.*not configured"):
        StoreEntryProvider(store, {"writers": MongoWriter})


def test_property_keys_unknown_and_records_are_jsonable(provider):
    assert provider.property_keys("writers")["id"] == "id"
    with pytest.raises(KeyError):
        provider.property_keys("nope")
    with pytest.raises(KeyError):
        list(provider.records("nope"))
    records = provider.records("books")
    assert iter(records) is records
    rows = _rows(provider, "books")
    assert len(rows) == 2
    json.dumps(rows)
    row = next(item for item in rows if item["_httk_custom_title"] == "Analytical Engines")
    assert row["_httk_custom_pages"] == 350
    assert row["_httk_custom_price"] == pytest.approx(float(Fraction(1, 3)))
    assert row["_httk_custom_published"] == "2026-07-24T12:30:00+00:00"
    assert row["_httk_custom_metric"] == [[1.0, 0.5], [0.0, 1.0]]
    assert row["_httk_custom_samples"] == [[0.0, 0.0], [0.5, 0.5]]
    assert row["_httk_custom_nkeywords"] == 2
    empty = next(item for item in rows if item["_httk_custom_title"] == "Silence")
    assert empty["_httk_custom_samples"] == [] and empty["_httk_custom_keywords"] == []
    writers = _rows(provider, "writers")
    assert {row["_httk_custom_name"] for row in writers} == {"Ada", "Boole", "Cara"}


def test_relationships_and_custom_ids(provider, store):
    related = provider.relationships("books")
    book_id = next(row["id"] for row in _rows(provider, "books") if row["_httk_custom_title"] == "Analytical Engines")
    assert related[book_id] == (
        RelatedEntry(
            "writers", next(row["id"] for row in _rows(provider, "writers") if row["_httk_custom_name"] == "Ada")
        ),
        RelatedEntry(
            "writers", next(row["id"] for row in _rows(provider, "writers") if row["_httk_custom_name"] == "Boole")
        ),
        RelatedEntry(
            "writers", next(row["id"] for row in _rows(provider, "writers") if row["_httk_custom_name"] == "Cara")
        ),
    )

    custom = StoreEntryProvider(
        store,
        {"books": MongoBook, "writers": MongoWriter},
        id_of=lambda kind, _sid, obj: f"{kind}/{obj.title if kind == 'books' else obj.name}",
    )
    assert {row["id"] for row in custom.records("books")} == {"books/Analytical Engines", "books/Silence"}
    assert custom.relationships("books")["books/Analytical Engines"][0] == RelatedEntry("writers", "writers/Ada")


def test_relationship_metadata_and_suppression(mongo_test_database):
    store = MongoStore(mongo_test_database, entry_records={MongoPeople: MongoPerson, MongoProjects: MongoProject})
    people = (MongoPerson("Ada"), MongoPerson("Boole"), MongoPerson("Cara"))
    project = MongoProject("Engine", people[0], people[1], [people[1], people[2]])
    for obj in (*people, project):
        store.save(obj)
    provider = StoreEntryProvider(store, {"people": MongoPerson, "projects": MongoProject})
    assert "_httk_backup" not in provider.entry_types()["projects"].properties
    source = next(iter(provider.relationships("projects")))
    assert provider.relationships("projects")[source] == (
        RelatedEntry(
            "people",
            next(row["id"] for row in provider.records("people") if row["_httk_custom_name"] == "Ada"),
            description="Project lead",
            role="lead",
        ),
        RelatedEntry(
            "people",
            next(row["id"] for row in provider.records("people") if row["_httk_custom_name"] == "Boole"),
            role="member",
        ),
        RelatedEntry(
            "people",
            next(row["id"] for row in provider.records("people") if row["_httk_custom_name"] == "Cara"),
            role="member",
        ),
    )


def test_absent_optional_storable_child_is_an_empty_relationship_set(mongo_test_database):
    store = MongoStore(mongo_test_database, entry_records={MongoPeople: MongoPerson, MongoProjects: MongoProject})
    store.save(MongoProject("No members"))
    provider = StoreEntryProvider(store, {"people": MongoPerson, "projects": MongoProject})
    assert provider.relationships("projects") == {}


def test_definitions_validation_and_record_validation(provider, store):
    generated = dict(provider.entry_types())
    assert (
        StoreEntryProvider(store, {"books": MongoBook, "writers": MongoWriter}, definitions=generated).entry_types()[
            "books"
        ]
        is generated["books"]
    )
    incomplete = EntryTypeDefinition("writers", "Writers.", {})
    with pytest.raises(ValueError):
        StoreEntryProvider(store, {"writers": MongoWriter}, definitions={"writers": incomplete})
    for kind, definition in provider.entry_types().items():
        keys = provider.property_keys(kind)
        for row in provider.records(kind):
            validate_record(definition, {name: row[key] for name, key in keys.items()})


def test_link_relationships_and_order(mongo_test_database):
    store = MongoStore(
        mongo_test_database, entry_records={MongoCompounds: MongoCompound, MongoCitations: MongoCitation}
    )
    c1, c2 = MongoCompound("NaCl"), MongoCompound("SiO2")
    z1, z2 = MongoCitation("10.1/a"), MongoCitation("10.1/b")
    for obj in (
        c1,
        c2,
        z1,
        z2,
        MongoCompoundCitation(c1, z1),
        MongoCompoundCitation(c1, z2),
        MongoCompoundCitation(c2, z1),
    ):
        store.save(obj)
    provider = StoreEntryProvider(
        store, {"compounds": MongoCompound, "citations": MongoCitation}, link_classes=[MongoCompoundCitation]
    )
    citations = {row["_httk_custom_doi"]: row["id"] for row in provider.records("citations")}
    compounds = {row["_httk_custom_formula"]: row["id"] for row in provider.records("compounds")}
    assert provider.relationships("compounds") == {
        compounds["NaCl"]: (
            RelatedEntry("citations", citations["10.1/a"], description="Cited by", role="citation"),
            RelatedEntry("citations", citations["10.1/b"], description="Cited by", role="citation"),
        ),
        compounds["SiO2"]: (RelatedEntry("citations", citations["10.1/a"], description="Cited by", role="citation"),),
    }


def test_link_none_endpoint_is_inverse_and_forward_reference_is_direct(mongo_test_database):
    store = MongoStore(
        mongo_test_database, entry_records={MongoCompounds: MongoCompound, MongoSimulations: MongoSimulation}
    )
    compound = MongoCompound("NaCl")
    store.save(compound)
    run = MongoSimulation("run-1", compound)
    store.save(run)
    provider = StoreEntryProvider(store, {"compounds": MongoCompound, "simulations": MongoSimulation})
    compound_id = next(row["id"] for row in provider.records("compounds"))
    simulation_id = next(row["id"] for row in provider.records("simulations"))
    assert provider.relationships("compounds") == {
        compound_id: (RelatedEntry("simulations", simulation_id, role="output"),)
    }
    assert provider.relationships("simulations") == {simulation_id: (RelatedEntry("compounds", compound_id),)}


def test_link_deduplication_custom_ids_and_link_class_validation(mongo_test_database):
    store = MongoStore(
        mongo_test_database, entry_records={MongoCompounds: MongoCompound, MongoCitations: MongoCitation}
    )
    compound, citation = MongoCompound("NaCl"), MongoCitation("10.1/a")
    for obj in (
        compound,
        citation,
        MongoCompoundTag(compound, citation),
        MongoCompoundTag(compound, citation),
        MongoCompoundCitation(compound, citation),
    ):
        store.save(obj)
    provider = StoreEntryProvider(
        store, {"compounds": MongoCompound, "citations": MongoCitation}, link_classes=[MongoCompoundTag]
    )
    assert provider.relationships("compounds") == {
        next(row["id"] for row in provider.records("compounds")): (
            RelatedEntry("citations", next(row["id"] for row in provider.records("citations")), role="primary"),
            RelatedEntry("citations", next(row["id"] for row in provider.records("citations")), role="secondary"),
        )
    }
    custom = StoreEntryProvider(
        store,
        {"compounds": MongoCompound, "citations": MongoCitation},
        link_classes=[MongoCompoundCitation],
        id_of=lambda kind, _sid, obj: f"{kind}/{obj.formula if kind == 'compounds' else obj.doi}",
    )
    assert custom.relationships("compounds") == {
        "compounds/NaCl": (RelatedEntry("citations", "citations/10.1/a", description="Cited by", role="citation"),)
    }
    with pytest.raises(ValueError, match="MongoPerson.*no relationship links"):
        StoreEntryProvider(store, {"compounds": MongoCompound}, link_classes=[MongoPerson])


def test_link_validation_and_no_relationships_for_unserved_target(mongo_test_database):
    store = MongoStore(mongo_test_database, entry_records={MongoCompounds: MongoCompound})
    with pytest.raises(ValueError, match="MongoCitation.*not served"):
        StoreEntryProvider(store, {"compounds": MongoCompound}, link_classes=[MongoCompoundCitation])
    provider = StoreEntryProvider(store, {"compounds": MongoCompound})
    assert provider.relationships("compounds") == {}


def test_configured_family_uses_the_mongo_stored_property_plan(mongo_test_database):
    from test_db_stored_properties import (
        FIRST,
        SECOND,
        CalculationEntry,
        GenericCalculationFirst,
        GenericCalculationSecond,
    )

    store = MongoStore(
        mongo_test_database, entry_records={CalculationEntry: (GenericCalculationFirst, GenericCalculationSecond)}
    )
    store.save(FIRST)
    store.save(SECOND)
    provider = StoreEntryProvider(store, {"calculations": CalculationEntry})
    rows = list(provider.records("calculations"))
    assert {row["immutable_id"] for row in rows} == {"first", "second"}
    assert {row["type"] for row in rows} == {"calculations"}
    assert all(len(row["id"]) == 64 for row in rows)
    assert {row["last_modified"] for row in rows} == {None, "2026-08-02T08:30:00+00:00"}
    with pytest.raises(ValueError, match="immutable_id"):
        StoreEntryProvider(
            store,
            {"calculations": CalculationEntry},
            definitions={"calculations": EntryTypeDefinition("calculations", "Empty.", {})},
        )


def test_optimade_adapter_consumes_mongo_provider_records(provider):
    pytest.importorskip("httk.serve.optimade")
    from httk.serve.optimade import adapter_from_providers
    from httk.serve.optimade.backend import execute_query
    from httk.serve.optimade.filter import parse_optimade_filter

    adapter = adapter_from_providers([provider])
    results = list(
        execute_query(
            adapter,
            ["books"],
            ["id", "_httk_custom_title"],
            [],
            100,
            0,
            parse_optimade_filter("_httk_custom_pages > 200"),
        )
    )
    assert len(results) == 1
    assert results[0].values["_httk_custom_title"] == "Analytical Engines"
