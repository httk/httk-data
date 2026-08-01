"""Domain-neutral stored-entry projection coverage for SQL providers and filters."""

import datetime
from dataclasses import dataclass
from fractions import Fraction
from typing import Annotated, ClassVar

import pytest
import sqlalchemy
from httk.core import FracVector, Shape, StorageInfo, StoredEntryProjection, StoredEntryValue

from httk.data.db import (
    Database,
    SqlStore,
    StoredSchemaRebuildRequiredError,
    StoreEntryProvider,
    optimade_filter_searcher,
)
from httk.data.optimade_query import FilterTranslationError

STRUCTURES_DEFINITION = "https://schemas.optimade.org/defs/v1.3/entrytypes/optimade/structures"


@dataclass(frozen=True)
class ProjectedSpecies(StoredEntryValue):
    name: str
    symbols: tuple[str, ...]
    concentrations: tuple[Fraction, ...]
    has_attached: bool = False
    attached_values: tuple[str, ...] = ()

    def to_stored_entry_value(self):
        value = {
            "name": self.name,
            "chemical_symbols": self.symbols,
            "concentration": self.concentrations,
        }
        if self.has_attached:
            value["attached"] = self.attached_values
            value["nattached"] = tuple(1 for _symbol in self.attached_values)
        return value


@dataclass(frozen=True)
class ProjectedAssemblies(StoredEntryValue):
    group_sites: tuple[int, ...]

    def to_stored_entry_value(self):
        return {"groups": [] if not self.group_sites else [{"sites_in_groups": [self.group_sites]}]}


@dataclass(frozen=True)
class ProjectedStructure:
    identifier: str
    modified: datetime.datetime | None
    formula: str
    element_values: tuple[str, ...]
    site_count: int
    feature_values: tuple[str, ...]
    coordinate_span: str
    symmetry_number: int | None
    optimization: str | None
    species_values: tuple[ProjectedSpecies, ...]
    assembly_values: ProjectedAssemblies | None
    positions: Annotated[FracVector, Shape(0, 3)]

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(
        storage_name="generic_structures_v3",
        dedup="none",
    )
    __httk_entry_projection__: ClassVar[StoredEntryProjection] = StoredEntryProjection(
        entry_type="structures",
        definition_id=STRUCTURES_DEFINITION,
        property_fields={
            "id": "identifier",
            "last_modified": "modified",
            "chemical_formula_reduced": "formula",
            "elements": "element_values",
            "nsites": "site_count",
            "structure_features": "feature_values",
            "site_coordinate_span": "coordinate_span",
            "space_group_it_number": "symmetry_number",
            "optimization_type": "optimization",
            "species": "species_values",
            "assemblies": "assembly_values",
            "fractional_site_positions": "positions",
        },
        filterable=frozenset(
            {
                "id",
                "last_modified",
                "chemical_formula_reduced",
                "elements",
                "nsites",
                "structure_features",
                "site_coordinate_span",
                "space_group_it_number",
                "optimization_type",
            }
        ),
        obsolete_storage_names=("generic_structures_v2",),
    )


STRUCTURE_A = ProjectedStructure(
    identifier="source/A",
    modified=datetime.datetime(2026, 1, 2, 5, 4, 5, tzinfo=datetime.timezone(datetime.timedelta(hours=2))),
    formula="Ge5Si3",
    element_values=("Ge", "Si"),
    site_count=8,
    feature_values=("disorder",),
    coordinate_span="unit_cell",
    symmetry_number=227,
    optimization="theoretical",
    species_values=(ProjectedSpecies("mixed", ("Ge", "Si"), (Fraction(5, 8), Fraction(3, 8))),),
    assembly_values=ProjectedAssemblies(()),
    positions=FracVector.create([[0, 0, 0], [Fraction(1, 2), Fraction(1, 2), Fraction(1, 2)]]),
)
STRUCTURE_B = ProjectedStructure(
    identifier="source/B",
    modified=None,
    formula="Si",
    element_values=("Si",),
    site_count=1,
    feature_values=(),
    coordinate_span="unit_cell",
    symmetry_number=None,
    optimization=None,
    species_values=(ProjectedSpecies("Si", ("Si",), (Fraction(1),), True, ()),),
    assembly_values=None,
    positions=FracVector.create([[0, 0, 0]]),
)


@pytest.fixture(params=("sqlite", "duckdb"))
def projected_store(request):
    if request.param == "duckdb":
        pytest.importorskip("duckdb_engine")
        manager = Database.duckdb()
    else:
        manager = Database.sqlite()
    with manager as database:
        store = SqlStore(database)
        store.save(STRUCTURE_A)
        store.save(STRUCTURE_B)
        yield store


def _results(searcher):
    return [item[0][0] for item in searcher]


def test_projected_provider_uses_standard_definition_ids_and_recursive_values(projected_store):
    provider = StoreEntryProvider(projected_store, {"structures": ProjectedStructure})
    definition = provider.entry_types()["structures"]
    assert definition.definition_id == STRUCTURES_DEFINITION
    assert set(provider.property_keys("structures")) == {
        "id",
        "type",
        *ProjectedStructure.__httk_entry_projection__.property_fields.keys(),
    }

    rows = {row["__id"]: row for row in provider.records("structures")}
    assert set(rows) == {"source/A", "source/B"}
    assert rows["source/A"]["last_modified"] == "2026-01-02T03:04:05+00:00"
    assert rows["source/A"]["species"] == [
        {"name": "mixed", "chemical_symbols": ["Ge", "Si"], "concentration": [0.625, 0.375]}
    ]
    assert rows["source/A"]["assemblies"] == {"groups": []}
    assert rows["source/B"]["assemblies"] is None
    assert rows["source/B"]["species"][0]["attached"] == []
    assert rows["source/B"]["fractional_site_positions"] == [[0.0, 0.0, 0.0]]


@pytest.mark.parametrize(
    ("filter_string", "expected"),
    (
        ('id = "source/A"', [STRUCTURE_A]),
        ('last_modified > "2025-06-01T00:00:00Z"', [STRUCTURE_A]),
        ('last_modified = "2026-01-02T03:04:05Z"', [STRUCTURE_A]),
        ('chemical_formula_reduced = "Ge5Si3"', [STRUCTURE_A]),
        ('elements HAS "Ge"', [STRUCTURE_A]),
        ("nsites >= 8", [STRUCTURE_A]),
        ('structure_features HAS "disorder"', [STRUCTURE_A]),
        ('site_coordinate_span = "unit_cell"', [STRUCTURE_A, STRUCTURE_B]),
        ("space_group_it_number = 227", [STRUCTURE_A]),
        ('optimization_type = "theoretical"', [STRUCTURE_A]),
        ('type = "structures"', [STRUCTURE_A, STRUCTURE_B]),
    ),
)
def test_projection_declared_standard_fields_translate_to_sql(projected_store, filter_string, expected):
    assert _results(optimade_filter_searcher(projected_store, ProjectedStructure, filter_string)) == expected


def test_projection_nonfilterable_mapped_values_are_recognized_but_not_translated(projected_store):
    for filter_string in ("species IS KNOWN", "fractional_site_positions IS KNOWN"):
        with pytest.raises(FilterTranslationError) as excinfo:
            optimade_filter_searcher(projected_store, ProjectedStructure, filter_string)
        assert excinfo.value.category == "not-implemented"


@pytest.mark.parametrize(
    ("filter_string", "expected"),
    (
        ("last_modified IS UNKNOWN", [STRUCTURE_B]),
        ("last_modified IS KNOWN", [STRUCTURE_A]),
        ("space_group_it_number IS UNKNOWN", [STRUCTURE_B]),
        ("space_group_it_number IS KNOWN", [STRUCTURE_A]),
        ("optimization_type IS UNKNOWN", [STRUCTURE_B]),
        ("optimization_type IS KNOWN", [STRUCTURE_A]),
    ),
)
def test_projection_nullable_fields_query_the_stored_column(projected_store, filter_string, expected):
    assert _results(optimade_filter_searcher(projected_store, ProjectedStructure, filter_string)) == expected


@pytest.mark.parametrize(
    "value",
    (
        "2026-01-01 00:00:00+00:00",
        "2026-01-01X00:00:00+00:00",
        "2026-01-01T00:00:00+00:00:30",
    ),
)
def test_projection_timestamp_filters_require_rfc3339(projected_store, value):
    with pytest.raises(FilterTranslationError) as excinfo:
        optimade_filter_searcher(projected_store, ProjectedStructure, f'last_modified = "{value}"')
    assert excinfo.value.category == "type-mismatch"


def test_projection_entry_type_mapping_is_authoritative(projected_store):
    with pytest.raises(ValueError, match="not provider mapping key"):
        StoreEntryProvider(projected_store, {"materials": ProjectedStructure})


def test_obsolete_projected_root_table_requires_rebuild_for_provider_and_filter():
    database = Database.sqlite()
    try:
        with database.engine.begin() as connection:
            connection.execute(sqlalchemy.text("CREATE TABLE generic_structures_v2 (sid INTEGER PRIMARY KEY)"))
        store = SqlStore(database)
        with pytest.raises(StoredSchemaRebuildRequiredError, match="rebuild/reimport"):
            StoreEntryProvider(store, {"structures": ProjectedStructure})
        with pytest.raises(StoredSchemaRebuildRequiredError, match="rebuild/reimport"):
            optimade_filter_searcher(store, ProjectedStructure, 'id = "source/A"')
    finally:
        database.dispose()


def test_obsolete_table_is_harmless_when_current_root_exists(projected_store):
    with projected_store._database.engine.begin() as connection:
        connection.execute(sqlalchemy.text("CREATE TABLE generic_structures_v2 (sid INTEGER PRIMARY KEY)"))
    provider = StoreEntryProvider(projected_store, {"structures": ProjectedStructure})
    assert {row["__id"] for row in provider.records("structures")} == {"source/A", "source/B"}
