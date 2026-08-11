"""Tests for definitions-bundled dataset exports."""

import hashlib
import json
import zipfile
from pathlib import Path

import pytest
from httk.core.files import FileEntry, FileRecord
from httk.core.project import initialize_project

from httk.store import export_dataset
from httk.store.db import Database, SqlStore


def _store(path: Path) -> None:
    with Database.sqlite(path) as database:
        store = SqlStore(database, entry_records={FileEntry: FileRecord})
        store.save(FileRecord("file:///presentation", "presentation", size=3, sha256="abc"))


def test_export_contains_store_manifest_and_every_definition(tmp_path: Path) -> None:
    source = tmp_path / "presentation.sqlite"
    output = tmp_path / "presentation.zip"
    _store(source)
    export_dataset(source, output)

    with zipfile.ZipFile(output) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        snapshot = archive.read("store/presentation.sqlite")
        assert hashlib.sha256(snapshot).hexdigest() == manifest["store"]["sha256"]
        assert manifest["store"]["snapshot_consistent"] is True
        assert manifest["store"]["snapshot_method"] == "sqlite-backup"
        names = set(archive.namelist())
        expected_ids = {
            "https://schemas.optimade.org/defs/v1.2/entrytypes/optimade/files",
            "https://schemas.optimade.org/defs/v1.2/properties/core/id",
            "https://schemas.optimade.org/defs/v1.2/properties/core/immutable_id",
            "https://schemas.optimade.org/defs/v1.2/properties/core/last_modified",
            "https://schemas.optimade.org/defs/v1.2/properties/core/type",
            "https://schemas.optimade.org/defs/v1.2/properties/optimade/files/atime",
            "https://schemas.optimade.org/defs/v1.2/properties/optimade/files/checksums",
            "https://schemas.optimade.org/defs/v1.2/properties/optimade/files/ctime",
            "https://schemas.optimade.org/defs/v1.2/properties/optimade/files/description",
            "https://schemas.optimade.org/defs/v1.2/properties/optimade/files/media_type",
            "https://schemas.optimade.org/defs/v1.2/properties/optimade/files/modification_timestamp",
            "https://schemas.optimade.org/defs/v1.2/properties/optimade/files/mtime",
            "https://schemas.optimade.org/defs/v1.2/properties/optimade/files/name",
            "https://schemas.optimade.org/defs/v1.2/properties/optimade/files/size",
            "https://schemas.optimade.org/defs/v1.2/properties/optimade/files/url",
            "https://schemas.optimade.org/defs/v1.2/properties/optimade/files/url_stable_until",
            "https://schemas.optimade.org/defs/v1.2/properties/optimade/files/version",
        }
        assert len(manifest["definitions"]) == 17
        assert {definition["id"] for definition in manifest["definitions"]} == expected_ids
        for definition in manifest["definitions"]:
            assert definition["path"] in names
            assert hashlib.sha256(archive.read(definition["path"])).hexdigest() == definition["sha256"]
        assert manifest["entry_record_declaration"][0]["records"] == ["core-file"]


def test_export_rejects_aliases_and_protected_project_files(tmp_path: Path) -> None:
    source = tmp_path / "presentation.sqlite"
    _store(source)
    with pytest.raises(ValueError, match="overwrite the source"):
        export_dataset(source, source)

    project = tmp_path / "project"
    initialize_project(project, name="export")
    project_store = project / "presentation.sqlite"
    _store(project_store)
    with pytest.raises(ValueError, match="protected project data"):
        export_dataset(project_store, project / "httk_project" / "project.json")


def test_export_refuses_duckdb_snapshot(tmp_path: Path) -> None:
    source = tmp_path / "store.duckdb"
    source.write_bytes(b"not a database")
    with pytest.raises(ValueError, match="DuckDB dataset export is refused"):
        export_dataset(source, tmp_path / "store.zip")


def test_export_is_reproducible(tmp_path: Path) -> None:
    source = tmp_path / "presentation.sqlite"
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"
    _store(source)
    export_dataset(source, first)
    export_dataset(source, second)
    assert first.read_bytes() == second.read_bytes()


def test_structure_export_contains_extended_family_definition(tmp_path: Path) -> None:
    atomistic = pytest.importorskip("httk.atomistic")
    source = tmp_path / "structures.sqlite"
    output = tmp_path / "structures.zip"
    from httk.atomistic import Cell, Sites, Species, StructureEntry, UnitcellStructure, UnitcellStructureRecord

    structure = UnitcellStructure(
        Cell([[4, 0, 0], [0, 4, 0], [0, 0, 4]]),
        Sites([[0, 0, 0]]),
        (Species("Na", ("Na",), (1,)),),
        ("Na",),
    )
    with Database.sqlite(source) as database:
        store = SqlStore(database, entry_records={StructureEntry: UnitcellStructureRecord})
        store.save(structure)

    export_dataset(source, output)
    with zipfile.ZipFile(output) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        definitions = {item["id"]: json.loads(archive.read(item["path"])) for item in manifest["definitions"]}
    entry = definitions[StructureEntry.definition_id]
    property_names = set(entry["properties"])
    assert len(property_names) == 40
    assert {"_httk_basis_precision", "_httk_charge", "_httk_site_moments"} <= property_names
    assert (
        atomistic.StructureEntry.entry_type_definition().as_optimade()["properties"].keys()
        == entry["properties"].keys()
    )
