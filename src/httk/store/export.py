"""Export a snapshot-consistent SQLite store with its definitions."""

import hashlib
import json
import os
import sqlite3
import sys
import tempfile
import zipfile
from collections.abc import Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, Any

from httk.core import load_entry_type_definition
from httk.core.project import PROJECT_DIRECTORY, discover_project
from httk.core.register import (
    known_property_definitions,
    load_property_definition,
)

if TYPE_CHECKING:
    from httk.core import CLIContext

__all__ = ["export_dataset"]


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def _definition_name(kind: str, definition_id: str) -> str:
    return f"definitions/{kind}-{_sha256(definition_id.encode('utf-8'))[:16]}.json"


def _add_definition_candidate(
    candidates: dict[tuple[str, str], dict[bytes, str]],
    kind: str,
    definition_id: str,
    document: dict[str, Any],
    source: str,
) -> None:
    canonical = _json_bytes(document)
    variants = candidates.setdefault((kind, definition_id), {})
    variants[canonical] = source
    if len(variants) > 1:
        raise ValueError(f"conflicting {kind} definitions for {definition_id!r}: {', '.join(variants.values())}")


def _definitions(
    store: Any,
) -> tuple[dict[str, bytes], list[dict[str, str]], list[dict[str, object]]]:
    """Collect declared entry-type and property documents for *store*."""

    entry_ids: set[str] = set()
    authoritative_definitions: dict[str, Any] = {}
    declaration: list[dict[str, object]] = []
    for family in store.entry_layout:
        family_id = family.definition_id
        record_ids = []
        for record_id in family.record_definition_ids:
            if record_id is not None:
                entry_ids.add(record_id)
                record_ids.append(record_id)
        if family_id is not None:
            entry_ids.add(family_id)
            factory = getattr(family.family, "entry_type_definition", None)
            if callable(factory):
                authoritative_definitions[family_id] = factory()
        elif not record_ids:
            raise ValueError(f"store family {family.name!r} has no entry-type definition")
        declaration.append(
            {
                "family": family.name,
                "records": list(family.record_names),
                "definition_ids": sorted({item for item in (family_id, *record_ids) if item is not None}),
            }
        )

    candidates: dict[tuple[str, str], dict[bytes, str]] = {}
    property_ids: set[str] = set()
    for definition_id in sorted(entry_ids):
        definition = authoritative_definitions.get(definition_id)
        source = f"family definition {definition_id}"
        if definition is None:
            definition = load_entry_type_definition(definition_id)
            source = f"entry registry {definition_id}"
        _add_definition_candidate(candidates, "entry-type", definition_id, definition.as_optimade(), source)
        for prop in definition.properties.values():
            property_ids.add(prop.definition_id)
            _add_definition_candidate(
                candidates,
                "property",
                prop.definition_id,
                prop.as_optimade(),
                f"embedded in {definition_id}",
            )

    registered_properties = set(known_property_definitions())
    for definition_id in sorted(property_ids):
        if definition_id in registered_properties:
            property_definition = load_property_definition(definition_id)
            _add_definition_candidate(
                candidates,
                "property",
                definition_id,
                property_definition.as_optimade(),
                f"property registry {definition_id}",
            )

    documents: dict[str, bytes] = {}
    manifest_definitions: list[dict[str, str]] = []
    for (kind, definition_id), variants in sorted(candidates.items()):
        document = next(iter(variants))
        path = _definition_name(kind, definition_id)
        documents[path] = document
        manifest_definitions.append(
            {
                "id": definition_id,
                "kind": "entry_type" if kind == "entry-type" else kind,
                "path": path,
                "sha256": _sha256(document),
            }
        )
    return documents, manifest_definitions, declaration


def _reject_output_collision(source: Path, destination: Path) -> None:
    if source == destination:
        raise ValueError("dataset output cannot overwrite the source database")
    if destination.exists():
        try:
            if os.path.samefile(source, destination):
                raise ValueError("dataset output cannot overwrite the source database")
        except OSError:
            pass
    try:
        project = discover_project(destination.parent)
    except ValueError:
        project = None
    if project is None:
        return
    try:
        relative = destination.relative_to(project)
    except ValueError:
        return
    if relative.parts and relative.parts[0].casefold() == PROJECT_DIRECTORY.casefold():
        raise ValueError(f"dataset output cannot overwrite protected project data: {destination}")


def _is_duckdb(path: Path) -> bool:
    if path.suffix.casefold() in {".duckdb", ".ddb"}:
        return True
    try:
        with path.open("rb") as stream:
            return stream.read(12)[8:12] == b"DUCK"
    except OSError:
        return False


def _sqlite_snapshot(source: Path, target: Path) -> None:
    source_connection: sqlite3.Connection | None = None
    target_connection: sqlite3.Connection | None = None
    try:
        source_connection = sqlite3.connect(f"{source.as_uri()}?mode=ro", uri=True)
        target_connection = sqlite3.connect(target)
        source_connection.backup(target_connection)
    except sqlite3.Error as error:
        raise ValueError(f"cannot create a consistent SQLite snapshot of {source}: {error}") from error
    finally:
        if source_connection is not None:
            source_connection.close()
        if target_connection is not None:
            target_connection.close()


def _write_zip_atomic(path: Path, entries: dict[str, bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        os.close(descriptor)
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for name in sorted(entries):
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.create_system = 3
                info.external_attr = 0o644 << 16
                archive.writestr(info, entries[name])
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def export_dataset(store_path: str | Path, out_path: str | Path) -> Path:
    """Export a snapshot-consistent SQLite store and its OPTIMADE definitions.

    :param store_path: SQLite store to copy.
    :param out_path: Destination zip file.
    :return: The destination path.
    :raises ValueError: If the store declaration has no resolvable definitions.
    """

    source = Path(store_path).expanduser().resolve()
    destination = Path(out_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    _reject_output_collision(source, destination)
    if _is_duckdb(source):
        raise ValueError(
            "DuckDB dataset export is refused: a safe single-file snapshot is not implemented; "
            "export from a SQLite store instead"
        )
    from httk.store.backend.sql import Backend, SqlStore

    with tempfile.TemporaryDirectory() as staging:
        snapshot = Path(staging) / source.name
        _sqlite_snapshot(source, snapshot)
        with Backend.sqlite(snapshot) as database:
            store = SqlStore(database)
            documents, definitions, declaration = _definitions(store)
        store_bytes = snapshot.read_bytes()
        store_name = source.name
        manifest = {
            "format": "httk-dataset",
            "format_version": 2,
            "store": {
                "path": f"store/{store_name}",
                "filename": store_name,
                "sha256": _sha256(store_bytes),
                "snapshot_consistent": True,
                "snapshot_method": "sqlite-backup",
            },
            "definitions": sorted(definitions, key=lambda item: (item["kind"], item["id"])),
            "entry_record_declaration": declaration,
            # The source mtime is stable for an identical input, making the complete
            # export reproducible while retaining a nanosecond snapshot timestamp.
            "created_at_ns": source.stat().st_mtime_ns,
            "packages": {"httk-core": _package_version("httk-core"), "httk-store": _package_version("httk-store")},
        }
        entries = {f"store/{store_name}": store_bytes, **documents, "manifest.json": _json_bytes(manifest)}
    _write_zip_atomic(destination, entries)
    return destination


def command(argv: Sequence[str], context: "CLIContext") -> int:
    """Handle ``httk store export``.

    :param argv: Arguments following ``store``.
    :param context: Root CLI invocation context.
    :return: Process-style exit status.
    """

    import argparse

    parser = argparse.ArgumentParser(prog=f"{context.program} store", description="export and inspect httk stores")
    subparsers = parser.add_subparsers(dest="command")
    export = subparsers.add_parser("export", help="export a definitions-bundled dataset")
    export.add_argument("store_file", metavar="STORE-FILE")
    export.add_argument("out_zip", metavar="OUT.ZIP")
    try:
        arguments = parser.parse_args(argv)
    except SystemExit as error:
        return error.code if isinstance(error.code, int) else 1
    try:
        if arguments.command != "export":
            parser.print_help()
            return 0
        output = export_dataset(arguments.store_file, arguments.out_zip)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"{parser.prog}: {error}", file=sys.stderr)
        return 2
    print(f"exported dataset to {output}")
    return 0
