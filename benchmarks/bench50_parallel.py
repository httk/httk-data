"""Benchmark ``SqlStore.bulk_ingest(workers=W)`` against the serial reference.

Builds the altermagnets material store (the ~9,000-material stream that the
serial ``bulk_ingest`` fast path already loads in
``altermagnets/tools/build_store.py``) into a fresh **file-backed DuckDB**
database for a range of worker counts, and reports the wall-clock split so the
parallel encode-plus-shard-merge path can be judged against the serial one.

Usage (from the httk-data checkout, with the sibling httk packages on
``PYTHONPATH``)::

    python benchmarks/bench50_parallel.py --workers 1 4 12 24 --repeats 1

The script adds the altermagnets ``src/functions`` directory to ``sys.path``
itself (as the packet specifies); point ``--altermagnets`` at a different
checkout if needed. Each build writes to a throwaway file inside a temporary
directory so shards share its filesystem, and the store is disposed between
runs.

Reported per run:

- ``total``    the whole ``with store.bulk_ingest(...)`` block;
- ``dispatch`` the caller's ``save()`` loop (serial: encode+buffer+flush;
  parallel: pickling objects onto the worker queue, overlapping worker encode);
- ``finalize`` the context exit (serial: index build; parallel: worker join +
  the set-wise shard merge + index build).
"""

import argparse
import dataclasses
import sys
import tempfile
import time
from pathlib import Path


def _add_altermagnets(path: str | None) -> None:
    root = Path(path) if path else Path(__file__).resolve().parents[2] / "altermagnets"
    functions = root / "src" / "functions"
    if not functions.is_dir():
        raise SystemExit(f"altermagnets functions directory not found: {functions}")
    if str(functions) not in sys.path:
        sys.path.insert(0, str(functions))


def _load_stream(replicate: int, mode: str) -> list[object]:
    """The layout record followed by ``replicate`` distinct copies of the base materials.

    Each replica perturbs the ``Unique`` ``id`` field so the copies are distinct
    materials. In ``shared`` mode they still share referenced substructure (cells,
    species, compositions) — a realistic offline build where the merge collapses
    many cross-worker duplicates. In ``distinct`` mode each replica also perturbs
    its structure's ``charge`` (part of the structure's content identity), giving
    distinct roots — each material and its structure distinct — while the atomic
    descendants (cells, sites, species, compositions) stay shared, so the merge
    collapses much less. Only ~180 materials' detail files are present in this
    checkout, hence replication.
    """
    from fractions import Fraction

    import material_store as ms

    base = ms._load_source_materials(ms.resolve_data_dir(), details_dir=ms.resolve_details_dir())
    stream: list[object] = [ms.StoreLayout(ms.STORE_LAYOUT_VERSION)]
    for copy in range(replicate):
        suffix = "" if copy == 0 else f"-rep{copy}"
        for material in base:
            if not suffix:
                stream.append(material)
                continue
            perturbed = dataclasses.replace(material, id=f"{material.id}{suffix}")
            if mode == "distinct" and perturbed.structure is not None:
                perturbed = dataclasses.replace(
                    perturbed, structure=dataclasses.replace(perturbed.structure, charge=Fraction(copy + 1, 1_000_003))
                )
            stream.append(perturbed)
    return stream


def _build(stream: list[object], workers: int, directory: Path, materials: int) -> dict[str, float]:
    import sqlalchemy
    from httk.data.db import Database, SqlStore
    from httk.data.db.mapping import CONTENT_ID_COLUMN

    target = directory / f"bench_w{workers}_{time.time_ns()}.duckdb"
    database = Database.duckdb(target)
    try:
        store = SqlStore(database, entry_records={})
        started = time.perf_counter()
        with store.bulk_ingest(workers=workers) as bulk:
            for obj in stream:
                bulk.save(obj)
            dispatched = time.perf_counter()
        finished = time.perf_counter()
        # Post-ingest verification: a silently lost task or a botched merge must
        # never flatter the timing. Every material must be present, and every
        # content-addressed table must hold each content id exactly once.
        with database.engine.connect() as connection:
            stored = connection.execute(
                sqlalchemy.text('SELECT count(*) FROM "altermagnets_material_records"')
            ).scalar_one()
            if int(stored) != materials:
                raise SystemExit(f"verification failed: stored {stored} materials, expected {materials}")
            for name, table in store._metadata.tables.items():
                if name.startswith("_httk_") or CONTENT_ID_COLUMN not in table.c:
                    continue
                duplicate = connection.execute(
                    sqlalchemy.select(table.c[CONTENT_ID_COLUMN])
                    .group_by(table.c[CONTENT_ID_COLUMN])
                    .having(sqlalchemy.func.count() > 1)
                    .limit(1)
                ).first()
                if duplicate is not None:
                    raise SystemExit(f"verification failed: duplicate content id in {name}")
        return {
            "total": finished - started,
            "dispatch": dispatched - started,
            "finalize": finished - dispatched,
        }
    finally:
        database.dispose()
        target.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 4, 12, 24])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--replicate", type=int, default=50, help="distinct copies of the base materials (50 ~= 9,000)")
    parser.add_argument(
        "--mode",
        choices=("shared", "distinct"),
        default="shared",
        help=(
            "shared: replicas share substructure (merge collapses many dups); "
            "distinct: distinct roots (material+structure) with shared atomic descendants"
        ),
    )
    parser.add_argument("--altermagnets", default=None, help="path to the altermagnets checkout")
    arguments = parser.parse_args()

    _add_altermagnets(arguments.altermagnets)
    load_started = time.perf_counter()
    stream = _load_stream(arguments.replicate, arguments.mode)
    load_elapsed = time.perf_counter() - load_started
    materials = len(stream) - 1
    print(f"loaded {materials} materials ({arguments.mode} mode) in {load_elapsed:.1f}s\n")

    with tempfile.TemporaryDirectory(prefix="bench50_") as directory:
        base = Path(directory)
        header = f"{'mode':>8} {'repeat':>6} {'total':>9} {'dispatch':>9} {'finalize':>9}"
        print(header)
        print("-" * len(header))
        best: dict[int, float] = {}
        for workers in arguments.workers:
            totals: list[float] = []
            for repeat in range(arguments.repeats):
                timing = _build(stream, workers, base, materials)
                totals.append(timing["total"])
                mode = "serial" if workers == 1 else f"w={workers}"
                print(
                    f"{mode:>8} {repeat:>6} "
                    f"{timing['total']:>8.1f}s {timing['dispatch']:>8.1f}s {timing['finalize']:>8.1f}s"
                )
            best[workers] = min(totals)
        print()
        serial = best.get(1)
        for workers, total in sorted(best.items()):
            speedup = f"{serial / total:.2f}x" if serial else "n/a"
            print(f"best workers={workers:>3}: {total:>7.1f}s   speedup vs serial: {speedup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
