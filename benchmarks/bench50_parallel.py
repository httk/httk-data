"""Benchmark ``SqlStore.bulk_ingest(workers=W)`` across supported bulk backends.

The normal DuckDB cells build into a throwaway file.  ClickHouse cells are
always deferred-finalize cells and create an isolated server database for every
worker/repeat cell.  Their Arrow stage inserts are instrumented at the harness
boundary, so the output includes each streamed shard's rows, payload bytes,
and throughput as well as the existing finalizer-step timings.

``--rss-scale N`` is the D12 cell.  It runs the same corpus in fresh child
interpreters at ``N`` and ``2N`` replicas with ``track_sids=False`` and prints
the client process peak RSS from each independent interpreter.
"""

import argparse
import dataclasses
import json
import os
import resource
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, nullcontext
from pathlib import Path

# Memory guard: DuckDB defaults its memory_limit to ~80% of system RAM per
# instance, and the parallel modes fork many processes; cap the store and all
# worker-inherited instances unless the caller sets an explicit value.
os.environ.setdefault("HTTK_DUCKDB_MEMORY_LIMIT", "4GB")


def _add_altermagnets(path: str | None) -> None:
    root = Path(path) if path else Path(__file__).resolve().parents[2] / "altermagnets"
    functions = root / "src" / "functions"
    if not functions.is_dir():
        raise SystemExit(f"altermagnets functions directory not found: {functions}")
    if str(functions) not in sys.path:
        sys.path.insert(0, str(functions))


def _load_stream(replicate: int, mode: str) -> list[object]:
    """Return the layout record followed by ``replicate`` copies of the base corpus."""
    from fractions import Fraction

    import material_store as ms

    base = ms._load_source_materials(ms.resolve_data_dir(), details_dir=ms.resolve_details_dir())
    stream: list[object] = [ms.StoreLayout(ms.STORE_LAYOUT_VERSION)]
    for copy in range(replicate):
        suffix = "" if copy == 0 else f"-rep{copy}"
        for material_index, material in enumerate(base):
            salt = copy * len(base) + material_index + 1
            if not suffix and mode != "all-distinct":
                stream.append(material)
                continue
            perturbed = dataclasses.replace(material, id=f"{material.id}{suffix}")
            if mode in {"distinct", "all-distinct"} and perturbed.structure is not None:
                perturbed = dataclasses.replace(
                    perturbed, structure=dataclasses.replace(perturbed.structure, charge=Fraction(copy + 1, 1_000_003))
                )
            if mode == "all-distinct" and perturbed.structure is not None:
                precision = Fraction(salt, 10_000_019)
                structure = perturbed.structure
                cell = dataclasses.replace(structure.cell, precision=precision)
                sites = dataclasses.replace(structure.sites, precision=precision)
                species = tuple(
                    dataclasses.replace(value, original_name=f"{value.original_name or value.name}-bench-{salt}")
                    for value in structure.species
                )
                perturbed = dataclasses.replace(
                    perturbed,
                    structure=dataclasses.replace(structure, cell=cell, sites=sites, species=species),
                )
            stream.append(perturbed)
    return stream


def _iter_storable_records(value: object) -> Iterator[object]:
    """Yield every storable record reachable from a benchmark input object."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        if hasattr(type(value), "__httk_storage__"):
            yield value
        for field in dataclasses.fields(value):
            yield from _iter_storable_records(getattr(value, field.name))
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield from _iter_storable_records(key)
            yield from _iter_storable_records(item)
        return
    if isinstance(value, (tuple, list, set, frozenset)):
        for item in value:
            yield from _iter_storable_records(item)


def _expected_table_counts(stream: list[object]) -> dict[str, int]:
    """Return expected unique content-ID rows by physical table."""
    from httk.core.storage import content_id

    from httk.store.backend.sql.schema import resolve_schema

    expected: dict[str, set[str]] = {}
    for value in stream:
        for record in _iter_storable_records(value):
            table_name = resolve_schema(type(record)).table_name
            expected.setdefault(table_name, set()).add(content_id(record))
    return {name: len(keys) for name, keys in expected.items()}


@contextmanager
def _clickhouse_database(uri: str) -> Iterator[object]:
    """Yield a fresh database with the deployment bootstrap check used by fixtures."""
    import sqlalchemy

    from httk.store.backend.sql import Backend

    source = sqlalchemy.engine.make_url(uri)
    if source.drivername.split("+")[0] != "clickhousedb":
        raise SystemExit("ClickHouse benchmark URI must use the clickhousedb:// dialect")
    name = f"httk_bench_{uuid.uuid4().hex}"
    admin = sqlalchemy.create_engine(source.set(database="default"))
    database = None
    created = False
    try:
        with admin.begin() as connection:
            present = connection.execute(
                sqlalchemy.text(
                    "SELECT count() FROM system.tables WHERE database = 'default' AND name = '_httk_bootstrap'"
                )
            ).scalar_one()
            if not present:
                raise SystemExit(
                    "ClickHouse benchmark requires deployment bootstrap table default._httk_bootstrap "
                    "(ENGINE=KeeperMap('/_httk_bootstrap') PRIMARY KEY key)"
                )
            connection.execute(sqlalchemy.text(f"CREATE DATABASE {name}"))
            created = True
        bootstrap = sqlalchemy.create_engine(source.set(database=name))
        try:
            with bootstrap.begin() as connection:
                connection.execute(
                    sqlalchemy.text(
                        "CREATE TABLE _httk_bootstrap (key String, value String) "
                        "ENGINE=KeeperMap('/_httk_bootstrap') PRIMARY KEY key"
                    )
                )
        finally:
            bootstrap.dispose()
        database = Backend.clickhouse(source, database=name)
        yield database
    finally:
        if database is not None:
            database.dispose()
        if created:
            with admin.begin() as connection:
                connection.execute(sqlalchemy.text(f"DROP DATABASE IF EXISTS {name}"))
        admin.dispose()


@contextmanager
def _measure_clickhouse_stream_load() -> Iterator[list[dict[str, object]]]:
    """Time Arrow insert calls without changing the ClickHouse adapter.

    ``load_parquet_stages`` issues one ``insert_arrow`` call per stage shard,
    so each successful call is reported as one client-streamed shard.
    """
    from httk.store.backend.clickhouse import support as clickhouse

    measurements: list[dict[str, object]] = []
    original = clickhouse._client_for_url

    class MeasuredClient:
        def __init__(self, client: object) -> None:
            self._client = client

        def insert_arrow(self, table: str, arrow: object) -> object:
            started = time.perf_counter()
            result = self._client.insert_arrow(table, arrow)  # type: ignore[attr-defined]
            elapsed = time.perf_counter() - started
            rows = int(arrow.num_rows)  # type: ignore[attr-defined]
            payload_bytes = int(arrow.nbytes)  # type: ignore[attr-defined]
            measurements.append(
                {
                    "table": table,
                    "rows": rows,
                    "bytes": payload_bytes,
                    "seconds": elapsed,
                }
            )
            return result

        def __getattr__(self, name: str) -> object:
            return getattr(self._client, name)

    def measured_client(url: object) -> MeasuredClient:
        return MeasuredClient(original(url))

    clickhouse._client_for_url = measured_client
    try:
        yield measurements
    finally:
        clickhouse._client_for_url = original


def _verify_counts(store: object, database: object, stream: list[object], materials: int) -> None:
    """Verify material and every content-addressed physical-table count."""
    import sqlalchemy

    from httk.store.backend.sql.mapping import CONTENT_ID_COLUMN

    expected_counts = _expected_table_counts(stream)
    with database.engine.connect() as connection:  # type: ignore[attr-defined]
        material_table = store._metadata.tables["altermagnets_material_records"]  # type: ignore[attr-defined]
        stored = connection.execute(sqlalchemy.select(sqlalchemy.func.count()).select_from(material_table)).scalar_one()
        if int(stored) != materials:
            raise SystemExit(f"verification failed: stored {stored} materials, expected {materials}")
        for name, table in store._metadata.tables.items():  # type: ignore[attr-defined]
            if name.startswith("_httk_") or CONTENT_ID_COLUMN not in table.c:
                continue
            actual = connection.execute(sqlalchemy.select(sqlalchemy.func.count()).select_from(table)).scalar_one()
            expected = expected_counts.get(name, 0)
            if int(actual) != expected:
                raise SystemExit(f"verification failed: stored {actual} rows in {name}, expected {expected}")
            duplicate = connection.execute(
                sqlalchemy.select(table.c[CONTENT_ID_COLUMN])
                .group_by(table.c[CONTENT_ID_COLUMN])
                .having(sqlalchemy.func.count() > 1)
                .limit(1)
            ).first()
            if duplicate is not None:
                raise SystemExit(f"verification failed: duplicate content id in {name}")
    structure_count = expected_counts.get("atomistic_unitcell_structure", 0)
    print(f"verified {len(expected_counts)} tables (structures={structure_count})")


def _point_lookup_seconds(connection: object, table: object, ids: list[str]) -> float:
    """Return elapsed time for one content-ID point lookup per supplied ID."""
    import sqlalchemy

    from httk.store.backend.sql.mapping import CONTENT_ID_COLUMN

    started = time.perf_counter()
    for content_id in ids:
        row = connection.execute(
            sqlalchemy.select(table.c.sid).where(table.c[CONTENT_ID_COLUMN] == content_id)  # type: ignore[attr-defined]
        ).first()
        if row is None:
            raise SystemExit(f"point lookup verification failed for content id {content_id}")
    return time.perf_counter() - started


def _bloom_lookup_benchmark(store: object, database: object, lookups: int) -> dict[str, object]:
    """Compare content-ID lookups before/after a benchmark-only Bloom index."""
    import sqlalchemy

    from httk.store.backend.sql.mapping import CONTENT_ID_COLUMN

    candidates = [
        table
        for name, table in store._metadata.tables.items()  # type: ignore[attr-defined]
        if not name.startswith("_httk_") and CONTENT_ID_COLUMN in table.c and "sid" in table.c
    ]
    if not candidates:
        raise SystemExit("bloom lookup benchmark found no content-addressed table")
    with database.engine.connect() as count_connection:  # type: ignore[attr-defined]
        table = max(
            candidates,
            key=lambda candidate: int(
                count_connection.execute(sqlalchemy.select(sqlalchemy.func.count()).select_from(candidate)).scalar_one()
            ),
        )
    with database.engine.begin() as connection:  # type: ignore[attr-defined]
        ids = [str(row[0]) for row in connection.execute(sqlalchemy.select(table.c[CONTENT_ID_COLUMN]))]
        if not ids:
            raise SystemExit("bloom lookup benchmark found an empty content-addressed table")
        query_ids = [ids[index % len(ids)] for index in range(lookups)]
        without_seconds = _point_lookup_seconds(connection, table, query_ids)
        index_name = "bench_content_id_bloom"
        quoted_table = connection.dialect.identifier_preparer.quote(table.name)
        connection.execute(
            sqlalchemy.text(
                f"ALTER TABLE {quoted_table} ADD INDEX {index_name} ({CONTENT_ID_COLUMN}) "
                "TYPE bloom_filter() GRANULARITY 1"
            )
        )
        connection.execute(sqlalchemy.text(f"ALTER TABLE {quoted_table} MATERIALIZE INDEX {index_name}"))
        with_seconds = _point_lookup_seconds(connection, table, query_ids)
    return {
        "table": table.name,
        "lookups": len(query_ids),
        "without_seconds": without_seconds,
        "with_seconds": with_seconds,
    }


def _build(
    stream: list[object],
    workers: int,
    directory: Path,
    materials: int,
    finalize: str,
    backend: str,
    clickhouse_uri: str | None,
    *,
    track_sids: bool,
    bloom_lookups: int,
) -> dict[str, object]:
    from httk.store.backend.sql import Backend, SqlStore

    @contextmanager
    def build_database() -> Iterator[object]:
        if backend == "clickhouse":
            assert clickhouse_uri is not None
            with _clickhouse_database(clickhouse_uri) as database:
                yield database
            return
        target = directory / f"bench_w{workers}_{time.time_ns()}.duckdb"
        database = Backend.duckdb(target)
        try:
            yield database
        finally:
            database.dispose()
            target.unlink(missing_ok=True)

    with build_database() as database:
        store = SqlStore(database, entry_records={})
        stream_measurements: list[dict[str, object]] = []
        started = time.perf_counter()
        stream_context = (
            _measure_clickhouse_stream_load() if backend == "clickhouse" else nullcontext(stream_measurements)
        )
        with (
            stream_context as stream_measurements,
            store.bulk_ingest(workers=workers, finalize=finalize, track_sids=track_sids) as bulk,
        ):
            for obj in stream:
                bulk.save(obj)
            dispatched = time.perf_counter()
        finished = time.perf_counter()
        _verify_counts(store, database, stream, materials)
        bloom = _bloom_lookup_benchmark(store, database, bloom_lookups) if backend == "clickhouse" else None
        return {
            "total": finished - started,
            "dispatch": dispatched - started,
            "finalize": finished - dispatched,
            "steps": dict(bulk.finalize_timings),
            "stream_loads": stream_measurements,
            "bloom": bloom,
        }


def _print_stream_loads(measurements: list[dict[str, object]]) -> None:
    if not measurements:
        return
    total_rows = sum(int(item["rows"]) for item in measurements)
    total_bytes = sum(int(item["bytes"]) for item in measurements)
    total_seconds = sum(float(item["seconds"]) for item in measurements)
    print("        client-stream shards (Arrow payload bytes)")
    for number, item in enumerate(measurements, start=1):
        elapsed = float(item["seconds"])
        rows = int(item["rows"])
        payload_bytes = int(item["bytes"])
        print(
            f"          {number:>3} {item['table']!s:<42} {rows:>9} rows {payload_bytes / 1_000_000:>8.3f} MB "
            f"{rows / elapsed if elapsed else float('inf'):>10.0f} rows/s "
            f"{payload_bytes / 1_000_000 / elapsed if elapsed else float('inf'):>8.2f} MB/s"
        )
    print(
        f"        total {'':<38} {total_rows:>9} rows {total_bytes / 1_000_000:>8.3f} MB "
        f"{total_rows / total_seconds if total_seconds else float('inf'):>10.0f} rows/s "
        f"{total_bytes / 1_000_000 / total_seconds if total_seconds else float('inf'):>8.2f} MB/s"
    )


def _print_bloom(bloom: dict[str, object] | None) -> None:
    if bloom is None:
        return
    without_seconds = float(bloom["without_seconds"])
    with_seconds = float(bloom["with_seconds"])
    ratio = without_seconds / with_seconds if with_seconds else float("inf")
    print(
        f"        bloom lookup {bloom['table']} ({bloom['lookups']} hits): "
        f"without={without_seconds:.3f}s with={with_seconds:.3f}s speedup={ratio:.2f}x"
    )


def _run_cells(arguments: argparse.Namespace, *, track_sids: bool = True) -> list[dict[str, object]]:
    _add_altermagnets(arguments.altermagnets)
    load_started = time.perf_counter()
    stream = _load_stream(arguments.replicate, arguments.mode)
    load_elapsed = time.perf_counter() - load_started
    materials = len(stream) - 1
    print(f"loaded {materials} materials ({arguments.mode} mode) in {load_elapsed:.1f}s\n")

    with tempfile.TemporaryDirectory(prefix="bench50_") as directory:
        base = Path(directory)
        header = f"{'backend':>10} {'workers':>8} {'repeat':>6} {'total':>9} {'dispatch':>9} {'finalize':>9}"
        print(header)
        print("-" * len(header))
        best: dict[int, float] = {}
        results: list[dict[str, object]] = []
        for workers in arguments.workers:
            totals: list[float] = []
            for repeat in range(arguments.repeats):
                cold_stream = _load_stream(arguments.replicate, arguments.mode)
                if len(cold_stream) - 1 != materials:
                    raise SystemExit("cold stream material count changed during benchmark")
                timing = _build(
                    cold_stream,
                    workers,
                    base,
                    materials,
                    arguments.finalize,
                    arguments.backend,
                    arguments.clickhouse_uri,
                    track_sids=track_sids,
                    bloom_lookups=arguments.bloom_lookups,
                )
                results.append(timing)
                totals.append(float(timing["total"]))
                mode = "serial" if workers == 1 else f"w={workers}"
                print(
                    f"{arguments.backend:>10} {mode:>8} {repeat:>6} "
                    f"{float(timing['total']):>8.1f}s {float(timing['dispatch']):>8.1f}s "
                    f"{float(timing['finalize']):>8.1f}s"
                )
                steps = timing["steps"]
                assert isinstance(steps, dict)
                if steps:
                    print(" " * 8 + "steps " + " ".join(f"{name}={value:.3f}s" for name, value in steps.items()))
                loads = timing["stream_loads"]
                assert isinstance(loads, list)
                _print_stream_loads(loads)
                bloom = timing["bloom"]
                assert bloom is None or isinstance(bloom, dict)
                _print_bloom(bloom)
            best[workers] = min(totals)
        print()
        serial = best.get(1)
        for workers, total in sorted(best.items()):
            speedup = f"{serial / total:.2f}x" if serial else "n/a"
            print(
                f"best backend={arguments.backend} workers={workers:>3}: {total:>7.1f}s   speedup vs serial: {speedup}"
            )
    return results


def _rss_peak_mib() -> float:
    # Linux ru_maxrss is KiB; macOS uses bytes.  The project test environment
    # is Linux, but retaining the platform distinction keeps the reported unit
    # honest when a developer runs the same harness elsewhere.
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1024 if sys.platform.startswith("linux") else peak / (1024 * 1024)


def _rss_command(arguments: argparse.Namespace, replicate: int) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--backend",
        arguments.backend,
        "--workers",
        str(arguments.workers[0]),
        "--repeats",
        "1",
        "--finalize",
        arguments.finalize,
        "--replicate",
        str(replicate),
        "--mode",
        arguments.mode,
        "--bloom-lookups",
        str(arguments.bloom_lookups),
        "--_rss-single",
    ]
    if arguments.altermagnets:
        command.extend(["--altermagnets", arguments.altermagnets])
    if arguments.clickhouse_uri:
        command.extend(["--clickhouse-uri", arguments.clickhouse_uri])
    return command


def _clickhouse_cell_command(arguments: argparse.Namespace, workers: int) -> list[str]:
    """Build one ClickHouse cell in a fresh interpreter before its worker fork."""
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--backend",
        "clickhouse",
        "--clickhouse-uri",
        arguments.clickhouse_uri,
        "--workers",
        str(workers),
        "--repeats",
        "1",
        "--finalize",
        arguments.finalize,
        "--replicate",
        str(arguments.replicate),
        "--mode",
        arguments.mode,
        "--bloom-lookups",
        str(arguments.bloom_lookups),
        "--_cell-process",
    ]
    if arguments.altermagnets:
        command.extend(["--altermagnets", arguments.altermagnets])
    return command


def _run_clickhouse_cells_isolated(arguments: argparse.Namespace) -> int:
    """Run each cold ClickHouse cell before a worker can inherit a client."""
    for workers in arguments.workers:
        for repeat in range(arguments.repeats):
            print(f"\nClickHouse fresh-process cell: workers={workers}, repeat={repeat}")
            completed = subprocess.run(_clickhouse_cell_command(arguments, workers), check=False)
            if completed.returncode:
                return completed.returncode
    return 0


def _run_rss_mode(arguments: argparse.Namespace) -> int:
    if len(arguments.workers) != 1:
        raise SystemExit("--rss-scale requires exactly one worker count so each scale has one peak RSS")
    if arguments.repeats != 1:
        raise SystemExit("--rss-scale requires --repeats 1")
    print("RSS-scaled ingest (track_sids=False; peak is this client interpreter, not the ClickHouse server)")
    print(f"{'replicate':>10} {'materials':>10} {'peak RSS':>12} {'total':>10}")
    print("-" * 48)
    for replicate in (arguments.rss_scale, arguments.rss_scale * 2):
        completed = subprocess.run(_rss_command(arguments, replicate), check=False, text=True, capture_output=True)
        print(completed.stdout, end="")
        if completed.returncode:
            print(completed.stderr, file=sys.stderr, end="")
            return completed.returncode
        result_lines = [line for line in completed.stdout.splitlines() if line.startswith("RSS_RESULT ")]
        if len(result_lines) != 1:
            raise SystemExit("RSS child did not emit exactly one RSS_RESULT record")
        result = json.loads(result_lines[0].removeprefix("RSS_RESULT "))
        print(
            f"{int(result['replicate']):>10} {int(result['materials']):>10} "
            f"{float(result['peak_rss_mib']):>9.1f} MiB {float(result['total']):>8.1f}s"
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("duckdb", "clickhouse"), default="duckdb")
    parser.add_argument("--clickhouse-uri", default=os.environ.get("HTTK_TEST_CLICKHOUSE_URI"))
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 4, 12, 24])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--finalize", choices=("auto", "parity", "deferred"), default="auto")
    parser.add_argument("--replicate", type=int, default=50, help="distinct copies of the base materials (50 ~= 9,000)")
    parser.add_argument(
        "--bloom-lookups", type=int, default=1_000, help="ClickHouse point lookups before/after Bloom index"
    )
    parser.add_argument(
        "--rss-scale", type=int, default=None, metavar="N", help="run D12 RSS cells at N and 2N replicas"
    )
    parser.add_argument("--_cell-process", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_rss-single", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--mode",
        choices=("shared", "distinct", "all-distinct"),
        default="shared",
        help=(
            "shared: replicas share substructure (merge collapses many dups); "
            "distinct: distinct roots (material+structure) with shared atomic descendants; "
            "all-distinct: salt each root's cell, sites, and species descendants"
        ),
    )
    parser.add_argument("--altermagnets", default=None, help="path to the altermagnets checkout")
    arguments = parser.parse_args()

    if arguments.replicate < 1 or arguments.rss_scale is not None and arguments.rss_scale < 1:
        raise SystemExit("--replicate and --rss-scale must be positive")
    if arguments.bloom_lookups < 1:
        raise SystemExit("--bloom-lookups must be positive")
    if arguments.backend == "clickhouse":
        if not arguments.clickhouse_uri:
            raise SystemExit(
                "ClickHouse benchmark requires --clickhouse-uri or HTTK_TEST_CLICKHOUSE_URI; "
                "run `make clickhouse-dev-server` first"
            )
        if arguments.finalize == "parity":
            raise SystemExit("ClickHouse benchmark cells are deferred-only; use --finalize deferred or auto")
        arguments.finalize = "deferred"
    if arguments._rss_single:
        results = _run_cells(arguments, track_sids=False)
        if len(results) != 1:
            raise SystemExit("RSS child requires exactly one benchmark cell")
        print(
            "RSS_RESULT "
            + json.dumps(
                {
                    "replicate": arguments.replicate,
                    "materials": len(_load_stream(arguments.replicate, arguments.mode)) - 1,
                    "peak_rss_mib": _rss_peak_mib(),
                    "total": results[0]["total"],
                },
                sort_keys=True,
            )
        )
        return 0
    if arguments.rss_scale is not None:
        return _run_rss_mode(arguments)
    if arguments.backend == "clickhouse" and not arguments._cell_process:
        return _run_clickhouse_cells_isolated(arguments)
    _run_cells(arguments)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
