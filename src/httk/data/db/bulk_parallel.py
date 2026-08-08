"""Parallel encode + shard-merge backend for :class:`~httk.data.db.bulk.BulkIngest`.

This module implements the ``workers > 1`` mode of
:meth:`~httk.data.db.store.SqlStore.bulk_ingest`. The serial ``workers = 1``
path in :mod:`httk.data.db.bulk` is untouched; :class:`~httk.data.db.bulk.BulkIngest`
delegates to the helpers here only when more than one worker is requested.

The design has three moving parts:

- **Workers** (``_worker_main``): forked processes that run the *pure* encoders
  (``_encode_parent_row`` / ``_encode_child_rows`` from
  :mod:`httk.data.db.store`) with a per-worker
  :class:`~httk.data.store_common.SaveProjection`. Each worker owns a disjoint
  sid block (``(worker_index + 1) << 26``) so its rows never collide with
  another worker's before the merge. A worker deduplicates *content-addressed
  records that carry no identity-excluded metadata* and *all by_value records*
  within its own stream (bounding shard size); records that carry a metadata
  plan are emitted per occurrence so the merge can verify every collision.
  Workers never touch the database — they only write shard files.

- **Shards** (``_ParquetShardWriter``, ``_SqliteShardWriter``): per-worker,
  per-table row files. DuckDB stores each flush as a pyarrow Parquet file
  (``pyarrow`` imported lazily; its absence raises the documented ``parallel``
  extra hint); SQLite stores one shard database per worker written with native
  ``executemany``. Shards live in a ``tempfile.TemporaryDirectory`` next to the
  target database file when it is file-backed, else the tempfile default, and
  are always removed.

- **Merge** (:func:`merge`): the main process, inside the ingest's spanning
  transaction, loads every shard into the freshly created (index-less) record
  tables under the workers' block sids, then collapses cross-worker duplicates
  set-wise (content-id and by_value) in foreign-key dependency order. Because
  referenced tables collapse before their referrers, two rows sharing a content
  id then differ only in their identity-excluded (``IdentitySkip``) columns, so
  each collision's metadata is verified with a single grouped scan per table
  rather than by reconstructing every duplicate record (the dominant cost at
  real-build scale); nested and ``descend`` conflicts surface at the target
  table where the skip metadata lives. The merge then sweeps rows orphaned by a
  collapsed duplicate's subtree and remaps the surviving block sids to a compact
  ``1..N`` range, rewriting every foreign-key column through the same map.

``workers > 1`` targets the offline *build* of a store: it requires a
physically empty target (no application table already holds rows). Incremental
appends into a populated store remain the serial path's domain, where the
per-record staging protocol and its metadata verification already live.
"""

import functools
import importlib
import math
import os
import pickle
import queue as queue_mod
import sqlite3
import tempfile
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import sqlalchemy
from httk.core.storage import StorageProjectionCycleError, resolve_storage_record

from httk.data.db.mapping import (
    CONTENT_ID_COLUMN,
    DISPATCH_CONTENT_ID_COLUMN,
    SID_COLUMN,
    backing_dispatch_column_name,
    entry_dispatch_table_name,
)
from httk.data.db.schema import TableSchema, resolve_schema
from httk.data.db.store import (
    SqlStore,
    _encode_child_rows,
    _encode_parent_row,
    _field_path,
)
from httk.data.store_common import (
    EntryDispatchIntegrityError,
    EntryMetadataConflictError,
    SaveProjection,
    _metadata_plan,
)

if TYPE_CHECKING:
    import httk.data.db.bulk
    from httk.data.db.bulk import BulkIngest

__all__ = ["ParallelController", "merge"]

# Each worker allocates sids from a disjoint high block so rows never collide
# before the merge; the merge remaps every block sid to a compact 1..N value.
# The store's sid column is a 32-bit integer (SQLAlchemy ``Integer`` renders as
# DuckDB ``INTEGER``), so worker ``w`` bases its block at ``(w + 1) << 26``:
# that fits a signed 32-bit integer for up to ~30 workers and leaves 2**26
# (~67M) rows per worker per table before the next worker's block.
_SID_BLOCK_BITS = 26
_SID_BLOCK = 1 << _SID_BLOCK_BITS
_MAX_WORKERS = (1 << 31) // _SID_BLOCK - 1

# Per-worker task queue depth (bounds in-flight buffering). A module constant so
# tests can shrink it to force a saturated queue.
_QUEUE_MAXSIZE = 64
# Upper bound on how long the main process waits for a worker to make progress
# (report a result, or accept a stop sentinel) before declaring the pool stalled.
_WORKER_STALL_TIMEOUT = 300.0


def _worker_base(worker_index: int) -> int:
    """The first sid a worker may allocate (its block is ``[base, base + 2**26)``)."""
    return (worker_index + 1) << _SID_BLOCK_BITS


def _references_reach(start: type, goal: type, seen: set[type] | None = None) -> bool:
    """Whether following reference edges from ``start`` reaches ``goal`` (a reference cycle test)."""
    if start is goal:
        return True
    seen = set() if seen is None else seen
    if start in seen:
        return False
    seen.add(start)
    return any(_references_reach(referenced, goal, seen) for referenced in resolve_schema(start).referenced_classes())


@functools.cache
def unsupported_metadata_reason(record_type: type) -> str | None:
    """Why the parallel merge cannot verify ``record_type``'s identity-excluded metadata, or ``None``.

    The set-wise merge verifies identity-excluded metadata with a grouped column
    scan (see :meth:`_Merger._verify_collision_metadata`). That covers scalar
    ``IdentitySkip`` columns and skipped references to content-addressed or
    by_value targets, and it delegates ``descend`` conflicts to the target
    table's own collapse. Three shapes fall outside it and are rejected up front
    (fail fast, naming ``workers=1``) rather than verified incorrectly:

    - an identity-excluded **child sequence** (no parent column to group on);
    - an identity-excluded **reference to a non-deduplicated** (``none``) record,
      or a **descend into** one (the target is never collapsed, so its metadata
      is never compared);
    - a **self-referential** identity-excluded reference (the target table is the
      one being collapsed, so its sids are not yet final when compared).

    :param record_type: The record class to classify.
    :return: A human-readable reason string, or ``None`` when the shape is supported.
    """
    plan = _metadata_plan(record_type)
    if plan is None:
        return None
    name = record_type.__name__
    for spec in plan.skipped_nested:
        if spec.role != "reference":
            return f"{name}.{spec.field} is an identity-excluded child sequence"
        if spec.target is None:
            continue
        if resolve_schema(spec.target).dedup not in ("content_id", "by_value"):
            return f"{name}.{spec.field} is an identity-excluded reference to a non-deduplicated record"
        if _references_reach(spec.target, record_type):
            return f"{name}.{spec.field} is a self-referential identity-excluded reference"
    for spec in plan.descend_specs:
        if spec.target is not None and resolve_schema(spec.target).dedup == "none":
            return f"{name}.{spec.field} descends into a non-deduplicated ('none') record's metadata"
    return None


@functools.cache
def _plain_float_skip_fields(record_type: type) -> tuple[tuple[str, str], ...]:
    """The ``(field, column)`` pairs of ``record_type``'s plain-``float`` ``IdentitySkip`` fields.

    Only the plain Python-``float`` codec is meant here — not the exact numeric
    codecs (``fraction``, ``fracscalar``, tensor codecs) that also carry a float
    column beside their exact text channel. A NaN in a plain-float column reads
    back as ``NaN`` on DuckDB and as ``NULL`` on SQLite (which has no NaN), so it
    cannot be told apart from a real ``None`` once stored; the worker therefore
    flags NaN-bearing content ids while it still holds the source value, and the
    merge treats a duplicated flagged content id as a conflict (serial's
    ``NaN != NaN``).
    """
    plan = _metadata_plan(record_type)
    if plan is None:
        return ()
    fields: list[tuple[str, str]] = []
    for spec in plan.skipped_specs:
        if spec.codec_name != "float":
            continue
        fields.extend((spec.field, column.name) for column in spec.columns if column.kind == "float")
    return tuple(fields)


@dataclass(frozen=True)
class _WorkerConfig:
    """Immutable per-run settings handed to every worker (fork-inherited)."""

    chunk_size: int
    shard_dir: str
    backend: str  # "duckdb" or "sqlite"


@dataclass
class _DispatchRow:
    """A buffered entry-dispatch row a worker produced (backing sid still a block sid)."""

    dispatch_name: str
    key: str
    column: str
    all_columns: tuple[str, ...]
    ref_table: str
    block_sid: int
    family_name: str


@dataclass
class _WorkerManifest:
    """What a finished worker reports to the main process."""

    worker_index: int
    token_sid: dict[int, tuple[str, int]]
    dispatch: list[_DispatchRow]
    tables: list[str]
    # DuckDB: table -> list of parquet file paths. SQLite: {"db": path}.
    shards: dict[str, Any]
    # (table, content_id, field) triples whose identity-excluded float held a NaN.
    nan_content: list[tuple[str, str, str]] = field(default_factory=list)


# Fork-inherited handles the worker reads from module scope (never pickled).
_PARENT_STORE: SqlStore | None = None
_PARENT_CONFIG: _WorkerConfig | None = None


# --------------------------------------------------------------------- shard writers


def _pa_type(column: sqlalchemy.Column[Any], pa: Any) -> Any:
    """Map a record column's SQLAlchemy type to the pyarrow type of its shard column."""
    type_ = column.type
    if isinstance(type_, sqlalchemy.Boolean):
        return pa.bool_()
    if isinstance(type_, sqlalchemy.Integer):
        return pa.int64()
    if isinstance(type_, sqlalchemy.Float):
        return pa.float64()
    if isinstance(type_, sqlalchemy.LargeBinary):
        return pa.binary()
    # Text / String and everything else stringly-typed.
    return pa.string()


class _ParquetShardWriter:
    """Write per-worker, per-table Parquet shards (the DuckDB backend hand-off)."""

    def __init__(self, store: SqlStore, worker_index: int, shard_dir: str) -> None:
        try:
            self._pa = importlib.import_module("pyarrow")
            self._pq = importlib.import_module("pyarrow.parquet")
        except ImportError as error:  # pragma: no cover - guarded before fork
            raise ImportError(
                "bulk_ingest(workers>1) on a DuckDB store needs pyarrow; "
                "install the 'httk-data[parallel]' extra to use it"
            ) from error
        self._store = store
        self._worker_index = worker_index
        self._dir = shard_dir
        self._schemas: dict[str, Any] = {}
        self._files: dict[str, list[str]] = {}
        self._sequence = 0

    def _schema_for(self, table_name: str) -> Any:
        schema = self._schemas.get(table_name)
        if schema is None:
            table = self._store._table(table_name)
            fields = [self._pa.field(column.name, _pa_type(column, self._pa)) for column in table.columns]
            schema = self._pa.schema(fields)
            self._schemas[table_name] = schema
        return schema

    def write(self, table_name: str, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        schema = self._schema_for(table_name)
        columns = [field_.name for field_ in schema]
        data = {name: self._pa.array([row.get(name) for row in rows], type=schema.field(name).type) for name in columns}
        table = self._pa.table(data, schema=schema)
        path = os.path.join(self._dir, f"w{self._worker_index}_{table_name}_{self._sequence}.parquet")
        self._sequence += 1
        self._pq.write_table(table, path)
        self._files.setdefault(table_name, []).append(path)

    def finalize(self) -> dict[str, Any]:
        return dict(self._files)


class _SqliteShardWriter:
    """Write one native-SQLite shard database per worker (the SQLite backend hand-off)."""

    def __init__(self, store: SqlStore, worker_index: int, shard_dir: str) -> None:
        self._store = store
        self._path = os.path.join(shard_dir, f"w{worker_index}.sqlite")
        self._connection = sqlite3.connect(self._path)
        self._connection.execute("PRAGMA journal_mode=OFF")
        self._connection.execute("PRAGMA synchronous=OFF")
        self._created: set[str] = set()

    def _columns(self, table_name: str) -> list[str]:
        return [column.name for column in self._store._table(table_name).columns]

    def write(self, table_name: str, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        columns = self._columns(table_name)
        if table_name not in self._created:
            definitions = ", ".join(f'"{name}"' for name in columns)
            self._connection.execute(f'CREATE TABLE "{table_name}" ({definitions})')
            self._created.add(table_name)
        placeholders = ", ".join("?" for _ in columns)
        self._connection.executemany(
            f'INSERT INTO "{table_name}" VALUES ({placeholders})',
            [tuple(row.get(name) for name in columns) for row in rows],
        )

    def finalize(self) -> dict[str, Any]:
        self._connection.commit()
        self._connection.close()
        return {"db": self._path, "tables": sorted(self._created)}


def _make_writer(store: SqlStore, worker_index: int, config: _WorkerConfig) -> Any:
    if config.backend == "duckdb":
        return _ParquetShardWriter(store, worker_index, config.shard_dir)
    return _SqliteShardWriter(store, worker_index, config.shard_dir)


# --------------------------------------------------------------------- worker encoder


class _WorkerEncoder:
    """Encode a worker's slice of the stream into shard rows with block sids.

    A stripped connection-free counterpart of
    :meth:`~httk.data.db.bulk.BulkIngest._encode_active`: no table DDL, sids from
    the worker's own block, and no metadata verification (the merge verifies
    every surviving collision). Content records that carry a metadata plan are
    emitted per occurrence — never deduplicated in the worker — so the merge
    sees, and can compare, all of them.
    """

    def __init__(self, store: SqlStore, worker_index: int, config: _WorkerConfig) -> None:
        self._store = store
        self._chunk_size = config.chunk_size
        self._writer = _make_writer(store, worker_index, config)
        self._base = _worker_base(worker_index)
        self._registered: set[type] = set()
        self._next_sid: dict[str, int] = {}
        self._rows: dict[str, list[dict[str, Any]]] = {}
        self._content_index: dict[str, dict[str, int]] = {}
        self._value_index: dict[str, dict[tuple[Any, ...], int]] = {}
        self._token_sid: dict[int, tuple[str, int]] = {}
        self._dispatch: list[_DispatchRow] = []
        self._tables: set[str] = set()
        self._nan_content: set[tuple[str, str, str]] = set()
        self._since_flush = 0

    # -- encoding

    def save(self, token: int, obj: Any, as_record: type | None) -> None:
        record_type = resolve_storage_record(obj, as_record=as_record)
        projection = SaveProjection()
        sid = self._encode(record_type, obj, projection, "")
        table_name = resolve_schema(record_type).table_name
        family = self._store._family_for_backing(record_type)
        if family is not None and len(family.records) > 1:
            self._dispatch.append(
                _DispatchRow(
                    dispatch_name=entry_dispatch_table_name(family.name),
                    key=projection.content_id(record_type, obj),
                    column=backing_dispatch_column_name(family.record_names[family.records.index(record_type)]),
                    all_columns=tuple(backing_dispatch_column_name(name) for name in family.record_names),
                    ref_table=table_name,
                    block_sid=sid,
                    family_name=family.name,
                )
            )
        self._token_sid[token] = (table_name, sid)
        self._since_flush += 1
        if self._since_flush >= self._chunk_size:
            self._flush()

    def _encode(self, record_type: type, source: Any, projection: SaveProjection, path: str) -> int:
        active_key = (record_type, id(source))
        if active_key in projection.active:
            raise StorageProjectionCycleError(path, record_type)
        projection.active.add(active_key)
        try:
            return self._encode_active(record_type, source, projection, path)
        finally:
            projection.active.remove(active_key)

    def _encode_active(self, record_type: type, source: Any, projection: SaveProjection, path: str) -> int:
        schema = resolve_schema(record_type)
        self._register(record_type)
        table_name = schema.table_name
        self._next_sid.setdefault(table_name, self._base + 1)
        projected = projection.projector(record_type, source)

        validation_key = (record_type, id(source))
        if type(source) is record_type and validation_key not in projection.validated:
            validator = vars(record_type).get("__httk_validate__")
            if validator is not None:
                validator.__get__(None, record_type)(source)
            projection.validated.add(validation_key)

        dedup_content = schema.dedup == "content_id" and _metadata_plan(record_type) is None
        key: str | None = None
        if schema.dedup == "content_id":
            key = projection.content_id(record_type, source)
            if dedup_content:
                existing = self._content_index.setdefault(table_name, {}).get(key)
                if existing is not None:
                    return existing

        def resolve_sid(referenced_type: type, value: Any, field_path: str) -> int:
            return self._encode(referenced_type, value, projection, field_path)

        values = _encode_parent_row(schema, source, projected, path, resolve_sid)

        if schema.dedup == "by_value":
            value_tuple = tuple(sorted(values.items()))
            existing = self._value_index.setdefault(table_name, {}).get(value_tuple)
            if existing is not None:
                return existing

        sid = self._next_sid[table_name]
        if sid - self._base >= _SID_BLOCK:
            raise RuntimeError(
                f"bulk_ingest worker exceeded its {_SID_BLOCK} sid block for table {table_name!r}; "
                "reduce the worker count or split the ingest"
            )
        self._next_sid[table_name] = sid + 1
        row = {SID_COLUMN: sid, **values}
        if key is not None:
            row[CONTENT_ID_COLUMN] = key
            for field_name, column_name in _plain_float_skip_fields(record_type):
                candidate = values.get(column_name)
                if isinstance(candidate, float) and math.isnan(candidate):
                    # Report every NaN field (not just the first): the merge picks
                    # the schema-order-first among them, deterministically.
                    self._nan_content.add((table_name, key, field_name))
            if dedup_content:
                self._content_index[table_name][key] = sid
        elif schema.dedup == "by_value":
            self._value_index[table_name][tuple(sorted(values.items()))] = sid
        self._buffer(table_name, row)

        for spec in schema.fields:
            if spec.role != "child":
                continue
            assert spec.child is not None
            child_rows = _encode_child_rows(
                schema,
                spec,
                sid,
                SqlStore._projected_value(record_type, source, projected, spec),
                _field_path(path, spec.field),
                resolve_sid,
            )
            for child_row in child_rows:
                self._buffer(spec.child.table_name, child_row)
        return sid

    def _register(self, record_type: type) -> None:
        if record_type in self._registered:
            return
        self._store._register_tables((record_type,))
        self._registered.add(record_type)

    def _buffer(self, table_name: str, row: dict[str, Any]) -> None:
        self._rows.setdefault(table_name, []).append(row)
        self._tables.add(table_name)

    def _flush(self) -> None:
        for table_name, rows in self._rows.items():
            if rows:
                self._writer.write(table_name, rows)
                rows.clear()
        self._since_flush = 0

    def finish(self) -> _WorkerManifest:
        self._flush()
        return _WorkerManifest(
            worker_index=-1,  # filled by the caller
            token_sid=self._token_sid,
            dispatch=self._dispatch,
            tables=sorted(self._tables),
            shards=self._writer.finalize(),
            nan_content=sorted(self._nan_content),
        )


# --------------------------------------------------------------------- worker process


def _worker_main(worker_index: int, task_queue: Any, result_queue: Any) -> None:
    """Worker process entry point: encode tasks into shards, then report a manifest.

    Tasks arrive as pickled ``(token, obj, as_record)`` byte strings (the main
    process pickles synchronously, so an unpicklable object fails the caller's
    ``save`` promptly instead of vanishing in a queue feeder thread). The worker
    never touches the store's database. On completion (or failure) it flushes its
    result onto ``result_queue`` and exits with :func:`os._exit` to skip
    interpreter finalizers that might disturb the fork-inherited engine.

    :param worker_index: The worker's index (its sid block and shard names).
    :param task_queue: This worker's task queue of pickled tasks (``None`` stops).
    :param result_queue: The queue the manifest or an exception is reported on.
    :return: None.
    """
    assert _PARENT_STORE is not None and _PARENT_CONFIG is not None
    encoder = _WorkerEncoder(_PARENT_STORE, worker_index, _PARENT_CONFIG)
    try:
        while True:
            item = task_queue.get()
            if item is None:
                break
            token, obj, as_record = pickle.loads(item)
            encoder.save(token, obj, as_record)
        manifest = encoder.finish()
        manifest.worker_index = worker_index
        _report(result_queue, (worker_index, "ok", manifest))
    except BaseException as error:  # faithfully relayed to the caller
        _report(result_queue, (worker_index, "error", _as_reportable(error)))
    os._exit(0)


def _report(result_queue: Any, payload: Any) -> None:
    result_queue.put(payload)
    result_queue.close()
    result_queue.join_thread()


def _as_reportable(error: BaseException) -> BaseException:
    """Return an exception that survives pickling back to the main process."""
    try:
        import pickle

        pickle.loads(pickle.dumps(error))
    except Exception:  # fall back to a faithful-typed surrogate
        return RuntimeError(f"{type(error).__name__}: {error}")
    return error


# --------------------------------------------------------------------- pool controller


class ParallelController:
    """Own the worker pool, task dispatch, and shard directory for one parallel ingest.

    Each worker has its own task queue; ``dispatch`` routes token ``k`` to worker
    ``k % workers`` (deterministic round-robin), so the record order the caller
    saves fully determines which worker encodes each record. A shared result
    queue carries each worker's manifest (or exception) back.
    """

    def __init__(self, store: SqlStore, *, workers: int, chunk_size: int, backend: str) -> None:
        import multiprocessing

        if workers > _MAX_WORKERS:
            raise ValueError(f"bulk_ingest supports at most {_MAX_WORKERS} workers (sid-block limit)")
        self._store = store
        self._workers = workers
        self._context = multiprocessing.get_context("fork")
        self._temp = tempfile.TemporaryDirectory(prefix="httk_bulk_", dir=_shard_parent_dir(store))
        self._config = _WorkerConfig(chunk_size=chunk_size, shard_dir=self._temp.name, backend=backend)
        self._queues: list[Any] = [self._context.Queue(maxsize=_QUEUE_MAXSIZE) for _ in range(workers)]
        self._result_queue: Any = self._context.Queue()
        self._processes: list[Any] = []
        # Every result consumed by health polling is cached here (not just errors),
        # so a worker that reports and exits cleanly while a sibling's queue is full
        # keeps its manifest and is not mistaken for a crash.
        self._results_cache: dict[int, tuple[str, Any]] = {}
        self._closed = False

    def start(self) -> None:
        import warnings

        global _PARENT_STORE, _PARENT_CONFIG
        _PARENT_STORE = self._store
        _PARENT_CONFIG = self._config
        try:
            for index in range(self._workers):
                process = self._context.Process(
                    target=_worker_main,
                    args=(index, self._queues[index], self._result_queue),
                    daemon=True,
                )
                with warnings.catch_warnings():
                    # Forking is required: workers inherit the (unpicklable) store
                    # and never touch its database, so Python 3.12's multi-threaded
                    # fork() advisory does not apply here.
                    warnings.simplefilter("ignore", DeprecationWarning)
                    process.start()
                self._processes.append(process)
        finally:
            # The children have forked; the parent no longer needs the globals.
            _PARENT_STORE = None
            _PARENT_CONFIG = None

    def dispatch(self, token: int, obj: Any, as_record: type | None) -> None:
        """Pickle the task synchronously and enqueue it on its worker (routed by token)."""
        # Pickle here, in the caller's thread: an unpicklable object raises out of
        # ``save`` promptly rather than being silently dropped by a queue feeder.
        payload = pickle.dumps((token, obj, as_record))
        queue = self._queues[token % self._workers]
        while True:
            self._raise_if_worker_broken()
            try:
                queue.put(payload, timeout=0.5)
                return
            except queue_mod.Full:
                continue

    def _raise_if_worker_broken(self) -> None:
        """Raise if any worker reported an error or exited *without* reporting (a crash or kill).

        Results are cached (both ``ok`` and ``error``), so a worker that reported
        and then exited cleanly — e.g. it took its stop sentinel while a sibling's
        queue was still full — is recognized as done, not misreported as a crash,
        and its manifest survives for :meth:`finish`.
        """
        self._drain_results(self._results_cache)
        for status, payload in self._results_cache.values():
            if status == "error":
                raise _forward(payload)
        for index, process in enumerate(self._processes):
            if process.exitcode is not None and index not in self._results_cache:
                raise RuntimeError(
                    "a bulk_ingest worker exited unexpectedly (crashed or was killed); the ingest is aborted"
                )

    def finish(self) -> list[_WorkerManifest]:
        """Signal completion, collect every worker's manifest, and re-raise the first error.

        A worker that exits without reporting (a crash or an external kill) is
        detected by its exit code and aborts the ingest, so a lost task can never
        reach the merge. Both the stop-sentinel sends and the result waits are
        bounded and interleaved with health checks, so a worker that dies with a
        full queue cannot deadlock the main process.
        """
        import time

        self._send_sentinels()
        # Start from whatever health polling already consumed (sending the
        # sentinels may have drained some workers' results into the cache).
        results: dict[int, tuple[str, Any]] = dict(self._results_cache)
        error: BaseException | None = None
        last_progress = time.monotonic()
        while len(results) < self._workers:
            try:
                worker_index, status, payload = self._result_queue.get(timeout=1.0)
                results[worker_index] = (status, payload)
                last_progress = time.monotonic()
            except queue_mod.Empty:
                if all(process.exitcode is not None for process in self._processes):
                    self._drain_results(results)
                    break
                if time.monotonic() - last_progress > _WORKER_STALL_TIMEOUT:
                    error = RuntimeError("bulk_ingest workers stopped making progress; aborting")
                    break
        if len(results) < self._workers and error is None:
            error = RuntimeError("a bulk_ingest worker exited without reporting a result (crashed or was killed)")
        for status, payload in results.values():
            if status == "error" and error is None:
                error = _forward(payload)
        for process in self._processes:
            process.join(timeout=30)
        if error is not None:
            raise error
        return [payload for status, payload in results.values() if status == "ok"]

    def _send_sentinels(self) -> None:
        """Put a stop sentinel on each worker queue, aborting if a worker died with a full queue."""
        import time

        pending = list(range(self._workers))
        deadline = time.monotonic() + _WORKER_STALL_TIMEOUT
        while pending:
            still_pending: list[int] = []
            for index in pending:
                try:
                    self._queues[index].put(None, timeout=0.1)
                except queue_mod.Full:
                    still_pending.append(index)
            pending = still_pending
            if not pending:
                return
            # A queue that will not accept the sentinel belongs to a worker that
            # is no longer draining it — detect the crash/kill and abort.
            self._raise_if_worker_broken()
            if time.monotonic() > deadline:
                raise RuntimeError("bulk_ingest could not signal completion to its workers; aborting")

    def _drain_results(self, results: dict[int, tuple[str, Any]]) -> None:
        """Absorb any results still queued after every worker has exited (avoids a report/exit race)."""
        try:
            while True:
                worker_index, status, payload = self._result_queue.get_nowait()
                results[worker_index] = (status, payload)
        except queue_mod.Empty:
            return

    def close(self) -> None:
        """Terminate any live workers and remove the shard directory (idempotent)."""
        if self._closed:
            return
        self._closed = True
        for process in self._processes:
            if process.is_alive():
                process.terminate()
        for process in self._processes:
            process.join(timeout=10)
        # Cancel each queue's feeder thread before closing: a queue left non-empty
        # by an aborted ingest would otherwise block ``close`` on the feeder join.
        for queue in (*self._queues, self._result_queue):
            queue.cancel_join_thread()
            queue.close()
        self._temp.cleanup()


def _forward(payload: Any) -> BaseException:
    return payload if isinstance(payload, BaseException) else RuntimeError(str(payload))


def _shard_parent_dir(store: SqlStore) -> str | None:
    """The directory shards are created in: next to a file-backed database, else the tempfile default."""
    try:
        database = store._database.engine.url.database
    except Exception:  # any odd URL falls back to the default temp root
        return None
    if not database or database == ":memory:":
        return None
    parent = os.path.dirname(os.path.abspath(database))
    return parent if os.path.isdir(parent) else None


# --------------------------------------------------------------------- merge (main process)


def merge(ingest: "httk.data.db.bulk.BulkIngest", manifests: list[_WorkerManifest]) -> None:
    """Load every worker shard, collapse cross-worker duplicates, and compact the sids.

    Runs in the main process inside the ingest's spanning transaction.

    :param ingest: The owning bulk-ingest context (its connection and store).
    :param manifests: One manifest per finished worker.
    :return: None.
    """
    _Merger(ingest, manifests).run()


class _Merger:
    """The set-wise shard merge for a parallel ingest (see :func:`merge`)."""

    def __init__(self, ingest: "BulkIngest", manifests: list[_WorkerManifest]) -> None:
        self._ingest = ingest
        self._store = ingest._store
        assert ingest._connection is not None
        self._connection = ingest._connection
        self._manifests = manifests
        self._fk_columns = ingest._build_fk_columns()
        self._referrers = self._invert_fk_columns()
        # (table, block_sid) -> keep_sid after cross-worker collapse.
        self._collapse: dict[tuple[str, int], int] = {}
        # (table, sid) -> compact_sid after final renumbering.
        self._compaction: dict[tuple[str, int], int] = {}
        # table -> {content id -> set of fields} whose identity-excluded float held
        # a NaN. A set (not last-manifest-wins) keeps attribution deterministic:
        # the merge names the schema-order-first field among the reported set.
        self._nan_content: dict[str, dict[str, set[str]]] = {}
        for manifest in manifests:
            for table_name, content_id, field_name in manifest.nan_content:
                self._nan_content.setdefault(table_name, {}).setdefault(content_id, set()).add(field_name)

    def _invert_fk_columns(self) -> dict[str, list[tuple[str, str]]]:
        """referenced table -> list of (referrer_table, referrer_column)."""
        result: dict[str, list[tuple[str, str]]] = {}
        for table_name, columns in self._fk_columns.items():
            for column, referenced in columns:
                result.setdefault(referenced, []).append((table_name, column))
        return result

    def run(self) -> None:
        self._load_shards()
        for table in self._store._metadata.sorted_tables:
            schema = self._ingest._parent_schema.get(table.name)
            if schema is None:
                continue
            if schema.dedup == "content_id":
                self._collapse_content(table, schema)
            elif schema.dedup == "by_value":
                self._collapse_by_value(table, schema)
        self._sweep_orphans()
        for table in self._store._metadata.sorted_tables:
            if SID_COLUMN in table.c:
                self._compact(table)
        self._merge_dispatch()
        self._populate_resolved_map()

    def _populate_resolved_map(self) -> None:
        """Map every sid ``save`` returned (a synthetic token) to its durable stored sid."""
        ingest = self._ingest
        for manifest in self._manifests:
            for token, (table_name, block_sid) in manifest.token_sid.items():
                ingest._resolved_map[(table_name, token)] = self._final_sid(table_name, block_sid)

    # -- shard loading

    def _load_shards(self) -> None:
        backend = self._connection.dialect.name
        if backend == "duckdb":
            self._load_parquet_shards()
        else:
            self._load_sqlite_shards()

    def _table_columns(self, table_name: str) -> list[str]:
        return [column.name for column in self._store._table(table_name).columns]

    def _load_parquet_shards(self) -> None:
        files_by_table: dict[str, list[str]] = {}
        for manifest in self._manifests:
            for table_name, files in manifest.shards.items():
                files_by_table.setdefault(table_name, []).extend(files)
        for table_name, files in files_by_table.items():
            if not files:
                continue
            columns = ", ".join(f'"{name}"' for name in self._table_columns(table_name))
            # Bind each shard path as a parameter — a path containing a quote must
            # not break or inject into the SQL.
            placeholders = ", ".join(f":f{index}" for index in range(len(files)))
            statement = sqlalchemy.text(
                f'INSERT INTO "{table_name}" ({columns}) SELECT {columns} FROM read_parquet([{placeholders}])'
            ).bindparams(**{f"f{index}": path for index, path in enumerate(files)})
            self._connection.execute(statement)

    def _load_sqlite_shards(self) -> None:
        # SQLite forbids DETACH inside a transaction, and the merge owns one
        # spanning transaction, so every shard is attached at once (its file is
        # unlinked when the shard directory is cleaned up). Raise the default
        # 10-database attach ceiling to accommodate many workers.
        # The ingest owns this connection until after the transaction closes, so
        # the post-transaction DETACH in _release_connection runs on the exact
        # connection that attached these shards.
        driver: Any = self._connection.connection.driver_connection
        try:
            driver.setlimit(sqlite3.SQLITE_LIMIT_ATTACHED, 125)
        except (AttributeError, sqlite3.NotSupportedError):  # pragma: no cover - platform dependent
            pass
        for index, manifest in enumerate(self._manifests):
            database = manifest.shards.get("db")
            tables = manifest.shards.get("tables", [])
            if not database or not tables:
                continue
            alias = f"httk_shard_{index}"
            # The path is bound (a quote in it must not break or inject); the alias
            # is a controlled identifier and cannot be a bound parameter.
            self._connection.execute(sqlalchemy.text(f"ATTACH DATABASE :db AS {alias}").bindparams(db=database))
            self._ingest._parallel_attached.append(alias)
            for table_name in tables:
                columns = ", ".join(f'"{name}"' for name in self._table_columns(table_name))
                self._connection.execute(
                    sqlalchemy.text(
                        f'INSERT INTO main."{table_name}" ({columns}) SELECT {columns} FROM {alias}."{table_name}"'
                    )
                )

    # -- cross-worker collapse

    def _collapse_content(self, table: sqlalchemy.Table, schema: TableSchema) -> None:
        keep = (
            sqlalchemy.select(table.c[CONTENT_ID_COLUMN], sqlalchemy.func.min(table.c[SID_COLUMN]).label("keep"))
            .group_by(table.c[CONTENT_ID_COLUMN])
            .subquery()
        )
        statement = (
            sqlalchemy.select(table.c[SID_COLUMN], keep.c.keep)
            .join_from(table, keep, table.c[CONTENT_ID_COLUMN] == keep.c[CONTENT_ID_COLUMN])
            .where(table.c[SID_COLUMN] != keep.c.keep)
        )
        pairs = [(int(row[0]), int(row[1])) for row in self._connection.execute(statement).all()]
        if not pairs:
            return
        if self._ingest._verify_metadata and _metadata_plan(schema.cls) is not None:
            self._verify_collision_metadata(table, schema)
        self._apply_collapse(table, schema, pairs)

    def _collapse_by_value(self, table: sqlalchemy.Table, schema: TableSchema) -> None:
        value_columns = [column.name for column in table.columns if column.name != SID_COLUMN]
        while True:
            keep = (
                sqlalchemy.select(
                    *(table.c[name] for name in value_columns),
                    sqlalchemy.func.min(table.c[SID_COLUMN]).label("keep"),
                )
                .group_by(*(table.c[name] for name in value_columns))
                .subquery()
            )
            condition = sqlalchemy.and_(*(table.c[name].is_not_distinct_from(keep.c[name]) for name in value_columns))
            statement = (
                sqlalchemy.select(table.c[SID_COLUMN], keep.c.keep)
                .join_from(table, keep, condition)
                .where(table.c[SID_COLUMN] != keep.c.keep)
            )
            pairs = [(int(row[0]), int(row[1])) for row in self._connection.execute(statement).all()]
            if not pairs:
                return
            self._apply_collapse(table, schema, pairs)

    def _apply_collapse(self, table: sqlalchemy.Table, schema: TableSchema, pairs: list[tuple[int, int]]) -> None:
        name = table.name
        for old, keep in pairs:
            self._collapse[(name, old)] = keep
        child_links = {(spec.child.table_name, f"{name}_{SID_COLUMN}") for spec in schema.fields if spec.child}
        map_table = self._make_map_table(pairs)
        try:
            for referrer_table, column in self._referrers.get(name, ()):
                if (referrer_table, column) in child_links:
                    # A collapsed parent's own child rows are dropped, not repointed;
                    # the surviving parent already carries its own children.
                    self._delete_where_in_map(referrer_table, column, map_table)
                else:
                    self._remap_column(referrer_table, column, map_table)
            self._delete_where_in_map(name, SID_COLUMN, map_table)
        finally:
            self._drop_map_table(map_table)

    def _verify_collision_metadata(self, table: sqlalchemy.Table, schema: TableSchema) -> None:
        """Set-wise verify that rows sharing a content id agree on their identity-excluded metadata.

        Only the columns that actually carry identity-excluded metadata are
        compared: the ``IdentitySkip`` scalar columns, and a skipped reference's
        sid column when its target is content-addressed (equal content collapses
        to one sid before this runs, so a differing sid is a differing skipped
        reference). ``descend`` conflicts — a non-skipped reference whose target
        carries the skip metadata — surface at that target table's own collapse,
        where the metadata lives, because equal-content parents reference
        equal-content (hence collapsed-together) targets. One grouped scan per
        content table therefore replaces reconstructing each duplicate record,
        the dominant cost at real-build scale.

        :param table: The content-addressed record table being collapsed.
        :param schema: The table's resolved schema (for the diagnostic record name).
        :raises httk.data.store_common.EntryMetadataConflictError: If a content id occurs with differing metadata.
        """
        compare_columns = self._metadata_compare_columns(schema)
        if not compare_columns:
            return
        # NaN scan first: serial treats ``NaN != NaN`` as a conflict, but SQL
        # equality groups a NaN with itself (DuckDB's total order) and SQLite
        # stores no NaN at all, so the exact scan below cannot see it. Running it
        # first also gives a NaN in an earlier schema field priority over a plain
        # value difference in a later one, matching serial's field-order checks.
        # A duplicated content id the workers flagged as NaN-bearing is a conflict;
        # the field named is the schema-order-first of the reported set.
        nan_by_content = self._nan_content.get(table.name)
        if nan_by_content:
            duplicated_nan = self._connection.execute(
                sqlalchemy.select(table.c[CONTENT_ID_COLUMN])
                .where(table.c[CONTENT_ID_COLUMN].in_(sorted(nan_by_content)))
                .group_by(table.c[CONTENT_ID_COLUMN])
                .having(sqlalchemy.func.count() > 1)
                .limit(1)
            ).first()
            if duplicated_nan is not None:
                key = duplicated_nan[0]
                reported = nan_by_content.get(key, set())
                field_name = next(
                    (field for _column, field in compare_columns if field in reported),
                    schema.cls.__name__,
                )
                raise EntryMetadataConflictError(
                    f"metadata conflict for {schema.cls.__name__}.{field_name}: content id {key!r} occurs with "
                    "a NaN identity-excluded value that never equals itself"
                )
        # Exact-difference scan: SQL ``=`` matches serial's scalar equality for
        # every finite value — ``-0.0 == 0.0`` and ``NULL``/``None`` both group as
        # equal — so a content id whose group has more than one distinct tuple
        # carries differing identity-excluded metadata.
        column_names = [column for column, _field in compare_columns]
        selected = [table.c[CONTENT_ID_COLUMN], *(table.c[name] for name in column_names)]
        distinct_rows = sqlalchemy.select(*selected).distinct().subquery()
        conflicting = self._connection.execute(
            sqlalchemy.select(distinct_rows.c[CONTENT_ID_COLUMN])
            .group_by(distinct_rows.c[CONTENT_ID_COLUMN])
            .having(sqlalchemy.func.count() > 1)
            .limit(1)
        ).first()
        if conflicting is not None:
            self._raise_metadata_conflict(table, schema, compare_columns, conflicting[0])

    def _raise_metadata_conflict(
        self, table: sqlalchemy.Table, schema: TableSchema, compare_columns: list[tuple[str, str]], key: str
    ) -> None:
        differing_field = self._first_differing_field(table, compare_columns, key)
        field_name = f"{schema.cls.__name__}.{differing_field}" if differing_field else schema.cls.__name__
        raise EntryMetadataConflictError(
            f"metadata conflict for {field_name}: content id {key!r} occurs with differing identity-excluded metadata"
        )

    @staticmethod
    def _metadata_compare_columns(schema: TableSchema) -> list[tuple[str, str]]:
        """The ``(column, field)`` pairs whose within-group difference is an identity-excluded conflict."""
        plan = _metadata_plan(schema.cls)
        if plan is None:
            return []
        columns: list[tuple[str, str]] = []
        for spec in plan.skipped_specs:
            if spec.codec_name == "float":
                # The plain-float codec stores an exact text companion for lossless
                # reconstruction, but ``-0.0`` and ``0.0`` differ there while serial's
                # ``_metadata_scalar_equal`` compares them with IEEE ``==``. Compare
                # only the float column (whose SQL equality is IEEE) and drop the
                # string companion; NaN is handled separately below. Exact numeric
                # codecs (fraction, fracscalar, tensors) keep their exact text
                # channel — dropping it would collapse them to a float approximation
                # and silently accept e.g. Fraction(2**53) vs Fraction(2**53 + 1).
                columns.extend((column.name, spec.field) for column in spec.columns if column.kind == "float")
            else:
                columns.extend((column.name, spec.field) for column in spec.columns)
        for spec in plan.skipped_nested:
            if (
                spec.role == "reference"
                and spec.target is not None
                and resolve_schema(spec.target).dedup in ("content_id", "by_value")
            ):
                columns.append((spec.columns[0].name, spec.field))
        return columns

    def _first_differing_field(
        self, table: sqlalchemy.Table, compare_columns: list[tuple[str, str]], key: str
    ) -> str | None:
        """The schema field of the first compared column that differs within the conflicting group."""
        for column, field_name in compare_columns:
            distinct = self._connection.execute(
                sqlalchemy.select(sqlalchemy.func.count(sqlalchemy.distinct(table.c[column]))).where(
                    table.c[CONTENT_ID_COLUMN] == key
                )
            ).scalar_one()
            if distinct is not None and int(distinct) > 1:
                return field_name
        return None

    # -- orphan sweep

    def _sweep_orphans(self) -> None:
        """Delete rows no longer reachable from a surviving top-level record.

        A collapsed duplicate parent drops its subtree; descendants that other
        surviving records also reach stay, but a duplicate's private ``dedup="none"``
        (or otherwise non-deduplicated) descendants become unreachable and must go,
        matching the per-record ``save()`` loop's result.
        """
        reach = self._reachable_table()
        seeds = self._survivor_seeds()
        self._insert_reach(reach, seeds)
        self._close_reachability(reach)
        self._delete_unreached(reach)
        self._connection.execute(sqlalchemy.schema.DropTable(reach, if_exists=True))

    def _survivor_seeds(self) -> list[tuple[str, int]]:
        seeds: set[tuple[str, int]] = set()
        for manifest in self._manifests:
            for table_name, block_sid in manifest.token_sid.values():
                seeds.add((table_name, self._collapse.get((table_name, block_sid), block_sid)))
        return sorted(seeds)

    def _reachable_table(self) -> sqlalchemy.Table:
        reach = sqlalchemy.Table(
            "_httk_bulk_reach",
            sqlalchemy.MetaData(),
            sqlalchemy.Column("tbl", sqlalchemy.Text, nullable=False),
            sqlalchemy.Column(SID_COLUMN, sqlalchemy.Integer, nullable=False),
        )
        self._connection.execute(sqlalchemy.schema.DropTable(reach, if_exists=True))
        self._connection.execute(sqlalchemy.schema.CreateTable(reach))
        self._ingest._staging_tables.add(reach.name)
        return reach

    def _insert_reach(self, reach: sqlalchemy.Table, seeds: list[tuple[str, int]]) -> None:
        if not seeds:
            return
        self._connection.execute(
            sqlalchemy.insert(reach), [{"tbl": table_name, SID_COLUMN: sid} for table_name, sid in seeds]
        )

    def _close_reachability(self, reach: sqlalchemy.Table) -> None:
        store = self._store
        # Forward edges: a reached row keeps every record it references, directly
        # (reference columns) or through its child rows (child-element columns).
        reference_edges: list[tuple[sqlalchemy.Table, str, str]] = []
        child_edges: list[tuple[sqlalchemy.Table, str, str, str]] = []
        for table_name, schema in self._ingest._parent_schema.items():
            table = store._table(table_name)
            for spec in schema.fields:
                if spec.role == "reference" and spec.target is not None:
                    reference_edges.append((table, spec.columns[0].name, resolve_schema(spec.target).table_name))
                elif spec.child is not None and spec.target is not None:
                    child = store._table(spec.child.table_name)
                    element_column = spec.child.element_columns[0].name
                    child_edges.append(
                        (child, f"{table_name}_{SID_COLUMN}", element_column, resolve_schema(spec.target).table_name)
                    )
        while True:
            before = self._connection.execute(
                sqlalchemy.select(sqlalchemy.func.count()).select_from(reach)
            ).scalar_one()
            for table, column, ref_table in reference_edges:
                self._grow_reach(reach, table, table.name, column, ref_table)
            for child, parent_column, element_column, ref_table in child_edges:
                self._grow_reach_via_child(reach, child, parent_column, element_column, ref_table)
            after = self._connection.execute(sqlalchemy.select(sqlalchemy.func.count()).select_from(reach)).scalar_one()
            if after == before:
                return

    def _grow_reach(
        self, reach: sqlalchemy.Table, table: sqlalchemy.Table, table_name: str, column: str, ref_table: str
    ) -> None:
        already = sqlalchemy.select(reach.c[SID_COLUMN]).where(reach.c.tbl == ref_table)
        source = (
            sqlalchemy.select(sqlalchemy.literal(ref_table).label("tbl"), table.c[column].label(SID_COLUMN))
            .join_from(
                table, reach, sqlalchemy.and_(reach.c.tbl == table_name, reach.c[SID_COLUMN] == table.c[SID_COLUMN])
            )
            .where(table.c[column].is_not(None))
            .where(table.c[column].not_in(already))
            .distinct()
        )
        self._connection.execute(sqlalchemy.insert(reach).from_select(["tbl", SID_COLUMN], source))

    def _grow_reach_via_child(
        self,
        reach: sqlalchemy.Table,
        child: sqlalchemy.Table,
        parent_column: str,
        element_column: str,
        ref_table: str,
    ) -> None:
        parent_table = parent_column[: -(len(SID_COLUMN) + 1)]
        already = sqlalchemy.select(reach.c[SID_COLUMN]).where(reach.c.tbl == ref_table)
        source = (
            sqlalchemy.select(sqlalchemy.literal(ref_table).label("tbl"), child.c[element_column].label(SID_COLUMN))
            .join_from(
                child,
                reach,
                sqlalchemy.and_(reach.c.tbl == parent_table, reach.c[SID_COLUMN] == child.c[parent_column]),
            )
            .where(child.c[element_column].is_not(None))
            .where(child.c[element_column].not_in(already))
            .distinct()
        )
        self._connection.execute(sqlalchemy.insert(reach).from_select(["tbl", SID_COLUMN], source))

    def _delete_unreached(self, reach: sqlalchemy.Table) -> None:
        store = self._store
        for table_name, schema in self._ingest._parent_schema.items():
            table = store._table(table_name)
            reached = sqlalchemy.select(reach.c[SID_COLUMN]).where(reach.c.tbl == table_name)
            self._connection.execute(sqlalchemy.delete(table).where(table.c[SID_COLUMN].not_in(reached)))
            for spec in schema.fields:
                if spec.child is None:
                    continue
                child = store._table(spec.child.table_name)
                parent_column = f"{table_name}_{SID_COLUMN}"
                surviving_parents = sqlalchemy.select(table.c[SID_COLUMN])
                self._connection.execute(
                    sqlalchemy.delete(child).where(child.c[parent_column].not_in(surviving_parents))
                )

    # -- compaction

    def _compact(self, table: sqlalchemy.Table) -> None:
        name = table.name
        start = self._ingest._initial_next_sid.get(name, 1)
        row_number = sqlalchemy.func.row_number().over(order_by=table.c[SID_COLUMN])
        statement = sqlalchemy.select(
            table.c[SID_COLUMN].label("old"),
            (row_number + (start - 1)).label("new"),
        ).where(table.c[SID_COLUMN] >= _SID_BLOCK)
        pairs = [(int(row[0]), int(row[1])) for row in self._connection.execute(statement).all()]
        surviving = self._connection.execute(sqlalchemy.select(sqlalchemy.func.count()).select_from(table)).scalar_one()
        self._ingest._inserted_count[name] = int(surviving)
        self._ingest._next_sid[name] = start + len(pairs)
        if not pairs:
            return
        for old, new in pairs:
            self._compaction[(name, old)] = new
        map_table = self._make_map_table(pairs)
        try:
            for referrer_table, column in self._referrers.get(name, ()):
                self._remap_column(referrer_table, column, map_table)
            self._remap_column(name, SID_COLUMN, map_table)
        finally:
            self._drop_map_table(map_table)

    # -- dispatch

    def _merge_dispatch(self) -> None:
        ingest = self._ingest
        for manifest in self._manifests:
            for row in manifest.dispatch:
                final_sid = self._final_sid(row.ref_table, row.block_sid)
                built: dict[str, Any] = {DISPATCH_CONTENT_ID_COLUMN: row.key}
                for column in row.all_columns:
                    built[column] = None
                built[row.column] = final_sid
                bucket = ingest._dispatch_rows.setdefault(row.dispatch_name, {})
                ingest._dispatch_family.setdefault(row.dispatch_name, self._family_named(row.family_name))
                existing = bucket.get(row.key)
                if existing is not None:
                    if existing != built:
                        raise EntryDispatchIntegrityError(
                            f"entry dispatch {row.family_name!r} maps content_id {row.key!r} "
                            f"to a conflicting backing row"
                        )
                    continue
                bucket[row.key] = built
        ingest._flush_dispatch()

    def _family_named(self, family_name: str) -> Any:
        for family in self._store.layout.families:
            if family.name == family_name:
                return family
        raise KeyError(family_name)  # pragma: no cover - families are declared up front

    def _final_sid(self, table_name: str, block_sid: int) -> int:
        keep = self._collapse.get((table_name, block_sid), block_sid)
        return self._compaction.get((table_name, keep), keep)

    # -- sid map helpers

    def _make_map_table(self, pairs: list[tuple[int, int]]) -> sqlalchemy.Table:
        map_table = sqlalchemy.Table(
            "_httk_bulk_sidmap",
            sqlalchemy.MetaData(),
            sqlalchemy.Column("old", sqlalchemy.Integer, nullable=False),
            sqlalchemy.Column("new", sqlalchemy.Integer, nullable=False),
        )
        self._connection.execute(sqlalchemy.schema.DropTable(map_table, if_exists=True))
        self._connection.execute(sqlalchemy.schema.CreateTable(map_table))
        self._ingest._staging_tables.add(map_table.name)
        self._connection.execute(sqlalchemy.insert(map_table), [{"old": old, "new": new} for old, new in pairs])
        return map_table

    def _drop_map_table(self, map_table: sqlalchemy.Table) -> None:
        self._connection.execute(sqlalchemy.schema.DropTable(map_table, if_exists=True))

    def _remap_column(self, table_name: str, column: str, map_table: sqlalchemy.Table) -> None:
        # A join-based ``UPDATE ... FROM`` (a single hash join), not a per-row
        # correlated subquery: the latter is quadratic on DuckDB and dominates
        # the whole merge on a real-scale build.
        table = self._store._table(table_name)
        self._connection.execute(
            sqlalchemy.update(table).where(table.c[column] == map_table.c.old).values({column: map_table.c.new})
        )

    def _delete_where_in_map(self, table_name: str, column: str, map_table: sqlalchemy.Table) -> None:
        table = self._store._table(table_name)
        member = sqlalchemy.select(map_table.c.old)
        self._connection.execute(sqlalchemy.delete(table).where(table.c[column].in_(member)))
