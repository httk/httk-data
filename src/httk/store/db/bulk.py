"""Bulk ingestion for :class:`~httk.store.db.store.SqlStore`.

:class:`BulkIngest` is the context manager returned by
:meth:`~httk.store.db.store.SqlStore.bulk_ingest`. It replaces the per-record
``save()`` loop: instead of one statement round-trip per row with an
in-database deduplication protocol, it encodes each object with the pure
encoders in :mod:`httk.store.db.store` (``_encode_parent_row`` and
``_encode_child_rows``), assigns sids from monotonic in-memory counters,
deduplicates set-wise, and appends buffered rows into the record tables with
executemany batches inside one transaction.

Two modes share the same encoder:

- *Empty store* (a fresh build): tables absent from the database are created
  without their separable indexes, buffered rows are appended directly, and the
  indexes (content-id uniqueness, ``ix_``/``uq_``, composite, child parent-sid)
  are built only once the stream has loaded — their creation is itself the
  uniqueness verification.
- *Populated store* (incremental append): tables that already hold rows keep
  their sid allocation above the current maximum. Each flushed chunk is staged
  into an ordinary ``bulkstage_<table>`` table and resolved set-wise against the
  target — a content-id anti-join (with in-memory
  :class:`~httk.core.storage.markers.IdentitySkip` metadata verification of the
  hits, reproducing :meth:`~httk.store.db.store.SqlStore.save`), a ``by_value``
  whole-parent-column anti-join with null-safe equality, and a sid remap that
  rewrites every still-buffered reference to a deduplicated existing sid before
  it is flushed. The ``index_strategy`` knob chooses whether existing tables
  keep their indexes during the append or rebuild them at the end.

Deduplication mirrors :meth:`~httk.store.db.store.SqlStore.save` set-wise: a
``"content_id"`` table keeps a ``content_id -> sid`` map (a hit returns the
mapped sid and buffers neither the parent row nor its children, and — unless
``verify_metadata`` is disabled — compares
:class:`~httk.core.storage.markers.IdentitySkip` metadata against the first
occurrence in memory, or against the stored row for a hit against existing data,
raising :class:`~httk.store.store_common.EntryMetadataConflictError`); a
``"by_value"`` table keeps a whole-parent-column-tuple map (a hit returns the
mapped sid with no metadata check); a ``"none"`` table always inserts.
Multi-record entry families buffer one deduplicated dispatch row per content id,
raising :class:`~httk.store.store_common.EntryDispatchIntegrityError` on a
conflicting backing.

A third, opt-in mode parallelizes the encode. ``bulk_ingest(workers=N)`` with
``N > 1`` forks a pool of worker processes (the ``fork`` start method, so each
inherits the unpicklable store and never touches its database) and pickles each
saved object onto a shared task queue. Every worker runs the *same* pure
encoders against a per-worker :class:`~httk.store.store_common.SaveProjection`,
allocating sids from a disjoint block and writing per-table shard files
(pyarrow Parquet on DuckDB — the optional ``parallel`` extra — or a native
SQLite database per worker). The main process then merges the shards inside the
ingest's spanning transaction: it loads every shard under the block sids,
collapses cross-worker duplicates set-wise (content-id and by_value), verifies
each surviving collision's identity-excluded metadata with a grouped scan,
sweeps rows orphaned by a collapsed duplicate's subtree, and renumbers the
survivors to a compact range.
Parallel mode targets the offline *build* of a store and requires a physically
empty target; incremental appends into a populated store stay on the serial
path. The implementation lives in :mod:`httk.store.db.bulk_parallel`; see its
module docstring for the full contract. On a fresh supported store, serial
``finalize="auto"`` selects the deferred finalizer; parallel ``auto`` remains
on the parity merge.

Identity caches are not populated by bulk ingestion (documented best-effort);
they are cleared on failure.

Two behaviors diverge from the per-record ``save()`` loop:

- *Returned sids are provisional until the context exits.* A record that
  deduplicates against a row the store already held is remapped to that existing
  sid at flush, so the sid :meth:`BulkIngest.save` returned is not durable for
  such a record. :meth:`BulkIngest.resolved_sid`, given the stored record type
  and a returned sid, maps it to its final stored sid once the context has
  exited cleanly.
- *Nested metadata-conflict messages carry the descendant's path.* Because the
  bulk encoder resolves referenced and child records eagerly and only discovers
  their existing-row hits at flush, an :class:`~httk.core.storage.markers.IdentitySkip`
  conflict reached through a ``descend`` field (a non-skipped reference whose
  target itself carries skipped metadata) is reported against the descendant
  record (at its own path, e.g. ``"Leaf.note"``) rather than the ancestor field
  path save() would use (e.g. ``"Root.primary.note"``). The exception type and
  abort-and-roll-back behavior are identical; the conflict message differs in
  its path prefix and, for some nested ``None``/length mismatches, in its
  detail text.
- *DuckDB never drops an existing table's indexes.* DuckDB reserves a dropped
  index's name until commit, so an in-transaction drop-then-recreate of the same
  index is rejected. Under ``index_strategy="rebuild"`` (or an ``"auto"`` rebuild
  decision) DuckDB therefore keeps the indexes in place through the append —
  relying on their incremental maintenance — and verifies content-id uniqueness
  with a duplicate-scan at finalize instead of an index rebuild. SQLite drops the
  separable indexes up front and recreates them at the end, where the creation is
  itself the uniqueness verification. Both leave the same final indexes present.
"""

import contextlib
import tempfile
import time
from collections.abc import Callable, Iterable, Mapping
from types import TracebackType
from typing import Any, Literal, Self, cast

import sqlalchemy
from httk.core.storage import (
    StorageProjectionCycleError,
    content_id,
    project_storage_record,
    resolve_storage_record,
)

from httk.store.db.graph import LogicalEdgeGraph
from httk.store.db.layout import METADATA_TABLE_NAME, actual_schema_objects, backend_facts_for_dialect
from httk.store.db.mapping import (
    CONTENT_ID_COLUMN,
    DISPATCH_CONTENT_ID_COLUMN,
    REPLACED_BY_COLUMN,
    ROLE_COLUMN,
    SID_COLUMN,
    TS_END_COLUMN,
    TS_START_COLUMN,
    backing_dispatch_column_name,
    entry_dispatch_table_name,
)
from httk.store.db.schema import FieldSpec, TableSchema, resolve_schema
from httk.store.db.store import (
    SqlStore,
    _encode_child_rows,
    _encode_parent_row,
    _encode_promoted_descendants,
    _field_path,
    _metadata_scalar_equal,
)
from httk.store.store_common import (
    EntryDispatchIntegrityError,
    EntryMetadataConflictError,
    SaveProjection,
    _metadata_plan,
    reject_cursor_proxy,
)

__all__ = ["BulkIngest"]

# The staged-rows-to-existing-rows ratio above which ``index_strategy="auto"``
# rebuilds an existing table's separable indexes rather than appending through
# them: a rebuild is chosen once ``staged_rows * _AUTO_REBUILD_DIVISOR`` exceeds
# the table's pre-ingest row count (i.e. staged rows exceed a quarter of the
# existing rows). This is a placeholder threshold; P4's benchmark phase
# calibrates it against measured index-build versus keep-and-append costs.
_AUTO_REBUILD_DIVISOR = 4


class _SerialDeferredStage:
    """Own a dependency-free serial stage directory and its worker encoder.

    Non-Parquet serial stages intentionally retain their identity/dispatch
    indexes: returning the same sid for a duplicate is part of the serial
    ``save()`` contract, so that retention is contract-inherent and bounded by
    the serial use-case.  The Parquet scale path spills all non-contract
    auxiliary occurrence data instead.
    """

    def __init__(self, temporary_directory: tempfile.TemporaryDirectory[str], encoder: Any) -> None:
        self._temporary_directory = temporary_directory
        self._encoder = encoder
        self._finished: Any = None

    def save(self, token: int, obj: Any, as_record: type | None, promote: frozenset[type]) -> int:
        # _WorkerEncoder deliberately exposes the assigned occurrence sid via
        # its token map, while retaining every metadata-bearing duplicate.
        return self._encoder.save(token, obj, as_record, promote)

    def finish(self) -> Any:
        if self._finished is None:
            self._finished = self._encoder.finish()
            self._finished.worker_index = 0
        return self._finished

    def close(self) -> None:
        self._temporary_directory.cleanup()


def _sid_sequence(table: sqlalchemy.Table) -> sqlalchemy.Sequence | None:
    """Return the sid primary key's attached sequence, or ``None`` for a child table."""
    if SID_COLUMN not in table.c:
        return None
    default = table.c[SID_COLUMN].default
    return default if isinstance(default, sqlalchemy.Sequence) else None


class BulkIngest:
    """Append a stream of storable objects into a store, then verify its indexes.

    Instances are produced by :meth:`~httk.store.db.store.SqlStore.bulk_ingest`
    and used as a context manager. Inside the ``with`` block, :meth:`save`
    encodes and buffers objects; on clean exit the buffered rows are flushed
    (staged and resolved set-wise against any existing rows), the separable
    indexes are created or rebuilt (verifying uniqueness), DuckDB sid sequences
    are resynchronized, per-table row counts are asserted against the encoder's
    bookkeeping, and the single spanning transaction commits on SQLite and
    DuckDB. On those backends, any exception rolls the transaction back, drops
    every table the context created, restores any index the context dropped,
    removes staging tables, and clears the store's identity caches, leaving the
    store exactly as it was before the context opened. ClickHouse is
    fresh-store-only and fail-closed through its KeeperMap marker; its P3
    loader/finalizer owns the durable ingest path.

    :param store: The store to ingest into.
    :param chunk_size: The number of top-level saves buffered before a flush.
    :param verify_metadata: Whether content-id hits compare identity-excluded metadata.
    :param index_strategy: How existing tables' separable indexes are handled during the append.
    :param on_progress: An optional ``(records_buffered_total, rows_flushed_total)`` callback invoked after each flush.
    :param workers: The number of worker processes; ``1`` (the default) is the serial path, ``>1`` encodes in parallel and merges shards.
    :param finalize: The finalization profile: ``"auto"`` selects the deferred finalizer on a fresh supported store for serial ingestion and the parity merge otherwise; ``"parity"`` and ``"deferred"`` force the respective profile.
    :param track_sids: Retain the per-save provisional-to-final sid mapping.  Disable it for bounded-memory offline builds when callers do not need :meth:`resolved_sid`.
    """

    def __init__(
        self,
        store: SqlStore,
        *,
        chunk_size: int = 100_000,
        verify_metadata: bool = True,
        index_strategy: Literal["auto", "keep", "rebuild"] = "auto",
        on_progress: Callable[[int, int], None] | None = None,
        workers: int = 1,
        finalize: Literal["auto", "parity", "deferred"] = "auto",
        track_sids: bool = True,
    ) -> None:
        if chunk_size < 1:
            raise ValueError("chunk_size must be a positive integer")
        if index_strategy not in ("auto", "keep", "rebuild"):
            raise ValueError("index_strategy must be one of 'auto', 'keep', or 'rebuild'")
        if workers < 1:
            raise ValueError("workers must be a positive integer")
        if finalize not in ("auto", "parity", "deferred"):
            raise ValueError("finalize must be one of 'auto', 'parity', or 'deferred'")
        if workers > 1 and on_progress is not None:
            raise ValueError(
                "on_progress is not supported with workers>1: worker processes encode asynchronously, "
                "so per-flush buffered/flushed counts are not observable from the main process"
            )
        self._store = store
        self._chunk_size = chunk_size
        self._verify_metadata = verify_metadata
        self._index_strategy = index_strategy
        self._on_progress = on_progress
        self._workers = workers
        self._parallel = workers > 1
        self._requested_finalize = finalize
        self._track_sids = track_sids
        self._finalize_profile = "parity"
        self._store_timestamp: int | None = None
        self._deferred = False
        # Parallel-mode state (unused on the serial path).
        self._controller: Any = None
        self._serial_stage: Any = None
        self._serial_public_stage_sid: dict[tuple[str, int], int] = {}
        self._serial_public_content: dict[tuple[str, str], int] = {}
        self._serial_public_value: dict[tuple[str, tuple[tuple[str, object], ...]], int] = {}
        self._serial_public_next: dict[str, int] = {}
        self._serial_next_token = 0
        self._deferred_top_types: set[type] = set()
        self._next_token = 0
        self._schema_graph_seen: set[type] = set()
        # External database aliases attached by deferred backends; detached in
        # _release_connection after the transaction closes.
        self._parallel_attached: list[str] = []
        self._connection: sqlalchemy.Connection | None = None
        self._transaction: Any = None
        self._closed = False
        self._entered = False

        # Physical bookkeeping.
        self._preexisting: frozenset[str] = frozenset()
        self._created: list[str] = []
        self._created_set: set[str] = set()
        self._ensured: set[type] = set()
        self._existing_scanned: set[str] = set()
        self._existing_row_count: dict[str, int] = {}
        self._initial_next_sid: dict[str, int] = {}
        self._dropped_indexes: list[sqlalchemy.Index] = []
        self._index_decided: set[str] = set()
        self._rebuild_scan_tables: set[str] = set()
        self._staging_tables: set[str] = set()
        self._marker_active = False
        self._marker_value: str | None = None
        self._entry_catalog: tuple[tuple[str, ...], ...] | None = None
        self._bulk_lock_held = False
        self._bulk_lifecycle_guard: Any = None
        self._preserve_clickhouse_fence = False
        self._clickhouse_stage_tables: dict[str, str] = {}

        # Encoder bookkeeping, keyed by table name.
        self._next_sid: dict[str, int] = {}
        self._rows: dict[str, list[dict[str, Any]]] = {}
        self._inserted_count: dict[str, int] = {}
        self._content_index: dict[str, dict[str, int]] = {}
        self._value_index: dict[str, dict[tuple[Any, ...], int]] = {}
        self._meta_values: dict[str, Mapping[str, object]] = {}
        self._meta_sources: dict[str, tuple[type, Any]] = {}
        self._parent_schema: dict[str, TableSchema] = {}
        self._dispatch_rows: dict[str, dict[str, dict[str, Any]]] = {}
        self._dispatch_family: dict[str, Any] = {}

        # Top-level records saved in the current (not-yet-flushed) chunk, keyed
        # by ``(table, sid)``; the garbage collector's roots for orphan sweeping.
        self._chunk_roots: list[tuple[str, int]] = []

        # Provisional-to-final sid resolution, keyed by ``(table, sid)`` (sids are
        # per table, so a bare int is ambiguous). One entry per remapped hit; all
        # other returned sids resolve to themselves.
        self._returned_sids: set[tuple[str, int]] = set()
        self._resolved_map: dict[tuple[str, int], int] = {}
        self._final_sids_ready = False
        # Debug/benchmark surface: populated only by deferred finalization.
        self.finalize_timings: dict[str, float] = {}

        # Progress counters.
        self._records_total = 0
        self._rows_flushed_total = 0
        self._since_flush = 0

    # ------------------------------------------------------------------ context management

    def __enter__(self) -> Self:
        store = self._store
        if store.write_profile == "degraded":
            raise RuntimeError(
                "bulk_ingest is not supported by the SQLite degraded write profile; use ordered save() or fsck()"
            )
        if store._current_connection() is not None:
            raise RuntimeError(
                "bulk_ingest cannot be opened inside an open store.transaction() or write scope on this thread; "
                "the ingest owns its own spanning transaction"
            )
        store._claim_bulk_context()
        try:
            self._after_bulk_context_claim()
        except BaseException:
            store._release_bulk_context()
            raise
        if store.write_profile == "bulk-fenced":
            try:
                store._mutation_lock.acquire()
            except BaseException:
                store._release_bulk_context()
                raise
            self._bulk_lock_held = True
            try:
                self._bulk_lifecycle_guard = store._degraded_lifecycle_guard()
                self._bulk_lifecycle_guard.__enter__()
            except BaseException:
                self._bulk_lifecycle_guard = None
                try:
                    self._release_bulk_ownership()
                finally:
                    store._release_bulk_context()
                raise
        connection = None
        try:
            with store._database.engine.connect() as probe:
                preexisting = self._scan_store(probe)
                self._entry_catalog = self._catalog_snapshot(probe)
                physically_empty = self._physically_empty(probe, preexisting)
                self._store_timestamp = store._capture_store_timestamp(probe)
            self._preexisting = preexisting
            self._select_finalize_profile()
            store._check_mutation_policy(
                "bulk_ingest",
                empty_deferred_bulk=self._deferred and physically_empty,
            )
            if store._database.engine.dialect.name == "clickhousedb" and self._deferred:
                # P2's deliberate boundary is after both durable KeeperMap
                # mutations. __enter__ raises here, so Python cannot invoke
                # __exit__; preserving these values is the crash-equivalent
                # path used by the recovery tests.
                self._acquire_clickhouse_lease()
                self._after_clickhouse_lease_acquired()
                self._write_ingest_marker()
                self._marker_active = True
                self._clickhouse_p3_boundary()
            if self._deferred and preexisting:
                raise RuntimeError(
                    'bulk_ingest(finalize="deferred") requires a physically empty store; use finalize="parity"'
                )
            if self._deferred:
                self._validate_declared_deferred_metadata()
            # The worker pool is forked before any main-store transaction opens,
            # so no child inherits an open database connection or transaction.
            if self._parallel:
                self._start_workers()
            if physically_empty and not self._marker_active:
                self._write_ingest_marker()
                self._marker_active = True
            connection = store._database.engine.connect()
            transaction = None if self._deferred else connection.begin()
        except BaseException:
            try:
                if connection is not None:
                    self._release_connection(connection)
                self._close_workers()
                if not self._preserve_clickhouse_fence:
                    self._clean_up_after_failure()
                    self._clear_marker_after_failure()
            finally:
                try:
                    self._release_bulk_ownership()
                finally:
                    store._release_bulk_context()
            raise
        self._transaction = transaction
        self._connection = connection
        try:
            if self._parallel:
                self._require_empty_store(connection)
        except BaseException:
            try:
                if transaction is not None:
                    transaction.rollback()
                self._release_connection(connection)
                self._close_workers()
                self._clean_up_after_failure()
                self._clear_marker_after_failure()
            finally:
                try:
                    self._release_bulk_ownership()
                finally:
                    store._release_bulk_context()
            raise
        self._entered = True
        return self

    def _clickhouse_p3_boundary(self) -> None:
        """P2's durable fence has been acquired; P3 continues with Parquet staging."""

    def _after_bulk_context_claim(self) -> None:
        """Test seam after atomic admission and before any lifecycle ownership."""

    def _acquire_clickhouse_lease(self) -> None:
        """Durably acquire the ClickHouse lease before the marker operation."""
        with self._store._database.engine.begin() as connection:
            self._store._ensure_degraded_lease(connection)

    def _after_clickhouse_lease_acquired(self) -> None:
        """Fault seam immediately after lease acquisition and before marker insert."""

    def _select_finalize_profile(self) -> None:
        """Resolve ``auto`` after the physical-empty probe, before any mutation.

        At current batch scales the parallel in-database merge is faster, so
        parallel ``auto`` stays on parity while serial ``auto`` gains about 36%
        from deferred finalization.
        """
        requested = self._requested_finalize
        if self._store.backend_facts.stage_load == "client-stream":
            if requested == "parity":
                raise RuntimeError("ClickHouse bulk_ingest is deferred-only; finalize='parity' is not supported")
            self._finalize_profile = "deferred"
            self._deferred = True
            return
        override = type(self._store).bulk_ingest_finalize_default
        if requested == "auto" and override != "auto":
            requested = override
        if requested == "deferred":
            if not self._store.backend_facts.supports_deferred_finalize:
                raise RuntimeError(
                    'bulk_ingest(finalize="deferred") is not supported by this backend; use finalize="parity"'
                )
            self._finalize_profile = "deferred"
            self._deferred = True
            return
        if (
            requested == "auto"
            and self._workers == 1
            and not self._preexisting
            and self._store.backend_facts.supports_deferred_finalize
            and not self._declared_unsupported_metadata_reason()
        ):
            # A declared unsupported shape is statically knowable.  Keep auto
            # backwards compatible rather than consuming a stream only to fail.
            self._finalize_profile = "deferred"
            self._deferred = True

    def _declared_unsupported_metadata_reason(self) -> str | None:
        if not self._verify_metadata:
            return None
        from httk.store.db.bulk_parallel import unsupported_metadata_reason

        seen: set[type] = set()

        def visit(record_type: type) -> str | None:
            if record_type in seen:
                return None
            seen.add(record_type)
            reason = unsupported_metadata_reason(record_type)
            if reason is not None:
                return reason
            schema = resolve_schema(record_type)
            for target in schema.referenced_classes():
                nested = visit(target)
                if nested is not None:
                    return nested
            return None

        for family in self._store.layout.families:
            for record_type in family.records:
                reason = visit(record_type)
                if reason is not None:
                    return reason
        return None

    def _validate_declared_deferred_metadata(self) -> None:
        reason = self._declared_unsupported_metadata_reason()
        if reason is not None:
            raise ValueError(
                "bulk_ingest(finalize=\"deferred\") cannot verify this identity-excluded metadata shape: "
                f"{reason}. Use finalize=\"parity\" for records of this kind, or open with verify_metadata=False."
            )

    def _start_workers(self) -> None:
        """Validate the parallel prerequisites and fork the worker pool."""
        from httk.store.db.bulk_parallel import ParallelController

        backend = self._store._database.engine.dialect.name
        if self._store.backend_facts.parallel_shard_format == "parquet":
            try:
                import importlib

                importlib.import_module("pyarrow")
            except ImportError as error:
                raise ImportError(
                    "bulk_ingest(workers>1) with Parquet staging needs pyarrow; "
                    "install the 'httk-store[parallel]' extra to use it"
                ) from error
        self._controller = ParallelController(
            self._store,
            workers=self._workers,
            chunk_size=self._chunk_size,
            backend=("parquet" if self._store.backend_facts.parallel_shard_format == "parquet" else backend),
            track_sids=self._track_sids,
            store_timestamp=self._store_timestamp,
            spill_deferred_auxiliary=(self._deferred and self._store.backend_facts.parallel_shard_format == "parquet"),
        )
        self._controller.start()

    def _close_workers(self) -> None:
        if self._controller is not None:
            self._controller.close()
            self._controller = None

    def _require_empty_store(self, connection: sqlalchemy.Connection) -> None:
        """Refuse parallel ingest into a store the merge cannot treat as a clean build.

        Parallel ingest requires a physically empty application store. On DuckDB
        this also avoids loading a pre-existing table into the offline merge;
        incremental appends remain on the serial path.
        """
        if not self._preexisting:
            return
        if connection.dialect.name == "duckdb":
            raise RuntimeError(
                "bulk_ingest(workers>1) on a DuckDB store requires no pre-existing application tables "
                f"(found {', '.join(sorted(self._preexisting))}); drop them or use workers=1."
            )
        for name in self._preexisting:
            count = connection.execute(sqlalchemy.text(f'SELECT count(*) FROM "{name}"')).scalar_one()
            if int(count) > 0:
                raise RuntimeError(
                    "bulk_ingest(workers>1) requires a physically empty store; "
                    f"table {name!r} already holds rows. Use workers=1 for incremental appends."
                )

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Finalize a clean ingest, or roll back and undo what a failed one did.

        :param exc_type: The exception class raised in the context, if any.
        :param exc: The exception instance raised in the context, if any.
        :param traceback: The traceback for the context exception, if any.
        :return: None.
        """
        store = self._store
        transaction = self._transaction
        connection = self._connection
        self._closed = True
        try:
            if self._deferred:
                if exc_type is None:
                    try:
                        self._deferred_finalize()
                    except BaseException as error:
                        # An Arrow response can be lost after the server has
                        # accepted a shard.  Retaining its nonce marker is the
                        # only safe recovery; never replay that ambiguous load.
                        from httk.store.db.clickhouse import (
                            ClickHouseBulkIntegrityError,
                            ClickHouseUncertainInsertError,
                        )

                        if isinstance(
                            error, (ClickHouseUncertainInsertError, ClickHouseBulkIntegrityError)
                        ) or not isinstance(error, Exception):
                            self._preserve_clickhouse_fence = True
                        self._release_connection(connection)
                        self._clean_up_after_failure()
                        if not self._preserve_clickhouse_fence:
                            self._clear_marker_after_failure()
                        raise
                    self._release_connection(connection)
                    self._clear_ingest_marker()
                    store._tables_present.update(self._created)
                    store._advance_store_timestamp_mark(self._store_timestamp)
                    self._final_sids_ready = True
                    return
                self._release_connection(connection)
                self._clean_up_after_failure()
                self._clear_marker_after_failure()
                return
            if exc_type is None:
                try:
                    self._finalize()
                except BaseException:
                    transaction.rollback()
                    self._release_connection(connection)  # detach shards, then release to the pool
                    self._clean_up_after_failure()
                    self._clear_marker_after_failure()
                    raise
                try:
                    transaction.commit()
                except BaseException:
                    # A failing commit still needs the created tables dropped,
                    # dropped indexes restored, and staging tables removed.
                    self._release_connection(connection)
                    self._clean_up_after_failure()
                    self._clear_marker_after_failure()
                    raise
                self._release_connection(connection)
                self._clear_ingest_marker()
                store._tables_present.update(self._created)
                store._advance_store_timestamp_mark(self._store_timestamp)
                self._final_sids_ready = True
                return
            transaction.rollback()
            self._release_connection(connection)
            self._clean_up_after_failure()
            self._clear_marker_after_failure()
        finally:
            try:
                self._release_connection(connection)  # idempotent: no-op if already released
            finally:
                try:
                    if self._serial_stage is not None:
                        self._serial_stage.close()
                finally:
                    self._serial_stage = None
                    try:
                        self._close_workers()
                    finally:
                        self._connection = None
                        self._transaction = None
                        try:
                            try:
                                self._release_bulk_ownership()
                            finally:
                                self._before_bulk_context_release()
                        finally:
                            store._release_bulk_context()

    def _release_bulk_ownership(self) -> None:
        """Release the in-memory bulk mutex and lifecycle guard, never the lease."""
        guard = self._bulk_lifecycle_guard
        self._bulk_lifecycle_guard = None
        try:
            if guard is not None:
                guard.__exit__(None, None, None)
        finally:
            if self._bulk_lock_held:
                self._bulk_lock_held = False
                self._store._mutation_lock.release()

    def _write_ingest_marker(self) -> None:
        """Commit ``ingest_state=bulk-ingest`` before an empty-store mutation."""
        with self._store._database.engine.begin() as connection:
            if self._store.write_profile == "bulk-fenced":
                from httk.store.db.clickhouse import write_ingest_marker

                self._store._ensure_degraded_lease(connection)
                self._marker_value = write_ingest_marker(connection, self._store._lease_value or "")
            else:
                connection.execute(
                    sqlalchemy.text(
                        "INSERT INTO \"_httk_store_metadata\" (key, value) VALUES ('ingest_state', 'bulk-ingest')"
                    )
                )

    def _clear_ingest_marker(self) -> None:
        """Clear the marker only after all successful finalize work has committed."""
        if not self._marker_active:
            return
        with self._store._database.engine.begin() as connection:
            if self._store.write_profile == "bulk-fenced":
                from httk.store.db.clickhouse import clear_ingest_marker, verify_lease

                verify_lease(connection, self._store._lease_value or "")
                self._before_clickhouse_marker_clear()
                clear_ingest_marker(connection, self._marker_value)
            else:
                connection.execute(sqlalchemy.text('DELETE FROM "_httk_store_metadata" WHERE key = \'ingest_state\''))
        self._marker_active = False
        self._marker_value = None

    def _before_clickhouse_marker_clear(self) -> None:
        """Fault seam immediately before the exact ClickHouse marker delete."""

    def _before_clickhouse_map_swap(self, table: str, boundary: str) -> None:
        """Fault seam at each durable ClickHouse map-rename boundary."""

    def _after_clickhouse_stage_load(self) -> None:
        """Fault seam after all Arrow stage inserts and before finalization."""

    def _after_clickhouse_projection(self) -> None:
        """Fault seam after durable projection and before working-table cleanup."""

    def _after_clickhouse_cleanup(self) -> None:
        """Fault seam after stage/working cleanup and before physical validation."""

    def _before_clickhouse_integrity_verification(self) -> None:
        """Raw-connection test seam before metadata-derived integrity checks."""

    def _after_clickhouse_physical_validation(self) -> None:
        """Fault seam after physical validation and before marker clear."""

    def _before_bulk_context_release(self) -> None:
        """Fault seam while admission remains closed during teardown."""

    def _clear_marker_after_failure(self) -> None:
        """Clear a failure marker only after restoring the complete entry catalog."""
        if not self._marker_active:
            return
        try:
            with self._store._database.engine.connect() as connection:
                if self._entry_catalog != self._catalog_snapshot(connection):
                    return
            if self._entry_catalog is None:
                return
            self._clear_ingest_marker()
        except BaseException:
            # The original ingest exception remains primary; the marker must
            # stay if emptiness or marker cleanup cannot be verified.
            return

    def _release_connection(self, connection: sqlalchemy.Connection | None) -> None:
        """Detach external stages on ``connection`` and return it to the pool (idempotent)."""
        if connection is None or connection.closed:
            return
        if self._parallel_attached:
            # Best-effort on the raw DB-API connection (a stale alias must never
            # mask the ingest's own outcome); ``exec_driver_sql`` would open a new
            # transaction, which DETACH forbids.
            with contextlib.suppress(Exception):
                raw: Any = connection.connection.driver_connection
                for alias in self._parallel_attached:
                    with contextlib.suppress(Exception):
                        raw.execute(f"DETACH DATABASE {alias}")
            self._parallel_attached = []
        if connection.dialect.name == "duckdb" and self._entry_catalog is not None:
            raw = connection.connection.driver_connection
            attached = tuple(
                sorted(str(row[0]) for row in raw.execute("SELECT database_name FROM duckdb_databases()").fetchall())
            )
            if attached != tuple(sorted(self._entry_catalog[2])):
                connection.close()
                raise RuntimeError("bulk_ingest failed to restore the DuckDB attached-database set")
        connection.close()

    def _scan_store(self, connection: sqlalchemy.Connection) -> frozenset[str]:
        """Return the application tables that already exist in the store.

        Per-table sid maxima and row counts are recorded lazily when a
        pre-existing table is first registered in :meth:`_ensure_tables`, so
        this scan only enumerates the physical tables (excluding the store's
        metadata marker).

        :param connection: The ingest transaction's connection.
        :return: The names of application tables present at context entry.
        """
        preexisting: set[str] = set()
        for name, kinds in actual_schema_objects(connection).items():
            if "table" not in kinds or name == METADATA_TABLE_NAME:
                continue
            preexisting.add(name)
        return frozenset(preexisting)

    def _physically_empty(self, connection: sqlalchemy.Connection, tables: Iterable[str]) -> bool:
        """Whether all application tables are empty, including pre-created SQLite tables."""
        return all(
            int(connection.execute(sqlalchemy.text(f'SELECT count(*) FROM "{name}"')).scalar_one()) == 0
            for name in tables
        )

    @staticmethod
    def _catalog_snapshot(connection: sqlalchemy.Connection) -> tuple[tuple[str, ...], ...]:
        """Capture durable and connection-local catalog state for marker recovery."""
        objects = tuple(
            sorted(f"{name}:{','.join(sorted(kinds))}" for name, kinds in actual_schema_objects(connection).items())
        )
        facts = backend_facts_for_dialect(connection.dialect.name)
        if facts.system_catalog == "sqlite":
            temporary = tuple(
                f"{row[0]}:{row[1]}"
                for row in connection.execute(
                    sqlalchemy.text(
                        "SELECT name, type FROM sqlite_temp_master WHERE name NOT LIKE 'sqlite_%' ORDER BY name"
                    )
                )
            )
            attached = tuple(
                f"{row[1]}:{row[2]}" for row in connection.execute(sqlalchemy.text("PRAGMA database_list"))
            )
        elif facts.system_catalog == "duckdb":
            temporary = ()
            attached = tuple(
                str(row[0])
                for row in connection.execute(
                    sqlalchemy.text("SELECT database_name FROM duckdb_databases() ORDER BY database_name")
                )
            )
        else:
            temporary = ()
            attached = ()
        return objects, temporary, attached

    def _clean_up_after_failure(self) -> None:
        """Undo a failed ingest: drop created and staging tables, restore dropped indexes, clear caches."""
        store = self._store
        store._clear_identity_caches()
        if not self._created and not self._dropped_indexes and not self._staging_tables:
            return
        # The spanning transaction has already unwound. This ordering matters
        # for SQLite, whose DDL can survive SQLAlchemy's outer rollback; opening
        # the cleanup transaction while the original one is still active would
        # merely fail against its shared connection. IF EXISTS keeps the DuckDB
        # path (already rolled back) harmless.
        try:
            with store._database.engine.begin() as cleanup:
                for name in self._staging_tables:
                    cleanup.execute(sqlalchemy.text(f'DROP TABLE IF EXISTS "{name}"'))
                for name in reversed(self._created):
                    table = store._table(name)
                    cleanup.execute(sqlalchemy.schema.DropTable(table, if_exists=True))
                    if cleanup.dialect.name in ("duckdb", "postgresql"):
                        sequence = _sid_sequence(table)
                        if sequence is not None:
                            cleanup.execute(sqlalchemy.text(f'DROP SEQUENCE IF EXISTS "{sequence.name}"'))
                for index in self._dropped_indexes:
                    # The rollback restored the table's original rows, so the
                    # unique index rebuilds cleanly; drop-then-create tolerates a
                    # dialect that kept the DROP inside the rolled-back span.
                    cleanup.execute(sqlalchemy.schema.DropIndex(index, if_exists=True))
                    cleanup.execute(sqlalchemy.schema.CreateIndex(index))
        except BaseException:
            # A residual object would refuse a later reopen; preserve the
            # original failure rather than masking it with a cleanup error.
            return

    # ------------------------------------------------------------------ saving

    def save(
        self,
        obj: Any,
        *,
        as_record: type | None = None,
        promote: type | Iterable[type] | None = None,
    ) -> int:
        """Encode and buffer ``obj``, returning its assigned or deduplicated sid.

        Mirrors :meth:`~httk.store.db.store.SqlStore.save`: an opted-in domain
        object is projected through its exact ``__httk_storage_record__`` and
        ``as_record`` selects an alternate record representation.

        The returned sid is **provisional** while the context is open. A newly
        inserted object keeps its returned sid, but an object that deduplicates
        against a row the store already held is remapped to that existing sid at
        the next flush, so its provisional sid is not the durable identifier.
        After the context exits cleanly, :meth:`resolved_sid` maps any returned
        sid — provisional or final — to the durable stored sid.

        :param obj: The object to store.
        :param as_record: The alternate record representation to use, if any.
        :param promote: A record class, or iterable of record classes, whose nested occurrences are also made
            top-level entries. The classes must be reachable from the selected outer record schema.
        :return: The provisional sid (see :meth:`resolved_sid` for the durable one).
        :raises RuntimeError: If the bulk context is not open.
        :raises TypeError: If ``obj`` is a cursor row that must be materialized first.
        :raises httk.store.store_common.EntryMetadataConflictError: If a content-id hit has conflicting metadata.
        :raises httk.store.store_common.EntryDispatchIntegrityError: If a dispatch content id maps to a conflicting backing.
        :raises httk.core.storage.identity.StorageProjectionCycleError: If projection reaches a reference cycle.
        """
        if not self._entered or self._closed:
            raise RuntimeError("bulk_ingest().save() is only usable inside an open bulk context")
        reject_cursor_proxy(obj)
        record_type = resolve_storage_record(obj, as_record=as_record)
        promoted = self._promoted_types(record_type, promote)
        if self._parallel:
            return self._parallel_save(obj, as_record, record_type, promoted)
        if self._deferred:
            return self._deferred_serial_save(obj, as_record, record_type, promoted)
        projection = SaveProjection(store_timestamp=self._store_timestamp)
        occurrences: list[tuple[type, Any, int]] = []
        seen: set[tuple[type, int, int]] = set()
        sid = self._encode(record_type, obj, projection, "", promoted, occurrences, seen)
        root = (record_type, id(obj), sid)
        if root not in seen:
            occurrences.append((record_type, obj, sid))
        for occurrence_type, source, occurrence_sid in occurrences:
            self._promote_occurrence(occurrence_type, source, projection, occurrence_sid)
        table_name = resolve_schema(record_type).table_name
        self._returned_sids.add((table_name, sid))
        self._records_total += 1
        self._since_flush += 1
        if self._since_flush >= self._chunk_size:
            self._flush()
        return sid

    def _deferred_serial_save(
        self,
        obj: Any,
        as_record: type | None,
        record_type: type,
        promote: frozenset[type],
    ) -> int:
        """Stage one serial occurrence without importing the parallel extra.

        The worker encoder is deliberately reused here: it is the only encoder
        which retains duplicate occurrences for the later grouped conflict
        scan.  Its SQLite writer is a dependency-free external artifact even
        when the target database is DuckDB.
        """
        from httk.store.db.bulk_parallel import _WorkerConfig, _WorkerEncoder

        self._record_schema_graph(record_type)
        self._deferred_top_types.add(record_type)
        self._deferred_top_types.update(promote)
        if self._serial_stage is None:
            temp = tempfile.TemporaryDirectory(prefix="httk_deferred_")
            stage_format = self._store.backend_facts.serial_stage_format
            stage_backend = (
                "duckdb-stage"
                if stage_format == "duckdb-attach"
                else ("parquet" if stage_format == "parquet" else "sqlite")
            )
            config = _WorkerConfig(
                chunk_size=self._chunk_size,
                shard_dir=temp.name,
                backend=stage_backend,
                track_sids=self._track_sids,
                store_timestamp=self._store_timestamp,
                spill_deferred_auxiliary=(self._deferred and stage_format == "parquet"),
            )
            self._serial_stage = _SerialDeferredStage(temp, _WorkerEncoder(self._store, 0, config))
        token = self._serial_next_token
        self._serial_next_token += 1
        stage_sid = self._serial_stage.save(token, obj, as_record, promote)
        table_name = resolve_schema(record_type).table_name
        # Worker block sids are deliberately never public.  Removing the
        # single serial block offset preserves the ordinary serial return value
        # for the common no-collapse case, while resolved_sid remains the
        # authoritative adapter after a collapse.
        public_sid = stage_sid - (1 << 26)
        if self._track_sids:
            schema = resolve_schema(record_type)
            if schema.dedup == "content_id":
                key = content_id(obj, as_record=record_type)
                public_sid = self._serial_public_content.get((table_name, key))
                if public_sid is None:
                    public_sid = self._serial_public_next.get(table_name, 1)
                    self._serial_public_next[table_name] = public_sid + 1
                    self._serial_public_content[(table_name, key)] = public_sid
            elif schema.dedup == "by_value":
                value_key = self._serial_by_value_key(record_type, obj)
                public_sid = self._serial_public_value.get((table_name, value_key))
                if public_sid is None:
                    public_sid = self._serial_public_next.get(table_name, 1)
                    self._serial_public_next[table_name] = public_sid + 1
                    self._serial_public_value[(table_name, value_key)] = public_sid
        if self._track_sids:
            self._serial_public_stage_sid.setdefault((table_name, public_sid), stage_sid)
            self._returned_sids.add((table_name, public_sid))
        self._records_total += 1
        return public_sid

    def _serial_by_value_key(self, record_type: type, obj: Any) -> tuple[tuple[str, object], ...]:
        """Build a lightweight logical parent key for the public by-value adapter.

        References are represented by their content identities rather than the
        occurrence sids assigned by the stage encoder.  Child rows are omitted,
        matching the by-value table policy; staging still retains every
        occurrence for the eventual fixpoint and conflict scan.
        """
        schema = resolve_schema(record_type)
        projection = SaveProjection(store_timestamp=self._store_timestamp)
        projected = projection.projector(record_type, obj)

        def reference_key(target: type, value: Any, _path: str) -> str:
            return content_id(value, as_record=target)

        # This adapter intentionally returns content IDs as logical key values, not allocated sids.
        values = _encode_parent_row(schema, obj, projected, "", cast(Callable[[type, Any, str], int], reference_key))
        return tuple(sorted(values.items()))

    def resolved_sid(self, record_type: type, sid: int) -> int:
        """Map a sid returned by :meth:`save` to its durable stored sid after the context exits.

        A newly inserted object's provisional sid resolves to itself; a sid that
        deduplicated against a pre-existing row resolves to that existing row's
        sid. This is the durable lookup for provisional sids (see :meth:`save`).

        Sids are allocated per table, so both the record type the sid was saved
        as (the same class :meth:`~httk.store.db.store.SqlStore.fetch` takes) and
        the sid are required to identify it unambiguously.

        :param record_type: The stored record class the sid was saved as.
        :param sid: A sid previously returned by :meth:`save`.
        :return: The durable stored sid.
        :raises RuntimeError: If the bulk context has not yet exited cleanly (resolution is incomplete).
        :raises KeyError: If ``(record_type, sid)`` was never returned by this ingest's :meth:`save`.
        """
        if not self._final_sids_ready:
            raise RuntimeError("resolved_sid is only available after the bulk_ingest context has exited cleanly")
        if not self._track_sids:
            raise RuntimeError("resolved_sid is unavailable because bulk_ingest(track_sids=False) did not retain sids")
        table_name = resolve_schema(record_type).table_name
        if (table_name, sid) not in self._returned_sids:
            raise KeyError((record_type, sid))
        return self._resolved_map.get((table_name, sid), sid)

    def _parallel_save(
        self,
        obj: Any,
        as_record: type | None,
        record_type: type,
        promote: frozenset[type],
    ) -> int:
        """Dispatch ``obj`` to a worker and return a provisional token resolved after the merge.

        In parallel mode the encode happens asynchronously in a worker, so the
        sid is not known synchronously. ``save`` instead returns a unique token
        that :meth:`resolved_sid` maps to the durable stored sid once the context
        has exited cleanly. The token is a proper stand-in: it is never a real
        row sid, and every equivalence guarantee flows through
        :meth:`resolved_sid`.

        :param obj: The object to store.
        :param as_record: The alternate record representation to use, if any.
        :param record_type: The already-resolved outer storage record class.
        :param promote: The validated nested record classes to make top-level.
        :return: A provisional token (see :meth:`resolved_sid`).
        """
        # Validate the metadata shape (and record the schema graph) before any DDL,
        # so a rejected type fails fast without leaving an empty table behind.
        self._record_schema_graph(record_type)
        if self._deferred:
            self._deferred_top_types.add(record_type)
            self._deferred_top_types.update(promote)
        else:
            self._ensure_tables(record_type)
            for promoted_type in promote:
                self._ensure_tables(promoted_type)
        table_name = resolve_schema(record_type).table_name
        token = self._next_token
        self._next_token += 1
        if self._track_sids:
            self._returned_sids.add((table_name, token))
        self._records_total += 1
        assert self._controller is not None
        self._controller.dispatch(token, obj, as_record, promote)
        return token

    @staticmethod
    def _promoted_types(record_type: type, promote: type | Iterable[type] | None) -> frozenset[type]:
        """Normalize and validate bulk-only nested top-level record classes."""
        if promote is None:
            return frozenset()
        try:
            candidates = (promote,) if isinstance(promote, type) else tuple(promote)
        except TypeError:
            raise TypeError("bulk_ingest().save(promote=...) accepts only record classes") from None
        if any(not isinstance(candidate, type) for candidate in candidates):
            raise TypeError("bulk_ingest().save(promote=...) accepts only record classes")
        requested = frozenset(candidates)
        reachable: set[type] = set()
        pending = [record_type]
        while pending:
            current = pending.pop()
            if current in reachable:
                continue
            reachable.add(current)
            pending.extend(resolve_schema(current).referenced_classes())
        missing = requested - reachable
        if missing:
            names = ", ".join(sorted(candidate.__name__ for candidate in missing))
            raise ValueError(f"promoted record class(es) are not reachable from {record_type.__name__}: {names}")
        return requested

    def _record_schema_graph(self, record_type: type) -> None:
        """Validate and record every record table's schema in the graph rooted at ``record_type``.

        Rejects, up front, any record type whose identity-excluded metadata shape
        the set-wise merge cannot verify (see
        :func:`~httk.store.db.bulk_parallel.unsupported_metadata_reason`), so an
        unsupported ingest fails on its first ``save`` rather than silently
        skipping a conflict check.
        """
        self._validate_schema_graph(record_type, set())

    def _validate_schema_graph(self, record_type: type, visiting: set[type]) -> None:
        """Depth-first validate the metadata graph, committing a type as seen only once its whole subgraph passes.

        A type is added to ``_schema_graph_seen`` (and ``_parent_schema``) *after*
        its entire referenced subgraph validates. If any descendant is rejected,
        the exception unwinds before any ancestor is committed, so a caller that
        catches the rejection inside the context and re-saves the same object is
        re-validated and rejected again — the fail-fast cannot be bypassed.
        ``visiting`` breaks reference cycles during the walk without prematurely
        marking a type validated.

        :param record_type: The record class to validate and record.
        :param visiting: The types on the current recursion path (cycle guard).
        """
        if record_type in self._schema_graph_seen or record_type in visiting:
            return
        visiting.add(record_type)
        if self._verify_metadata:
            from httk.store.db.bulk_parallel import unsupported_metadata_reason

            reason = unsupported_metadata_reason(record_type)
            if reason is not None:
                remedy = 'finalize="parity"' if self._deferred else "workers=1"
                raise ValueError(
                    f"bulk_ingest({'finalize="deferred"' if self._deferred else 'workers>1'}) cannot verify "
                    f"this identity-excluded metadata shape: {reason}. Use {remedy} for records of this kind, "
                    "or open with verify_metadata=False."
                )
        schema = resolve_schema(record_type)
        for target in schema.referenced_classes():
            self._validate_schema_graph(target, visiting)
        for spec in schema.fields:
            if spec.child is not None and spec.target is not None:
                self._validate_schema_graph(spec.target, visiting)
        # The whole subgraph validated: only now commit this type.
        self._schema_graph_seen.add(record_type)
        self._parent_schema[schema.table_name] = schema

    def _encode(
        self,
        record_type: type,
        source: Any,
        projection: SaveProjection,
        path: str,
        promote: frozenset[type],
        occurrences: list[tuple[type, Any, int]],
        seen: set[tuple[type, int, int]],
    ) -> int:
        active_key = (record_type, id(source))
        if active_key in projection.active:
            raise StorageProjectionCycleError(path, record_type)
        projection.active.add(active_key)
        try:
            sid = self._encode_active(record_type, source, projection, path, promote, occurrences, seen)
            key = (record_type, id(source), sid)
            if record_type in promote and key not in seen:
                seen.add(key)
                occurrences.append((record_type, source, sid))
            return sid
        finally:
            projection.active.remove(active_key)

    def _encode_active(
        self,
        record_type: type,
        source: Any,
        projection: SaveProjection,
        path: str,
        promote: frozenset[type],
        occurrences: list[tuple[type, Any, int]],
        seen: set[tuple[type, int, int]],
    ) -> int:
        schema = resolve_schema(record_type)
        self._ensure_tables(record_type)
        table_name = schema.table_name
        self._next_sid.setdefault(table_name, 1)
        self._content_index.setdefault(table_name, {})
        self._value_index.setdefault(table_name, {})
        self._parent_schema.setdefault(table_name, schema)
        projected = projection.projector(record_type, source)

        validation_key = (record_type, id(source))
        if type(source) is record_type and validation_key not in projection.validated:
            validator = vars(record_type).get("__httk_validate__")
            if validator is not None:
                # Bind the descriptor from the class's own dict; the own-dict
                # lookup keeps inherited validators out, exactly as save() does.
                validator.__get__(None, record_type)(source)
            projection.validated.add(validation_key)

        def resolve_sid(referenced_type: type, value: Any, field_path: str) -> int:
            return self._encode(referenced_type, value, projection, field_path, promote, occurrences, seen)

        key: str | None = None
        if schema.dedup == "content_id":
            key = projection.content_id(record_type, source)
            existing = self._content_index[table_name].get(key)
            if existing is not None:
                if self._verify_metadata:
                    self._check_hit_metadata(record_type, key, projected, source, existing)
                _encode_promoted_descendants(
                    schema, source, projected, path, existing, promote, resolve_sid, references=True
                )
                return existing

        values = _encode_parent_row(schema, source, projected, path, resolve_sid)

        if schema.dedup == "by_value":
            value_tuple = tuple(sorted(values.items()))
            existing = self._value_index[table_name].get(value_tuple)
            if existing is not None:
                _encode_promoted_descendants(
                    schema, source, projected, path, existing, promote, resolve_sid, references=False
                )
                return existing

        sid = self._next_sid[table_name]
        self._next_sid[table_name] = sid + 1
        row = {SID_COLUMN: sid, ROLE_COLUMN: 0, **values}
        if projection.store_timestamp is not None:
            row[TS_START_COLUMN] = projection.store_timestamp
        if key is not None:
            row[CONTENT_ID_COLUMN] = key
            self._content_index[table_name][key] = sid
            if self._verify_metadata and _metadata_plan(record_type) is not None:
                self._meta_values[key] = projected
                self._meta_sources[key] = (record_type, source)
        elif schema.dedup == "by_value":
            self._value_index[table_name][tuple(sorted(values.items()))] = sid
        self._buffer_row(table_name, row)

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
                self._buffer_row(spec.child.table_name, child_row)
        return sid

    def _promote_buffered_role(self, table_name: str, sid: int) -> bool:
        """Mark a top-level bulk occurrence main without making role a dedup key."""
        for row in self._rows.get(table_name, ()):
            if row[SID_COLUMN] == sid:
                row[ROLE_COLUMN] = 1
                return True
        return False

    def _promote_occurrence(
        self,
        record_type: type,
        source: Any,
        projection: SaveProjection,
        sid: int,
    ) -> None:
        """Give one encoded occurrence ordinary top-level role and dispatch semantics."""
        table_name = resolve_schema(record_type).table_name
        if not self._promote_buffered_role(table_name, sid):
            assert self._connection is not None
            table = self._store._table(table_name)
            self._connection.execute(
                sqlalchemy.update(table)
                .where(table.c[SID_COLUMN] == sid, table.c[ROLE_COLUMN] == 0)
                .values(_httk_role=1)
            )
        self._chunk_roots.append((table_name, sid))
        family = self._store._family_for_backing(record_type)
        if family is not None and len(family.records) > 1:
            self._buffer_dispatch(family, record_type, sid, projection.content_id(record_type, source))

    def _buffer_row(self, table_name: str, row: dict[str, Any]) -> None:
        self._rows.setdefault(table_name, []).append(row)

    def _buffer_dispatch(self, family: Any, backing: type, sid: int, key: str) -> None:
        dispatch_name = entry_dispatch_table_name(family.name)
        column = backing_dispatch_column_name(family.record_names[family.records.index(backing)])
        row: dict[str, Any] = {DISPATCH_CONTENT_ID_COLUMN: key}
        for backing_name in family.record_names:
            row[backing_dispatch_column_name(backing_name)] = None
        row[column] = sid
        self._dispatch_family.setdefault(dispatch_name, family)
        bucket = self._dispatch_rows.setdefault(dispatch_name, {})
        existing = bucket.get(key)
        if existing is not None:
            if existing != row:
                raise EntryDispatchIntegrityError(
                    f"entry dispatch {family.name!r} maps content_id {key!r} to a conflicting backing row"
                )
            return
        bucket[key] = row

    # ------------------------------------------------------------------ table creation

    def _ensure_tables(self, record_type: type) -> None:
        if record_type in self._ensured:
            return
        candidate = self._store._register_tables((record_type,))
        # Reject a record whose table claims a reserved ``_httk_`` name, exactly
        # as the ordinary write path does before creating tables.
        self._store._validate_table_names(frozenset(candidate.tables))
        order = LogicalEdgeGraph.from_store(self._store, (resolve_schema(record_type),)).dependency_order(
            candidate.tables
        )
        for name in order:
            table = candidate.tables[name]
            name = table.name
            if name in self._created_set:
                continue
            if name in self._preexisting:
                self._scan_existing_table(table)
                continue
            self._create_physical_table(self._store._table(name))
            self._created.append(name)
            self._created_set.add(name)
        self._ensured.add(record_type)

    def _scan_existing_table(self, table: sqlalchemy.Table) -> None:
        """Record a pre-existing table's row count and sid maximum on first registration."""
        name = table.name
        if name in self._existing_scanned:
            return
        self._existing_scanned.add(name)
        assert self._connection is not None
        count = self._connection.execute(sqlalchemy.text(f'SELECT count(*) FROM "{name}"')).scalar_one()
        self._existing_row_count[name] = int(count)
        if SID_COLUMN in table.c:
            maximum = self._connection.execute(
                sqlalchemy.text(f'SELECT max("{SID_COLUMN}") FROM "{name}"')
            ).scalar_one()
            start = int(maximum) + 1 if maximum is not None else 1
            self._next_sid[name] = start
            self._initial_next_sid[name] = start

    def _create_physical_table(self, table: sqlalchemy.Table) -> None:
        assert self._connection is not None
        if self._connection.dialect.name in ("duckdb", "postgresql"):
            sequence = _sid_sequence(table)
            if sequence is not None:
                # DuckDB renders the sid column as DEFAULT nextval(<seq>) and
                # needs the sequence at CREATE TABLE time for that default to
                # bind.  PostgreSQL's bare CreateTable emits no default, but an
                # ordinary post-ingest save still draws its sid via nextval, so
                # every built table (even one that stays empty this ingest) must
                # own its sequence.  SQLite ignores the sequence entirely.
                #
                # This table is being created fresh, so PostgreSQL creates the
                # sequence without IF NOT EXISTS: a leftover sequence is a stale
                # remnant of a failed prior bulk and must be a hard error (like
                # the bare CreateTable below), not silently reused.  DuckDB keeps
                # IF NOT EXISTS to tolerate its rolled-back-DDL cleanup ordering.
                if self._connection.dialect.name == "duckdb":
                    self._connection.execute(sqlalchemy.text(f'CREATE SEQUENCE IF NOT EXISTS "{sequence.name}"'))
                else:
                    self._connection.execute(sqlalchemy.text(f'CREATE SEQUENCE "{sequence.name}"'))
        # A bare CreateTable (not create_all / Table.create) so the separable
        # indexes stay out until the deferred post-load build.
        self._connection.execute(sqlalchemy.schema.CreateTable(table))

    # ------------------------------------------------------------------ flushing and finalization

    def _flush(self) -> None:
        assert self._connection is not None
        if any(self._rows.values()):
            self._resolve_and_insert()
        # Metadata caches (and the chunk's roots) live only for the chunk that
        # buffered them: bound their memory to one chunk. A later chunk's hit on
        # an already-flushed content id verifies against the stored row instead.
        self._meta_values.clear()
        self._meta_sources.clear()
        self._chunk_roots = []
        self._since_flush = 0
        if self._on_progress is not None:
            self._on_progress(self._records_total, self._rows_flushed_total)

    def _resolve_and_insert(self) -> None:
        """Resolve this chunk's set-wise dedup against existing rows, then append the survivors."""
        assert self._connection is not None
        store = self._store
        fk_columns = self._build_fk_columns()
        had_hits = False
        # Resolve every pre-existing content-addressed table first, in FK
        # dependency order (referenced tables before referrers) so a parent's
        # tuples and by_value keys are computed against final, remapped sids.
        for name in self._logical_graph().dependency_order(store._metadata.tables):
            table = store._metadata.tables[name]
            name = table.name
            rows = self._rows.get(name)
            if not rows or name not in self._preexisting:
                continue
            schema = self._parent_schema.get(name)
            if schema is None:
                continue  # A pre-existing child table: its element sids are remapped by their targets.
            if schema.dedup == "content_id":
                had_hits |= self._dedup_content(table, schema, rows, fk_columns)
            elif schema.dedup == "by_value":
                had_hits |= self._dedup_by_value(table, schema, fk_columns)
        if had_hits:
            # A dropped hit can orphan descendants the eager encoder buffered
            # (referenced records, none-policy records, child element records)
            # that save() would never have created; sweep those unreachable rows.
            self._collect_garbage(fk_columns)
        self._refresh_value_index()
        for name in self._logical_graph().dependency_order(store._metadata.tables):
            table = store._metadata.tables[name]
            name = table.name
            rows = self._rows.get(name)
            if not rows:
                continue
            self._decide_index(name)
            self._connection.execute(sqlalchemy.insert(table), rows)
            self._inserted_count[name] = self._inserted_count.get(name, 0) + len(rows)
            self._rows_flushed_total += len(rows)
            rows.clear()

    def _finalize(self) -> None:
        if self._parallel:
            self._parallel_finalize()
            return
        self._flush()
        self._flush_dispatch()
        self._create_new_indexes()
        self._recreate_dropped_indexes()
        self._verify_rebuild_scans()
        self._resync_sequences()
        self._assert_counts()

    def _parallel_finalize(self) -> None:
        """Join the workers, merge their shards set-wise, then build indexes and verify (parallel mode)."""
        from httk.store.db.bulk_parallel import merge

        assert self._controller is not None
        manifests = self._controller.finish()  # re-raises the first worker exception
        self._assert_no_lost_tasks(manifests)
        merge(self, manifests)
        self._create_new_indexes()
        self._resync_sequences()
        self._assert_counts()

    def _deferred_finalize(self) -> None:
        """Build an empty store from external occurrence-preserving stage data.

        Staging has completed before this point.  The main database is touched
        only in this final transaction (plus the separately committed marker),
        which is required by DuckDB's one-writable-attached-database rule.
        """
        assert self._connection is not None
        if self._parallel:
            assert self._controller is not None
            started = time.perf_counter()
            manifests = self._controller.finish()
            self.finalize_timings["stage_finish"] = time.perf_counter() - started
            self._assert_no_lost_tasks(manifests)
        else:
            started = time.perf_counter()
            manifests = [] if self._serial_stage is None else [self._serial_stage.finish()]
            self.finalize_timings["stage_finish"] = time.perf_counter() - started
        transaction = None if self._connection.dialect.name == "clickhousedb" else self._connection.begin()
        finalizer: Any = None
        finalizer_cleaned = False
        try:
            # Register and create schema-faithful ordinary tables only now,
            # after staging is complete.  There are no physical FKs, so table
            # insertion order is intentionally unconstrained.
            started = time.perf_counter()
            for record_type in sorted(self._deferred_top_types, key=lambda value: resolve_schema(value).table_name):
                self._ensure_tables(record_type)
            self.finalize_timings["ddl"] = time.perf_counter() - started
            if self._store.backend_facts.stage_load == "client-stream":
                from httk.store.db.clickhouse import load_parquet_stages

                self._clickhouse_stage_tables = load_parquet_stages(self._store, manifests)
                self._staging_tables.update(self._clickhouse_stage_tables.values())
                self._after_clickhouse_stage_load()
            from httk.store.db.bulk_deferred import DeferredFinalizer

            finalizer = DeferredFinalizer(self, manifests)
            try:
                try:
                    finalizer.run()
                except EntryMetadataConflictError as error:
                    if not self._parallel:
                        message = self._serial_conflict_message(str(error))
                        if message != str(error):
                            raise EntryMetadataConflictError(message) from error
                    raise
                # Serial callers receive a table-scoped public provisional sid,
                # while stage manifests use unique root tokens.
                if not self._parallel:
                    for (table, public_sid), stage_sid in self._serial_public_stage_sid.items():
                        self._resolved_map[(table, public_sid)] = finalizer._final_for_stage(table, stage_sid)
                finalizer_timings = dict(finalizer.finalize_timings)
                self.finalize_timings.update(finalizer_timings)
                started = time.perf_counter()
                self._create_new_indexes()
                self._resync_sequences()
                self.finalize_timings["indexes"] = time.perf_counter() - started
                started = time.perf_counter()
                self._assert_counts()
                if self._connection.dialect.name == "clickhousedb":
                    self._before_clickhouse_integrity_verification()
                    from httk.store.db.clickhouse import verify_bulk_integrity

                    verify_bulk_integrity(self._connection, [self._store._table(name) for name in self._created])
                # ClickHouse working and stage relations are durable tables.
                # They must be gone before physical validation and marker clear.
                finalizer.cleanup()
                finalizer_cleaned = True
                self._after_clickhouse_cleanup()
                self._validate_deferred_staging_cleared()
                self._validate_deferred_physical()
                if self._connection.dialect.name == "clickhousedb":
                    from httk.store.db.clickhouse import validate_metadata_table

                    # Validate the KeeperMap shape in this same durable
                    # pre-marker-clear pass, not merely while opening a store.
                    validate_metadata_table(self._connection)
                self._after_clickhouse_physical_validation()
                self.finalize_timings["validation"] = time.perf_counter() - started
            finally:
                if not finalizer_cleaned:
                    finalizer.cleanup()
                self.finalize_timings.update(finalizer.finalize_timings)
            started = time.perf_counter()
            if transaction is not None:
                transaction.commit()
            self.finalize_timings["commit"] = time.perf_counter() - started
        except BaseException:
            if transaction is not None:
                transaction.rollback()
            raise

    def _serial_conflict_message(self, message: str) -> str:
        for schema in self._parent_schema.values():
            prefix = f"metadata conflict for {schema.cls.__name__}."
            if not message.startswith(prefix):
                continue
            field = message[len(prefix) :].split(":", 1)[0]
            for parent in self._parent_schema.values():
                for spec in parent.fields:
                    if spec.role == "reference" and spec.target is schema.cls:
                        return message.replace(f"{schema.cls.__name__}.{field}", f"{spec.field}.{field}", 1)
        return message

    def _validate_deferred_physical(self) -> None:
        """Verify the full declared physical shape before marker clear.

        Deferred load intentionally uses bare table DDL followed by the load
        and index build.  Validate all parts that can otherwise be made
        durable by an interrupted or externally-contended finalize: column
        shape/defaults/nullability, constraints, indexes, sequences, and the
        absence of physical FKs or stage relations.
        """
        assert self._connection is not None
        objects = actual_schema_objects(self._connection)
        expected = set(self._created) | {METADATA_TABLE_NAME}
        actual_tables = {name for name, kinds in objects.items() if "table" in kinds}
        if actual_tables != expected:
            raise RuntimeError(
                "deferred finalize schema tables differ from declaration: "
                f"expected {', '.join(sorted(expected))}; found {', '.join(sorted(actual_tables))}"
            )
        if self._connection.dialect.name == "duckdb" and self._entry_catalog is not None:
            attached = {
                str(row[0])
                for row in self._connection.execute(sqlalchemy.text("SELECT database_name FROM duckdb_databases()"))
            }
            expected_attached = set(self._entry_catalog[2]) | set(self._parallel_attached)
            if attached != expected_attached:
                raise RuntimeError(
                    "deferred finalize found unexpected DuckDB attachments: "
                    f"expected {', '.join(sorted(expected_attached))}; found {', '.join(sorted(attached))}"
                )
        for name in self._created:
            table = self._store._table(name)
            if self._connection.dialect.name == "sqlite":
                self._validate_deferred_sqlite_table(name, table)
            elif self._connection.dialect.name == "duckdb":
                self._validate_deferred_duckdb_table(name, table)
            elif self._connection.dialect.name == "postgresql":
                self._validate_deferred_pg_table(name, table)
            else:
                from httk.store.db.clickhouse import validate_bulk_tables

                validate_bulk_tables(self._connection, [table])

    def _validate_deferred_staging_cleared(self) -> None:
        """Ensure temporary finalizer relations did not escape its cleanup."""
        assert self._connection is not None
        lingering = sorted(
            name
            for name in actual_schema_objects(self._connection)
            if name.startswith(("_httk_deferred_", "_httk_stage_"))
        )
        if lingering:
            raise RuntimeError(f"deferred finalize found lingering staging objects: {', '.join(lingering)}")
        if self._connection.dialect.name == "sqlite":
            temporary = (
                self._connection.execute(
                    sqlalchemy.text("SELECT name FROM sqlite_temp_master WHERE name NOT LIKE 'sqlite_%' ORDER BY name")
                )
                .scalars()
                .all()
            )
            if temporary:
                raise RuntimeError(f"deferred finalize found lingering SQLite temp objects: {', '.join(temporary)}")

    @staticmethod
    def _physical_sql(value: object | None) -> str | None:
        """Canonicalize catalog SQL enough for stable cross-dialect checks."""
        if value is None:
            return None
        normalized = (
            " ".join(str(value).replace('"', "").lower().split())
            .replace("double precision", "double")
            .replace("varchar", "text")
            .replace("bytea", "blob")
        )
        # DuckDB expands CASE expressions and wraps CHECK expressions in
        # implementation parentheses; neither changes the declared check.
        if "case" in normalized or normalized.startswith("check"):
            return normalized.replace("check", "").replace("(", "").replace(")", "").replace(" ", "")
        return normalized

    def _physical_failure(self, name: str, detail: str) -> None:
        raise RuntimeError(f"deferred finalize physical validation failed for {name!r}: {detail}")

    def _expected_unique_columns(self, table: sqlalchemy.Table) -> set[tuple[str, ...]]:
        expected = {
            tuple(column.name for column in constraint.columns)
            for constraint in table.constraints
            if isinstance(constraint, sqlalchemy.UniqueConstraint)
        }
        expected.update((column.name,) for column in table.columns if column.unique)
        return expected

    def _expected_sqlite_unique_columns(self, table: sqlalchemy.Table) -> set[tuple[str, ...]]:
        expected = self._expected_unique_columns(table)
        primary = tuple(table.primary_key.columns)
        # SQLite's single INTEGER primary key aliases rowid and deliberately
        # has no index_list entry; other primary keys do have an autoindex.
        if primary and not (len(primary) == 1 and isinstance(primary[0].type, sqlalchemy.Integer)):
            expected.add(tuple(column.name for column in table.primary_key.columns))
        expected.update(tuple(column.name for column in index.columns) for index in table.indexes if index.unique)
        return expected

    def _expected_checks(self, table: sqlalchemy.Table) -> set[str]:
        return {
            self._physical_check(constraint.sqltext)
            for constraint in table.constraints
            if isinstance(constraint, sqlalchemy.CheckConstraint)
        }

    @classmethod
    def _physical_check(cls, value: object) -> str:
        """Normalize non-semantic SQLite/DuckDB parenthesis and whitespace differences."""
        return (cls._physical_sql(value) or "").replace("(", "").replace(")", "").replace(" ", "")

    @classmethod
    def _pg_check_norm(cls, value: str) -> str:
        """Canonicalize a check-constraint definition for cross-form comparison.

        Reduces both a declared ``x IN (0, 1)`` and PostgreSQL's stored
        ``CHECK ((x = ANY (ARRAY[0, 1])))`` rewrite to the same token string, so
        a genuinely different predicate still compares unequal.

        :param value: A ``pg_get_constraintdef`` string or a declared ``sqltext``.
        :return: The canonical token string.
        """
        normalized = cls._physical_check(value).replace("[", "").replace("]", "")
        return normalized.removeprefix("check").replace("=anyarray", "in")

    @classmethod
    def _sqlite_check_clauses(cls, sql: str) -> list[str]:
        """Extract top-level SQLite ``CHECK(...)`` expressions without keyword false positives."""
        result: list[str] = []
        index = 0
        quote: str | None = None
        while index < len(sql):
            character = sql[index]
            if quote is not None:
                if character == quote:
                    if quote in "'\"" and index + 1 < len(sql) and sql[index + 1] == quote:
                        index += 2
                        continue
                    quote = None
                index += 1
                continue
            if character in "'\"`":
                quote = character
                index += 1
                continue
            if character == "[":
                quote = "]"
                index += 1
                continue
            if character.isalpha() or character == "_":
                start = index
                index += 1
                while index < len(sql) and (sql[index].isalnum() or sql[index] == "_"):
                    index += 1
                if sql[start:index].lower() != "check":
                    continue
                cursor = index
                while cursor < len(sql) and sql[cursor].isspace():
                    cursor += 1
                if cursor >= len(sql) or sql[cursor] != "(":
                    continue
                depth, expression_start, cursor = 1, cursor + 1, cursor + 1
                nested_quote: str | None = None
                while cursor < len(sql) and depth:
                    nested = sql[cursor]
                    if nested_quote is not None:
                        if nested == nested_quote:
                            if nested_quote in "'\"" and cursor + 1 < len(sql) and sql[cursor + 1] == nested_quote:
                                cursor += 2
                                continue
                            nested_quote = None
                    elif nested in "'\"`":
                        nested_quote = nested
                    elif nested == "[":
                        nested_quote = "]"
                    elif nested == "(":
                        depth += 1
                    elif nested == ")":
                        depth -= 1
                    cursor += 1
                if depth == 0:
                    result.append(cls._physical_check(sql[expression_start : cursor - 1]))
                    index = cursor
                continue
            index += 1
        return result

    def _validate_deferred_sqlite_table(self, name: str, table: sqlalchemy.Table) -> None:
        assert self._connection is not None
        rows = self._connection.execute(sqlalchemy.text(f'PRAGMA table_info("{name}")')).mappings().all()
        if [str(row["name"]) for row in rows] != [column.name for column in table.columns]:
            self._physical_failure(name, "column declaration")
        for row, column in zip(rows, table.columns, strict=True):
            expected_type = self._physical_sql(column.type.compile(dialect=self._connection.dialect))
            if self._physical_sql(row["type"]) != expected_type:
                self._physical_failure(name, f"type for {column.name!r}")
            if bool(row["notnull"]) != (not column.nullable):
                self._physical_failure(name, f"nullability for {column.name!r}")
            # The internal table builder supplies only DefaultClause-compatible defaults here.
            expected_default = self._physical_sql(
                cast(sqlalchemy.DefaultClause, column.server_default).arg if column.server_default else None
            )
            if self._physical_sql(row["dflt_value"]) != expected_default:
                self._physical_failure(name, f"default for {column.name!r}")
        expected_pk = [column.name for column in table.primary_key.columns]
        actual_pk = [str(row["name"]) for row in sorted(rows, key=lambda row: int(row["pk"])) if row["pk"]]
        if actual_pk != expected_pk:
            self._physical_failure(name, "primary key")
        indexes = self._connection.execute(sqlalchemy.text(f'PRAGMA index_list("{name}")')).mappings().all()
        declared_indexes = {index.name: bool(index.unique) for index in table.indexes}
        actual_indexes = {str(row["name"]): bool(row["unique"]) for row in indexes if str(row["origin"]) == "c"}
        if actual_indexes != declared_indexes:
            self._physical_failure(name, "indexes")
        actual_unique = {
            tuple(
                str(row["name"])
                for row in self._connection.execute(sqlalchemy.text(f'PRAGMA index_info("{index["name"]}")'))
                .mappings()
                .all()
            )
            for index in indexes
            if bool(index["unique"])
        }
        if actual_unique != self._expected_sqlite_unique_columns(table):
            self._physical_failure(name, "unique constraints")
        create_sql = self._connection.execute(
            sqlalchemy.text("SELECT sql FROM sqlite_master WHERE type = 'table' AND name = :name"), {"name": name}
        ).scalar_one()
        if sorted(self._sqlite_check_clauses(str(create_sql))) != sorted(self._expected_checks(table)):
            self._physical_failure(name, "check constraints")
        if self._connection.execute(sqlalchemy.text(f'PRAGMA foreign_key_list("{name}")')).first() is not None:
            self._physical_failure(name, "foreign keys are forbidden")

    def _validate_deferred_duckdb_table(self, name: str, table: sqlalchemy.Table) -> None:
        assert self._connection is not None
        rows = (
            self._connection.execute(
                sqlalchemy.text(
                    "SELECT column_name, data_type, is_nullable, column_default FROM information_schema.columns "
                    "WHERE table_catalog = current_database() AND table_schema = current_schema() "
                    "AND table_name = :name ORDER BY ordinal_position"
                ),
                {"name": name},
            )
            .mappings()
            .all()
        )
        if [str(row["column_name"]) for row in rows] != [column.name for column in table.columns]:
            self._physical_failure(name, "column declaration")
        for row, column in zip(rows, table.columns, strict=True):
            expected_type = self._physical_sql(column.type.compile(dialect=self._connection.dialect))
            if self._physical_sql(row["data_type"]) != expected_type:
                self._physical_failure(name, f"type for {column.name!r}")
            if (str(row["is_nullable"]) == "YES") != column.nullable:
                self._physical_failure(name, f"nullability for {column.name!r}")
            # The internal table builder supplies only DefaultClause-compatible defaults here.
            expected_default = self._physical_sql(
                cast(sqlalchemy.DefaultClause, column.server_default).arg if column.server_default else None
            )
            if self._physical_sql(row["column_default"]) != expected_default:
                self._physical_failure(name, f"default for {column.name!r}")
        constraints = (
            self._connection.execute(
                sqlalchemy.text(
                    "SELECT constraint_type, constraint_text, constraint_column_names FROM duckdb_constraints() "
                    "WHERE database_name = current_database() AND schema_name = current_schema() AND table_name = :name"
                ),
                {"name": name},
            )
            .mappings()
            .all()
        )
        by_type: dict[str, list[Any]] = {}
        for constraint in constraints:
            by_type.setdefault(str(constraint["constraint_type"]), []).append(constraint)
        expected_pk = [column.name for column in table.primary_key.columns]
        actual_pk = next((list(row["constraint_column_names"]) for row in by_type.get("PRIMARY KEY", ())), None)
        if (actual_pk or []) != expected_pk:
            self._physical_failure(name, "primary key")
        actual_unique = {
            tuple(str(column) for column in row["constraint_column_names"]) for row in by_type.get("UNIQUE", ())
        }
        if actual_unique != self._expected_unique_columns(table):
            self._physical_failure(name, "unique constraints")
        actual_checks = {self._physical_check(row["constraint_text"]) for row in by_type.get("CHECK", ())}
        if actual_checks != self._expected_checks(table):
            self._physical_failure(name, "check constraints")
        if by_type.get("FOREIGN KEY"):
            self._physical_failure(name, "foreign keys are forbidden")
        indexes = (
            self._connection.execute(
                sqlalchemy.text(
                    "SELECT index_name, is_unique FROM duckdb_indexes() "
                    "WHERE database_name = current_database() AND schema_name = current_schema() AND table_name = :name"
                ),
                {"name": name},
            )
            .mappings()
            .all()
        )
        actual_indexes = {str(row["index_name"]): bool(row["is_unique"]) for row in indexes}
        expected_indexes = {index.name: bool(index.unique) for index in table.indexes}
        if actual_indexes != expected_indexes:
            self._physical_failure(name, "indexes")
        sequence = _sid_sequence(table)
        if sequence is not None:
            sequence_row = self._connection.execute(
                sqlalchemy.text(
                    "SELECT start_value FROM duckdb_sequences() WHERE database_name = current_database() "
                    "AND schema_name = current_schema() AND sequence_name = :name"
                ),
                {"name": sequence.name},
            ).first()
            if sequence_row is None or int(sequence_row[0]) != self._next_sid.get(name, 1):
                self._physical_failure(name, f"sequence {sequence.name!r}")

    def _validate_deferred_pg_table(self, name: str, table: sqlalchemy.Table) -> None:
        """Validate a deferred-built PostgreSQL table against its declaration.

        Mirrors :meth:`_validate_deferred_duckdb_table` using the standard
        ``information_schema`` (columns) and ``pg_catalog`` (constraints,
        indexes, sequence).  Check constraints are compared by name rather than
        expression text: PostgreSQL rewrites e.g. ``x IN (0, 1)`` to
        ``x = ANY (ARRAY[0, 1])``, which no purely textual normalization
        recovers, whereas the deterministic constraint names do not drift.
        """
        assert self._connection is not None
        rows = (
            self._connection.execute(
                sqlalchemy.text(
                    "SELECT column_name, data_type, is_nullable, column_default FROM information_schema.columns "
                    "WHERE table_catalog = current_database() AND table_schema = current_schema() "
                    "AND table_name = :name ORDER BY ordinal_position"
                ),
                {"name": name},
            )
            .mappings()
            .all()
        )
        if [str(row["column_name"]) for row in rows] != [column.name for column in table.columns]:
            self._physical_failure(name, "column declaration")
        for row, column in zip(rows, table.columns, strict=True):
            expected_type = self._physical_sql(column.type.compile(dialect=self._connection.dialect))
            if self._physical_sql(row["data_type"]) != expected_type:
                self._physical_failure(name, f"type for {column.name!r}")
            if (str(row["is_nullable"]) == "YES") != column.nullable:
                self._physical_failure(name, f"nullability for {column.name!r}")
            # The internal table builder supplies only DefaultClause-compatible defaults here.
            expected_default = self._physical_sql(
                cast(sqlalchemy.DefaultClause, column.server_default).arg if column.server_default else None
            )
            if self._physical_sql(row["column_default"]) != expected_default:
                self._physical_failure(name, f"default for {column.name!r}")
        constraints = (
            self._connection.execute(
                sqlalchemy.text(
                    "SELECT con.contype::text AS contype, con.conname AS conname, "
                    "pg_get_constraintdef(con.oid) AS condef, "
                    "ARRAY(SELECT a.attname FROM unnest(con.conkey) WITH ORDINALITY AS k(attnum, ord) "
                    "JOIN pg_attribute a ON a.attrelid = con.conrelid AND a.attnum = k.attnum "
                    "ORDER BY k.ord) AS cols "
                    "FROM pg_constraint con JOIN pg_class t ON t.oid = con.conrelid "
                    "JOIN pg_namespace ns ON ns.oid = t.relnamespace "
                    "WHERE t.relname = :name AND ns.nspname = current_schema()"
                ),
                {"name": name},
            )
            .mappings()
            .all()
        )
        by_type: dict[str, list[Any]] = {}
        for constraint in constraints:
            by_type.setdefault(str(constraint["contype"]), []).append(constraint)
        # The mapper emits only primary-key, unique, and check constraints; any
        # other constraint kind (e.g. a foreign key) is a corrupted schema. NOT
        # NULL is tolerated: PostgreSQL 18 records it as a "n" pg_constraint row,
        # and column nullability is already validated separately above.
        if set(by_type) - {"p", "u", "c", "n"}:
            self._physical_failure(name, "unexpected constraint kind")
        expected_pk = [column.name for column in table.primary_key.columns]
        actual_pk = next((list(row["cols"]) for row in by_type.get("p", ())), None)
        if (actual_pk or []) != expected_pk:
            self._physical_failure(name, "primary key")
        actual_unique = {tuple(str(column) for column in row["cols"]) for row in by_type.get("u", ())}
        if actual_unique != self._expected_unique_columns(table):
            self._physical_failure(name, "unique constraints")
        # Compare check constraints by definition, not merely by name: a
        # same-named but semantically wrong check must fail.  ``_pg_check_norm``
        # canonicalizes PostgreSQL's ``x = ANY (ARRAY[...])`` rewrite of the
        # mapper's ``x IN (...)`` so the declared and stored forms compare equal.
        actual_checks = {self._pg_check_norm(str(row["condef"])) for row in by_type.get("c", ())}
        expected_checks = {
            self._pg_check_norm(str(constraint.sqltext))
            for constraint in table.constraints
            if isinstance(constraint, sqlalchemy.CheckConstraint)
        }
        if actual_checks != expected_checks:
            self._physical_failure(name, "check constraints")
        # Only pure ``Index`` objects; exclude the indexes that back a primary
        # key or unique constraint (validated above).  Compare each index's
        # ordered column list, not just its name and uniqueness.
        indexes = (
            self._connection.execute(
                sqlalchemy.text(
                    "SELECT c.relname AS index_name, i.indisunique AS is_unique, "
                    "ARRAY(SELECT a.attname FROM unnest(i.indkey) WITH ORDINALITY AS k(attnum, ord) "
                    "JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = k.attnum "
                    "ORDER BY k.ord) AS cols "
                    "FROM pg_index i JOIN pg_class c ON c.oid = i.indexrelid "
                    "JOIN pg_class t ON t.oid = i.indrelid JOIN pg_namespace ns ON ns.oid = t.relnamespace "
                    "WHERE t.relname = :name AND ns.nspname = current_schema() "
                    "AND NOT EXISTS (SELECT 1 FROM pg_constraint con "
                    "WHERE con.conindid = i.indexrelid AND con.contype IN ('p', 'u'))"
                ),
                {"name": name},
            )
            .mappings()
            .all()
        )
        actual_indexes = {
            str(row["index_name"]): (tuple(str(column) for column in row["cols"]), bool(row["is_unique"]))
            for row in indexes
        }
        expected_indexes = {
            index.name: (tuple(column.name for column in index.columns), bool(index.unique)) for index in table.indexes
        }
        if actual_indexes != expected_indexes:
            self._physical_failure(name, "indexes")
        sequence = _sid_sequence(table)
        if sequence is not None:
            exists = self._connection.execute(
                sqlalchemy.text(
                    "SELECT 1 FROM pg_sequences WHERE schemaname = current_schema() AND sequencename = :name"
                ),
                {"name": sequence.name},
            ).first()
            if exists is None:
                self._physical_failure(name, f"sequence {sequence.name!r}")
            else:
                # ``is_called`` lives on the sequence relation, not pg_sequences:
                # the next drawn value is ``last_value`` when it is false (fresh
                # or ``setval(..., false)``), else ``last_value + 1``.
                last_value, is_called = self._connection.execute(
                    sqlalchemy.text(f'SELECT last_value, is_called FROM "{sequence.name}"')
                ).one()
                next_value = int(last_value) + (1 if is_called else 0)
                if next_value != self._next_sid.get(name, 1):
                    self._physical_failure(name, f"sequence {sequence.name!r}")

    def _assert_no_lost_tasks(self, manifests: list[Any]) -> None:
        """Abort (never commit) if any dispatched task did not come back encoded in a worker manifest."""
        # Deferred Parquet stages carry one root row per submitted task.  Count
        # those sidecars directly, avoiding a second O(N) Python token set on
        # the scale path (``token_sid`` itself is retained only when the public
        # resolved_sid contract requested it).
        if any("_httk_roots" in manifest.shards for manifest in manifests):
            roots = 0
            for manifest in manifests:
                for path in manifest.shards.get("_httk_roots", ()):
                    from pyarrow.parquet import ParquetFile

                    roots += ParquetFile(path).metadata.num_rows
            if roots != self._records_total:
                raise RuntimeError(
                    "bulk_ingest lost tasks between dispatch and deferred finalize: "
                    f"expected {self._records_total} roots, found {roots}"
                )
            return
        if not self._track_sids:
            encoded_count = sum(manifest.encoded_count for manifest in manifests)
            if encoded_count != self._records_total:
                raise RuntimeError(
                    "bulk_ingest(workers>1) lost tasks between dispatch and merge: "
                    f"expected {self._records_total} encoded record(s), found {encoded_count}; "
                    "the ingest is aborted rather than committing a partial store"
                )
            return
        dispatched = {token for _table, token in self._returned_sids}
        encoded: set[int] = set()
        for manifest in manifests:
            encoded.update(manifest.token_sid)
        if encoded != dispatched:
            lost = len(dispatched - encoded)
            extra = len(encoded - dispatched)
            raise RuntimeError(
                "bulk_ingest(workers>1) lost tasks between dispatch and merge: "
                f"{lost} dispatched record(s) were never encoded"
                + (f" and {extra} unexpected token(s) were reported" if extra else "")
                + "; the ingest is aborted rather than committing a partial store"
            )

    def _flush_dispatch(self) -> None:
        assert self._connection is not None
        store = self._store
        for dispatch_name, bucket in self._dispatch_rows.items():
            if not bucket:
                continue
            table = store._table(dispatch_name)
            if dispatch_name not in self._preexisting:
                rows = list(bucket.values())
                self._connection.execute(sqlalchemy.insert(table), rows)
                self._inserted_count[dispatch_name] = self._inserted_count.get(dispatch_name, 0) + len(rows)
                continue
            family = self._dispatch_family[dispatch_name]
            to_insert: list[dict[str, Any]] = []
            for key, row in bucket.items():
                existing = (
                    self._connection.execute(sqlalchemy.select(table).where(table.c[DISPATCH_CONTENT_ID_COLUMN] == key))
                    .mappings()
                    .one_or_none()
                )
                if existing is None:
                    to_insert.append(row)
                    continue
                existing_backing, existing_sid = store._dispatch_target(family, existing, key)
                new_backing, new_sid = store._dispatch_target(family, row, key)
                if existing_backing is not new_backing or existing_sid != new_sid:
                    raise EntryDispatchIntegrityError(
                        f"entry dispatch {family.name!r} maps content_id {key!r} to a conflicting backing row"
                    )
            if to_insert:
                self._connection.execute(sqlalchemy.insert(table), to_insert)
                self._inserted_count[dispatch_name] = self._inserted_count.get(dispatch_name, 0) + len(to_insert)

    def _create_new_indexes(self) -> None:
        assert self._connection is not None
        for name in self._created:
            table = self._store._table(name)
            for index in table.indexes:
                # Creating a unique content-id index over the loaded rows is the
                # uniqueness verification; a duplicate aborts the whole ingest.
                self._connection.execute(sqlalchemy.schema.CreateIndex(index))

    def _recreate_dropped_indexes(self) -> None:
        assert self._connection is not None
        for index in self._dropped_indexes:
            # Recreating the (unique) index over the appended rows re-verifies
            # global uniqueness for the rebuild strategy; a duplicate aborts.
            self._connection.execute(sqlalchemy.schema.CreateIndex(index))

    def _verify_rebuild_scans(self) -> None:
        """Verify content-id uniqueness for rebuild tables whose index was kept (the DuckDB path)."""
        assert self._connection is not None
        for name in self._rebuild_scan_tables:
            table = self._store._table(name)
            if CONTENT_ID_COLUMN not in table.c:
                continue
            column = table.c[CONTENT_ID_COLUMN]
            duplicate = self._connection.execute(
                sqlalchemy.select(column).group_by(column).having(sqlalchemy.func.count() > 1).limit(1)
            ).first()
            if duplicate is not None:
                raise RuntimeError(
                    f"bulk_ingest uniqueness verification failed for table {name!r}: "
                    f"content_id {duplicate[0]!r} occurs more than once"
                )

    def _resync_sequences(self) -> None:
        assert self._connection is not None
        dialect = self._connection.dialect.name
        if dialect not in ("duckdb", "postgresql"):
            # SQLite's rowid self-syncs to max+1; only the explicit DuckDB and
            # PostgreSQL sequences must be advanced past the pre-assigned sids.
            return
        for name, next_sid in self._next_sid.items():
            sequence = _sid_sequence(self._store._table(name))
            if sequence is None:
                continue
            if dialect == "duckdb":
                self._connection.execute(
                    sqlalchemy.text(f'CREATE OR REPLACE SEQUENCE "{sequence.name}" START WITH {next_sid}')
                )
            else:
                # A bare ``CreateTable`` does not emit the parent sequence, so
                # create it and point it past the bulk-assigned sids: an
                # ordinary ``save()`` afterwards then draws a fresh sid via
                # ``nextval`` rather than colliding.  ``is_called=false`` makes
                # the next drawn value exactly ``next_sid``.
                self._connection.execute(sqlalchemy.text(f'CREATE SEQUENCE IF NOT EXISTS "{sequence.name}"'))
                self._connection.execute(
                    sqlalchemy.text("SELECT setval(:seq, :value, false)"),
                    {"seq": sequence.name, "value": next_sid},
                )

    def _assert_counts(self) -> None:
        assert self._connection is not None
        names = set(self._inserted_count) | set(self._existing_row_count)
        for name in names:
            inserted = self._inserted_count.get(name, 0)
            existing = self._existing_row_count.get(name, 0)
            if inserted == 0 and name not in self._existing_row_count:
                continue
            expected = existing + inserted
            actual = self._connection.execute(sqlalchemy.text(f'SELECT count(*) FROM "{name}"')).scalar_one()
            if actual != expected:
                raise RuntimeError(
                    f"bulk_ingest row-count verification failed for table {name!r}: "
                    f"expected {expected} (existing {existing} + inserted {inserted}), stored {actual}"
                )

    # ------------------------------------------------------------------ set-wise deduplication against existing rows

    def _dedup_content(
        self,
        table: sqlalchemy.Table,
        schema: TableSchema,
        rows: list[dict[str, Any]],
        fk_columns: Mapping[str, list[tuple[str, str]]],
    ) -> bool:
        """Anti-join this chunk's content-addressed rows against ``table``, dropping the hits.

        Returns whether any hit was found (and rows therefore dropped).
        """
        hits = self._stage_content_hits(table, rows)
        if not hits:
            return False
        name = table.name
        sid_map: dict[int, int] = {}
        content_map = self._content_index.setdefault(name, {})
        for staged_sid, existing_sid, key in hits:
            sid_map[staged_sid] = existing_sid
            content_map[key] = existing_sid
        if self._verify_metadata:
            self._verify_existing_metadata(hits)
        self._promote_existing_roles(table, rows, sid_map)
        for staged_sid, existing_sid in sid_map.items():
            self._resolved_map[(name, staged_sid)] = existing_sid
        self._drop_hit_rows(name, schema, sid_map)
        self._apply_remap(name, sid_map, fk_columns)
        return True

    def _dedup_by_value(
        self,
        table: sqlalchemy.Table,
        schema: TableSchema,
        fk_columns: Mapping[str, list[tuple[str, str]]],
    ) -> bool:
        """Anti-join this chunk's by_value rows against ``table`` on all parent columns, dropping the hits.

        A by_value key is the whole parent-column tuple, so a self-referential
        table needs the stage-join and remap iterated to a fixpoint: remapping a
        hit's sid rewrites the reference column of another staged row, which can
        expose a match the previous pass missed. Each pass drops at least one row,
        so the loop terminates. Returns whether any hit was found.
        """
        name = table.name
        value_map = self._value_index.setdefault(name, {})
        found_any = False
        while True:
            rows = self._rows.get(name)
            if not rows:
                break
            row_by_sid = {row[SID_COLUMN]: row for row in rows}
            hits = self._stage_by_value_hits(table, rows)
            if not hits:
                break
            found_any = True
            sid_map: dict[int, int] = {}
            for staged_sid, existing_sid in hits:
                sid_map[staged_sid] = existing_sid
                value_map[_value_tuple(row_by_sid[staged_sid])] = existing_sid
                self._resolved_map[(name, staged_sid)] = existing_sid
            self._promote_existing_roles(table, rows, sid_map)
            self._drop_hit_rows(name, schema, sid_map)
            self._apply_remap(name, sid_map, fk_columns)
        return found_any

    def _promote_existing_roles(
        self, table: sqlalchemy.Table, staged_rows: list[dict[str, Any]], sid_map: Mapping[int, int]
    ) -> None:
        """Propagate a collapsed staged main occurrence to an existing winner.

        This runs after content metadata verification, so a rejected staged
        content-id hit cannot mutate the existing row.  By-value has no
        metadata comparison, matching ordinary by-value save semantics.
        """
        assert self._connection is not None
        main_existing = {
            existing_sid
            for row in staged_rows
            if row[SID_COLUMN] in sid_map and int(row.get(ROLE_COLUMN, 0)) == 1
            for existing_sid in (sid_map[row[SID_COLUMN]],)
        }
        if main_existing:
            self._connection.execute(
                sqlalchemy.update(table)
                .where(table.c[SID_COLUMN].in_(main_existing), table.c[ROLE_COLUMN] == 0)
                .values({ROLE_COLUMN: 1})
            )

    def _stage_content_hits(self, table: sqlalchemy.Table, rows: list[dict[str, Any]]) -> list[tuple[int, int, str]]:
        """Stage ``rows`` and return ``(staged_sid, existing_sid, content_id)`` for each content-id hit."""
        assert self._connection is not None
        stage = self._create_stage(table, rows)
        try:
            statement = sqlalchemy.select(
                stage.c[SID_COLUMN], stage.c[CONTENT_ID_COLUMN], table.c[SID_COLUMN]
            ).join_from(stage, table, stage.c[CONTENT_ID_COLUMN] == table.c[CONTENT_ID_COLUMN])
            return [(int(row[0]), int(row[2]), str(row[1])) for row in self._connection.execute(statement).all()]
        finally:
            self._drop_stage(stage)

    def _stage_by_value_hits(self, table: sqlalchemy.Table, rows: list[dict[str, Any]]) -> list[tuple[int, int]]:
        """Stage ``rows`` and return ``(staged_sid, existing_sid)`` for each whole-parent-column hit."""
        assert self._connection is not None
        stage = self._create_stage(table, rows)
        try:
            value_columns = [
                column.name
                for column in table.columns
                if column.name not in (SID_COLUMN, ROLE_COLUMN, TS_START_COLUMN, TS_END_COLUMN, REPLACED_BY_COLUMN)
            ]
            condition = sqlalchemy.and_(*(stage.c[name].is_not_distinct_from(table.c[name]) for name in value_columns))
            statement = (
                sqlalchemy.select(stage.c[SID_COLUMN], sqlalchemy.func.min(table.c[SID_COLUMN]))
                .join_from(stage, table, condition)
                .group_by(stage.c[SID_COLUMN])
            )
            return [(int(row[0]), int(row[1])) for row in self._connection.execute(statement).all()]
        finally:
            self._drop_stage(stage)

    def _create_stage(self, table: sqlalchemy.Table, rows: list[dict[str, Any]]) -> sqlalchemy.Table:
        """Create an index-less ``bulkstage_<table>`` clone and load ``rows`` into it."""
        assert self._connection is not None
        stage_name = f"bulkstage_{table.name}"
        stage = sqlalchemy.Table(
            stage_name,
            sqlalchemy.MetaData(),
            *(sqlalchemy.Column(column.name, column.type) for column in table.columns),
        )
        self._connection.execute(sqlalchemy.schema.DropTable(stage, if_exists=True))
        self._connection.execute(sqlalchemy.schema.CreateTable(stage))
        self._staging_tables.add(stage_name)
        self._connection.execute(sqlalchemy.insert(stage), rows)
        return stage

    def _drop_stage(self, stage: sqlalchemy.Table) -> None:
        assert self._connection is not None
        # The name stays tracked in ``_staging_tables`` so failure cleanup can
        # drop it by exact name: on SQLite a rolled-back transaction can revive a
        # staging table this drop already removed, and globbing ``bulkstage_*``
        # would risk a user table that legitimately uses the prefix.
        self._connection.execute(sqlalchemy.schema.DropTable(stage, if_exists=True))

    def _drop_hit_rows(self, name: str, schema: TableSchema, sid_map: Mapping[int, int]) -> None:
        """Drop the deduplicated parent rows and suppress their buffered child rows."""
        hit_sids = set(sid_map)
        rows = self._rows.get(name)
        if rows is not None:
            rows[:] = [row for row in rows if row[SID_COLUMN] not in hit_sids]
        parent_column = f"{name}_sid"
        for spec in schema.fields:
            if spec.role != "child":
                continue
            assert spec.child is not None
            child_rows = self._rows.get(spec.child.table_name)
            if child_rows:
                child_rows[:] = [row for row in child_rows if row.get(parent_column) not in hit_sids]

    def _apply_remap(
        self, ref_table: str, sid_map: Mapping[int, int], fk_columns: Mapping[str, list[tuple[str, str]]]
    ) -> None:
        """Rewrite every still-buffered sid that references ``ref_table`` to its deduplicated existing sid."""
        for table_name, buffered in self._rows.items():
            columns = [column for column, target in fk_columns.get(table_name, ()) if target == ref_table]
            if not columns:
                continue
            for row in buffered:
                for column in columns:
                    value = row.get(column)
                    if value is not None and value in sid_map:
                        row[column] = sid_map[value]
        for dispatch_name, bucket in self._dispatch_rows.items():
            columns = [column for column, target in fk_columns.get(dispatch_name, ()) if target == ref_table]
            if not columns:
                continue
            for row in bucket.values():
                for column in columns:
                    value = row.get(column)
                    if value is not None and value in sid_map:
                        row[column] = sid_map[value]

    def _collect_garbage(self, fk_columns: Mapping[str, list[tuple[str, str]]]) -> None:
        """Sweep buffered rows no longer reachable from a surviving top-level save of this chunk.

        A flush-time dedup hit drops the hit parent (and its child rows), which
        can orphan descendants the eager encoder buffered — referenced records,
        ``dedup="none"`` records, and child-element records — that ``save()``
        would never have created because its hit short-circuits before them. This
        marks every buffered row reachable from a surviving chunk root and drops
        the rest, converging the final state to the per-record loop.

        :param fk_columns: Each table's ``(column, referenced_table)`` sid foreign keys.
        """
        # child table -> (parent table, parent-sid column)
        child_of: dict[str, tuple[str, str]] = {}
        for parent_name, schema in self._parent_schema.items():
            for spec in schema.fields:
                if spec.role == "child" and spec.child is not None:
                    child_of[spec.child.table_name] = (parent_name, f"{parent_name}_sid")

        parent_by_key: dict[tuple[str, int], dict[str, Any]] = {}
        for name, rows in self._rows.items():
            if name in self._parent_schema:
                for row in rows:
                    parent_by_key[(name, row[SID_COLUMN])] = row
        children_index: dict[tuple[str, int], list[tuple[str, dict[str, Any]]]] = {}
        for child_table, (parent_table, parent_column) in child_of.items():
            for row in self._rows.get(child_table, ()):
                parent_sid = row.get(parent_column)
                if parent_sid is not None:
                    children_index.setdefault((parent_table, parent_sid), []).append((child_table, row))

        marked: set[int] = set()  # id() of rows to keep
        seen: set[tuple[str, int]] = set()
        stack = [key for key in self._chunk_roots if key in parent_by_key]
        while stack:
            key = stack.pop()
            if key in seen:
                continue
            seen.add(key)
            parent_row = parent_by_key.get(key)
            if parent_row is None:
                continue
            marked.add(id(parent_row))
            for column, ref_table in fk_columns.get(key[0], ()):
                value = parent_row.get(column)
                if value is not None and (ref_table, value) in parent_by_key:
                    stack.append((ref_table, value))
            for child_table, child_row in children_index.get(key, ()):
                marked.add(id(child_row))
                for column, ref_table in fk_columns.get(child_table, ()):
                    value = child_row.get(column)
                    if value is not None and (ref_table, value) in parent_by_key:
                        stack.append((ref_table, value))

        for name, rows in self._rows.items():
            if not rows or all(id(row) in marked for row in rows):
                continue
            sweep_schema = self._parent_schema.get(name)
            kept: list[dict[str, Any]] = []
            for row in rows:
                if id(row) in marked:
                    kept.append(row)
                    continue
                # An orphaned parent row must also drop its dedup-index entry so
                # a later chunk re-encodes the record fresh rather than resolving
                # to a swept, never-inserted sid.
                if sweep_schema is not None and sweep_schema.dedup == "content_id":
                    content_map = self._content_index.get(name)
                    content_key = row.get(CONTENT_ID_COLUMN)
                    if (
                        content_map is not None
                        and isinstance(content_key, str)
                        and content_map.get(content_key) == row[SID_COLUMN]
                    ):
                        del content_map[content_key]
                elif sweep_schema is not None and sweep_schema.dedup == "by_value":
                    value_map = self._value_index.get(name)
                    if value_map is not None:
                        value_key = _value_tuple(row)
                        if value_map.get(value_key) == row[SID_COLUMN]:
                            del value_map[value_key]
            rows[:] = kept

    def _refresh_value_index(self) -> None:
        """Re-key each surviving buffered by_value row after remapping, so later chunks still deduplicate in memory."""
        for name, rows in self._rows.items():
            schema = self._parent_schema.get(name)
            if schema is None or schema.dedup != "by_value" or not rows:
                continue
            value_map = self._value_index.setdefault(name, {})
            for row in rows:
                value_map[_value_tuple(row)] = row[SID_COLUMN]

    def _logical_graph(self, extra: Iterable[TableSchema] = ()) -> LogicalEdgeGraph:
        """Return the schema-derived graph for this ingest's registered tables."""
        schemas = tuple(self._parent_schema.values()) + tuple(extra)
        return LogicalEdgeGraph.from_store(self._store, schemas)

    def _build_fk_columns(self) -> dict[str, list[tuple[str, str]]]:
        """Compatibility name for the logical sid-column map used by remapping."""
        return self._logical_graph().sid_columns()

    def _decide_index(self, name: str) -> None:
        """Before a pre-existing table's first append, drop its separable indexes if the strategy asks."""
        if name not in self._preexisting or name in self._index_decided:
            return
        self._index_decided.add(name)
        if self._index_strategy == "keep":
            return
        if self._index_strategy == "auto":
            existing = self._existing_row_count.get(name, 0)
            staged = self._next_sid.get(name, 1) - self._initial_next_sid.get(name, 1)
            if staged * _AUTO_REBUILD_DIVISOR <= existing:
                return
        assert self._connection is not None
        if self._connection.dialect.name == "duckdb":
            # DuckDB reserves a dropped index's name until commit, so an
            # in-transaction drop-then-recreate of the same index is rejected.
            # Keep the indexes (DuckDB maintains them incrementally through the
            # append) and verify content-id uniqueness with a duplicate scan.
            self._rebuild_scan_tables.add(name)
            return
        table = self._store._table(name)
        for index in table.indexes:
            self._connection.execute(sqlalchemy.schema.DropIndex(index, if_exists=True))
            self._dropped_indexes.append(index)

    # ------------------------------------------------------------------ in-memory metadata comparison

    def _verify_existing_metadata(self, hits: list[tuple[int, int, str]]) -> None:
        """Compare each content-id hit's identity-excluded metadata against the stored row, like save()."""
        assert self._connection is not None
        store = self._store
        stack = store._connection_stack()
        stack.append(self._connection)
        try:
            for _staged_sid, existing_sid, key in hits:
                entry = self._meta_sources.get(key)
                if entry is None:
                    continue
                record_type, source = entry
                store._check_metadata(self._connection, record_type, existing_sid, source, SaveProjection())
        finally:
            stack.pop()

    def _check_hit_metadata(
        self, record_type: type, key: str, incoming: Mapping[str, object], source: Any, existing_sid: int
    ) -> None:
        """Verify an in-memory content hit's identity-excluded metadata against the first occurrence.

        Within the chunk the first occurrence is still buffered, so the
        comparison runs in memory. Once the first occurrence has flushed (its
        projected metadata pruned), a later chunk's hit verifies against the
        stored row instead — exactly as :meth:`~httk.store.db.store.SqlStore.save`.

        :param record_type: The record type of the hit.
        :param key: The content id that hit.
        :param incoming: The projected fields of the current occurrence.
        :param source: The current occurrence's source object (for the stored-row comparison).
        :param existing_sid: The sid the content id resolves to.
        """
        stored = self._meta_values.get(key)
        if stored is not None:
            self._compare_metadata(record_type, incoming, stored, record_type.__name__)
            return
        if _metadata_plan(record_type) is None:
            return
        store = self._store
        assert self._connection is not None
        stack = store._connection_stack()
        stack.append(self._connection)
        try:
            store._check_metadata(self._connection, record_type, existing_sid, source, SaveProjection())
        finally:
            stack.pop()

    def _compare_metadata(
        self,
        record_type: type,
        incoming: Mapping[str, object],
        stored: Mapping[str, object],
        path: str,
    ) -> None:
        plan = _metadata_plan(record_type)
        if plan is None:
            return
        schema = resolve_schema(record_type)
        skipped = {spec.field for spec in plan.skipped_specs}
        skipped_nested = {spec.field for spec in plan.skipped_nested}
        descend = {spec.field for spec in plan.descend_specs}
        for spec in schema.fields:
            if spec.derived:
                continue
            field_path = _field_path(path, spec.field)
            if spec.field in skipped:
                incoming_value = incoming[spec.field]
                stored_value = stored[spec.field]
                if not _metadata_scalar_equal(incoming_value, stored_value):
                    raise EntryMetadataConflictError(
                        f"metadata conflict for {field_path}: stored {stored_value!r}, received {incoming_value!r}"
                    )
            elif spec.field in skipped_nested:
                self._compare_nested(spec, incoming[spec.field], stored[spec.field], field_path, compare_content=True)
            elif spec.field in descend:
                self._compare_nested(spec, incoming[spec.field], stored[spec.field], field_path, compare_content=False)

    def _compare_nested(
        self,
        spec: FieldSpec,
        incoming: Any,
        stored: Any,
        path: str,
        *,
        compare_content: bool,
    ) -> None:
        if spec.role == "reference":
            assert spec.target is not None
            if incoming is None or stored is None:
                if incoming is not None or stored is not None:
                    raise EntryMetadataConflictError(
                        f"metadata conflict for {path}: stored {stored!r}, received {incoming!r}"
                    )
                return
            self._compare_target(spec.target, incoming, stored, path, compare_content=compare_content)
            return
        if spec.target is None:
            # A non-storable child sequence: compare the projected values whole,
            # exactly as save() compares the decoded child list.
            if not _metadata_scalar_equal(incoming, stored):
                raise EntryMetadataConflictError(
                    f"metadata conflict for {path}: stored {stored!r}, received {incoming!r}"
                )
            return
        if incoming is None or stored is None:
            if incoming is not stored:
                raise EntryMetadataConflictError(
                    f"metadata conflict for {path}: stored {stored!r}, received {incoming!r}"
                )
            return
        if len(incoming) != len(stored):
            raise EntryMetadataConflictError(f"metadata conflict for {path}: stored {stored!r}, received {incoming!r}")
        for index, (incoming_item, stored_item) in enumerate(zip(incoming, stored, strict=True)):
            self._compare_target(
                spec.target, incoming_item, stored_item, f"{path}[{index}]", compare_content=compare_content
            )

    def _compare_target(
        self,
        record_type: type,
        incoming: Any,
        stored: Any,
        path: str,
        *,
        compare_content: bool,
    ) -> None:
        if compare_content and content_id(incoming, as_record=record_type) != content_id(stored, as_record=record_type):
            raise EntryMetadataConflictError(f"metadata conflict for {path}: stored {stored!r}, received {incoming!r}")
        if _metadata_plan(record_type) is not None:
            self._compare_metadata(
                record_type,
                project_storage_record(record_type, incoming),
                project_storage_record(record_type, stored),
                path,
            )


def _value_tuple(row: Mapping[str, Any]) -> tuple[Any, ...]:
    """The whole-parent-column dedup key of a by_value row (its sid excluded)."""
    return tuple(
        sorted(
            (name, value)
            for name, value in row.items()
            if name not in (SID_COLUMN, ROLE_COLUMN, TS_START_COLUMN, TS_END_COLUMN, REPLACED_BY_COLUMN)
        )
    )
