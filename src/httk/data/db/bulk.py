"""Bulk ingestion for :class:`~httk.data.db.store.SqlStore`.

:class:`BulkIngest` is the context manager returned by
:meth:`~httk.data.db.store.SqlStore.bulk_ingest`. It replaces the per-record
``save()`` loop: instead of one statement round-trip per row with an
in-database deduplication protocol, it encodes each object with the pure
encoders in :mod:`httk.data.db.store` (``_encode_parent_row`` and
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
  hits, reproducing :meth:`~httk.data.db.store.SqlStore.save`), a ``by_value``
  whole-parent-column anti-join with null-safe equality, and a sid remap that
  rewrites every still-buffered reference to a deduplicated existing sid before
  it is flushed. The ``index_strategy`` knob chooses whether existing tables
  keep their indexes during the append or rebuild them at the end.

Deduplication mirrors :meth:`~httk.data.db.store.SqlStore.save` set-wise: a
``"content_id"`` table keeps a ``content_id -> sid`` map (a hit returns the
mapped sid and buffers neither the parent row nor its children, and — unless
``verify_metadata`` is disabled — compares
:class:`~httk.core.storage.markers.IdentitySkip` metadata against the first
occurrence in memory, or against the stored row for a hit against existing data,
raising :class:`~httk.data.store_common.EntryMetadataConflictError`); a
``"by_value"`` table keeps a whole-parent-column-tuple map (a hit returns the
mapped sid with no metadata check); a ``"none"`` table always inserts.
Multi-record entry families buffer one deduplicated dispatch row per content id,
raising :class:`~httk.data.store_common.EntryDispatchIntegrityError` on a
conflicting backing.

A third, opt-in mode parallelizes the encode. ``bulk_ingest(workers=N)`` with
``N > 1`` forks a pool of worker processes (the ``fork`` start method, so each
inherits the unpicklable store and never touches its database) and pickles each
saved object onto a shared task queue. Every worker runs the *same* pure
encoders against a per-worker :class:`~httk.data.store_common.SaveProjection`,
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
path. The implementation lives in :mod:`httk.data.db.bulk_parallel`; see its
module docstring for the full contract. ``workers=1`` (the default) is exactly
the serial path described above, unchanged.

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
from collections.abc import Callable, Mapping
from types import TracebackType
from typing import Any, Literal, Self

import sqlalchemy
from httk.core.storage import (
    StorageProjectionCycleError,
    content_id,
    project_storage_record,
    resolve_storage_record,
)

from httk.data.db.layout import METADATA_TABLE_NAME, actual_schema_objects
from httk.data.db.mapping import (
    CONTENT_ID_COLUMN,
    DISPATCH_CONTENT_ID_COLUMN,
    SID_COLUMN,
    backing_dispatch_column_name,
    entry_dispatch_table_name,
)
from httk.data.db.schema import FieldSpec, TableSchema, resolve_schema
from httk.data.db.store import (
    SqlStore,
    _encode_child_rows,
    _encode_parent_row,
    _field_path,
    _metadata_scalar_equal,
)
from httk.data.store_common import (
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


def _foreign_key_free_clone(table: sqlalchemy.Table) -> sqlalchemy.Table:
    """A structural copy of ``table`` without foreign-key constraints (indexes are built separately).

    Preserves each column's type, nullability, primary-key membership, and the
    sid column's sequence default, but drops every ``ForeignKey`` so the parallel
    merge can renumber and delete rows in place. The separable indexes are added
    later by :meth:`BulkIngest._create_new_indexes`, exactly as for the serial
    empty-store path.

    :param table: The registered record or child table to clone.
    :return: A detached table of the same name and columns without foreign keys.
    """
    columns: list[sqlalchemy.Column[Any]] = []
    for column in table.columns:
        arguments: list[Any] = []
        default = column.default
        if isinstance(default, sqlalchemy.Sequence):
            arguments.append(sqlalchemy.Sequence(default.name))
        columns.append(
            sqlalchemy.Column(
                column.name,
                column.type,
                *arguments,
                primary_key=column.primary_key,
                nullable=column.nullable,
                autoincrement=column.autoincrement,
            )
        )
    return sqlalchemy.Table(table.name, sqlalchemy.MetaData(), *columns)


def _sid_sequence(table: sqlalchemy.Table) -> sqlalchemy.Sequence | None:
    """Return the sid primary key's attached sequence, or ``None`` for a child table."""
    if SID_COLUMN not in table.c:
        return None
    default = table.c[SID_COLUMN].default
    return default if isinstance(default, sqlalchemy.Sequence) else None


class BulkIngest:
    """Append a stream of storable objects into a store, then verify its indexes.

    Instances are produced by :meth:`~httk.data.db.store.SqlStore.bulk_ingest`
    and used as a context manager. Inside the ``with`` block, :meth:`save`
    encodes and buffers objects; on clean exit the buffered rows are flushed
    (staged and resolved set-wise against any existing rows), the separable
    indexes are created or rebuilt (verifying uniqueness), DuckDB sid sequences
    are resynchronized, per-table row counts are asserted against the encoder's
    bookkeeping, and the single spanning transaction commits. On any exception
    the transaction rolls back, every table the context created is dropped, any
    index the context dropped is restored, staging tables are removed, and the
    store's identity caches are cleared, leaving the store exactly as it was
    before the context opened.

    :param store: The store to ingest into.
    :param chunk_size: The number of top-level saves buffered before a flush.
    :param verify_metadata: Whether content-id hits compare identity-excluded metadata.
    :param index_strategy: How existing tables' separable indexes are handled during the append.
    :param on_progress: An optional ``(records_buffered_total, rows_flushed_total)`` callback invoked after each flush.
    :param workers: The number of worker processes; ``1`` (the default) is the serial path, ``>1`` encodes in parallel and merges shards.
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
    ) -> None:
        if chunk_size < 1:
            raise ValueError("chunk_size must be a positive integer")
        if index_strategy not in ("auto", "keep", "rebuild"):
            raise ValueError("index_strategy must be one of 'auto', 'keep', or 'rebuild'")
        if workers < 1:
            raise ValueError("workers must be a positive integer")
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
        # Parallel-mode state (unused on the serial path).
        self._controller: Any = None
        self._next_token = 0
        self._schema_graph_seen: set[type] = set()
        # SQLite shard aliases the merge attached; detached in _release_connection
        # after the transaction closes (SQLite forbids DETACH inside a transaction).
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

        # Progress counters.
        self._records_total = 0
        self._rows_flushed_total = 0
        self._since_flush = 0

    # ------------------------------------------------------------------ context management

    def __enter__(self) -> Self:
        store = self._store
        if store._bulk_active:
            raise RuntimeError("this SqlStore already has an open bulk_ingest context")
        if store._current_connection() is not None:
            raise RuntimeError(
                "bulk_ingest cannot be opened inside an open store.transaction() or write scope on this thread; "
                "the ingest owns its own spanning transaction"
            )
        store._bulk_active = True
        # The worker pool is forked before the spanning transaction opens, so no
        # child inherits an open database connection or transaction.
        if self._parallel:
            try:
                self._start_workers()
            except BaseException:
                store._bulk_active = False
                raise
        # Own the connection explicitly (not engine.begin(), which returns it to
        # the pool on commit): the SQLite shard DETACH must run on this exact
        # connection *after* the transaction closes but *before* it is released,
        # so no other thread can check it out in that window.
        connection = None
        try:
            connection = store._database.engine.connect()
            transaction = connection.begin()
        except BaseException:
            if connection is not None:
                connection.close()
            self._close_workers()
            store._bulk_active = False
            raise
        self._transaction = transaction
        self._connection = connection
        try:
            self._preexisting = self._scan_store(connection)
            if self._parallel:
                self._require_empty_store(connection)
                self._require_no_foreign_key_enforcement(connection)
        except BaseException:
            transaction.rollback()
            connection.close()
            self._close_workers()
            store._bulk_active = False
            raise
        self._entered = True
        return self

    def _start_workers(self) -> None:
        """Validate the parallel prerequisites and fork the worker pool."""
        from httk.data.db.bulk_parallel import ParallelController

        backend = self._store._database.engine.dialect.name
        if backend == "duckdb":
            try:
                import importlib

                importlib.import_module("pyarrow")
            except ImportError as error:
                raise ImportError(
                    "bulk_ingest(workers>1) on a DuckDB store needs pyarrow; "
                    "install the 'httk-data[parallel]' extra to use it"
                ) from error
        self._controller = ParallelController(
            self._store, workers=self._workers, chunk_size=self._chunk_size, backend=backend
        )
        self._controller.start()

    def _close_workers(self) -> None:
        if self._controller is not None:
            self._controller.close()
            self._controller = None

    def _require_empty_store(self, connection: sqlalchemy.Connection) -> None:
        """Refuse parallel ingest into a store the merge cannot treat as a clean build.

        On DuckDB a pre-existing application table already carries its foreign-key
        constraints, which the merge's in-place collapse and renumber cannot work
        through, so *any* pre-existing application table is refused. On SQLite —
        which does not enforce foreign keys unless the engine turns them on (see
        :meth:`_require_no_foreign_key_enforcement`) — only pre-existing rows are
        refused.
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
            if exc_type is None:
                try:
                    self._finalize()
                except BaseException:
                    transaction.rollback()
                    self._release_connection(connection)  # detach shards, then release to the pool
                    self._clean_up_after_failure()
                    raise
                try:
                    transaction.commit()
                except BaseException:
                    # A failing commit still needs the created tables dropped,
                    # dropped indexes restored, and staging tables removed.
                    self._release_connection(connection)
                    self._clean_up_after_failure()
                    raise
                self._release_connection(connection)
                store._tables_present.update(self._created)
                self._final_sids_ready = True
                return
            transaction.rollback()
            self._release_connection(connection)
            self._clean_up_after_failure()
        finally:
            self._release_connection(connection)  # idempotent: no-op if already released
            self._close_workers()
            self._connection = None
            self._transaction = None
            store._bulk_active = False

    def _release_connection(self, connection: sqlalchemy.Connection | None) -> None:
        """Detach any SQLite shards on ``connection`` and return it to the pool (idempotent).

        SQLite forbids ``DETACH`` inside a transaction, so this runs only after
        the spanning transaction has committed or rolled back — and on the exact
        connection that ran the ``ATTACH``, before it is released, so no other
        thread can check it out with the shards still attached.
        """
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
        connection.close()

    def _require_no_foreign_key_enforcement(self, connection: sqlalchemy.Connection) -> None:
        """Refuse a parallel SQLite ingest when the engine enforces foreign keys.

        The merge collapses and renumbers rows in place; a live foreign key would
        block that. SQLite does not enforce foreign keys by default, but a
        user-supplied engine can turn ``PRAGMA foreign_keys`` on, so this checks
        the ingest connection and refuses rather than silently corrupting.
        """
        if connection.dialect.name != "sqlite":
            return
        enforced = connection.exec_driver_sql("PRAGMA foreign_keys").scalar()
        if enforced:
            raise RuntimeError(
                "bulk_ingest(workers>1) on SQLite requires foreign-key enforcement to be off "
                "(PRAGMA foreign_keys=OFF): the merge collapses and renumbers rows in place, which a "
                "live foreign key would block. Disable enforcement on the engine, or use workers=1."
            )

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
                    if cleanup.dialect.name == "duckdb":
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

    def save(self, obj: Any, *, as_record: type | None = None) -> int:
        """Encode and buffer ``obj``, returning its assigned or deduplicated sid.

        Mirrors :meth:`~httk.data.db.store.SqlStore.save`: an opted-in domain
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
        :return: The provisional sid (see :meth:`resolved_sid` for the durable one).
        :raises RuntimeError: If the bulk context is not open.
        :raises TypeError: If ``obj`` is a cursor row that must be materialized first.
        :raises httk.data.store_common.EntryMetadataConflictError: If a content-id hit has conflicting metadata.
        :raises httk.data.store_common.EntryDispatchIntegrityError: If a dispatch content id maps to a conflicting backing.
        :raises httk.core.storage.identity.StorageProjectionCycleError: If projection reaches a reference cycle.
        """
        if not self._entered or self._closed:
            raise RuntimeError("bulk_ingest().save() is only usable inside an open bulk context")
        reject_cursor_proxy(obj)
        if self._parallel:
            return self._parallel_save(obj, as_record)
        record_type = resolve_storage_record(obj, as_record=as_record)
        projection = SaveProjection()
        sid = self._encode(record_type, obj, projection, "")
        table_name = resolve_schema(record_type).table_name
        family = self._store._family_for_backing(record_type)
        if family is not None and len(family.records) > 1:
            self._buffer_dispatch(family, record_type, sid, projection.content_id(record_type, obj))
        self._returned_sids.add((table_name, sid))
        self._chunk_roots.append((table_name, sid))
        self._records_total += 1
        self._since_flush += 1
        if self._since_flush >= self._chunk_size:
            self._flush()
        return sid

    def resolved_sid(self, record_type: type, sid: int) -> int:
        """Map a sid returned by :meth:`save` to its durable stored sid after the context exits.

        A newly inserted object's provisional sid resolves to itself; a sid that
        deduplicated against a pre-existing row resolves to that existing row's
        sid. This is the durable lookup for provisional sids (see :meth:`save`).

        Sids are allocated per table, so both the record type the sid was saved
        as (the same class :meth:`~httk.data.db.store.SqlStore.fetch` takes) and
        the sid are required to identify it unambiguously.

        :param record_type: The stored record class the sid was saved as.
        :param sid: A sid previously returned by :meth:`save`.
        :return: The durable stored sid.
        :raises RuntimeError: If the bulk context has not yet exited cleanly (resolution is incomplete).
        :raises KeyError: If ``(record_type, sid)`` was never returned by this ingest's :meth:`save`.
        """
        if not self._final_sids_ready:
            raise RuntimeError("resolved_sid is only available after the bulk_ingest context has exited cleanly")
        table_name = resolve_schema(record_type).table_name
        if (table_name, sid) not in self._returned_sids:
            raise KeyError((record_type, sid))
        return self._resolved_map.get((table_name, sid), sid)

    def _parallel_save(self, obj: Any, as_record: type | None) -> int:
        """Dispatch ``obj`` to a worker and return a provisional token resolved after the merge.

        In parallel mode the encode happens asynchronously in a worker, so the
        sid is not known synchronously. ``save`` instead returns a unique token
        that :meth:`resolved_sid` maps to the durable stored sid once the context
        has exited cleanly. The token is a proper stand-in: it is never a real
        row sid, and every equivalence guarantee flows through
        :meth:`resolved_sid`.

        :param obj: The object to store.
        :param as_record: The alternate record representation to use, if any.
        :return: A provisional token (see :meth:`resolved_sid`).
        """
        record_type = resolve_storage_record(obj, as_record=as_record)
        # Validate the metadata shape (and record the schema graph) before any DDL,
        # so a rejected type fails fast without leaving an empty table behind.
        self._record_schema_graph(record_type)
        self._ensure_tables(record_type)
        table_name = resolve_schema(record_type).table_name
        token = self._next_token
        self._next_token += 1
        self._returned_sids.add((table_name, token))
        self._records_total += 1
        assert self._controller is not None
        self._controller.dispatch(token, obj, as_record)
        return token

    def _record_schema_graph(self, record_type: type) -> None:
        """Validate and record every record table's schema in the graph rooted at ``record_type``.

        Rejects, up front, any record type whose identity-excluded metadata shape
        the set-wise merge cannot verify (see
        :func:`~httk.data.db.bulk_parallel.unsupported_metadata_reason`), so an
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
            from httk.data.db.bulk_parallel import unsupported_metadata_reason

            reason = unsupported_metadata_reason(record_type)
            if reason is not None:
                raise ValueError(
                    f"bulk_ingest(workers>1) cannot verify this identity-excluded metadata shape: {reason}. "
                    "Use workers=1 for records of this kind, or open with verify_metadata=False."
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

        key: str | None = None
        if schema.dedup == "content_id":
            key = projection.content_id(record_type, source)
            existing = self._content_index[table_name].get(key)
            if existing is not None:
                if self._verify_metadata:
                    self._check_hit_metadata(record_type, key, projected, source, existing)
                return existing

        def resolve_sid(referenced_type: type, value: Any, field_path: str) -> int:
            return self._encode(referenced_type, value, projection, field_path)

        values = _encode_parent_row(schema, source, projected, path, resolve_sid)

        if schema.dedup == "by_value":
            value_tuple = tuple(sorted(values.items()))
            existing = self._value_index[table_name].get(value_tuple)
            if existing is not None:
                return existing

        sid = self._next_sid[table_name]
        self._next_sid[table_name] = sid + 1
        row = {SID_COLUMN: sid, **values}
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
        for table in candidate.sorted_tables:
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
        if self._connection.dialect.name == "duckdb":
            sequence = _sid_sequence(table)
            if sequence is not None:
                # The sid column renders as DEFAULT nextval(<seq>); DuckDB needs
                # the sequence to exist at CREATE TABLE time for that default to
                # bind, otherwise ordinary post-ingest saves lose their sid
                # allocator. SQLite ignores the sequence entirely.
                self._connection.execute(sqlalchemy.text(f'CREATE SEQUENCE IF NOT EXISTS "{sequence.name}"'))
        create = table
        if self._parallel and self._connection.dialect.name == "duckdb" and not table.name.startswith("_httk_"):
            # The parallel merge remaps and deletes rows in place while collapsing
            # cross-worker duplicates and compacting the block sids. DuckDB refuses
            # to delete or renumber a row a foreign key still references, so on
            # DuckDB record and child tables are created without their reference
            # constraints. SQLite keeps them: it does not enforce foreign keys
            # unless the engine turns them on, which a parallel ingest refuses.
            # Every other structure — the sid primary key and sequence, the
            # content-id uniqueness index, and the separable indexes — is unchanged.
            create = _foreign_key_free_clone(table)
        # A bare CreateTable (not create_all / Table.create) so the separable
        # indexes stay out until the deferred post-load build.
        self._connection.execute(sqlalchemy.schema.CreateTable(create))

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
        for table in store._metadata.sorted_tables:
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
        for table in store._metadata.sorted_tables:
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
        from httk.data.db.bulk_parallel import merge

        assert self._controller is not None
        manifests = self._controller.finish()  # re-raises the first worker exception
        self._assert_no_lost_tasks(manifests)
        merge(self, manifests)
        self._create_new_indexes()
        self._resync_sequences()
        self._assert_counts()

    def _assert_no_lost_tasks(self, manifests: list[Any]) -> None:
        """Abort (never commit) if any dispatched task did not come back encoded in a worker manifest."""
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
        if self._connection.dialect.name != "duckdb":
            # SQLite's rowid self-syncs to max+1; only DuckDB's explicit
            # sequence must be advanced past the pre-assigned sids.
            return
        for name, next_sid in self._next_sid.items():
            sequence = _sid_sequence(self._store._table(name))
            if sequence is None:
                continue
            self._connection.execute(
                sqlalchemy.text(f'CREATE OR REPLACE SEQUENCE "{sequence.name}" START WITH {next_sid}')
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
            self._drop_hit_rows(name, schema, sid_map)
            self._apply_remap(name, sid_map, fk_columns)
        return found_any

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
            value_columns = [column.name for column in table.columns if column.name != SID_COLUMN]
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

    def _build_fk_columns(self) -> dict[str, list[tuple[str, str]]]:
        """Map each registered table to its ``(column, referenced_table)`` sid foreign keys."""
        result: dict[str, list[tuple[str, str]]] = {}
        for table_name, table in self._store._metadata.tables.items():
            columns: list[tuple[str, str]] = []
            for column in table.columns:
                for foreign_key in column.foreign_keys:
                    columns.append((column.name, foreign_key.column.table.name))
            if columns:
                result[table_name] = columns
        return result

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
        stored row instead — exactly as :meth:`~httk.data.db.store.SqlStore.save`.

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
    return tuple(sorted((name, value) for name, value in row.items() if name != SID_COLUMN))
