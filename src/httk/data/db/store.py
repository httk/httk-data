"""The SQL store: save and fetch storable frozen dataclasses through a :class:`~httk.data.db.engine.Database`.

:class:`SqlStore` is the object-level storage API on top of the schema IR
(:mod:`httk.data.db.schema`), the value codecs (:mod:`httk.data.db.codecs`),
the content identity (:mod:`httk.core.storage`), and the SQLAlchemy table
mapping (:mod:`httk.data.db.mapping`):

- :meth:`SqlStore.save` writes an instance (recursing into referenced and
  child-element storables) and returns its integer ``sid``, deduplicating per
  the class's :attr:`~httk.core.storage.StorageInfo.dedup` policy;
- :meth:`SqlStore.fetch` reconstructs the instance stored under a ``sid`` —
  exactly, via the ``*_exact`` companion columns for rationals — with an
  identity guarantee: while an instance is alive, fetching its sid again
  returns the very same object;
- :meth:`SqlStore.transaction` scopes several operations into one database
  transaction (commit on exit, roll back on exception); outside of it every
  operation autocommits;
- :meth:`SqlStore.referring` finds join-objects (tags, references) pointing at
  a stored instance, replacing v1's implicit codependent-data machinery;
- :meth:`SqlStore.searcher` starts a query through the search DSL
  (:mod:`httk.data.db.searcher`), implementing the :mod:`httk.data.query`
  protocols.

Deduplication semantics (ported from v1): under ``"content_id"`` an equal
instance maps to the existing row (children are not re-inserted); under
``"by_value"`` a row matching **all parent-table columns** is reused — child
table contents are *not* part of the match, mirroring v1 which matched key
columns only; under ``"none"`` every save inserts a new row.

Identity caches are best-effort; content-addressed :meth:`SqlStore.sid_of`
lookups fall back to the database.
"""

import contextlib
import datetime
import threading
import typing
from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import TYPE_CHECKING, Any, Literal, cast

import sqlalchemy
from httk.core import (
    FracVector,
)
from httk.core.storage import (
    Shape,
    StorageProjectionCycleError,
    resolve_storage_record,
)
from sqlalchemy.exc import SQLAlchemyError

from httk.data.db.codecs import (
    codec_named,
    decode_fracvector_exact,
    encode_fracvector_exact,
    encode_fracvector_floats,
)
from httk.data.db.engine import Database
from httk.data.db.layout import (
    METADATA_TABLE_NAME,
    STORAGE_PROTOCOL_VERSION,
    EntryFamilyLayout,
    StorageLayout,
    StorageLayoutUpgradeRequiredError,
    actual_schema_objects,
    actual_table_names,
    declaration_json,
    expected_metadata,
    metadata_table_for,
    normalize_entry_records,
    read_store_metadata,
)
from httk.data.db.mapping import (
    CONTENT_ID_COLUMN,
    DISPATCH_CONTENT_ID_COLUMN,
    SID_COLUMN,
    backing_dispatch_column_name,
    dispatch_table_for,
    entry_dispatch_table_name,
    table_for,
)
from httk.data.db.rows import RowHydrator, StaleResultError, decode_field, is_lazy_row, lazy_row_identity
from httk.data.db.schema import FieldSpec, SchemaError, TableSchema, resolve_schema
from httk.data.db.searcher import SqlSearcher
from httk.data.store_common import (
    _MISSING_METADATA,
    EntryDispatchIntegrityError,
    EntryMetadataConflictError,
    IdentityCaches,
    SaveProjection,
    _metadata_plan,
    _MetadataPlan,
    reject_cursor_proxy,
)

if TYPE_CHECKING:
    from httk.data.db.bulk import BulkIngest

_Projection = SaveProjection

# A sid resolver assigns (by saving recursively, or by an in-memory allocator)
# an integer sid to a referenced record: ``(record_type, source, path) -> sid``.
type SidResolver = Callable[[type, Any, str], int]

__all__ = [
    "EntryDispatchIntegrityError",
    "EntryMetadataConflictError",
    "SqlStore",
]


def _schema_object_type(kinds: frozenset[str]) -> object:
    return next(iter(kinds)) if len(kinds) == 1 else tuple(sorted(kinds))


class SqlStore:
    """Object storage for storable frozen dataclasses in a relational :class:`~httk.data.db.engine.Database`.

    A store starts with an explicit, versioned entry declaration. Ordinary
    unconfigured frozen-dataclass tables remain on-demand, but only after the
    layout marker has been initialized on a physically empty database. Schemas
    edited out-of-band fail at use time with the database's own errors.

    The first open of a database requires ``entry_records``. The store stamps
    the canonical JSON declaration and protocol version, then trusts that
    declaration on reopen: a supplied declaration must be byte-identical, and
    mismatches raise :class:`~httk.data.db.layout.StorageLayoutUpgradeRequiredError`.
    Reopening does not diff or migrate record schemas. Read paths never issue
    DDL; missing ordinary tables behave as empty results or missing rows, while
    table creation happens only through writes or :meth:`ensure_tables`.

    :param database: The database used for storage.
    :param entry_records: The required entry-family declaration when first opening a database.
    :raises TypeError: If the first open omits ``entry_records``.
    :raises httk.data.db.layout.StorageLayoutUpgradeRequiredError: If the trusted declaration or protocol does not match.
    """

    def __init__(
        self,
        database: Database,
        *,
        entry_records: Mapping[type, type | tuple[type, ...]] | None = None,
    ) -> None:
        self._database = database
        self._metadata = sqlalchemy.MetaData()
        self._layout: StorageLayout | None = None
        self._managed_table_names: frozenset[str] = frozenset()
        self._tables_present: set[str] = set()
        self._initialized = False
        self._initialization_ddl_journal: list[sqlalchemy.Table] = []
        self._identity = IdentityCaches()
        self._local = threading.local()
        self._bulk_active = False
        self._initialize_layout(entry_records)

    @property
    def layout(self) -> StorageLayout:
        """Return the immutable persisted entry declaration and resolved classes.

        :return: The persisted storage layout.
        """
        assert self._layout is not None
        return self._layout

    @property
    def _instances(self) -> Any:
        """Compatibility view of the shared instance cache for the SQL hydrator."""
        return self._identity._instances

    @property
    def _sids_by_identity(self) -> Any:
        """Compatibility view of the unhashable-instance reverse cache."""
        return self._identity._sids_by_identity

    @property
    def entry_layout(self) -> tuple[EntryFamilyLayout, ...]:
        """Return configured entry-family layouts in deterministic stable-name order.

        :return: The configured entry-family layouts.
        """
        return self.layout.families

    @property
    def entry_records(self) -> Mapping[type, tuple[type, ...]]:
        """Return configured entry-family classes mapped to ordered concrete records.

        :return: The entry-family to concrete-record mapping.
        """
        return self.layout.entry_records

    def _initialize_layout(self, entry_records: Mapping[type, type | tuple[type, ...]] | None) -> None:
        supplied = normalize_entry_records(entry_records) if entry_records is not None else None
        try:
            with self._database.engine.begin() as connection:
                self._initialize_layout_on_connection(connection, supplied)
        except BaseException:
            created_tables = tuple(self._initialization_ddl_journal)
            self._initialization_ddl_journal.clear()
            # The transaction context has now unwound. This ordering matters
            # for SQLite, whose DDL can survive SQLAlchemy's outer rollback;
            # opening the cleanup transaction while the original one is still
            # active would merely fail against its shared connection.
            if created_tables:
                self._cleanup_initialization_tables(created_tables)
            self._metadata = sqlalchemy.MetaData()
            self._layout = None
            self._managed_table_names = frozenset()
            self._tables_present.clear()
            self._initialized = False
            self._clear_identity_caches()
            raise
        self._initialization_ddl_journal.clear()

    def _initialize_layout_on_connection(
        self,
        connection: sqlalchemy.Connection,
        supplied: StorageLayout | None,
    ) -> None:
        objects_before = actual_schema_objects(connection)
        names_before = frozenset(name for name, kinds in objects_before.items() if "table" in kinds)
        if METADATA_TABLE_NAME in names_before:
            try:
                stored = read_store_metadata(connection)
            except (ValueError, SQLAlchemyError) as error:
                raise StorageLayoutUpgradeRequiredError(
                    {"declaration": {"metadata": "malformed", "error": str(error)}}
                ) from error
            assert stored is not None
            self._open_marked_layout(connection, stored, supplied)
            return

        if not objects_before:
            if supplied is None:
                raise TypeError("entry_records is required when opening an uninitialized database")
            expected = expected_metadata(supplied)
            expected.tables[METADATA_TABLE_NAME].create(connection, checkfirst=False)
            self._initialization_ddl_journal.append(expected.tables[METADATA_TABLE_NAME])
            self._stamp_layout(connection, supplied)
            self._install_layout(supplied, expected, names_before | {METADATA_TABLE_NAME})
            return

        schema: dict[str, object] = {METADATA_TABLE_NAME: {"missing": True}}
        for name, kinds in sorted(objects_before.items()):
            object_type = _schema_object_type(kinds)
            if name.startswith("_httk_"):
                schema[name] = {
                    "reserved": True,
                    "object_type": object_type,
                    "message": "unexpected schema object uses the SqlStore-reserved _httk_ prefix",
                }
            else:
                schema[name] = {
                    "unversioned": True,
                    "object_type": object_type,
                    "message": "a nonempty database without SqlStore metadata cannot be adopted",
                }
        raise StorageLayoutUpgradeRequiredError(
            {
                "protocol": {"expected": STORAGE_PROTOCOL_VERSION, "actual": None},
                "declaration": {
                    "expected": declaration_json(supplied) if supplied is not None else "explicit entry_records",
                    "actual": None,
                },
                "schema": schema,
            }
        )

    def _open_marked_layout(
        self,
        connection: sqlalchemy.Connection,
        stored: Mapping[str, str],
        supplied: StorageLayout | None,
    ) -> None:
        required_keys = {"protocol", "entry_declaration"}
        diff: dict[str, object] = {}
        if set(stored) != required_keys:
            diff["declaration"] = {
                "metadata_keys": {"expected": tuple(sorted(required_keys)), "actual": tuple(sorted(stored))}
            }
        if stored.get("protocol") != STORAGE_PROTOCOL_VERSION:
            diff["protocol"] = {"expected": STORAGE_PROTOCOL_VERSION, "actual": stored.get("protocol")}
        persisted: StorageLayout | None = None
        try:
            persisted = self._layout_from_stored_declaration(stored.get("entry_declaration"))
        except (TypeError, ValueError) as error:
            diff["declaration"] = {
                "expected": "canonical registered declaration",
                "actual": stored.get("entry_declaration"),
                "error": str(error),
            }
        if persisted is not None and supplied is not None and declaration_json(persisted) != declaration_json(supplied):
            diff["declaration"] = {
                "expected": declaration_json(persisted),
                "actual": declaration_json(supplied),
            }
        if diff:
            raise StorageLayoutUpgradeRequiredError(diff)
        assert persisted is not None
        objects_before = actual_schema_objects(connection)
        names_before = frozenset(name for name, kinds in objects_before.items() if "table" in kinds)
        declaration_owned = {
            METADATA_TABLE_NAME,
            *(entry_dispatch_table_name(family.name) for family in persisted.families if len(family.records) > 1),
        }
        object_problems: dict[str, object] = {}
        for name, kinds in objects_before.items():
            if name.startswith("_httk_") and (name not in declaration_owned or kinds != {"table"}):
                object_problems[name] = {
                    "reserved": True,
                    "object_type": _schema_object_type(kinds),
                    "message": "unexpected schema object uses the SqlStore-reserved _httk_ prefix",
                }
        if object_problems:
            raise StorageLayoutUpgradeRequiredError({"schema": object_problems})
        self._install_layout(persisted, expected_metadata(persisted), names_before)

    @staticmethod
    def _layout_from_stored_declaration(value: str | None) -> StorageLayout:
        # The canonical JSON parser is intentionally private to layout.py; a
        # no-op explicit declaration round-trip uses the public normalizer.
        from httk.data.db.layout import _layout_from_declaration

        if value is None:
            raise ValueError("metadata is missing entry_declaration")
        return _layout_from_declaration(value)

    def _stamp_layout(
        self,
        connection: sqlalchemy.Connection,
        layout: StorageLayout,
    ) -> None:
        table = metadata_table_for(sqlalchemy.MetaData())
        connection.execute(
            sqlalchemy.insert(table),
            (
                {"key": "protocol", "value": STORAGE_PROTOCOL_VERSION},
                {"key": "entry_declaration", "value": declaration_json(layout)},
            ),
        )

    def _install_layout(
        self,
        layout: StorageLayout,
        metadata: sqlalchemy.MetaData,
        table_names: Iterable[str],
    ) -> None:
        self._layout = layout
        self._metadata = metadata
        self._managed_table_names = frozenset(metadata.tables)
        self._tables_present = set(table_names)
        self._initialized = True

    def _cleanup_initialization_tables(self, created_tables: tuple[sqlalchemy.Table, ...]) -> None:
        # SQLite DDL can escape SQLAlchemy's outer rollback helper. A fresh
        # explicit cleanup transaction means a failed first initialization is
        # never left with a usable partial marker/layout. DuckDB has already
        # rolled it back at this point; IF EXISTS makes the same path harmless.
        try:
            with self._database.engine.begin() as cleanup:
                for table in reversed(created_tables):
                    cleanup.execute(sqlalchemy.schema.DropTable(table, if_exists=True))
        except BaseException:
            # Preserve the original initialization error. A remaining table
            # still has no marker and will be refused as unversioned.
            return

    # ------------------------------------------------------------------ tables and transactions

    def _reject_during_bulk(self) -> None:
        """Refuse ordinary writes while a :meth:`bulk_ingest` context owns the store.

        :raises RuntimeError: If a bulk-ingest context is currently open.
        """
        if self._bulk_active:
            raise RuntimeError(
                "this SqlStore has an open bulk_ingest context; ordinary save/ensure_tables/transaction "
                "operations are refused until it exits"
            )

    def bulk_ingest(
        self,
        *,
        chunk_size: int = 100_000,
        verify_metadata: bool = True,
        index_strategy: Literal["auto", "keep", "rebuild"] = "auto",
        on_progress: Callable[[int, int], None] | None = None,
        workers: int = 1,
    ) -> "BulkIngest":
        """Return a context manager that appends a stream of objects into this store.

        The returned :class:`~httk.data.db.bulk.BulkIngest` exposes
        ``save(obj, *, as_record=None) -> int`` mirroring :meth:`save`, but
        buffers encoded rows with pre-assigned sids and appends them in
        executemany batches. On a physically empty store the record tables are
        created index-less and their separable indexes are built once the stream
        completes; on a populated store each flushed chunk is staged and
        resolved set-wise against the existing rows (content-id anti-join with
        metadata verification, ``by_value`` whole-column anti-join, and a sid
        remap of the surviving references) before it is appended. While the
        context is open the store's ordinary write path is exclusively owned:
        :meth:`save`, :meth:`ensure_tables` and :meth:`transaction` raise
        :class:`RuntimeError`.

        A sid returned by ``save`` inside the context is provisional: a record
        that deduplicates against a pre-existing row is remapped at flush, so its
        durable sid is obtained from :meth:`~httk.data.db.bulk.BulkIngest.resolved_sid`
        once the context has exited cleanly.

        :param chunk_size: The number of top-level saves buffered before a flush.
        :param verify_metadata: Whether content-id hits compare identity-excluded metadata.
        :param index_strategy: How an existing table's separable indexes are handled during the append —
            ``"keep"`` appends through them, ``"rebuild"`` drops and recreates them, and ``"auto"`` picks
            per table by the staged-to-existing row ratio. On DuckDB, which reserves a dropped index's name
            until commit, ``"rebuild"`` instead keeps the indexes and verifies content-id uniqueness with a
            duplicate scan; the final indexes are the same either way.
        :param on_progress: An optional ``(records_buffered_total, rows_flushed_total)`` callback invoked after each flush.
        :param workers: The number of worker processes. ``1`` (the default) is the serial path with byte-for-byte
            unchanged semantics; ``>1`` encodes the stream in forked worker processes and merges their per-table
            shards set-wise. Parallel mode requires a physically empty target store (the offline-build use case) and,
            on DuckDB, the ``httk-data[parallel]`` extra (pyarrow); incremental appends stay on the serial path.
        :return: A bulk-ingest context manager bound to this store.
        """
        from httk.data.db.bulk import BulkIngest

        return BulkIngest(
            self,
            chunk_size=chunk_size,
            verify_metadata=verify_metadata,
            index_strategy=index_strategy,
            on_progress=on_progress,
            workers=workers,
        )

    def ensure_tables(self, *classes: type) -> None:
        r"""Create the requested tables as an explicit write operation.

        :param \*classes: The storable classes whose tables should exist.
        :return: None.
        :raises RuntimeError: If a :meth:`bulk_ingest` context is currently open.
        """
        self._reject_during_bulk()
        with self._write_connection() as connection:
            self._create_tables_for_write(connection, classes)

    def transaction(self) -> contextlib.AbstractContextManager[None]:
        """Return a context manager for one database transaction.

        :return: A transaction context manager that commits on normal exit and rolls back on failure.
        """
        return self._transaction_scope()

    @contextlib.contextmanager
    def _transaction_scope(self) -> Iterator[None]:
        self._reject_during_bulk()
        stack = self._connection_stack()
        if stack:
            yield
            return
        pending = self._pending_table_names()
        try:
            with self._database.engine.begin() as connection:
                stack.append(connection)
                try:
                    yield
                finally:
                    stack.pop()
            self._tables_present.update(pending)
        except BaseException:
            pending.clear()
            self._tables_present.clear()
            self._clear_identity_caches()
            raise
        finally:
            pending.clear()

    def _connection_stack(self) -> list[sqlalchemy.Connection]:
        stack = getattr(self._local, "stack", None)
        if stack is None:
            stack = []
            self._local.stack = stack
        return cast(list[sqlalchemy.Connection], stack)

    def _current_connection(self) -> sqlalchemy.Connection | None:
        stack = self._connection_stack()
        return stack[-1] if stack else None

    @contextlib.contextmanager
    def _write_connection(self) -> Iterator[sqlalchemy.Connection]:
        current = self._current_connection()
        if current is not None:
            yield current
            return
        pending = self._pending_table_names()
        try:
            with self._database.engine.begin() as connection:
                stack = self._connection_stack()
                stack.append(connection)
                try:
                    yield connection
                finally:
                    stack.pop()
            self._tables_present.update(pending)
        except BaseException:
            pending.clear()
            self._tables_present.clear()
            self._clear_identity_caches()
            raise
        finally:
            pending.clear()

    def _pending_table_names(self) -> set[str]:
        pending = getattr(self._local, "pending_tables", None)
        if pending is None:
            pending = set()
            self._local.pending_tables = pending
        return cast(set[str], pending)

    @contextlib.contextmanager
    def _read_connection(self) -> Iterator[sqlalchemy.Connection]:
        current = self._current_connection()
        if current is not None:
            yield current
            return
        with self._database.engine.connect() as connection:
            stack = self._connection_stack()
            stack.append(connection)
            try:
                yield connection
            finally:
                stack.pop()

    def _candidate_metadata(self, classes: Iterable[type]) -> sqlalchemy.MetaData:
        candidate = sqlalchemy.MetaData()
        requested = tuple(classes)
        for cls in requested:
            table_for(resolve_schema(cls), candidate)
        for family in self.layout.families:
            if not any(record in family.records for record in requested):
                continue
            schemas = tuple(resolve_schema(record) for record in family.records)
            for schema in schemas:
                table_for(schema, candidate)
            if len(schemas) > 1:
                dispatch_table_for(family.name, tuple(zip(family.record_names, schemas, strict=True)), candidate)
        return candidate

    def _register_tables(self, classes: Iterable[type]) -> sqlalchemy.MetaData:
        requested = tuple(classes)
        candidate = self._candidate_metadata(requested)
        for cls in requested:
            table_for(resolve_schema(cls), self._metadata)
        for family in self.layout.families:
            if not any(record in family.records for record in requested):
                continue
            schemas = tuple(resolve_schema(record) for record in family.records)
            for schema in schemas:
                table_for(schema, self._metadata)
            if len(schemas) > 1:
                dispatch_table_for(family.name, tuple(zip(family.record_names, schemas, strict=True)), self._metadata)
        return candidate

    def _validate_table_names(self, names: Iterable[str]) -> None:
        forbidden = sorted(
            name for name in names if name.startswith("_httk_") and name not in self._managed_table_names
        )
        if forbidden:
            raise ValueError(f"ordinary records may not claim reserved SqlStore table names: {', '.join(forbidden)}")

    def _create_tables_for_write(self, connection: sqlalchemy.Connection, classes: Iterable[type]) -> None:
        """Register and create missing record tables for the caller's write operation.

        SQLite's legacy transaction mode may commit DDL eagerly, so a failed
        save can leave empty or partial declaration-shaped tables. Stamp trust
        accepts that residue; the next write's ``checkfirst`` completes it.
        """
        candidate = self._register_tables(classes)
        candidate_names = frozenset(candidate.tables)
        self._validate_table_names(candidate_names)
        pending = self._pending_table_names()
        missing = candidate_names - self._tables_present - pending
        if missing:
            pending.update(actual_table_names(connection))
            missing = candidate_names - self._tables_present - pending
        if missing:
            candidate.create_all(connection, checkfirst=True)
            # Publish only after the owning transaction commits. SQLite may
            # retain empty or partial declaration-shaped tables after rollback;
            # stamp trust accepts that residue and the next write completes it.
            self._pending_table_names().update(missing)

    def _missing_tables_for_read(self, classes: Iterable[type]) -> bool:
        """Register tables and report absence without issuing DDL."""
        candidate = self._register_tables(classes)
        candidate_names = frozenset(candidate.tables)
        self._validate_table_names(candidate_names)
        pending = self._pending_table_names()
        missing = candidate_names - self._tables_present - pending
        if missing:
            current = self._current_connection()
            if current is not None:
                # Keep transaction-local catalog observations in the overlay;
                # publishing them before commit would make rollback unsafe.
                pending.update(actual_table_names(current))
            else:
                self._refresh_committed_table_names()
            missing = candidate_names - self._tables_present - pending
        return bool(missing)

    def _refresh_committed_table_names(self) -> None:
        """Refresh the shared table cache from a connection outside this transaction."""
        with self._database.engine.connect() as connection:
            self._tables_present.update(actual_table_names(connection))

    def _table(self, name: str) -> sqlalchemy.Table:
        return self._metadata.tables[name]

    # ------------------------------------------------------------------ saving

    def save(self, obj: Any, *, as_record: type | None = None) -> int:
        """Store ``obj`` (deduplicating per its class's policy) and return its integer sid.

        An opted-in domain object is projected through its exact
        ``__httk_storage_record__``; ``as_record`` selects an alternate record
        representation explicitly. Referenced records and record-valued child
        elements are saved recursively without constructing intermediate
        record instances.

        A content-id deduplication hit compares metadata marked with
        :class:`~httk.core.storage.markers.IdentitySkip` in schema order.
        Nested plans are cached per record type, and a mismatch raises
        :class:`~httk.data.db.store.EntryMetadataConflictError` without replacing the row.

        :param obj: The object to store.
        :param as_record: The alternate record representation to use, if any.
        :return: The stored row's sid.
        :raises TypeError: If ``obj`` is a cursor row that must be materialized first.
        :raises httk.data.db.store.EntryMetadataConflictError: If a deduplication hit has conflicting metadata.
        :raises httk.core.storage.identity.StorageProjectionCycleError: If projection reaches a reference cycle.
        :raises RuntimeError: If a :meth:`bulk_ingest` context is currently open.
        """
        self._reject_during_bulk()
        reject_cursor_proxy(obj)
        record_type = resolve_storage_record(obj, as_record=as_record)
        projection = SaveProjection()
        with self._write_connection() as connection:
            self._create_tables_for_write(connection, (record_type,))
            sid = self._save(connection, record_type, obj, projection, "")
            family = self._family_for_backing(record_type)
            if family is not None:
                self._save_entry_dispatch(connection, family, record_type, sid, projection.content_id(record_type, obj))
            return sid

    def _save(
        self,
        connection: sqlalchemy.Connection,
        record_type: type,
        source: Any,
        projection: _Projection,
        path: str,
    ) -> int:
        active_key = (record_type, id(source))
        if active_key in projection.active:
            raise StorageProjectionCycleError(path, record_type)
        projection.active.add(active_key)
        try:
            return self._save_active(connection, record_type, source, projection, path)
        finally:
            projection.active.remove(active_key)

    def _save_active(
        self,
        connection: sqlalchemy.Connection,
        record_type: type,
        source: Any,
        projection: _Projection,
        path: str,
    ) -> int:
        schema = resolve_schema(record_type)
        table = self._table(schema.table_name)
        projected = projection.projector(record_type, source)
        validation_key = (record_type, id(source))
        if type(source) is record_type and validation_key not in projection.validated:
            validator = vars(record_type).get("__httk_validate__")
            if validator is not None:
                # Bind the descriptor fetched from the class's own dict; the
                # own-dict lookup (not getattr) keeps inherited validators out.
                validator.__get__(None, record_type)(source)
            projection.validated.add(validation_key)

        key: str | None = None
        if schema.dedup == "content_id":
            key = projection.content_id(record_type, source)
            found = connection.execute(
                sqlalchemy.select(table.c[SID_COLUMN]).where(table.c[CONTENT_ID_COLUMN] == key)
            ).first()
            if found is not None:
                sid = int(found[0])
                self._check_metadata(connection, record_type, sid, source, projection)
                self._remember(record_type, sid, source, cache_instance=type(source) is record_type)
                return sid

        checkpoint = len(projection.inserted)
        values = self._parent_row(connection, schema, source, projected, projection, path)

        if schema.dedup == "by_value":
            # v1 semantics: a by_value match compares the parent table's stored
            # columns only; child-table contents are not part of the match.
            conditions = [
                table.c[name].is_(None) if value is None else table.c[name] == value for name, value in values.items()
            ]
            statement = sqlalchemy.select(table.c[SID_COLUMN])
            if conditions:
                statement = statement.where(*conditions)
            found = connection.execute(statement.limit(1)).first()
            if found is not None:
                sid = int(found[0])
                self._discard_inserted(connection, projection, checkpoint)
                self._remember(record_type, sid, source, cache_instance=type(source) is record_type)
                return sid

        if key is not None:
            values[CONTENT_ID_COLUMN] = key
            sid, inserted = self._insert_content_row(connection, table, values, key)
            if not inserted:
                self._discard_inserted(connection, projection, checkpoint)
                self._check_metadata(connection, record_type, sid, source, projection)
                self._remember(record_type, sid, source, cache_instance=type(source) is record_type)
                return sid
        else:
            insert = sqlalchemy.insert(table).values(values) if values else sqlalchemy.insert(table)
            result = connection.execute(insert)
            sid = int(cast(Any, result.inserted_primary_key)[0])
        projection.inserted.append((record_type, sid))
        for spec in schema.fields:
            if spec.role == "child":
                self._insert_child_rows(
                    connection,
                    schema,
                    spec,
                    sid,
                    self._projected_value(record_type, source, projected, spec),
                    projection,
                    _field_path(path, spec.field),
                )
        self._remember(record_type, sid, source, cache_instance=type(source) is record_type)
        return sid

    def _insert_content_row(
        self,
        connection: sqlalchemy.Connection,
        table: sqlalchemy.Table,
        values: dict[str, Any],
        key: str,
    ) -> tuple[int, bool]:
        """Insert one content-addressed row, returning the race winner safely."""
        dialect = connection.dialect.name
        if dialect == "sqlite":
            from sqlalchemy.dialects.sqlite import insert as sqlite_insert

            statement: Any = (
                sqlite_insert(table).values(values).on_conflict_do_nothing(index_elements=[CONTENT_ID_COLUMN])
            )
        elif dialect in {"duckdb", "postgresql"}:
            from sqlalchemy.dialects.postgresql import insert as postgresql_insert

            statement = (
                postgresql_insert(table).values(values).on_conflict_do_nothing(index_elements=[CONTENT_ID_COLUMN])
            )
        else:
            result = connection.execute(sqlalchemy.insert(table).values(values))
            return int(cast(Any, result.inserted_primary_key)[0]), True

        result = connection.execute(statement.returning(table.c[SID_COLUMN]))
        inserted_sid = result.scalar_one_or_none()
        if inserted_sid is not None:
            return int(inserted_sid), True
        found = connection.execute(
            sqlalchemy.select(table.c[SID_COLUMN]).where(table.c[CONTENT_ID_COLUMN] == key)
        ).scalar_one()
        return int(found), False

    def _parent_row(
        self,
        connection: sqlalchemy.Connection,
        schema: TableSchema,
        source: Any,
        projected: Mapping[str, object],
        projection: _Projection,
        path: str,
    ) -> dict[str, Any]:
        """Encode projected parent columns, saving referenced records recursively."""

        def resolve_sid(record_type: type, value: Any, field_path: str) -> int:
            return self._save(connection, record_type, value, projection, field_path)

        return _encode_parent_row(schema, source, projected, path, resolve_sid)

    @staticmethod
    def _projected_value(record_type: type, source: Any, projected: Mapping[str, object], spec: FieldSpec) -> Any:
        if spec.field in projected:
            return projected[spec.field]
        if spec.derived:
            try:
                return getattr(source, spec.field)
            except AttributeError:
                raise TypeError(
                    f"projecting {type(source).__name__} as {record_type.__name__} requires the source "
                    f"to expose derived stored property {spec.field!r}"
                ) from None
        raise ValueError(f"projection for {type(source).__name__} omitted stored field {spec.field!r}")

    def _insert_child_rows(
        self,
        connection: sqlalchemy.Connection,
        schema: TableSchema,
        spec: FieldSpec,
        sid: int,
        value: Any,
        projection: _Projection,
        path: str,
    ) -> None:
        assert spec.child is not None

        def resolve_sid(record_type: type, element: Any, element_path: str) -> int:
            return self._save(connection, record_type, element, projection, element_path)

        rows = _encode_child_rows(schema, spec, sid, value, path, resolve_sid)
        if rows:
            table = self._table(spec.child.table_name)
            connection.execute(sqlalchemy.insert(table), rows)

    # ------------------------------------------------------------------ fetching

    def fetch[T](self, cls: type[T], sid: int) -> T:
        """Reconstruct the ``cls`` instance stored under ``sid``.

        While a previously fetched (or saved) instance for this ``(class,
        sid)`` is alive, the very same object is returned. Raises
        :class:`KeyError` (carrying the class and sid) when no such row exists.
        A missing table therefore has the same result as a missing row.

        :param cls: The storable class to reconstruct.
        :param sid: The stored row identifier.
        :return: The reconstructed instance.
        :raises KeyError: If no row exists for ``cls`` and ``sid``.
        """
        with self._read_connection() as connection:
            return cast(T, self._fetch(connection, cls, sid))

    def fetch_by_content_id[T](self, cls: type[T], key: str) -> T | None:
        """Return the ``cls`` instance whose content identity is ``key``, or None if not stored.

        Only classes with the ``"content_id"`` dedup policy carry a content
        identity column; :class:`~httk.data.db.schema.SchemaError` is raised
        for any other class.

        :param cls: The storable class to search.
        :param key: The content identity to find.
        :return: The stored instance, or ``None`` when no row matches.
        :raises httk.data.db.schema.SchemaError: If the class does not use content-id deduplication.
        """
        schema = resolve_schema(cls)
        if schema.dedup != "content_id":
            raise SchemaError(
                f"{cls.__name__} has dedup policy {schema.dedup!r}; only classes with the "
                f"'content_id' policy have a content identity column"
            )
        with self._read_connection() as connection:
            if self._missing_tables_for_read((cls,)):
                return None
            table = self._table(schema.table_name)
            found = connection.execute(
                sqlalchemy.select(table.c[SID_COLUMN]).where(table.c[CONTENT_ID_COLUMN] == key)
            ).first()
            if found is None:
                return None
            return cast(T, self._fetch(connection, cls, int(found[0])))

    def fetch_entry(self, family_cls: type, content_id: str) -> object | None:
        """Return the concrete configured record for an entry-family content identity.

        The result is the actual frozen record class, not the family protocol.
        A single-record family can query that record directly; only
        multi-record families use their reserved one-of-many dispatch table,
        whose constraint permits exactly one backing sid per content identity.

        :param family_cls: The configured entry-family class.
        :param content_id: The entry content identity to find.
        :return: The concrete stored record, or ``None`` when no row matches.
        :raises ValueError: If ``family_cls`` is not configured for this store.
        :raises EntryDispatchIntegrityError: If a dispatch row is inconsistent with its backing row.
        """
        family = next((item for item in self.layout.families if item.family is family_cls), None)
        if family is None:
            raise ValueError(f"{family_cls.__name__} is not a configured entry family in this SqlStore")
        with self._read_connection() as connection:
            if self._missing_tables_for_read(family.records):
                return None
            if len(family.records) == 1:
                backing = family.records[0]
                schema = resolve_schema(backing)
                table = self._table(schema.table_name)
                sid = connection.execute(
                    sqlalchemy.select(table.c[SID_COLUMN]).where(table.c[CONTENT_ID_COLUMN] == content_id)
                ).scalar_one_or_none()
                return None if sid is None else self._fetch(connection, backing, int(sid))
            table = self._table(entry_dispatch_table_name(family.name))
            row = (
                connection.execute(sqlalchemy.select(table).where(table.c[DISPATCH_CONTENT_ID_COLUMN] == content_id))
                .mappings()
                .one_or_none()
            )
            if row is None:
                for backing in family.records:
                    backing_table = self._table(resolve_schema(backing).table_name)
                    found = connection.execute(
                        sqlalchemy.select(backing_table.c[SID_COLUMN])
                        .where(backing_table.c[CONTENT_ID_COLUMN] == content_id)
                        .limit(1)
                    ).first()
                    if found is not None:
                        raise EntryDispatchIntegrityError(
                            f"entry dispatch {family.name!r} is missing for stored content_id {content_id!r}"
                        )
                return None
            backing, sid = self._dispatch_target(family, row, content_id)
            backing_table = self._table(resolve_schema(backing).table_name)
            backing_content_id = connection.execute(
                sqlalchemy.select(backing_table.c[CONTENT_ID_COLUMN]).where(backing_table.c[SID_COLUMN] == sid)
            ).scalar_one_or_none()
            if backing_content_id != content_id:
                raise EntryDispatchIntegrityError(
                    f"entry dispatch {family.name!r} maps content_id {content_id!r} to backing sid {sid} "
                    f"whose content_id is {backing_content_id!r}"
                )
            return self._fetch(connection, backing, sid)

    def sid_of(self, obj: Any, *, as_record: type | None = None) -> int | None:
        """Return this store's sid for ``obj``'s record identity, if present.

        :param obj: The object whose stored identity should be looked up.
        :param as_record: The alternate record representation to use, if any.
        :return: The stored sid, or ``None`` when no matching row is known.
        """
        record_type = resolve_storage_record(obj, as_record=as_record)
        lazy_identity = lazy_row_identity(obj)
        if is_lazy_row(obj) and record_type is resolve_storage_record(obj):
            return lazy_identity[1] if lazy_identity is not None and lazy_identity[0] is self else None
        try:
            cached = self._identity._sids.get(obj, {}).get(record_type)
        except TypeError:
            cached = None
        if cached is None:
            cached = self._identity._sids_by_identity.get((record_type, id(obj)))
        if cached is not None:
            return cached

        schema = resolve_schema(record_type)
        if schema.dedup != "content_id":
            return cached
        projection = _Projection()
        key = projection.content_id(record_type, obj)
        with self._read_connection() as connection:
            if self._missing_tables_for_read((record_type,)):
                return None
            table = self._table(schema.table_name)
            found = connection.execute(
                sqlalchemy.select(table.c[SID_COLUMN]).where(table.c[CONTENT_ID_COLUMN] == key)
            ).scalar_one_or_none()
        if found is None:
            return None
        sid = int(found)
        self._remember(record_type, sid, obj, cache_instance=type(obj) is record_type)
        return sid

    def searcher(self) -> SqlSearcher:
        """Return a new :class:`~httk.data.db.searcher.SqlSearcher` querying this store.

        The searcher runs on this store's read path — inside an open
        :meth:`transaction` block it sees uncommitted writes — and
        reconstructs matched objects through :meth:`fetch`, so the identity
        cache applies.

        :return: A new SQL searcher bound to this store.
        """
        return SqlSearcher(self)

    def stored_property_plan(self, family: type) -> Any:
        """Return the SQL stored-property plan for one configured entry family.

        :param family: The logical entry-family class to plan.
        :return: The validated SQL stored-property plan.
        """
        from httk.data.db.stored_properties import stored_property_sql_plan

        return stored_property_sql_plan(self, family)

    def referring(self, cls: type, *, field: str, to: Any) -> list[Any]:
        """Return all stored ``cls`` instances whose reference field ``field`` points at ``to``.

        ``field`` must be a reference field of ``cls`` targeting ``to``'s class
        (:class:`~httk.data.db.schema.SchemaError` otherwise), and ``to`` must
        be known to this store — saved or fetched through it — else
        :class:`ValueError` is raised. Results are ordered by sid.

        :param cls: The storable class whose references should be searched.
        :param field: The reference field to match.
        :param to: The stored target instance.
        :return: The referring stored instances ordered by sid.
        :raises httk.data.db.schema.SchemaError: If ``field`` is not a compatible reference field.
        :raises ValueError: If ``to`` is not known to this store.
        """
        schema = resolve_schema(cls)
        spec = schema.field(field)
        if spec.role != "reference":
            raise SchemaError(f"{cls.__name__}.{field} is not a reference field (its role is {spec.role!r})")
        assert spec.target is not None
        if not isinstance(to, spec.target):
            raise SchemaError(f"{cls.__name__}.{field} references {spec.target.__name__}, not {type(to).__name__}")
        sid = self.sid_of(to)
        if sid is None:
            raise ValueError(f"the {type(to).__name__} instance has not been stored or fetched through this store")
        with self._read_connection() as connection:
            if self._missing_tables_for_read((cls,)):
                return []
            table = self._table(schema.table_name)
            found = connection.execute(
                sqlalchemy.select(table.c[SID_COLUMN])
                .where(table.c[spec.columns[0].name] == sid)
                .order_by(table.c[SID_COLUMN])
            ).all()
            return [self._fetch(connection, cls, int(row[0])) for row in found]

    def _fetch(self, connection: sqlalchemy.Connection, cls: type, sid: int) -> Any:
        sid = int(sid)
        cached = self._identity._instances.get((cls, sid))
        if cached is not None:
            return cached
        # The hydrator owns exact decoding and child/reference batching; this
        # path still materializes and validates the real base dataclass.
        try:
            instance = RowHydrator(self, cls, (sid,)).materialize(sid)
        except StaleResultError:
            raise KeyError(cls, sid) from None
        self._remember(cls, sid, instance)
        return instance

    def _family_for_backing(self, record_type: type) -> EntryFamilyLayout | None:
        for family in self.layout.families:
            if any(backing is record_type for backing in family.records):
                return family
        return None

    def _save_entry_dispatch(
        self,
        connection: sqlalchemy.Connection,
        family: EntryFamilyLayout,
        backing: type,
        sid: int,
        key: str,
    ) -> None:
        if len(family.records) == 1:
            return
        table = self._table(entry_dispatch_table_name(family.name))
        column_name = backing_dispatch_column_name(family.record_names[family.records.index(backing)])
        existing = (
            connection.execute(sqlalchemy.select(table).where(table.c[DISPATCH_CONTENT_ID_COLUMN] == key))
            .mappings()
            .one_or_none()
        )
        if existing is not None:
            found_backing, found_sid = self._dispatch_target(family, existing, key)
            if found_backing is not backing or found_sid != sid:
                raise EntryDispatchIntegrityError(
                    f"entry dispatch {family.name!r} maps content_id {key!r} to a conflicting backing row"
                )
            return
        values = {DISPATCH_CONTENT_ID_COLUMN: key, column_name: sid}
        inserted = self._insert_dispatch_row(connection, table, values)
        if inserted:
            return
        # ``ON CONFLICT DO NOTHING`` keeps SQLite, DuckDB and PostgreSQL
        # transactions usable after either uniqueness conflict. Diagnose the
        # content-id path first, then the per-backing UNIQUE sid path.
        existing = (
            connection.execute(sqlalchemy.select(table).where(table.c[DISPATCH_CONTENT_ID_COLUMN] == key))
            .mappings()
            .one_or_none()
        )
        if existing is not None:
            found_backing, found_sid = self._dispatch_target(family, existing, key)
            if found_backing is backing and found_sid == sid:
                return
            raise EntryDispatchIntegrityError(
                f"entry dispatch {family.name!r} maps content_id {key!r} to a conflicting backing row"
            )
        sid_owner = (
            connection.execute(sqlalchemy.select(table).where(table.c[column_name] == sid)).mappings().one_or_none()
        )
        if sid_owner is not None:
            owner_content_id = str(sid_owner[DISPATCH_CONTENT_ID_COLUMN])
            raise EntryDispatchIntegrityError(
                f"entry dispatch {family.name!r} already maps backing sid {sid} to content_id {owner_content_id!r}, "
                f"not {key!r}"
            )
        raise EntryDispatchIntegrityError(
            f"entry dispatch {family.name!r} declined content_id {key!r} without a discoverable conflicting row"
        )

    @staticmethod
    def _insert_dispatch_row(
        connection: sqlalchemy.Connection,
        table: sqlalchemy.Table,
        values: Mapping[str, object],
    ) -> bool:
        """Insert one dispatch association, safely detecting every uniqueness conflict."""
        dialect = connection.dialect.name
        if dialect == "sqlite":
            from sqlalchemy.dialects.sqlite import insert as sqlite_insert

            statement: Any = sqlite_insert(table).values(values).on_conflict_do_nothing()
        elif dialect in {"duckdb", "postgresql"}:
            from sqlalchemy.dialects.postgresql import insert as postgresql_insert

            statement = postgresql_insert(table).values(values).on_conflict_do_nothing()
        else:
            result = connection.execute(sqlalchemy.insert(table).values(values))
            return result.rowcount == 1
        result = connection.execute(statement.returning(table.c[DISPATCH_CONTENT_ID_COLUMN]))
        return result.scalar_one_or_none() is not None

    @staticmethod
    def _dispatch_target(
        family: EntryFamilyLayout,
        row: Any,
        content_id: str,
    ) -> tuple[type, int]:
        populated: list[tuple[type, int]] = []
        for backing_name, backing in zip(family.record_names, family.records, strict=True):
            value = row[backing_dispatch_column_name(backing_name)]
            if value is not None:
                populated.append((backing, int(value)))
        if len(populated) != 1:
            raise EntryDispatchIntegrityError(
                f"entry dispatch {family.name!r} has {len(populated)} backing rows for content_id {content_id!r}"
            )
        return populated[0]

    def _check_metadata(
        self,
        connection: sqlalchemy.Connection,
        record_type: type,
        sid: int,
        source: Any,
        projection: _Projection,
    ) -> None:
        plan = _metadata_plan(record_type)
        if plan is None:
            return
        self._check_metadata_at(connection, record_type, sid, source, projection, record_type.__name__, plan)

    def _check_metadata_at(
        self,
        connection: sqlalchemy.Connection,
        record_type: type,
        sid: int,
        source: Any,
        projection: _Projection,
        path: str,
        plan: _MetadataPlan | None = None,
    ) -> None:
        plan = _metadata_plan(record_type) if plan is None else plan
        if plan is None:
            return
        schema = resolve_schema(record_type)
        row = self._metadata_parent_row(connection, record_type, sid, plan, projection)
        values = projection.projector(record_type, source)
        skipped_specs = {spec.field for spec in plan.skipped_specs}
        skipped_nested = {spec.field for spec in plan.skipped_nested}
        descend_specs = {spec.field for spec in plan.descend_specs}
        for spec in schema.fields:
            if spec.derived:
                continue
            field_path = _field_path(path, spec.field)
            if spec.field in skipped_specs:
                incoming = values[spec.field]
                existing = decode_field(self, schema, spec, sid, row)
                if not _metadata_scalar_equal(incoming, existing):
                    raise EntryMetadataConflictError(
                        f"metadata conflict for {field_path}: stored {existing!r}, received {incoming!r}"
                    )
            elif spec.field in skipped_nested:
                self._check_metadata_nested(
                    connection, schema, row, sid, spec, values[spec.field], projection, field_path, True
                )
            elif spec.field in descend_specs:
                self._check_metadata_nested(
                    connection, schema, row, sid, spec, values[spec.field], projection, field_path, False
                )

    def _metadata_parent_row(
        self,
        connection: sqlalchemy.Connection,
        record_type: type,
        sid: int,
        plan: _MetadataPlan,
        projection: _Projection,
    ) -> Mapping[str, Any]:
        key = (record_type, int(sid))
        cached = projection.metadata_rows.get(key)
        if cached is not None:
            return cached
        schema = resolve_schema(record_type)
        table = self._table(schema.table_name)
        specs = (*plan.skipped_specs, *plan.skipped_nested, *plan.descend_specs)
        columns: list[sqlalchemy.Column[Any]] = []
        seen: set[str] = set()
        for spec in specs:
            if spec.role == "child":
                if spec.optional:
                    name = f"{spec.field}_present"
                    if name not in seen:
                        columns.append(table.c[name])
                        seen.add(name)
                continue
            for column in spec.columns:
                if column.name not in seen:
                    columns.append(table.c[column.name])
                    seen.add(column.name)
        if not columns:
            row: Mapping[str, Any] = {}
            projection.metadata_rows[key] = row
            return row
        result = connection.execute(sqlalchemy.select(*columns).where(table.c[SID_COLUMN] == sid)).mappings().first()
        if result is None:
            raise KeyError(record_type, sid)
        row = cast(Mapping[str, Any], result)
        projection.metadata_rows[key] = row
        return row

    def _metadata_child_value(
        self,
        connection: sqlalchemy.Connection,
        schema: TableSchema,
        sid: int,
        spec: FieldSpec,
        parent_row: Mapping[str, Any],
        projection: _Projection,
    ) -> Any:
        key = (schema.cls, int(sid), spec.field)
        cached = projection.metadata_children.get(key, _MISSING_METADATA)
        if cached is not _MISSING_METADATA:
            return cached
        if spec.optional and not parent_row[f"{spec.field}_present"]:
            projection.metadata_children[key] = None
            return None
        assert spec.child is not None
        table = self._table(spec.child.table_name)
        parent_column = f"{schema.table_name}_sid"
        index_column = f"{spec.field}_index"
        columns = tuple(table.c[column.name] for column in spec.child.element_columns)
        rows = connection.execute(
            sqlalchemy.select(*columns).where(table.c[parent_column] == sid).order_by(table.c[index_column])
        ).mappings()
        decoded = [self._metadata_child_element(spec, cast(Mapping[str, Any], row)) for row in rows]
        if spec.shape is not None:
            value: Any = FracVector.create(decoded)
        elif typing.get_origin(spec.python_type) is tuple:
            value = tuple(decoded)
        else:
            value = decoded
        projection.metadata_children[key] = value
        return value

    def _metadata_child_records(
        self,
        connection: sqlalchemy.Connection,
        spec: FieldSpec,
        stored: Any,
    ) -> Any:
        if stored is None:
            return None
        assert spec.target is not None
        records = [self._fetch(connection, spec.target, int(stored_sid)) for stored_sid in stored]
        return tuple(records) if typing.get_origin(spec.python_type) is tuple else records

    @staticmethod
    def _metadata_child_element(spec: FieldSpec, row: Mapping[str, Any]) -> Any:
        assert spec.child is not None
        if spec.target is not None:
            return int(row[spec.child.element_columns[0].name])
        if spec.shape is not None:
            assert spec.shape is not None
            return decode_fracvector_exact(row[f"{spec.field}_exact"], 1, spec.shape.cols).to_fractions()[0]
        if spec.codec_name is not None:
            return codec_named(spec.codec_name).decode(tuple(row[column.name] for column in spec.child.element_columns))
        return row[spec.child.element_columns[0].name]

    def _check_metadata_nested(
        self,
        connection: sqlalchemy.Connection,
        schema: TableSchema,
        parent_row: Mapping[str, Any],
        sid: int,
        spec: FieldSpec,
        incoming: Any,
        projection: _Projection,
        path: str,
        compare_content: bool,
    ) -> None:
        if spec.role == "reference":
            assert spec.target is not None
            stored_sid = parent_row[spec.columns[0].name]
            if incoming is None or stored_sid is None:
                if incoming is not None or stored_sid is not None:
                    if compare_content:
                        stored = None if stored_sid is None else self._fetch(connection, spec.target, int(stored_sid))
                        raise EntryMetadataConflictError(
                            f"metadata conflict for {path}: stored {stored!r}, received {incoming!r}"
                        )
                    raise EntryMetadataConflictError(f"metadata conflict for {path}")
                return
            self._check_metadata_target(
                connection, spec.target, int(stored_sid), incoming, projection, path, compare_content
            )
            return

        stored = self._metadata_child_value(connection, schema, sid, spec, parent_row, projection)
        if spec.target is None:
            if not _metadata_scalar_equal(incoming, stored):
                raise EntryMetadataConflictError(
                    f"metadata conflict for {path}: stored {stored!r}, received {incoming!r}"
                )
            return
        if incoming is None or stored is None:
            if incoming is not stored:
                if compare_content:
                    existing = self._metadata_child_records(connection, spec, stored)
                    raise EntryMetadataConflictError(
                        f"metadata conflict for {path}: stored {existing!r}, received {incoming!r}"
                    )
                raise EntryMetadataConflictError(f"metadata conflict for {path}")
            return
        if len(incoming) != len(stored):
            if compare_content:
                existing = self._metadata_child_records(connection, spec, stored)
                raise EntryMetadataConflictError(
                    f"metadata conflict for {path}: stored {existing!r}, received {incoming!r}"
                )
            raise EntryMetadataConflictError(f"metadata conflict for {path}")
        for index, (incoming_item, stored_sid) in enumerate(zip(incoming, stored, strict=True)):
            item_path = f"{path}[{index}]"
            self._check_metadata_target(
                connection, spec.target, int(stored_sid), incoming_item, projection, item_path, compare_content
            )

    def _check_metadata_target(
        self,
        connection: sqlalchemy.Connection,
        record_type: type,
        sid: int,
        source: Any,
        projection: _Projection,
        path: str,
        compare_content: bool,
    ) -> None:
        if compare_content:
            stored_content_id = self._metadata_content_id(connection, record_type, sid, projection)
            incoming_content_id = projection.content_id(record_type, source)
            if incoming_content_id != stored_content_id:
                stored = self._fetch(connection, record_type, sid)
                raise EntryMetadataConflictError(
                    f"metadata conflict for {path}: stored {stored!r}, received {source!r}"
                )
        plan = _metadata_plan(record_type)
        if plan is not None:
            self._check_metadata_at(connection, record_type, sid, source, projection, path, plan)

    def _metadata_content_id(
        self,
        connection: sqlalchemy.Connection,
        record_type: type,
        sid: int,
        projection: _Projection,
    ) -> str:
        key = (record_type, int(sid))
        cached = projection.metadata_content_ids.get(key)
        if cached is not None:
            return cached
        schema = resolve_schema(record_type)
        table = self._table(schema.table_name)
        if schema.dedup == "content_id":
            stored_content_id = connection.execute(
                sqlalchemy.select(table.c[CONTENT_ID_COLUMN]).where(table.c[SID_COLUMN] == sid)
            ).scalar_one()
            result = str(stored_content_id)
        else:
            result = projection.content_id(record_type, self._fetch(connection, record_type, sid))
        projection.metadata_content_ids[key] = result
        return result

    # ------------------------------------------------------------------ identity caches

    def _discard_inserted(self, connection: sqlalchemy.Connection, projection: _Projection, checkpoint: int) -> None:
        if checkpoint == len(projection.inserted):
            return
        for record_type, sid in reversed(projection.inserted[checkpoint:]):
            schema = resolve_schema(record_type)
            for spec in schema.fields:
                if spec.role == "child":
                    assert spec.child is not None
                    table = self._table(spec.child.table_name)
                    connection.execute(sqlalchemy.delete(table).where(table.c[f"{schema.table_name}_sid"] == sid))
            table = self._table(schema.table_name)
            connection.execute(sqlalchemy.delete(table).where(table.c[SID_COLUMN] == sid))
        del projection.inserted[checkpoint:]
        self._clear_identity_caches()

    def _clear_identity_caches(self) -> None:
        self._identity._clear_identity_caches()

    def _remember(self, cls: type, sid: int, obj: Any, *, cache_instance: bool = True) -> None:
        self._identity._remember(cls, sid, obj, cache_instance=cache_instance)


def _as_fixed_tensor(schema: TableSchema, spec: FieldSpec, shape: Shape, value: Any) -> FracVector:
    """Normalize a fixed-shape field value to a ``(rows, cols)`` FracVector, validating its shape."""
    tensor = FracVector.use(value)
    dim = tensor.dim
    if dim == (shape.rows, shape.cols):
        return tensor
    if shape.rows == 1 and dim == (shape.cols,):
        return FracVector((tensor.noms,), tensor.denom)
    raise ValueError(
        f"{schema.cls.__name__}.{spec.field}: expected a FracVector of shape ({shape.rows}, {shape.cols}), got {dim}"
    )


def _tensor_rows(schema: TableSchema, spec: FieldSpec, shape: Shape, value: Any) -> list[FracVector]:
    """The rows of a variable-rows (``Shape(0, c)``) field value, each as a ``(c,)`` FracVector."""
    if value is None:
        return []
    tensor = FracVector.use(value)
    dim = tensor.dim
    if dim == () or dim == (0,):
        return []
    if len(dim) != 2 or dim[1] != shape.cols:
        raise ValueError(
            f"{schema.cls.__name__}.{spec.field}: expected a FracVector with {shape.cols} columns per row, "
            f"got shape {dim}"
        )
    rows = cast(tuple[tuple[int, ...], ...], tensor.noms)  # dim was validated two-dimensional above
    return [FracVector(noms_row, tensor.denom) for noms_row in rows]


def _field_path(path: str, field: str) -> str:
    return f"{path}.{field}" if path else field


def _encode_parent_row(
    schema: TableSchema,
    source: Any,
    projected: Mapping[str, object],
    path: str,
    resolve_sid: SidResolver,
) -> dict[str, Any]:
    """Encode projected parent columns, resolving referenced records through ``resolve_sid``.

    Connection-free counterpart of :meth:`SqlStore._parent_row`: every branch
    but the reference one is pure, and references defer to ``resolve_sid`` so
    the same encoder serves both recursive-save and bulk-allocation callers.

    :param schema: The parent record's resolved table schema.
    :param source: The instance (or projection source) being encoded.
    :param projected: The projected field mapping for ``source``.
    :param path: The projection path prefix used for diagnostics.
    :param resolve_sid: The callback assigning a sid to each referenced record.
    :return: The encoded parent-table column values.
    """
    values: dict[str, Any] = {}
    for spec in schema.fields:
        if spec.role == "child":
            if spec.optional:
                values[f"{spec.field}_present"] = (
                    SqlStore._projected_value(schema.cls, source, projected, spec) is not None
                )
            continue
        value = SqlStore._projected_value(schema.cls, source, projected, spec)
        if value is None:
            for column in spec.columns:
                values[column.name] = None
        elif spec.role == "scalar":
            values[spec.columns[0].name] = value
        elif spec.role == "encoded":
            assert spec.codec_name is not None
            encoded = codec_named(spec.codec_name).encode(value)
            for column, part in zip(spec.columns, encoded, strict=True):
                values[column.name] = part
        elif spec.role == "fixed_array":
            assert spec.shape is not None
            tensor = _as_fixed_tensor(schema, spec, spec.shape, value)
            for i, part in enumerate(encode_fracvector_floats(tensor)):
                values[f"{spec.field}_{i}"] = part
            values[f"{spec.field}_exact"] = encode_fracvector_exact(tensor)
        else:  # reference
            assert spec.target is not None
            values[spec.columns[0].name] = resolve_sid(spec.target, value, _field_path(path, spec.field))
    return values


def _encode_child_rows(
    schema: TableSchema,
    spec: FieldSpec,
    sid: int,
    value: Any,
    path: str,
    resolve_sid: SidResolver,
) -> list[dict[str, Any]]:
    """Build the child-table rows for one child field, resolving element records through ``resolve_sid``.

    Connection-free counterpart of the row-building loop in
    :meth:`SqlStore._insert_child_rows`: tensor and codec branches are pure, and
    storable-element references defer to ``resolve_sid``; the caller owns the
    ``executemany`` on the returned rows.

    :param schema: The parent record's resolved table schema.
    :param spec: The child field specification being encoded.
    :param sid: The parent row's sid, stamped into every child row.
    :param value: The child field value (a sequence, tensor, or ``None``).
    :param path: The projection path prefix used for diagnostics.
    :param resolve_sid: The callback assigning a sid to each referenced element record.
    :return: The encoded child-table rows in element order.
    """
    assert spec.child is not None
    parent_column = f"{schema.table_name}_sid"
    index_column = f"{spec.field}_index"
    rows: list[dict[str, Any]] = []
    if spec.shape is not None:
        for position, row_tensor in enumerate(_tensor_rows(schema, spec, spec.shape, value)):
            row: dict[str, Any] = {parent_column: sid, index_column: position}
            for i, part in enumerate(encode_fracvector_floats(row_tensor)):
                row[f"{spec.field}_{i}"] = part
            row[f"{spec.field}_exact"] = encode_fracvector_exact(row_tensor)
            rows.append(row)
    else:
        codec = codec_named(spec.codec_name) if spec.codec_name is not None else None
        for position, element in enumerate(value if value is not None else ()):
            row = {parent_column: sid, index_column: position}
            if spec.target is not None:
                row[spec.child.element_columns[0].name] = resolve_sid(spec.target, element, f"{path}[{position}]")
            elif codec is not None:
                for column, part in zip(spec.child.element_columns, codec.encode(element), strict=True):
                    row[column.name] = part
            else:
                row[spec.child.element_columns[0].name] = element
            rows.append(row)
    return rows


def _metadata_scalar_equal(left: Any, right: Any) -> bool:
    if isinstance(left, list | tuple) or isinstance(right, list | tuple):
        if type(left) is not type(right) or len(left) != len(right):
            return False
        return all(_metadata_scalar_equal(left_item, right_item) for left_item, right_item in zip(left, right))
    if isinstance(left, datetime.datetime) and isinstance(right, datetime.datetime):
        left_aware = left.utcoffset() is not None
        right_aware = right.utcoffset() is not None
        if left_aware != right_aware:
            return False
        if left_aware:
            return left.astimezone(datetime.UTC) == right.astimezone(datetime.UTC)
    return bool(left == right)
