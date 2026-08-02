"""The SQL store: save and fetch storable frozen dataclasses through a :class:`~httk.data.db.engine.Database`.

:class:`SqlStore` is the object-level storage API on top of the schema IR
(:mod:`httk.data.db.schema`), the value codecs (:mod:`httk.data.db.codecs`),
the content identity (:mod:`httk.data.db.identity`), and the SQLAlchemy table
mapping (:mod:`httk.data.db.mapping`):

- :meth:`SqlStore.save` writes an instance (recursing into referenced and
  child-element storables) and returns its integer ``sid``, deduplicating per
  the class's :attr:`~httk.core.StorageInfo.dedup` policy;
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

One small, documented liberty: an optional child-table field saved as
``None`` comes back as an empty container (the relational layout cannot tell
the two apart). Identity caches are best-effort; content-addressed
:meth:`SqlStore.sid_of` lookups fall back to the database.
"""

import contextlib
import datetime
import json
import threading
import types
import typing
import weakref
from collections.abc import Iterable, Iterator, Mapping
from typing import Annotated, Any, Literal, cast

import sqlalchemy
from httk.core import (
    FracVector,
    IdentitySkip,
    Shape,
    StorageProjectionCycleError,
    content_id,
    project_storage_record,
    resolve_storage_record,
)
from sqlalchemy.exc import SQLAlchemyError

from httk.data.db.codecs import (
    codec_named,
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
    normalize_entry_backings,
    read_store_metadata,
    validate_expected_tables,
)
from httk.data.db.mapping import (
    CONTENT_ID_COLUMN,
    DISPATCH_CONTENT_ID_COLUMN,
    SID_COLUMN,
    backing_dispatch_column_name,
    table_for,
)
from httk.data.db.rows import RowHydrator, StaleResultError, is_lazy_row, lazy_row_identity
from httk.data.db.schema import FieldSpec, SchemaError, TableSchema, resolve_schema
from httk.data.db.searcher import SqlSearcher

__all__ = [
    "EntryDispatchIntegrityError",
    "EntryMetadataConflictError",
    "SqlStore",
]


class EntryMetadataConflictError(ValueError):
    """Stored identity-excluded metadata differs from a repeated save."""


class EntryDispatchIntegrityError(RuntimeError):
    """A persisted entry dispatch row does not name exactly its expected backing."""


class _Projection:
    """One-save projection cache shared by core identity and SQL encoding."""

    def __init__(self) -> None:
        self.values_by_source: dict[tuple[type, int], Mapping[str, object]] = {}
        self.active: set[tuple[type, int]] = set()
        self.inserted: list[tuple[type, int]] = []

    def projector(self, record_type: type, source: Any) -> Mapping[str, object]:
        key = (record_type, id(source))
        values = self.values_by_source.get(key)
        if values is None:
            values = project_storage_record(record_type, source)
            self.values_by_source[key] = values
        return values

    def content_id(self, record_type: type, source: Any) -> str:
        return content_id(source, as_record=record_type, projector=self.projector)


def _schema_object_type(kinds: frozenset[str]) -> object:
    return next(iter(kinds)) if len(kinds) == 1 else tuple(sorted(kinds))


class SqlStore:
    """Object storage for storable frozen dataclasses in a relational :class:`~httk.data.db.engine.Database`.

    A store starts with an explicit, versioned entry declaration. Ordinary
    unconfigured frozen-dataclass tables remain on-demand, but only after the
    layout marker has been initialized on a physically empty database.
    """

    def __init__(
        self,
        database: Database,
        *,
        entry_backings: Mapping[type, type | tuple[type, ...]] | None = None,
        layout_mode: Literal["ensure", "verify"] = "ensure",
    ) -> None:
        if not isinstance(layout_mode, str):
            raise TypeError("layout_mode must be 'ensure' or 'verify'")
        if layout_mode not in {"ensure", "verify"}:
            raise ValueError("layout_mode must be 'ensure' or 'verify'")
        self._database = database
        self._layout_mode: Literal["ensure", "verify"] = layout_mode
        self._metadata = sqlalchemy.MetaData()
        self._layout: StorageLayout | None = None
        self._managed_table_names: frozenset[str] = frozenset()
        self._unclaimed_table_names: frozenset[str] = frozenset()
        self._validated_table_names: set[str] = set()
        self._initialized = False
        self._initialization_ddl_journal: list[sqlalchemy.Table] = []
        self._instances: weakref.WeakValueDictionary[tuple[type, int], Any] = weakref.WeakValueDictionary()
        self._sids: weakref.WeakKeyDictionary[Any, dict[type, int]] = weakref.WeakKeyDictionary()
        self._sids_by_identity: dict[tuple[type, int], int] = {}
        """Reverse cache for instances that cannot be hashed (e.g. they hold a list).

        Keyed on ``id()``, with a finalizer dropping each entry when its
        instance dies, so a recycled id can never resolve to a stale sid.
        """
        self._local = threading.local()
        self._initialize_layout(entry_backings)

    @property
    def layout(self) -> StorageLayout:
        """The immutable persisted entry declaration and its resolved classes."""
        assert self._layout is not None
        return self._layout

    @property
    def entry_layout(self) -> tuple[EntryFamilyLayout, ...]:
        """Configured entry-family layouts in deterministic stable-name order."""
        return self.layout.families

    @property
    def entry_backings(self) -> Mapping[type, tuple[type, ...]]:
        """Configured entry-family classes mapped to ordered concrete record backings."""
        return self.layout.entry_backings

    def _initialize_layout(self, entry_backings: Mapping[type, type | tuple[type, ...]] | None) -> None:
        supplied = normalize_entry_backings(entry_backings) if entry_backings is not None else None
        if self._layout_mode == "verify":
            with self._database.engine.connect() as connection:
                self._initialize_layout_on_connection(connection, supplied, write=False)
            return
        try:
            with self._database.engine.begin() as connection:
                self._initialize_layout_on_connection(connection, supplied, write=True)
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
            self._unclaimed_table_names = frozenset()
            self._validated_table_names.clear()
            self._initialized = False
            self._clear_identity_caches()
            raise
        self._initialization_ddl_journal.clear()

    def _initialize_layout_on_connection(
        self,
        connection: sqlalchemy.Connection,
        supplied: StorageLayout | None,
        *,
        write: bool,
    ) -> None:
        objects_before = actual_schema_objects(connection)
        names_before = frozenset(name for name, kinds in objects_before.items() if "table" in kinds)
        marker_metadata = sqlalchemy.MetaData()
        metadata_table_for(marker_metadata)
        if METADATA_TABLE_NAME in names_before:
            marker_schema = validate_expected_tables(connection, marker_metadata)
            if marker_schema:
                raise StorageLayoutUpgradeRequiredError({"schema": marker_schema})
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
                raise TypeError("entry_backings is required when opening an uninitialized database")
            expected = expected_metadata(supplied)
            if not write:
                raise StorageLayoutUpgradeRequiredError(
                    {
                        "protocol": {"expected": STORAGE_PROTOCOL_VERSION, "actual": None},
                        "declaration": {"expected": declaration_json(supplied), "actual": None},
                        "schema": {name: {"missing": True} for name in expected.tables},
                    }
                )
            for table in expected.sorted_tables:
                table.create(connection, checkfirst=False)
                self._initialization_ddl_journal.append(table)
            self._stamp_layout(connection, supplied, ())
            self._install_layout(supplied, expected, ())
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
                    "expected": declaration_json(supplied) if supplied is not None else "explicit entry_backings",
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
        required_keys = {"protocol", "entry_declaration", "unclaimed_tables"}
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
        unclaimed: tuple[str, ...] = ()
        try:
            unclaimed = self._unclaimed_from_stored_value(stored.get("unclaimed_tables"))
        except (TypeError, ValueError) as error:
            diff["declaration"] = {
                "expected": "canonical unclaimed-table list",
                "actual": stored.get("unclaimed_tables"),
                "error": str(error),
            }
        if persisted is not None and supplied is not None and declaration_json(persisted) != declaration_json(supplied):
            diff["declaration"] = {
                "expected": declaration_json(persisted),
                "actual": declaration_json(supplied),
            }
        expected: sqlalchemy.MetaData | None = None
        if persisted is not None:
            expected = expected_metadata(persisted)
            overlap = tuple(sorted(set(unclaimed) & set(expected.tables)))
            if overlap:
                diff["declaration"] = {
                    "unclaimed_tables": {
                        "overlap_with_protocol_tables": overlap,
                    }
                }
        if diff:
            raise StorageLayoutUpgradeRequiredError(diff)
        assert persisted is not None and expected is not None
        object_problems: dict[str, object] = {}
        for name, kinds in actual_schema_objects(connection).items():
            if name in expected.tables and kinds != frozenset(("table",)):
                object_problems[name] = {
                    "object_type": _schema_object_type(kinds),
                    "message": "a managed table name is also used by an unexpected schema object",
                }
            elif name.startswith("_httk_") and name not in expected.tables:
                object_problems[name] = {
                    "reserved": True,
                    "object_type": _schema_object_type(kinds),
                    "message": "unexpected schema object uses the SqlStore-reserved _httk_ prefix",
                }
        if object_problems:
            raise StorageLayoutUpgradeRequiredError({"schema": object_problems})
        schema_difference = validate_expected_tables(connection, expected)
        if schema_difference:
            raise StorageLayoutUpgradeRequiredError({"schema": schema_difference})
        self._audit_dispatch_data(connection, persisted, expected)
        self._install_layout(persisted, expected, unclaimed)

    @staticmethod
    def _layout_from_stored_declaration(value: str | None) -> StorageLayout:
        # The canonical JSON parser is intentionally private to layout.py; a
        # no-op explicit declaration round-trip uses the public normalizer.
        from httk.data.db.layout import _layout_from_declaration

        if value is None:
            raise ValueError("metadata is missing entry_declaration")
        return _layout_from_declaration(value)

    @staticmethod
    def _unclaimed_from_stored_value(value: str | None) -> tuple[str, ...]:
        if value is None:
            raise ValueError("metadata is missing unclaimed_tables")
        parsed = json.loads(value)
        if not isinstance(parsed, list) or any(not isinstance(name, str) for name in parsed):
            raise ValueError("unclaimed_tables must be a JSON string list")
        names = tuple(parsed)
        if names != tuple(sorted(set(names))):
            raise ValueError("unclaimed_tables is not canonically sorted and unique")
        reserved = tuple(name for name in names if name.startswith("_httk_"))
        if reserved:
            raise ValueError("unclaimed_tables may not include SqlStore-reserved _httk_ names")
        return names

    def _stamp_layout(
        self,
        connection: sqlalchemy.Connection,
        layout: StorageLayout,
        unclaimed: tuple[str, ...],
    ) -> None:
        table = metadata_table_for(sqlalchemy.MetaData())
        connection.execute(
            sqlalchemy.insert(table),
            (
                {"key": "protocol", "value": STORAGE_PROTOCOL_VERSION},
                {"key": "entry_declaration", "value": declaration_json(layout)},
                {"key": "unclaimed_tables", "value": json.dumps(list(unclaimed), separators=(",", ":"))},
            ),
        )

    def _install_layout(
        self,
        layout: StorageLayout,
        metadata: sqlalchemy.MetaData,
        unclaimed: tuple[str, ...],
    ) -> None:
        self._layout = layout
        self._metadata = metadata
        self._managed_table_names = frozenset(metadata.tables)
        self._unclaimed_table_names = frozenset(unclaimed)
        self._validated_table_names = set(metadata.tables)
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

    @staticmethod
    def _audit_dispatch_data(
        connection: sqlalchemy.Connection,
        layout: StorageLayout,
        metadata: sqlalchemy.MetaData,
    ) -> None:
        """Verify that every configured multi-backing row has one exact dispatch row.

        The dispatch constraints ensure a row selects at most one concrete
        backing, but they cannot enforce the inverse: a backing may be saved
        without a dispatch row or a valid foreign key may point at a backing
        whose content id differs.  Both conditions make entry lookup
        incomplete or wrong, so opening a store refuses them.
        """
        problems: dict[str, object] = {}
        for family in layout.families:
            if len(family.backings) < 2:
                continue
            assert family.dispatch_table_name is not None
            dispatch = metadata.tables[family.dispatch_table_name]
            data: dict[str, object] = {}
            dispatch_columns = tuple(
                dispatch.c[backing_dispatch_column_name(backing_name)] for backing_name in family.backing_names
            )
            cardinality = sum(
                (sqlalchemy.case((column.is_not(None), 1), else_=0) for column in dispatch_columns),
                sqlalchemy.literal(0),
            )
            invalid = (
                connection.execute(
                    sqlalchemy.select(dispatch.c[DISPATCH_CONTENT_ID_COLUMN], *dispatch_columns)
                    .where(cardinality != 1)
                    .limit(1)
                )
                .mappings()
                .first()
            )
            if invalid is not None:
                data["invalid_dispatch_rows"] = (
                    {
                        "content_id": str(invalid[DISPATCH_CONTENT_ID_COLUMN]),
                        "populated_backings": tuple(
                            backing_name
                            for backing_name, column in zip(family.backing_names, dispatch_columns, strict=True)
                            if invalid[column.name] is not None
                        ),
                    },
                )

            for backing_name, backing in zip(family.backing_names, family.backings, strict=True):
                backing_table = metadata.tables[resolve_schema(backing).table_name]
                dispatch_column = dispatch.c[backing_dispatch_column_name(backing_name)]
                matching_dispatch = sqlalchemy.exists(
                    sqlalchemy.select(sqlalchemy.literal(1))
                    .select_from(dispatch)
                    .where(
                        dispatch.c[DISPATCH_CONTENT_ID_COLUMN] == backing_table.c[CONTENT_ID_COLUMN],
                        dispatch_column == backing_table.c[SID_COLUMN],
                    )
                )
                missing = connection.execute(
                    sqlalchemy.select(backing_table.c[SID_COLUMN], backing_table.c[CONTENT_ID_COLUMN])
                    .where(~matching_dispatch)
                    .limit(1)
                ).first()
                if missing is not None and "missing_dispatch_rows" not in data:
                    data["missing_dispatch_rows"] = (
                        {"content_id": str(missing[1]), "backing": backing_name, "sid": int(missing[0])},
                    )

                matching_backing = sqlalchemy.exists(
                    sqlalchemy.select(sqlalchemy.literal(1))
                    .select_from(backing_table)
                    .where(
                        backing_table.c[SID_COLUMN] == dispatch_column,
                        backing_table.c[CONTENT_ID_COLUMN] == dispatch.c[DISPATCH_CONTENT_ID_COLUMN],
                    )
                )
                mismatch = connection.execute(
                    sqlalchemy.select(dispatch.c[DISPATCH_CONTENT_ID_COLUMN], dispatch_column)
                    .where(dispatch_column.is_not(None), ~matching_backing)
                    .limit(1)
                ).first()
                if mismatch is not None and "orphaned_or_mismatched_dispatch_rows" not in data:
                    data["orphaned_or_mismatched_dispatch_rows"] = (
                        {"content_id": str(mismatch[0]), "backing": backing_name, "sid": int(mismatch[1])},
                    )

            if data:
                problems[family.dispatch_table_name] = {"data": data}
        if problems:
            raise StorageLayoutUpgradeRequiredError({"schema": problems})

    # ------------------------------------------------------------------ tables and transactions

    def ensure_tables(self, *classes: type) -> None:
        """Resolve each class's schema and create its tables (and those it references, transitively).

        Existing tables are validated and left alone. Saving or fetching calls
        this implicitly, but calling it up front is useful to establish and
        validate the physical layout before a data transaction.
        """
        self._ensure_tables(self._current_connection(), classes)

    def transaction(self) -> contextlib.AbstractContextManager[None]:
        """Scope the operations of a ``with`` block into one database transaction.

        Every :meth:`save`/:meth:`fetch` (on this thread) inside the block runs
        on the same open connection; the transaction commits when the block
        exits normally and rolls back if it raises. On a rollback the identity
        caches are flushed, since they may name rows that no longer exist.
        Nesting is flat: an inner ``transaction()`` block simply joins the
        outer transaction. Outside any transaction block, each operation runs
        (and autocommits) on its own.
        """
        return self._transaction_scope()

    @contextlib.contextmanager
    def _transaction_scope(self) -> Iterator[None]:
        stack = self._connection_stack()
        if stack:
            yield
            return
        try:
            with self._database.engine.begin() as connection:
                stack.append(connection)
                try:
                    yield
                finally:
                    stack.pop()
        except BaseException:
            self._invalidate_private_table_validations()
            self._clear_identity_caches()
            raise

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
        try:
            with self._database.engine.begin() as connection:
                stack = self._connection_stack()
                stack.append(connection)
                try:
                    yield connection
                finally:
                    stack.pop()
        except BaseException:
            self._invalidate_private_table_validations()
            self._clear_identity_caches()
            raise

    @contextlib.contextmanager
    def _read_connection(self) -> Iterator[sqlalchemy.Connection]:
        current = self._current_connection()
        if current is not None:
            yield current
            return
        try:
            # A read path can establish a missing private table in ``ensure``
            # mode. Commit that DDL on success; on failure DuckDB rolls it back
            # and the validation cache must forget it.
            with self._database.engine.begin() as connection:
                stack = self._connection_stack()
                stack.append(connection)
                try:
                    yield connection
                finally:
                    stack.pop()
        except BaseException:
            self._invalidate_private_table_validations()
            self._clear_identity_caches()
            raise

    def _invalidate_private_table_validations(self) -> None:
        """Keep only initialization-validated protocol tables after a transaction failure."""
        self._validated_table_names.intersection_update(self._managed_table_names)

    def _ensure_tables(self, connection: sqlalchemy.Connection | None, classes: Iterable[type]) -> None:
        """Validate or create ordinary record tables after protocol initialization.

        Configured entry tables are installed at construction.  Every other
        class is private storage: an existing table is always compared to the
        SQLAlchemy-derived layout before use, and a missing one is created
        only by an ``ensure`` store.
        """
        if not self._initialized:
            raise RuntimeError("SqlStore layout has not been initialized")
        if connection is None:
            # RowHydrator calls this helper before it obtains its own nested
            # read scope. Reuse the caller's connection when one is active:
            # DuckDB permits only one transaction on its shared in-memory
            # connection, and this also preserves transaction visibility.
            connection = self._current_connection()
        candidate = sqlalchemy.MetaData()
        for cls in classes:
            table_for(resolve_schema(cls), candidate)
        candidate_names = frozenset(candidate.tables)
        forbidden = sorted(
            name for name in candidate_names if name.startswith("_httk_") and name not in self._managed_table_names
        )
        if forbidden:
            raise ValueError(f"ordinary records may not claim reserved SqlStore table names: {', '.join(forbidden)}")
        collisions = tuple(sorted(candidate_names & self._unclaimed_table_names))
        if collisions:
            raise StorageLayoutUpgradeRequiredError(
                {
                    "schema": {
                        name: {
                            "unversioned": True,
                            "message": "pre-existing unclaimed table cannot be adopted as private storage",
                        }
                        for name in collisions
                    }
                }
            )
        pending_validation = candidate_names - self._validated_table_names
        if not pending_validation:
            for cls in classes:
                table_for(resolve_schema(cls), self._metadata)
            return

        def ensure_on(active: sqlalchemy.Connection) -> None:
            names = actual_table_names(active)
            existing = tuple(sorted(pending_validation & names))
            difference = validate_expected_tables(active, candidate, table_names=existing)
            if difference:
                raise StorageLayoutUpgradeRequiredError({"schema": difference})
            missing = tuple(sorted(pending_validation - names))
            if missing:
                if self._layout_mode == "verify":
                    raise StorageLayoutUpgradeRequiredError({"schema": {name: {"missing": True} for name in missing}})
                # The candidate contains all transitive targets. ``checkfirst``
                # is safe only after the explicit validation above, and lets
                # SQLAlchemy order foreign-key creation for each dialect.
                candidate.create_all(active, checkfirst=True)

        if connection is not None:
            ensure_on(connection)
        else:
            if self._layout_mode == "verify":
                with self._database.engine.connect() as active:
                    ensure_on(active)
            else:
                with self._database.engine.begin() as active:
                    ensure_on(active)
        self._validated_table_names.update(candidate_names)
        for cls in classes:
            table_for(resolve_schema(cls), self._metadata)

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
        """
        if getattr(obj, "__httk_cursor_proxy__", False):
            raise TypeError("cursor rows cannot be saved; materialize the record first")
        record_type = resolve_storage_record(obj, as_record=as_record)
        projection = _Projection()
        with self._write_connection() as connection:
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
        self._ensure_tables(connection, (record_type,))
        table = self._table(schema.table_name)
        projected = projection.projector(record_type, source)

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
        values: dict[str, Any] = {}
        for spec in schema.fields:
            if spec.role == "child":
                continue
            value = self._projected_value(schema.cls, source, projected, spec)
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
                values[spec.columns[0].name] = self._save(
                    connection, spec.target, value, projection, _field_path(path, spec.field)
                )
        return values

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
        table = self._table(spec.child.table_name)
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
                    row[spec.child.element_columns[0].name] = self._save(
                        connection, spec.target, element, projection, f"{path}[{position}]"
                    )
                elif codec is not None:
                    for column, part in zip(spec.child.element_columns, codec.encode(element), strict=True):
                        row[column.name] = part
                else:
                    row[spec.child.element_columns[0].name] = element
                rows.append(row)
        if rows:
            connection.execute(sqlalchemy.insert(table), rows)

    # ------------------------------------------------------------------ fetching

    def fetch[T](self, cls: type[T], sid: int) -> T:
        """Reconstruct the ``cls`` instance stored under ``sid``.

        While a previously fetched (or saved) instance for this ``(class,
        sid)`` is alive, the very same object is returned. Raises
        :class:`KeyError` (carrying the class and sid) when no such row exists.
        """
        with self._read_connection() as connection:
            return cast(T, self._fetch(connection, cls, sid))

    def fetch_by_content_id[T](self, cls: type[T], key: str) -> T | None:
        """The ``cls`` instance whose content identity is ``key``, or None if not stored.

        Only classes with the ``"content_id"`` dedup policy carry a content
        identity column; :class:`~httk.data.db.schema.SchemaError` is raised
        for any other class.
        """
        schema = resolve_schema(cls)
        if schema.dedup != "content_id":
            raise SchemaError(
                f"{cls.__name__} has dedup policy {schema.dedup!r}; only classes with the "
                f"'content_id' policy have a content identity column"
            )
        with self._read_connection() as connection:
            self._ensure_tables(connection, (cls,))
            table = self._table(schema.table_name)
            found = connection.execute(
                sqlalchemy.select(table.c[SID_COLUMN]).where(table.c[CONTENT_ID_COLUMN] == key)
            ).first()
            if found is None:
                return None
            return cast(T, self._fetch(connection, cls, int(found[0])))

    def fetch_entry(self, family_cls: type, content_id: str) -> object | None:
        """Return the concrete configured backing for an entry-family content identity.

        The result is the actual frozen record class, not the family protocol.
        A single-backing family can query that backing directly; only
        multi-backing families need their reserved one-of-many dispatch table.
        """
        family = next((item for item in self.layout.families if item.family is family_cls), None)
        if family is None:
            raise ValueError(f"{family_cls.__name__} is not a configured entry family in this SqlStore")
        with self._read_connection() as connection:
            if len(family.backings) == 1:
                backing = family.backings[0]
                schema = resolve_schema(backing)
                self._ensure_tables(connection, (backing,))
                table = self._table(schema.table_name)
                sid = connection.execute(
                    sqlalchemy.select(table.c[SID_COLUMN]).where(table.c[CONTENT_ID_COLUMN] == content_id)
                ).scalar_one_or_none()
                return None if sid is None else self._fetch(connection, backing, int(sid))
            assert family.dispatch_table_name is not None
            table = self._table(family.dispatch_table_name)
            row = (
                connection.execute(sqlalchemy.select(table).where(table.c[DISPATCH_CONTENT_ID_COLUMN] == content_id))
                .mappings()
                .one_or_none()
            )
            if row is None:
                for backing in family.backings:
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
        """Return this store's sid for ``obj``'s record identity, if present."""
        record_type = resolve_storage_record(obj, as_record=as_record)
        lazy_identity = lazy_row_identity(obj)
        if is_lazy_row(obj) and record_type is resolve_storage_record(obj):
            return lazy_identity[1] if lazy_identity is not None and lazy_identity[0] is self else None
        try:
            cached = self._sids.get(obj, {}).get(record_type)
        except TypeError:
            cached = None
        if cached is None:
            cached = self._sids_by_identity.get((record_type, id(obj)))
        if cached is not None:
            return cached

        schema = resolve_schema(record_type)
        if schema.dedup != "content_id":
            return cached
        projection = _Projection()
        key = projection.content_id(record_type, obj)
        with self._read_connection() as connection:
            self._ensure_tables(connection, (record_type,))
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
        """A new :class:`~httk.data.db.searcher.SqlSearcher` querying this store.

        The searcher runs on this store's read path — inside an open
        :meth:`transaction` block it sees uncommitted writes — and
        reconstructs matched objects through :meth:`fetch`, so the identity
        cache applies.
        """
        return SqlSearcher(self)

    def referring(self, cls: type, *, field: str, to: Any) -> list[Any]:
        """All stored ``cls`` instances whose reference field ``field`` points at ``to``.

        ``field`` must be a reference field of ``cls`` targeting ``to``'s class
        (:class:`~httk.data.db.schema.SchemaError` otherwise), and ``to`` must
        be known to this store — saved or fetched through it — else
        :class:`ValueError` is raised. Results are ordered by sid.
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
            self._ensure_tables(connection, (cls,))
            table = self._table(schema.table_name)
            found = connection.execute(
                sqlalchemy.select(table.c[SID_COLUMN])
                .where(table.c[spec.columns[0].name] == sid)
                .order_by(table.c[SID_COLUMN])
            ).all()
            return [self._fetch(connection, cls, int(row[0])) for row in found]

    def _fetch(self, connection: sqlalchemy.Connection, cls: type, sid: int) -> Any:
        sid = int(sid)
        cached = self._instances.get((cls, sid))
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
            if any(backing is record_type for backing in family.backings):
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
        if len(family.backings) == 1:
            return
        assert family.dispatch_table_name is not None
        table = self._table(family.dispatch_table_name)
        column_name = backing_dispatch_column_name(family.backing_names[family.backings.index(backing)])
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
        for backing_name, backing in zip(family.backing_names, family.backings, strict=True):
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
        stored = self._fetch(connection, record_type, sid)
        self._check_record_metadata(record_type, source, stored, projection, record_type.__name__)

    def _check_record_metadata(
        self,
        record_type: type,
        source: Any,
        stored: Any,
        projection: _Projection,
        path: str,
    ) -> None:
        values = projection.projector(record_type, source)
        hints = typing.get_type_hints(record_type, include_extras=True)
        for spec in resolve_schema(record_type).fields:
            if spec.derived:
                continue
            incoming = values[spec.field]
            existing = getattr(stored, spec.field)
            field_path = f"{path}.{spec.field}"
            identity_skipped = _has_identity_skip(hints[spec.field])
            if identity_skipped and not self._metadata_value_equal(spec, incoming, existing, projection, field_path):
                raise EntryMetadataConflictError(
                    f"metadata conflict for {field_path}: stored {existing!r}, received {incoming!r}"
                )
            if not identity_skipped and spec.target is not None:
                self._check_nested_metadata(spec, incoming, existing, projection, field_path)

    def _metadata_value_equal(
        self,
        spec: FieldSpec,
        incoming: Any,
        existing: Any,
        projection: _Projection,
        path: str,
    ) -> bool:
        if spec.role == "child" and (
            (incoming is None and isinstance(existing, list | tuple) and not existing)
            or (existing is None and isinstance(incoming, list | tuple) and not incoming)
        ):
            return True
        if incoming is None or existing is None:
            return incoming is existing
        if spec.target is not None:
            try:
                self._check_nested_metadata(spec, incoming, existing, projection, path, compare_content=True)
            except EntryMetadataConflictError:
                return False
            return True
        return _metadata_scalar_equal(incoming, existing)

    def _check_nested_metadata(
        self,
        spec: FieldSpec,
        incoming: Any,
        existing: Any,
        projection: _Projection,
        path: str,
        *,
        compare_content: bool = False,
    ) -> None:
        assert spec.target is not None
        if incoming is None or existing is None:
            if incoming is not existing:
                raise EntryMetadataConflictError(f"metadata conflict for {path}")
            return
        pairs: Iterable[tuple[Any, Any]]
        if spec.role == "reference":
            pairs = ((incoming, existing),)
        else:
            if len(incoming) != len(existing):
                raise EntryMetadataConflictError(f"metadata conflict for {path}")
            pairs = zip(incoming, existing, strict=True)
        for index, (incoming_item, existing_item) in enumerate(pairs):
            item_path = path if spec.role == "reference" else f"{path}[{index}]"
            if compare_content and projection.content_id(spec.target, incoming_item) != projection.content_id(
                spec.target, existing_item
            ):
                raise EntryMetadataConflictError(f"metadata conflict for {item_path}")
            self._check_record_metadata(spec.target, incoming_item, existing_item, projection, item_path)

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
        self._instances.clear()
        self._sids.clear()
        self._sids_by_identity.clear()

    def _remember(self, cls: type, sid: int, obj: Any, *, cache_instance: bool = True) -> None:
        if cache_instance:
            try:
                self._instances[(cls, sid)] = obj
            except TypeError:
                return  # Not weak-referenceable; identity caching is best-effort.
        try:
            sids = self._sids.setdefault(obj, {})
            sids[cls] = sid
        except TypeError:
            # Unhashable (a storable class holding a list field is): key the
            # reverse cache on identity instead, dropping the entry when the
            # instance dies. Without this, sid_of() — and so referring() —
            # would report a just-saved instance as never stored.
            key = (cls, id(obj))
            try:
                weakref.finalize(obj, self._sids_by_identity.pop, key, None)
            except TypeError:
                return  # Tuples and other non-weakrefable sources use database lookup.
            self._sids_by_identity[key] = sid


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


def _has_identity_skip(annotation: Any) -> bool:
    origin = typing.get_origin(annotation)
    if origin is Annotated:
        arguments = typing.get_args(annotation)
        return any(isinstance(marker, IdentitySkip) for marker in arguments[1:]) or _has_identity_skip(arguments[0])
    if origin in (typing.Union, types.UnionType):
        return any(_has_identity_skip(argument) for argument in typing.get_args(annotation))
    return False


def _field_path(path: str, field: str) -> str:
    return f"{path}.{field}" if path else field


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
