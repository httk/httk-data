"""MongoDB store layout initialization and collection preparation."""

import contextlib
import datetime
import logging
import threading
import time
import typing
from collections.abc import Mapping
from typing import Any

from httk.core import FracVector
from httk.core.storage import StorageProjectionCycleError, resolve_storage_record
from pymongo import IndexModel
from pymongo.errors import CollectionInvalid, DuplicateKeyError, PyMongoError
from pymongo.read_concern import ReadConcern
from pymongo.write_concern import WriteConcern

from httk.data.db.codecs import codec_named, decode_fracvector_exact
from httk.data.db.schema import SchemaError, resolve_schema
from httk.data.storage_layout import (
    DECLARATION_PROTOCOL_VERSION,
    EntryFamilyLayout,
    StorageLayout,
    StorageLayoutUpgradeRequiredError,
    _layout_from_declaration,
    declaration_json,
    normalize_entry_records,
)
from httk.data.store_common import (
    EntryDispatchIntegrityError,
    EntryMetadataConflictError,
    IdentityCaches,
    SaveProjection,
    _metadata_plan,
    reject_cursor_proxy,
)

from .database import MongoDatabase, TransactionsUnavailableError
from .documents import decode_record, encode_record, preflight_document
from .fsck import FsckSummary
from .leases import WriterLease, acquire_writer, clear_stale_lock
from .mapping import (
    COUNTERS_COLLECTION,
    METADATA_COLLECTION,
    collection_name_for,
    counter_next,
    dispatch_index_specs,
    dispatch_validator_for,
    document_fields_for,
    entry_dispatch_table_name,
    index_specs_for,
    validator_for,
)

__all__ = ["MongoStore"]

_DOCUMENT_LAYOUT = "mongo-v2"
_RESERVED_PREFIX = "_httk_"
_METADATA_KEYS = frozenset({"_id", "protocol", "entry_declaration", "document_layout", "generation"})
_LOGGER = logging.getLogger("httk.data.mongo")
_TRANSACTION_ATTEMPTS = 5


class _HydrationContext:
    """Per-fetch document cache shared by one recursive hydration."""

    def __init__(self) -> None:
        self.documents: dict[tuple[type, int], Mapping[str, Any]] = {}


class _TransactionState:
    """Thread-local transaction session and deferred identity-cache entries."""

    def __init__(self, session: Any, lease: WriterLease | None) -> None:
        self.session = session
        self.lease = lease
        self.pending: dict[tuple[type, int], tuple[Any, bool]] = {}
        self.pending_sids: dict[tuple[type, int], int] = {}


class MongoStore:
    """Object store foundation for MongoDB-backed storable records.

    Construction stamps a new empty database or validates the existing layout
    declaration.  Record collections are deliberately created only by the
    explicit :meth:`ensure_collections` operation; save and fetch belong to a
    later phase.

    :param database: The MongoDB database wrapper.
    :param entry_records: The required entry-family declaration on first open.
    :raises TypeError: If the first open omits ``entry_records``.
    :raises ~httk.data.storage_layout.StorageLayoutUpgradeRequiredError: If the
        persisted layout is not trusted by this implementation.
    """

    supports_page = True
    """Whether this backend implements keyset result paging."""

    def __init__(
        self,
        database: MongoDatabase,
        *,
        entry_records: Mapping[type, type | tuple[type, ...]] | None = None,
    ) -> None:
        self._database = database
        self._layout: StorageLayout | None = None
        self._collections_ready: set[str] = set()
        # Layout declarations describe the persistent roots; this additional
        # set lets fsck also attribute arbitrary record classes saved through
        # this live store instance.
        self._known_record_types: set[type] = set()
        self._identity = IdentityCaches()
        self._write_lock = threading.RLock()
        self._local = threading.local()
        self._failed_identities: set[tuple[type, int]] = set()
        hello = database.client.admin.command("hello")
        self._max_bson_size = int(hello.get("maxBsonObjectSize", 16 * 1024 * 1024))
        if not database.supports_transactions:
            _LOGGER.warning(
                "MongoStore is running in degraded mode without multi-document transactions",
                extra={"context": "storage"},
            )
        supplied = normalize_entry_records(entry_records) if entry_records is not None else None
        self._initialize_layout(supplied)
        for family in self.layout.families:
            self._known_record_types.update(family.records)
        self._last_generation = self._layout_generation()

    @property
    def layout(self) -> StorageLayout:
        """Return the immutable persisted entry declaration.

        :return: The normalized storage layout.
        """
        assert self._layout is not None
        return self._layout

    @property
    def entry_layout(self) -> tuple[EntryFamilyLayout, ...]:
        """Return configured entry-family layouts in stable order.

        :return: The configured entry-family layouts.
        """
        return self.layout.families

    @property
    def entry_records(self) -> Mapping[type, tuple[type, ...]]:
        """Return configured family classes mapped to backing classes.

        :return: The normalized entry declaration keyed by family class.
        """
        return self.layout.entry_records

    def _initialize_layout(self, supplied: StorageLayout | None) -> None:
        database = self._database.database
        names = {name for name in database.list_collection_names() if not name.startswith("system.")}
        metadata_exists = METADATA_COLLECTION in names
        stored = database[METADATA_COLLECTION].find_one({"_id": "layout"}) if metadata_exists else None

        if not metadata_exists and not names:
            if supplied is None:
                raise TypeError("entry_records is required when opening an uninitialized database")
            self._validate_layout_names(supplied)
            document = {
                "_id": "layout",
                "protocol": DECLARATION_PROTOCOL_VERSION,
                "entry_declaration": declaration_json(supplied),
                "document_layout": _DOCUMENT_LAYOUT,
                "generation": 0,
            }
            try:
                database[METADATA_COLLECTION].insert_one(document)
            except DuplicateKeyError:
                # Another opener won the single-document first-open race.
                stored = database[METADATA_COLLECTION].find_one({"_id": "layout"})
            else:
                self._install_layout(supplied)
                return

        if stored is None:
            self._raise_unversioned(names)
        assert stored is not None
        self._open_marked_layout(stored, supplied, names)

    def _open_marked_layout(
        self,
        stored: Mapping[str, Any],
        supplied: StorageLayout | None,
        collection_names: set[str],
    ) -> None:
        diff: dict[str, object] = {}
        if set(stored) != _METADATA_KEYS:
            diff["declaration"] = {
                "metadata_keys": {"expected": tuple(sorted(_METADATA_KEYS)), "actual": tuple(sorted(stored))}
            }
        protocol_actual = stored.get("protocol")
        document_layout_actual = stored.get("document_layout")
        if protocol_actual != DECLARATION_PROTOCOL_VERSION or document_layout_actual != _DOCUMENT_LAYOUT:
            diff["protocol"] = {
                "expected": {"protocol": DECLARATION_PROTOCOL_VERSION, "document_layout": _DOCUMENT_LAYOUT},
                "actual": {"protocol": protocol_actual, "document_layout": document_layout_actual},
            }

        persisted: StorageLayout | None = None
        declaration = stored.get("entry_declaration")
        try:
            if not isinstance(declaration, str):
                raise ValueError("metadata is missing entry_declaration")
            persisted = _layout_from_declaration(declaration)
            if declaration_json(persisted) != declaration:
                raise ValueError("stored entry declaration is not in its canonical deterministic encoding")
        except (TypeError, ValueError) as error:
            diff["declaration"] = {
                "expected": "canonical registered declaration",
                "actual": declaration,
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
        self._validate_layout_names(persisted)
        expected_reserved = {
            METADATA_COLLECTION,
            COUNTERS_COLLECTION,
            *(entry_dispatch_table_name(family.name) for family in persisted.families if len(family.records) > 1),
        }
        problems: dict[str, object] = {}
        for name in collection_names:
            if name.startswith(_RESERVED_PREFIX) and name not in expected_reserved:
                problems[name] = {
                    "reserved": True,
                    "message": "unexpected collection uses the MongoStore-reserved _httk_ prefix",
                }
        if problems:
            raise StorageLayoutUpgradeRequiredError({"schema": problems})
        self._install_layout(persisted)

    def _raise_unversioned(self, collection_names: set[str]) -> None:
        schema: dict[str, object] = {METADATA_COLLECTION: {"missing": True}}
        for name in sorted(collection_names):
            schema[name] = (
                {
                    "reserved": True,
                    "message": "unexpected collection uses the MongoStore-reserved _httk_ prefix",
                }
                if name.startswith(_RESERVED_PREFIX)
                else {
                    "unversioned": True,
                    "message": "a nonempty database without MongoStore metadata cannot be adopted",
                }
            )
        raise StorageLayoutUpgradeRequiredError(
            {
                "protocol": {"expected": DECLARATION_PROTOCOL_VERSION, "actual": None},
                "declaration": {
                    "expected": "canonical registered declaration",
                    "actual": None,
                },
                "schema": schema,
            }
        )

    @staticmethod
    def _validate_layout_names(layout: StorageLayout) -> None:
        owners: dict[str, type] = {}
        visited: set[type] = set()

        def visit(record: type) -> None:
            if record in visited:
                return
            visited.add(record)
            schema = resolve_schema(record)
            names = [collection_name_for(schema)]
            names.extend(spec.child.table_name for spec in schema.fields if spec.child is not None)
            for name in names:
                if name.startswith(_RESERVED_PREFIX):
                    raise ValueError(f"record {record.__name__} claims reserved MongoStore collection name {name!r}")
                previous = owners.get(name)
                if previous is not None and previous is not record:
                    raise ValueError(
                        f"records {previous.__name__} and {record.__name__} collide on physical collection name {name!r}"
                    )
                owners[name] = record
            for target in schema.referenced_classes():
                visit(target)

        for family in layout.families:
            for record in family.records:
                visit(record)
            dispatch_name = entry_dispatch_table_name(family.name) if len(family.records) > 1 else None
            if dispatch_name is not None:
                if dispatch_name in owners:
                    raise ValueError(
                        f"entry family {family.name!r} dispatch collection collides with a record collection"
                    )
                owners[dispatch_name] = family.family

    def _install_layout(self, layout: StorageLayout) -> None:
        self._layout = layout

    def ensure_collections(self, *classes: type) -> None:
        r"""Synchronously create or update record collections and their indexes.

        :param \*classes: Storable record classes whose collections should be
            prepared.  A configured multi-record family also prepares its
            dispatch collection.
        :return: None.
        :raises ValueError: If a requested physical name is reserved.
        """
        requested: list[tuple[str, dict[str, Any], list[Any]]] = []
        seen: set[str] = set()
        for cls in classes:
            schema = resolve_schema(cls)
            name = collection_name_for(schema)
            if name not in seen:
                requested.append((name, validator_for(schema), index_specs_for(schema)))
                seen.add(name)
            for family in self.layout.families:
                if cls not in family.records or len(family.records) < 2:
                    continue
                dispatch_name = entry_dispatch_table_name(family.name)
                if dispatch_name in seen:
                    continue
                requested.append((dispatch_name, dispatch_validator_for(family), dispatch_index_specs(family)))
                seen.add(dispatch_name)

        for name, validator, specs in requested:
            if name in self._collections_ready:
                continue
            self._ensure_collection(name, validator)
            collection = self._database.database[name]
            models = []
            for spec in specs:
                options: dict[str, Any] = {"name": spec.name, "unique": spec.unique}
                if spec.partial_filter_expression is not None:
                    options["partialFilterExpression"] = spec.partial_filter_expression
                models.append(IndexModel(list(spec.keys), **options))
            if models:
                collection.create_indexes(models)
            self._collections_ready.add(name)

    def _ensure_collection(self, name: str, validator: dict[str, Any]) -> None:
        database = self._database.database
        try:
            database.create_collection(
                name,
                validator=validator,
                validationLevel="strict",
                validationAction="error",
            )
        except CollectionInvalid:
            database.command(
                "collMod",
                name,
                validator=validator,
                validationLevel="strict",
                validationAction="error",
            )

    # ------------------------------------------------------------------ leases and transactions

    def _layout_generation(self) -> int:
        document = self._database.database[METADATA_COLLECTION].find_one({"_id": "layout"}, {"generation": 1})
        if document is None or not isinstance(document.get("generation"), int):
            raise RuntimeError("MongoStore metadata layout document is missing its generation counter")
        return int(document["generation"])

    def _observe_generation(self, generation: int) -> None:
        if generation != self._last_generation:
            self._identity._clear_identity_caches()
            self._last_generation = generation

    def _transaction_stack(self) -> list[_TransactionState]:
        stack = getattr(self._local, "transactions", None)
        if stack is None:
            stack = []
            self._local.transactions = stack
        return typing.cast(list[_TransactionState], stack)

    def _current_transaction(self) -> _TransactionState | None:
        stack = self._transaction_stack()
        return stack[-1] if stack else None

    def _session_kwargs(self) -> dict[str, Any]:
        transaction = self._current_transaction()
        if transaction is not None:
            return {"session": transaction.session}
        session = getattr(self._local, "write_session", None)
        return {} if session is None else {"session": session}

    def _write_session(self) -> Any:
        transaction = self._current_transaction()
        return getattr(self._local, "write_session", None) if transaction is None else transaction.session

    @staticmethod
    def _has_label(error: BaseException, label: str) -> bool:
        return isinstance(error, PyMongoError) and error.has_error_label(label)

    def _is_protocol_duplicate(self, error: BaseException) -> bool:
        """Return whether a duplicate belongs to a MongoStore protocol index."""
        if not isinstance(error, DuplicateKeyError):
            return False
        details = error.details or {}
        key_pattern = details.get("keyPattern")
        if key_pattern == {"content_id": 1}:
            return True
        message = str(details.get("errmsg", error))
        return any(entry_dispatch_table_name(family.name) in message for family in self.layout.families)

    def _start_transaction(self, session: Any) -> None:
        session.start_transaction(read_concern=ReadConcern("majority"), write_concern=WriteConcern("majority", j=True))

    def _commit(self, session: Any) -> None:
        while True:
            try:
                session.commit_transaction()
                return
            except BaseException as error:
                if self._has_label(error, "UnknownTransactionCommitResult"):
                    continue
                raise

    @staticmethod
    def _abort(session: Any) -> None:
        try:
            session.abort_transaction()
        except PyMongoError:
            pass

    def _publish_transaction_cache(self, transaction: _TransactionState) -> None:
        for (cls, sid), (obj, cache_instance) in transaction.pending.items():
            self._identity._remember(cls, sid, obj, cache_instance=cache_instance)
        self._failed_identities.difference_update(transaction.pending_sids)

    @contextlib.contextmanager
    def _transaction_scope(self) -> typing.Iterator[None]:
        current = self._current_transaction()
        if current is not None:
            yield
            return
        if not self._database.supports_transactions:
            raise TransactionsUnavailableError("MongoDB transactions require a replica-set deployment")
        with self._write_lock:
            lease = acquire_writer(self._database.database)
            try:
                self._observe_generation(lease.generation)
                with self._database.client.start_session(causal_consistency=True) as session:
                    transaction = _TransactionState(session, lease)
                    stack = self._transaction_stack()
                    stack.append(transaction)
                    try:
                        self._start_transaction(session)
                        yield
                        self._commit(session)
                    except BaseException:
                        self._abort(session)
                        self._identity._clear_identity_caches()
                        self._failed_identities.update(transaction.pending_sids)
                        raise
                    else:
                        self._publish_transaction_cache(transaction)
                    finally:
                        stack.pop()
            finally:
                lease.release()

    def transaction(self) -> contextlib.AbstractContextManager[None]:
        """Return a flat explicit MongoDB transaction context manager.

        :return: A context that commits on normal exit and aborts on exception.
        :raises TransactionsUnavailableError: If this store is in degraded mode.
        """
        return self._transaction_scope()

    def clear_stale_lock(self) -> None:
        """Clear a stale fsck lease after verifying its owner is dead.

        This is an administrative operation. Clearing a merely slow fsck can
        corrupt the store because the lease protocol intentionally has no
        fencing token.

        :return: None.
        :raises StoreLockedError: If the fsck lease is still fresh.
        """
        clear_stale_lock(self._database.database)

    def fsck(
        self,
        *,
        repair: bool = True,
        collect_garbage: bool = True,
        repair_conflicts: bool = False,
        force: bool = False,
        known_types: tuple[type, ...] = (),
    ) -> FsckSummary:
        """Exclusively repair dispatch integrity and collect orphan dependencies.

        Main-role records and dispatch-addressed records are roots. Only
        dependency-role documents are eligible for collection; fsck never
        creates a dispatch for a dependency-role backing.

        :param repair: Insert missing dispatches for main multi-family backings.
        :param collect_garbage: Delete unmarked dependency documents.
        :param repair_conflicts: Delete invalid dispatch documents after reporting them.
        :param force: Administrative stale-lease override for the fsck handshake.
        :param known_types: Record classes that attribute ordinary collections
            from earlier store sessions, allowing a safe sweep after reopen.
        :return: An immutable :class:`~httk.data.mongo.fsck.FsckSummary`.
        """
        from .fsck import run_fsck

        return run_fsck(
            self,
            repair=repair,
            collect_garbage=collect_garbage,
            repair_conflicts=repair_conflicts,
            force=force,
            known_types=known_types,
        )

    def _refresh_writer_lease(self) -> None:
        transaction = self._current_transaction()
        if transaction is not None and transaction.lease is not None:
            transaction.lease.refresh_heartbeat()
            return
        lease = getattr(self._local, "writer_lease", None)
        if lease is not None:
            lease.refresh_heartbeat()

    # ------------------------------------------------------------------ object storage

    def save(self, obj: Any, *, as_record: type | None = None) -> int:
        """Store an object graph and return its integer sid.

        :param obj: The object or projected domain object to store.
        :param as_record: An explicit alternate storage-record class.
        :return: The stored sid.
        :raises TypeError: If ``obj`` is a cursor proxy.
        :raises ~httk.core.storage.StorageProjectionCycleError: If the projected graph cycles.
        :raises ~httk.data.store_common.EntryMetadataConflictError: If identity-excluded metadata conflicts.
        """
        reject_cursor_proxy(obj)
        record_type = resolve_storage_record(obj, as_record=as_record)
        if self._current_transaction() is not None:
            self._ensure_graph_collections(record_type)
            self._ensure_counter_collection()
            return self._save_once(record_type, obj)
        with self._write_lock:
            lease = acquire_writer(self._database.database)
            previous_lease = getattr(self._local, "writer_lease", None)
            self._local.writer_lease = lease
            try:
                self._observe_generation(lease.generation)
                self._ensure_graph_collections(record_type)
                self._ensure_counter_collection()
                if not self._database.supports_transactions:
                    with self._database.client.start_session(causal_consistency=True) as session:
                        previous_session = getattr(self._local, "write_session", None)
                        self._local.write_session = session
                        try:
                            return self._save_once(record_type, obj)
                        finally:
                            self._local.write_session = previous_session
                return self._save_implicit_transaction(record_type, obj, lease)
            finally:
                self._local.writer_lease = previous_lease
                lease.release()

    def _save_once(self, record_type: type, obj: Any) -> int:
        projection = SaveProjection()
        self._projection_state(projection)
        try:
            sid = self._save(record_type, obj, projection, "", top_level=True)
            family = self._family_for_backing(record_type)
            if family is not None:
                self._save_entry_dispatch(family, record_type, sid, projection.content_id(record_type, obj))
            for (saved_type, identity), saved_sid in self._projection_sids(projection).items():
                source = self._projection_sources(projection)[(saved_type, identity)]
                self._remember(saved_type, saved_sid, source, cache_instance=type(source) is saved_type)
                self._failed_identities.discard((saved_type, identity))
            return sid
        except BaseException:
            self._failed_identities.update(self._projection_sources(projection))
            raise

    def _save_implicit_transaction(self, record_type: type, obj: Any, lease: WriterLease) -> int:
        last_error: BaseException | None = None
        for attempt in range(_TRANSACTION_ATTEMPTS):
            with self._database.client.start_session(causal_consistency=True) as session:
                transaction = _TransactionState(session, lease)
                stack = self._transaction_stack()
                stack.append(transaction)
                try:
                    self._start_transaction(session)
                    sid = self._save_once(record_type, obj)
                    self._commit(session)
                except BaseException as error:
                    self._abort(session)
                    last_error = error
                    # A duplicate content-id within a transaction is retried as
                    # a fresh callback so its first lookup can observe the winner.
                    if self._is_protocol_duplicate(error) or self._has_label(error, "TransientTransactionError"):
                        time.sleep(min(0.01 * (2**attempt), 0.1))
                        continue
                    raise
                else:
                    self._publish_transaction_cache(transaction)
                    return sid
                finally:
                    stack.pop()
        assert last_error is not None
        self._identity._clear_identity_caches()
        raise last_error

    @staticmethod
    def _projection_state(projection: SaveProjection) -> None:
        projection.__dict__["mongo_sids"] = {}
        projection.__dict__["mongo_sources"] = {}

    @staticmethod
    def _projection_sids(projection: SaveProjection) -> dict[tuple[type, int], int]:
        return typing.cast(dict[tuple[type, int], int], projection.__dict__["mongo_sids"])

    @staticmethod
    def _projection_sources(projection: SaveProjection) -> dict[tuple[type, int], Any]:
        return typing.cast(dict[tuple[type, int], Any], projection.__dict__["mongo_sources"])

    def _ensure_counter_collection(self) -> None:
        try:
            self._database.database.create_collection(COUNTERS_COLLECTION)
        except CollectionInvalid:
            pass

    def _ensure_graph_collections(self, record_type: type) -> None:
        pending = [record_type]
        seen: set[type] = set()
        while pending:
            current = pending.pop()
            if current in seen:
                continue
            seen.add(current)
            self._known_record_types.add(current)
            schema = resolve_schema(current)
            self.ensure_collections(current)
            pending.extend(spec.target for spec in schema.fields if spec.target is not None)

    def _save(self, record_type: type, source: Any, projection: SaveProjection, path: str, *, top_level: bool) -> int:
        self._refresh_writer_lease()
        key = (record_type, id(source))
        if key in projection.active:
            raise StorageProjectionCycleError(path, record_type)
        existing = self._projection_sids(projection).get(key)
        if existing is not None:
            return existing
        projection.active.add(key)
        self._projection_sources(projection)[key] = source
        try:
            return self._save_active(record_type, source, projection, path, top_level=top_level)
        finally:
            projection.active.remove(key)

    def _save_active(
        self,
        record_type: type,
        source: Any,
        projection: SaveProjection,
        path: str,
        *,
        top_level: bool,
    ) -> int:
        schema = resolve_schema(record_type)
        projected = projection.projector(record_type, source)
        validation_key = (record_type, id(source))
        if type(source) is record_type and validation_key not in projection.validated:
            validator = vars(record_type).get("__httk_validate__")
            if validator is not None:
                validator.__get__(None, record_type)(source)
            projection.validated.add(validation_key)

        content_key: str | None = None
        collection = self._database.database[collection_name_for(schema)]
        if schema.dedup == "content_id":
            content_key = projection.content_id(record_type, source)
            found = collection.find_one({"content_id": content_key}, **self._session_kwargs())
            if found is not None:
                sid = int(found["_id"])
                self._check_metadata(record_type, sid, source, projection)
                self._projection_sids(projection)[validation_key] = sid
                if top_level and found.get("_httk_role") == "dep":
                    collection.update_one({"_id": sid}, {"$set": {"_httk_role": "main"}}, **self._session_kwargs())
                return sid

        checkpoint = len(projection.inserted)
        f_document = encode_record(
            schema,
            projected,
            source,
            record_type,
            lambda target, value, field: self._save(
                target, value, projection, self._field_path(path, field), top_level=False
            ),
        )
        if schema.dedup == "by_value":
            query = self._by_value_query(schema, f_document)
            found = collection.find_one(query, {"_id": 1, "_httk_role": 1}, **self._session_kwargs())
            if found is not None:
                sid = int(found["_id"])
                self._discard_inserts(projection, checkpoint)
                self._projection_sids(projection)[validation_key] = sid
                if top_level and found.get("_httk_role") == "dep":
                    collection.update_one({"_id": sid}, {"$set": {"_httk_role": "main"}}, **self._session_kwargs())
                return sid

        sid = counter_next(self._database.database, schema.table_name, session=self._write_session())
        document: dict[str, Any] = {"_id": sid, "_httk_role": "main" if top_level else "dep", "f": f_document}
        if content_key is not None:
            document["content_id"] = content_key
        preflight_document(document, self._max_bson_size, record_type)
        try:
            collection.insert_one(document, **self._session_kwargs())
        except DuplicateKeyError as error:
            if not self._is_protocol_duplicate(error) or content_key is None:
                raise
            winner = collection.find_one({"content_id": content_key}, **self._session_kwargs())
            if winner is None:
                raise
            sid = int(winner["_id"])
            self._discard_inserts(projection, checkpoint)
            self._check_metadata(record_type, sid, source, projection)
            self._projection_sids(projection)[validation_key] = sid
            if top_level and winner.get("_httk_role") == "dep":
                collection.update_one({"_id": sid}, {"$set": {"_httk_role": "main"}}, **self._session_kwargs())
            return sid
        projection.inserted.append((record_type, sid))
        self._projection_sids(projection)[validation_key] = sid
        return sid

    @staticmethod
    def _field_path(path: str, field: str) -> str:
        return f"{path}.{field}" if path else field

    @staticmethod
    def _by_value_query(schema: Any, f_document: Mapping[str, Any]) -> dict[str, Any]:
        query: dict[str, Any] = {}
        child_fields = {spec.field: spec for spec in schema.fields if spec.role == "child"}
        field_plans = {plan.field: plan for plan in document_fields_for(schema)}
        for key, value in f_document.items():
            if key not in child_fields:
                query[f"f.{key}"] = value
        for spec in schema.fields:
            if spec.role == "child" or not spec.optional:
                continue
            plan = field_plans[spec.field]
            for key in plan.keys:
                if key not in f_document:
                    query[f"f.{key}"] = None
        for spec in child_fields.values():
            if spec.optional and spec.field not in f_document:
                query[f"f.{spec.field}"] = {"$exists": False}
            elif spec.field in f_document:
                query[f"f.{spec.field}"] = {"$type": "array"}
        return query

    def _discard_inserts(self, projection: SaveProjection, checkpoint: int) -> None:
        if checkpoint == len(projection.inserted):
            return
        if self._current_transaction() is None:
            return
        sids = self._projection_sids(projection)
        for record_type, sid in reversed(projection.inserted[checkpoint:]):
            self._database.database[collection_name_for(resolve_schema(record_type))].delete_one(
                {"_id": sid}, **self._session_kwargs()
            )
            for key, value in tuple(sids.items()):
                if value == sid and key[0] is record_type:
                    del sids[key]
        del projection.inserted[checkpoint:]

    def _family_for_backing(self, record_type: type) -> Any:
        return next((family for family in self.layout.families if record_type in family.records), None)

    def _save_entry_dispatch(self, family: Any, backing: type, sid: int, key: str) -> None:
        if len(family.records) == 1:
            return
        collection = self._database.database[entry_dispatch_table_name(family.name)]
        record_name = family.record_names[family.records.index(backing)]
        existing = collection.find_one({"_id": key}, **self._session_kwargs())
        if existing is not None:
            if existing.get("record") == record_name and int(existing.get("sid", -1)) == sid:
                return
            raise EntryDispatchIntegrityError(
                f"entry dispatch {family.name!r} maps content_id {key!r} to a conflicting backing row"
            )
        try:
            collection.insert_one({"_id": key, "record": record_name, "sid": sid}, **self._session_kwargs())
            return
        except DuplicateKeyError:
            existing = collection.find_one({"_id": key}, **self._session_kwargs())
            if existing is not None:
                if existing.get("record") == record_name and int(existing.get("sid", -1)) == sid:
                    return
                raise EntryDispatchIntegrityError(
                    f"entry dispatch {family.name!r} maps content_id {key!r} to a conflicting backing row"
                ) from None
            owner = collection.find_one({"record": record_name, "sid": sid}, **self._session_kwargs())
            if owner is not None:
                raise EntryDispatchIntegrityError(
                    f"entry dispatch {family.name!r} already maps backing sid {sid} to content_id {owner['_id']!r}, "
                    f"not {key!r}"
                ) from None
            raise EntryDispatchIntegrityError(
                f"entry dispatch {family.name!r} declined content_id {key!r} without a discoverable conflicting row"
            ) from None

    # ------------------------------------------------------------------ reads

    def fetch[T](self, cls: type[T], sid: int) -> T:
        """Fetch and eagerly hydrate ``cls`` at ``sid``.

        :param cls: The storable record class.
        :param sid: The integer sid.
        :return: The hydrated record.
        :raises KeyError: If the record does not exist.
        """
        return typing.cast(T, self._fetch(cls, int(sid), _HydrationContext()))

    def _fetch(self, cls: type, sid: int, context: _HydrationContext | None = None) -> Any:
        if context is None:
            context = _HydrationContext()
        key = (cls, int(sid))
        transaction = self._current_transaction()
        pending = None if transaction is None else transaction.pending.get(key)
        cached = None if pending is None else pending[0]
        if cached is None:
            cached = self._identity._instances.get(key)
        if cached is not None:
            return cached
        schema = resolve_schema(cls)
        document = context.documents.get(key)
        if document is None:
            document = self._database.database[collection_name_for(schema)].find_one(
                {"_id": int(sid)}, **self._session_kwargs()
            )
        if document is None:
            raise KeyError(cls, int(sid))
        self._prefetch_references(schema, document, context)
        instance = decode_record(schema, document, lambda target, target_sid: self._fetch(target, target_sid, context))
        self._remember(cls, int(sid), instance)
        return instance

    def _prefetch_references(self, schema: Any, document: Mapping[str, Any], context: _HydrationContext) -> None:
        targets: dict[type, set[int]] = {}
        embedded = document.get("f", {})
        for spec in schema.fields:
            if spec.target is None:
                continue
            if spec.role == "reference":
                sid = embedded.get(spec.columns[0].name)
                if sid is not None:
                    targets.setdefault(spec.target, set()).add(int(sid))
            elif spec.role == "child" and spec.child is not None:
                for element in embedded.get(spec.field, ()):
                    sid = element.get(spec.child.element_columns[0].name)
                    if sid is not None:
                        targets.setdefault(spec.target, set()).add(int(sid))
        cache = context.documents
        transaction = self._current_transaction()
        for target, sids in targets.items():
            missing = [
                sid
                for sid in sids
                if (target, sid) not in cache
                and (transaction is None or (target, sid) not in transaction.pending)
                and self._identity._instances.get((target, sid)) is None
            ]
            if not missing:
                continue
            target_schema = resolve_schema(target)
            for item in self._database.database[collection_name_for(target_schema)].find(
                {"_id": {"$in": missing}}, **self._session_kwargs()
            ):
                cache[(target, int(item["_id"]))] = item

    def fetch_by_content_id[T](self, cls: type[T], key: str) -> T | None:
        """Fetch a content-addressed record, or return ``None``.

        :param cls: The storable record class.
        :param key: The content identity.
        :return: The hydrated record or ``None``.
        :raises ~httk.data.db.schema.SchemaError: If ``cls`` is not content-id deduplicated.
        """
        schema = resolve_schema(cls)
        if schema.dedup != "content_id":
            raise SchemaError(
                f"{cls.__name__} has dedup policy {schema.dedup!r}; only classes with the "
                f"'content_id' policy have a content identity column"
            )
        document = self._database.database[collection_name_for(schema)].find_one(
            {"content_id": key}, {"_id": 1}, **self._session_kwargs()
        )
        return None if document is None else self.fetch(cls, int(document["_id"]))

    def fetch_entry(self, family_cls: type, content_id: str) -> object | None:
        """Fetch the concrete backing record for an entry-family identity.

        :param family_cls: The configured entry-family class.
        :param content_id: The entry content identity.
        :return: The backing record or ``None``.
        :raises ValueError: If the family is not configured.
        :raises ~httk.data.store_common.EntryDispatchIntegrityError: If dispatch and backing disagree.
        """
        family = next((item for item in self.layout.families if item.family is family_cls), None)
        if family is None:
            raise ValueError(f"{family_cls.__name__} is not a configured entry family in this MongoStore")
        if len(family.records) == 1:
            return self.fetch_by_content_id(family.records[0], content_id)
        dispatch = self._database.database[entry_dispatch_table_name(family.name)]
        row = dispatch.find_one({"_id": content_id}, **self._session_kwargs())
        if row is None:
            for backing in family.records:
                found = self._database.database[collection_name_for(resolve_schema(backing))].find_one(
                    {"content_id": content_id}, {"_id": 1}, **self._session_kwargs()
                )
                if found is not None:
                    raise EntryDispatchIntegrityError(
                        f"entry dispatch {family.name!r} is missing for stored content_id {content_id!r}"
                    )
            return None
        record_name = row.get("record")
        try:
            index = family.record_names.index(record_name)
        except ValueError:
            raise EntryDispatchIntegrityError(
                f"entry dispatch {family.name!r} names an unknown backing {record_name!r}"
            ) from None
        backing = family.records[index]
        sid = row.get("sid")
        if not isinstance(sid, int):
            raise EntryDispatchIntegrityError(
                f"entry dispatch {family.name!r} has an invalid sid for content_id {content_id!r}"
            )
        backing_document = self._database.database[collection_name_for(resolve_schema(backing))].find_one(
            {"_id": sid}, {"content_id": 1}, **self._session_kwargs()
        )
        backing_key = None if backing_document is None else backing_document.get("content_id")
        if backing_key != content_id:
            raise EntryDispatchIntegrityError(
                f"entry dispatch {family.name!r} maps content_id {content_id!r} to backing sid {sid} "
                f"whose content_id is {backing_key!r}"
            )
        return self.fetch(backing, sid)

    def sid_of(self, obj: Any, *, as_record: type | None = None) -> int | None:
        """Return the sid known for ``obj``, using content lookup when allowed.

        :param obj: The object whose sid is requested.
        :param as_record: An explicit alternate record class.
        :return: The sid, or ``None``.
        """
        record_type = resolve_storage_record(obj, as_record=as_record)
        if (record_type, id(obj)) in self._failed_identities:
            return None
        transaction = self._current_transaction()
        if transaction is not None:
            pending = transaction.pending_sids.get((record_type, id(obj)))
            if pending is not None:
                return pending
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
            return None
        projection = SaveProjection()
        key = projection.content_id(record_type, obj)
        document = self._database.database[collection_name_for(schema)].find_one(
            {"content_id": key}, {"_id": 1}, **self._session_kwargs()
        )
        if document is None:
            return None
        sid = int(document["_id"])
        self._remember(record_type, sid, obj, cache_instance=type(obj) is record_type)
        return sid

    def searcher(self) -> Any:
        """Return a Mongo searcher bound to this store's read path.

        Queries use the active transaction session when one is open, so they
        see that transaction's uncommitted writes, and object outputs hydrate
        through :meth:`fetch`, preserving the identity-cache contract.

        :return: A new MongoDB searcher bound to this store.
        """
        from .searcher import MongoSearcher

        return MongoSearcher(self)

    def stored_property_plan(self, family: type) -> Any:
        """Return the Mongo stored-property plan for one configured entry family.

        :param family: The logical entry-family class.
        :return: Its validated Mongo stored-property plan.
        """
        from .stored_properties import stored_property_mongo_plan

        return stored_property_mongo_plan(self, family)

    def referring(self, cls: type, *, field: str, to: Any) -> list[Any]:
        """Return records whose reference field points at ``to``, ordered by sid.

        :param cls: The referring record class.
        :param field: The reference field.
        :param to: The stored target instance.
        :return: Matching records ordered by sid.
        :raises ~httk.data.db.schema.SchemaError: If the field or target class is incompatible.
        :raises ValueError: If ``to`` is not stored or fetched here.
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
        path = f"f.{spec.columns[0].name}"
        collection = self._database.database[collection_name_for(schema)]
        return [
            self.fetch(cls, int(document["_id"]))
            for document in collection.find({path: sid}, {"_id": 1}, **self._session_kwargs()).sort("_id", 1)
        ]

    def _remember(self, cls: type, sid: int, obj: Any, *, cache_instance: bool = True) -> None:
        transaction = self._current_transaction()
        if transaction is None:
            self._identity._remember(cls, sid, obj, cache_instance=cache_instance)
            return
        transaction.pending[(cls, sid)] = (obj, cache_instance)
        transaction.pending_sids[(cls, id(obj))] = sid

    # ------------------------------------------------------------------ metadata comparison

    def _check_metadata(self, record_type: type, sid: int, source: Any, projection: SaveProjection) -> None:
        plan = _metadata_plan(record_type)
        if plan is not None:
            self._check_metadata_at(record_type, sid, source, projection, record_type.__name__, plan)

    def _metadata_parent_document(
        self, record_type: type, sid: int, plan: Any, projection: SaveProjection
    ) -> Mapping[str, Any]:
        key = (record_type, int(sid))
        cached = projection.metadata_rows.get(key)
        if cached is not None:
            return cached
        paths: dict[str, int] = {}
        for spec in (*plan.skipped_specs, *plan.skipped_nested, *plan.descend_specs):
            if spec.role == "child":
                paths[f"f.{spec.field}"] = 1
            else:
                for column in spec.columns:
                    paths[f"f.{column.name}"] = 1
        collection = self._database.database[collection_name_for(resolve_schema(record_type))]
        document = collection.find_one({"_id": int(sid)}, paths, **self._session_kwargs())
        if document is None:
            raise KeyError(record_type, int(sid))
        projection.metadata_rows[key] = document
        return document

    @staticmethod
    def _metadata_scalar_equal(left: Any, right: Any) -> bool:
        if isinstance(left, list | tuple) or isinstance(right, list | tuple):
            return (
                type(left) is type(right)
                and len(left) == len(right)
                and all(MongoStore._metadata_scalar_equal(a, b) for a, b in zip(left, right, strict=True))
            )
        if isinstance(left, datetime.datetime) and isinstance(right, datetime.datetime):
            if (left.utcoffset() is None) != (right.utcoffset() is None):
                return False
            if left.utcoffset() is not None:
                return left.astimezone(datetime.UTC) == right.astimezone(datetime.UTC)
        return bool(left == right)

    def _metadata_value(self, spec: Any, document: Mapping[str, Any]) -> Any:
        embedded = document.get("f", {})
        if spec.role == "child":
            if spec.optional and spec.field not in embedded:
                return None
            entries = embedded.get(spec.field, [])
            assert spec.child is not None
            if spec.shape is not None:
                return FracVector(
                    [
                        decode_fracvector_exact(item[f"{spec.field}_exact"], 1, spec.shape.cols).to_fractions()[0]
                        for item in entries
                    ]
                )
            if spec.target is not None:
                return [int(item[spec.child.element_columns[0].name]) for item in entries]
            if spec.codec_name is not None:
                codec = codec_named(spec.codec_name)
                values = [
                    codec.decode(tuple(item[column.name] for column in spec.child.element_columns)) for item in entries
                ]
            else:
                values = [item[spec.child.element_columns[0].name] for item in entries]
            return tuple(values) if typing.get_origin(spec.python_type) is tuple else values
        embedded = document.get("f", {})
        if spec.role == "scalar":
            return embedded.get(spec.columns[0].name)
        if spec.role == "encoded":
            parts = tuple(embedded.get(column.name) for column in spec.columns)
            return None if all(part is None for part in parts) else codec_named(spec.codec_name).decode(parts)
        if spec.role == "fixed_array":
            exact = embedded.get(f"{spec.field}_exact")
            return None if exact is None else decode_fracvector_exact(exact, spec.shape.rows, spec.shape.cols)
        return embedded.get(spec.columns[0].name)

    def _check_metadata_at(
        self, record_type: type, sid: int, source: Any, projection: SaveProjection, path: str, plan: Any
    ) -> None:
        schema = resolve_schema(record_type)
        document = self._metadata_parent_document(record_type, sid, plan, projection)
        values = projection.projector(record_type, source)
        skipped = {spec.field for spec in plan.skipped_specs}
        nested = {spec.field for spec in plan.skipped_nested}
        descend = {spec.field for spec in plan.descend_specs}
        for spec in schema.fields:
            if spec.derived:
                continue
            field_path = self._field_path(path, spec.field)
            if spec.field in skipped:
                incoming = values[spec.field]
                stored = self._metadata_value(spec, document)
                if not self._metadata_scalar_equal(incoming, stored):
                    raise EntryMetadataConflictError(
                        f"metadata conflict for {field_path}: stored {stored!r}, received {incoming!r}"
                    )
            elif spec.field in nested:
                self._check_metadata_nested(
                    schema, document, sid, spec, values[spec.field], projection, field_path, True
                )
            elif spec.field in descend:
                self._check_metadata_nested(
                    schema, document, sid, spec, values[spec.field], projection, field_path, False
                )

    def _check_metadata_nested(
        self,
        schema: Any,
        document: Mapping[str, Any],
        sid: int,
        spec: Any,
        incoming: Any,
        projection: SaveProjection,
        path: str,
        compare_content: bool,
    ) -> None:
        stored = self._metadata_value(spec, document)
        if spec.role == "reference":
            if incoming is None or stored is None:
                if incoming is not None or stored is not None:
                    if compare_content:
                        existing = None if stored is None else self._fetch(spec.target, int(stored))
                        raise EntryMetadataConflictError(
                            f"metadata conflict for {path}: stored {existing!r}, received {incoming!r}"
                        )
                    raise EntryMetadataConflictError(f"metadata conflict for {path}")
                return
            self._check_metadata_target(spec.target, int(stored), incoming, projection, path, compare_content)
            return
        if spec.target is None:
            if not self._metadata_scalar_equal(incoming, stored):
                raise EntryMetadataConflictError(
                    f"metadata conflict for {path}: stored {stored!r}, received {incoming!r}"
                )
            return
        if incoming is None or stored is None:
            if incoming is not stored:
                if compare_content:
                    existing = None if stored is None else [self._fetch(spec.target, int(item)) for item in stored]
                    raise EntryMetadataConflictError(
                        f"metadata conflict for {path}: stored {existing!r}, received {incoming!r}"
                    )
                raise EntryMetadataConflictError(f"metadata conflict for {path}")
            return
        if len(incoming) != len(stored):
            if compare_content:
                existing = [self._fetch(spec.target, int(item)) for item in stored]
                raise EntryMetadataConflictError(
                    f"metadata conflict for {path}: stored {existing!r}, received {incoming!r}"
                )
            raise EntryMetadataConflictError(f"metadata conflict for {path}")
        for index, (incoming_item, stored_sid) in enumerate(zip(incoming, stored, strict=True)):
            self._check_metadata_target(
                spec.target, int(stored_sid), incoming_item, projection, f"{path}[{index}]", compare_content
            )

    def _check_metadata_target(
        self, record_type: type, sid: int, source: Any, projection: SaveProjection, path: str, compare_content: bool
    ) -> None:
        if compare_content:
            collection = self._database.database[collection_name_for(resolve_schema(record_type))]
            stored = collection.find_one({"_id": sid}, {"content_id": 1}, **self._session_kwargs())
            if resolve_schema(record_type).dedup == "content_id":
                stored_key = None if stored is None else stored.get("content_id")
            else:
                stored_key = projection.content_id(record_type, self._fetch(record_type, sid))
            incoming_key = projection.content_id(record_type, source)
            if incoming_key != stored_key:
                existing = self._fetch(record_type, sid)
                raise EntryMetadataConflictError(
                    f"metadata conflict for {path}: stored {existing!r}, received {source!r}"
                )
        plan = _metadata_plan(record_type)
        if plan is not None:
            self._check_metadata_at(record_type, sid, source, projection, path, plan)
