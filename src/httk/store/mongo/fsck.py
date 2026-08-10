"""Integrity repair and mark/sweep collection for :mod:`httk.store.mongo`."""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from pymongo import ReturnDocument
from pymongo.errors import DuplicateKeyError

from httk.store.db.schema import TableSchema, resolve_schema

from .leases import acquire_fsck
from .mapping import COUNTERS_COLLECTION, METADATA_COLLECTION, collection_name_for, entry_dispatch_table_name

if TYPE_CHECKING:
    from .store import MongoStore

__all__ = ["FsckCollectionSummary", "FsckSummary", "run_fsck"]

_LOGGER = logging.getLogger("httk.store.mongo")
_BATCH_SIZE = 500


@dataclass(frozen=True)
class FsckCollectionSummary:
    """Counters collected for one record or dispatch collection.

    :param examined: Documents inspected in the collection.
    :param repaired: Documents inserted or removed by integrity repair.
    :param conflicts: Integrity violations found in the collection.
    :param deleted: Documents removed, including conflict removals and swept dependencies.
    """

    examined: int = 0
    repaired: int = 0
    conflicts: int = 0
    deleted: int = 0


@dataclass(frozen=True)
class FsckSummary:
    """The immutable result of one :meth:`~httk.store.mongo.store.MongoStore.fsck` run.

    :param generation: Metadata generation after this run's single increment.
    :param collections: Per-collection repair and collection counters.
    :param violations: Human-readable violations reported during the run.
    """

    generation: int
    collections: Mapping[str, FsckCollectionSummary]
    violations: tuple[str, ...]


class _Counters:
    """Mutable implementation counters converted to public summaries at return."""

    __slots__ = ("conflicts", "deleted", "examined", "repaired")

    def __init__(self) -> None:
        self.examined = 0
        self.repaired = 0
        self.conflicts = 0
        self.deleted = 0

    def freeze(self) -> FsckCollectionSummary:
        """Return the corresponding immutable public counter value."""
        return FsckCollectionSummary(self.examined, self.repaired, self.conflicts, self.deleted)


def _schemas(store: MongoStore, known_types: tuple[type, ...]) -> dict[str, TableSchema]:
    """Return schemas known through the layout, this session, or the caller."""
    result: dict[str, TableSchema] = {}
    pending = [*store._known_record_types, *known_types]
    seen: set[type] = set()
    while pending:
        record = pending.pop()
        if record in seen:
            continue
        seen.add(record)
        schema = resolve_schema(record)
        result[collection_name_for(schema)] = schema
        pending.extend(schema.referenced_classes())
    return result


def _record_violation(violations: list[str], counters: dict[str, _Counters], collection: str, message: str) -> None:
    """Store and report one fsck violation."""
    counters[collection].conflicts += 1
    violations.append(message)
    _LOGGER.warning("MongoStore fsck: %s", message, extra={"context": "storage"})


def _record_repair(counters: dict[str, _Counters], collection: str, message: str) -> None:
    """Store and report one fsck repair."""
    counters[collection].repaired += 1
    _LOGGER.warning("MongoStore fsck repaired: %s", message, extra={"context": "storage"})


def _valid_sid(value: Any) -> bool:
    """Return whether ``value`` is an integer Mongo sid (but not a boolean)."""
    return isinstance(value, int) and not isinstance(value, bool)


def _integrity_pass(
    store: MongoStore,
    schemas: Mapping[str, TableSchema],
    counters: dict[str, _Counters],
    violations: list[str],
    *,
    repair_conflicts: bool,
    lease: Any,
) -> None:
    """Verify multi-record dispatches and repair only missing main-role dispatches."""
    database = store._database.database
    for family in store.layout.families:
        if len(family.records) < 2:
            continue
        lease.refresh_heartbeat()
        dispatch_name = entry_dispatch_table_name(family.name)
        dispatch = database[dispatch_name]
        record_names = dict(zip(family.record_names, family.records, strict=True))
        for row in dispatch.find({}, {"_id": 1, "record": 1, "sid": 1}):
            counters[dispatch_name].examined += 1
            content_id = row.get("_id")
            record_name = row.get("record")
            sid = row.get("sid")
            backing = record_names.get(record_name)
            document: Mapping[str, Any] | None = None
            if backing is not None and _valid_sid(sid):
                document = database[collection_name_for(resolve_schema(backing))].find_one(
                    {"_id": sid}, {"content_id": 1}
                )
            if isinstance(content_id, str) and document is not None and document.get("content_id") == content_id:
                continue
            message = (
                f"dispatch {dispatch_name!r} content_id {content_id!r} does not name a matching registered backing "
                f"(record={record_name!r}, sid={sid!r})"
            )
            _record_violation(violations, counters, dispatch_name, message)
            if repair_conflicts:
                result = dispatch.delete_one({"_id": content_id})
                if result.deleted_count:
                    counters[dispatch_name].deleted += 1
                    _record_repair(counters, dispatch_name, f"removed conflicting {message}")

        # A dependency backing is deliberately not considered here.  It may be
        # a family record reached from another record and has no entry identity.
        for backing, record_name in zip(family.records, family.record_names, strict=True):
            collection_name = collection_name_for(resolve_schema(backing))
            collection = database[collection_name]
            for document in collection.find(
                {"_httk_role": "main", "content_id": {"$exists": True}}, {"_id": 1, "content_id": 1}
            ):
                counters[collection_name].examined += 1
                assert document is not None
                sid = document["_id"]
                content_id = document.get("content_id")
                if not isinstance(content_id, str):
                    _record_violation(
                        violations,
                        counters,
                        collection_name,
                        f"main backing {collection_name!r}/{sid!r} has a non-string content_id {content_id!r}",
                    )
                    continue
                existing = dispatch.find_one({"_id": content_id}, {"_id": 1})
                if existing is not None:
                    continue
                try:
                    dispatch.insert_one({"_id": content_id, "record": record_name, "sid": sid})
                except DuplicateKeyError:
                    # Another raw writer can race only outside the advisory
                    # lease protocol; retain it as an integrity conflict.
                    _record_violation(
                        violations,
                        counters,
                        dispatch_name,
                        f"dispatch {dispatch_name!r} changed while repairing content_id {content_id!r}",
                    )
                else:
                    _record_repair(
                        counters,
                        dispatch_name,
                        f"inserted missing dispatch for {collection_name!r}/{sid!r}",
                    )


def _mark(
    store: MongoStore,
    schemas: Mapping[str, TableSchema],
    counters: dict[str, _Counters],
    violations: list[str],
    *,
    lease: Any,
) -> dict[str, set[int]]:
    """Mark schema-known documents reachable from main roles and dispatches."""
    database = store._database.database
    marked: dict[str, set[int]] = defaultdict(set)
    pending: dict[str, deque[int]] = defaultdict(deque)

    def enqueue(collection: str, sid: Any) -> None:
        if collection in schemas and _valid_sid(sid) and sid not in marked[collection]:
            marked[collection].add(sid)
            pending[collection].append(sid)

    for collection_name in schemas:
        lease.refresh_heartbeat()
        collection = database[collection_name]
        for document in collection.find({}, {"_id": 1, "_httk_role": 1}):
            counters[collection_name].examined += 1
            role = document.get("_httk_role")
            if role == "main":
                enqueue(collection_name, document.get("_id"))
            elif role != "dep":
                _record_violation(
                    violations,
                    counters,
                    collection_name,
                    f"record {collection_name!r}/{document.get('_id')!r} lacks a valid _httk_role marker",
                )

    for family in store.layout.families:
        if len(family.records) < 2:
            continue
        dispatch_name = entry_dispatch_table_name(family.name)
        targets = dict(zip(family.record_names, family.records, strict=True))
        for row in database[dispatch_name].find({}, {"record": 1, "sid": 1}):
            backing = targets.get(row.get("record"))
            if backing is not None:
                enqueue(collection_name_for(resolve_schema(backing)), row.get("sid"))

    while any(pending.values()):
        for collection_name, queue in tuple(pending.items()):
            if not queue:
                continue
            lease.refresh_heartbeat()
            batch = [queue.popleft() for _ in range(min(_BATCH_SIZE, len(queue)))]
            schema = schemas[collection_name]
            projection: dict[str, int] = {"_id": 1, "_httk_role": 1, "f": 1}
            for document in database[collection_name].find({"_id": {"$in": batch}}, projection):
                if document.get("_httk_role") not in {"main", "dep"}:
                    continue
                fields = document.get("f")
                if not isinstance(fields, Mapping):
                    _record_violation(
                        violations,
                        counters,
                        collection_name,
                        f"record {collection_name!r}/{document.get('_id')!r} has a non-document 'f' field",
                    )
                    continue
                for spec in schema.fields:
                    if spec.target is None:
                        continue
                    target_collection = collection_name_for(resolve_schema(spec.target))
                    if spec.role == "child":
                        child = fields.get(spec.field)
                        if not isinstance(child, list) or spec.child is None:
                            continue
                        key = spec.child.element_columns[0].name
                        for element in child:
                            if isinstance(element, Mapping):
                                enqueue(target_collection, element.get(key))
                    else:
                        # A reference role has exactly one physical sid key.
                        key = spec.columns[0].name
                        enqueue(target_collection, fields.get(key))
    return marked


def _report_unattributed_collections(
    store: MongoStore, schemas: Mapping[str, TableSchema], counters: dict[str, _Counters], violations: list[str]
) -> tuple[str, ...]:
    """Report unknown collections and return non-reserved sweep blockers."""
    expected_dispatch = {
        entry_dispatch_table_name(family.name) for family in store.layout.families if len(family.records) > 1
    }
    reserved = {METADATA_COLLECTION, COUNTERS_COLLECTION, *expected_dispatch}
    unattributed: list[str] = []
    for name in store._database.database.list_collection_names():
        if name.startswith("system.") or name in schemas or name in reserved:
            continue
        if name.startswith("_httk_"):
            message = f"unrecognized reserved collection {name!r} was left untouched"
        else:
            unattributed.append(name)
            message = f"collection {name!r} cannot be attributed to a known schema and blocks fsck sweep"
        _record_violation(violations, counters, name, message)
    return tuple(sorted(unattributed))


def _sweep(
    store: MongoStore,
    schemas: Mapping[str, TableSchema],
    marked: Mapping[str, set[int]],
    counters: dict[str, _Counters],
    *,
    lease: Any,
) -> None:
    """Delete unmarked dependency-role documents from positively known collections."""
    database = store._database.database
    for collection_name in schemas:
        lease.refresh_heartbeat()
        collection = database[collection_name]
        survivors = marked.get(collection_name, set())
        query: dict[str, Any] = {"_httk_role": "dep"}
        if survivors:
            query["_id"] = {"$nin": list(survivors)}
        result = collection.delete_many(query)
        counters[collection_name].deleted += result.deleted_count


def run_fsck(
    store: MongoStore,
    *,
    repair: bool = True,
    collect_garbage: bool = True,
    repair_conflicts: bool = False,
    force: bool = False,
    known_types: tuple[type, ...] = (),
) -> FsckSummary:
    """Run the exclusive MongoStore integrity repair and garbage collector.

    :param store: The owning MongoStore instance.
    :param repair: Whether missing main-role dispatch entries are repaired.
    :param collect_garbage: Whether unmarked dependency documents are swept.
    :param repair_conflicts: Whether invalid dispatch documents are deleted.
    :param force: Administrative override passed to the fsck lease protocol.
    :param known_types: Record classes needed to attribute ordinary collections
        after reopening a store.
    :return: Immutable per-collection counters and all reported violations.
    """
    counters: dict[str, _Counters] = defaultdict(_Counters)
    violations: list[str] = []
    with store._write_lock:
        lease = acquire_fsck(store._database.database, force=force)
        try:
            layout = store._database.database[METADATA_COLLECTION].find_one_and_update(
                {"_id": "layout"},
                {"$inc": {"generation": 1}},
                return_document=ReturnDocument.AFTER,
            )
            if layout is None or not isinstance(layout.get("generation"), int):
                raise RuntimeError("MongoStore metadata layout document is missing its generation counter")
            generation = int(layout["generation"])
            schemas = _schemas(store, known_types)
            for collection_name in schemas:
                counters[collection_name]
            for family in store.layout.families:
                if len(family.records) > 1:
                    counters[entry_dispatch_table_name(family.name)]
            unattributed = _report_unattributed_collections(store, schemas, counters, violations)
            if repair:
                _integrity_pass(store, schemas, counters, violations, repair_conflicts=repair_conflicts, lease=lease)
            marked = _mark(store, schemas, counters, violations, lease=lease)
            if collect_garbage and unattributed:
                message = (
                    "sweep aborted because these collections cannot be attributed to a schema: "
                    f"{', '.join(repr(name) for name in unattributed)}; rerun fsck(known_types=(...)) "
                    "with their record classes"
                )
                violations.append(message)
                _LOGGER.warning("MongoStore fsck: %s", message, extra={"context": "storage"})
            elif collect_garbage:
                _sweep(store, schemas, marked, counters, lease=lease)
            store._identity._clear_identity_caches()
            store._last_generation = generation
            return FsckSummary(
                generation=generation,
                collections=MappingProxyType({name: value.freeze() for name, value in sorted(counters.items())}),
                violations=tuple(violations),
            )
        finally:
            lease.release()
