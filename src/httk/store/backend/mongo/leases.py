"""Cross-process writer and fsck leases for :mod:`httk.store.backend.mongo`.

The lease protocol deliberately uses MongoDB's server clock.  A writer first
inserts itself and only then looks for the fsck singleton; fsck does the
opposite half of that handshake by installing the singleton before draining
fresh writers.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any

from pymongo.errors import DuplicateKeyError

from .mapping import METADATA_COLLECTION

__all__ = [
    "FsckLease",
    "Lease",
    "LeaseLostError",
    "LeaseTiming",
    "StoreLockedError",
    "WriterLease",
    "acquire_fsck",
    "acquire_writer",
    "clear_stale_lock",
]


class StoreLockedError(RuntimeError):
    """An fsck lease or a live writer prevents the requested operation."""


class LeaseLostError(StoreLockedError):
    """A lease owner tried to refresh after its lease was displaced."""


@dataclass(frozen=True)
class LeaseTiming:
    """Server-clock lease timings.

    :param refresh_interval: Maximum normal interval between heartbeats.
    :param stale_multiplier: Number of refresh intervals before a writer is stale.
    :param poll_initial: Initial fsck writer-drain sleep.
    :param poll_max: Maximum fsck writer-drain sleep.
    """

    refresh_interval: float = 2.0
    stale_multiplier: int = 4
    poll_initial: float = 0.02
    poll_max: float = 0.25

    @property
    def stale_after(self) -> float:
        """Return the age after which a writer heartbeat is stale."""
        return self.refresh_interval * self.stale_multiplier


DEFAULT_TIMING = LeaseTiming()


@dataclass
class Lease:
    """Shared lease state and heartbeat operations for writer and fsck leases."""

    database: Any
    identifier: str
    owner: str
    timing: LeaseTiming
    _last_refresh: float = 0.0

    def refresh_heartbeat(self, *, force: bool = False) -> None:
        """Refresh this lease using MongoDB's ``$$NOW`` server timestamp."""
        if not force and time.monotonic() - self._last_refresh < self.timing.refresh_interval / 2:
            return
        result = self.database[METADATA_COLLECTION].update_one(
            {"_id": self.identifier, "owner": self.owner}, [{"$set": {"heartbeat": "$$NOW"}}]
        )
        if result.matched_count == 0:
            raise LeaseLostError(f"lease {self.identifier!r} was lost by its owner")
        self._last_refresh = time.monotonic()

    def release(self) -> None:
        """Remove this process's lease document."""
        self.database[METADATA_COLLECTION].delete_one({"_id": self.identifier, "owner": self.owner})


@dataclass
class WriterLease(Lease):
    """A writer registration plus the generation seen during acquisition."""

    generation: int = 0


@dataclass
class FsckLease(Lease):
    """The singleton fsck exclusion lease."""


def _owner() -> str:
    return str(uuid.uuid4())


def _fresh_writer_documents(database: Any, timing: LeaseTiming) -> list[dict[str, Any]]:
    cutoff_ms = int(timing.stale_after * 1000)
    return list(
        database[METADATA_COLLECTION].aggregate(
            [
                {"$match": {"kind": "writer"}},
                {"$match": {"$expr": {"$gt": ["$heartbeat", {"$subtract": ["$$NOW", cutoff_ms]}]}}},
            ]
        )
    )


def _stale_writer_documents(database: Any, timing: LeaseTiming) -> list[dict[str, Any]]:
    fresh = {document["_id"] for document in _fresh_writer_documents(database, timing)}
    return [
        document for document in database[METADATA_COLLECTION].find({"kind": "writer"}) if document["_id"] not in fresh
    ]


def acquire_writer(database: Any, *, timing: LeaseTiming = DEFAULT_TIMING) -> WriterLease:
    """Insert a writer lease, then check fsck and observe the layout generation.

    The ordering is the writer half of the insert-then-check handshake.  A
    stale fsck lease intentionally still blocks writers: clearing it is an
    explicit administrative assertion that its owner is dead.
    """
    owner = _owner()
    identifier = f"lease/{uuid.uuid4()}"
    collection = database[METADATA_COLLECTION]
    collection.update_one(
        {"_id": identifier},
        [{"$set": {"kind": "writer", "owner": owner, "heartbeat": "$$NOW"}}],
        upsert=True,
    )
    lease = WriterLease(database, identifier, owner, timing)
    lease._last_refresh = time.monotonic()
    try:
        documents = {
            document["_id"]: document
            for document in collection.find({"_id": {"$in": ["lease/fsck", "layout"]}}, {"_id": 1, "generation": 1})
        }
        if "lease/fsck" in documents:
            raise StoreLockedError("MongoStore is locked by an fsck lease")
        layout = documents.get("layout")
        if layout is None or not isinstance(layout.get("generation"), int):
            raise RuntimeError("MongoStore metadata layout document is missing its generation counter")
        lease.generation = int(layout["generation"])
        return lease
    except BaseException:
        lease.release()
        raise


def acquire_fsck(database: Any, *, force: bool = False, timing: LeaseTiming = DEFAULT_TIMING) -> FsckLease:
    """Install the fsck singleton and drain fresh writer registrations.

    ``force`` may replace a pre-existing fsck lease and may proceed past stale
    writer residue.  It is an administrative assertion that those owners are
    dead; using it against a still-running owner can corrupt the store.
    """
    owner = _owner()
    collection = database[METADATA_COLLECTION]
    try:
        collection.insert_one({"_id": "lease/fsck", "owner": owner, "heartbeat": None})
    except DuplicateKeyError:
        if not force:
            raise StoreLockedError("MongoStore already has an fsck lease") from None
        cutoff_ms = int(timing.stale_after * 1000)
        replaced = collection.find_one_and_update(
            {
                "_id": "lease/fsck",
                "$expr": {"$lte": ["$heartbeat", {"$subtract": ["$$NOW", cutoff_ms]}]},
            },
            [{"$set": {"owner": owner, "heartbeat": "$$NOW"}}],
        )
        if replaced is None:
            raise StoreLockedError("the existing fsck lease is fresh and cannot be force-replaced") from None
    lease = FsckLease(database, "lease/fsck", owner, timing)
    if force:
        lease._last_refresh = time.monotonic()
    try:
        lease.refresh_heartbeat(force=True)
        delay = timing.poll_initial
        while True:
            fresh = _fresh_writer_documents(database, timing)
            if not fresh:
                stale = _stale_writer_documents(database, timing)
                if stale and not force:
                    raise StoreLockedError("stale writer leases require fsck(force=True) to override")
                return lease
            time.sleep(delay)
            delay = min(timing.poll_max, delay * 2)
            lease.refresh_heartbeat()
    except BaseException:
        lease.release()
        raise


def clear_stale_lock(database: Any, *, timing: LeaseTiming = DEFAULT_TIMING) -> None:
    """Clear a stale fsck lock after an administrator verified its owner died.

    This has no fencing: clearing a merely slow fsck can corrupt the store.
    """
    collection = database[METADATA_COLLECTION]
    cutoff_ms = int(timing.stale_after * 1000)
    result = collection.delete_one(
        {
            "_id": "lease/fsck",
            "$expr": {"$lte": ["$heartbeat", {"$subtract": ["$$NOW", cutoff_ms]}]},
        }
    )
    if result.deleted_count == 0 and collection.find_one({"_id": "lease/fsck"}) is not None:
        raise StoreLockedError("the fsck lease is fresh and cannot be cleared")
