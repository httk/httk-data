"""Backend-neutral save-path machinery shared by storage backends."""

import functools
import types
import typing
import weakref
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated, Any, Protocol, TypeVar, runtime_checkable

from httk.core.storage import IdentitySkip, project_storage_record
from httk.core.storage.identity import _trusted_content_id

from httk.store.backend.schema import FieldSpec, resolve_schema
from httk.store.storage_layout import EntryFamilyLayout

__all__ = [
    "EntryDispatchIntegrityError",
    "EntryMetadataConflictError",
    "EntryReplacementError",
    "EntryStore",
    "IdentityCaches",
    "SaveProjection",
    "reject_cursor_proxy",
]


_StoredRecord = TypeVar("_StoredRecord")


@runtime_checkable
class EntryStore(Protocol):
    """The store surface consumed by :mod:`httk.store.backend.sql.stored_federation`.

    This protocol deliberately describes the small backend seam used by the
    federation.  The stored-property plan and candidate-stream objects remain
    backend-specific; their SQL implementations use :meth:`searcher`.
    """

    @property
    def entry_layout(self) -> tuple[EntryFamilyLayout, ...]:
        """Return the configured entry-family layouts in stable order."""
        ...

    def searcher(self, *, as_of: object = None) -> Any:
        """Return a backend searcher used to build candidate ID streams."""
        ...

    def fetch(self, cls: type[_StoredRecord], sid: int, *, eager: bool = False) -> _StoredRecord:
        """Fetch the stored record of ``cls`` identified by ``sid``.

        :param cls: The storable record class.
        :param sid: The stored row identifier to fetch.
        :param eager: Whether to fully materialize the record instead of returning a lazy row.
        :return: The reconstructed instance.
        """
        ...

    def fetch_many(self, cls: type[_StoredRecord], sids: Sequence[int], *, eager: bool = False) -> list[_StoredRecord]:
        """Fetch the stored records of ``cls`` identified by ``sids``.

        Batched counterpart of :meth:`fetch`.

        :param cls: The storable record class.
        :param sids: The stored row identifiers to fetch.
        :param eager: Whether to fully materialize each record instead of returning lazy rows.
        :return: The reconstructed instances in ``sids`` order.
        :raises KeyError: When any requested row is absent.
        """
        ...

    def stored_property_plan(self, family: type) -> Any:
        """Return the backend-specific stored-property plan for one family.

        :param family: The logical entry-family class to plan.
        :return: The validated stored-property plan consumed by federation.
        """
        ...


class EntryMetadataConflictError(ValueError):
    """Stored identity-excluded metadata differs from a repeated save."""


class EntryReplacementError(ValueError):
    """A replacement deduplicated onto a row from a different lineage.

    :param table_name: The table (or collection) whose replacement failed.
    :param predecessor_logical_id: The logical_id of the intended predecessor.
    :param conflicting_logical_id: The logical_id of the row actually hit.
    """

    def __init__(self, table_name: str, predecessor_logical_id: int, conflicting_logical_id: int) -> None:
        self.table_name = table_name
        self.predecessor_logical_id = predecessor_logical_id
        self.conflicting_logical_id = conflicting_logical_id
        super().__init__(
            f"replacement in table {table_name!r} deduplicated onto an existing row with logical_id "
            f"{conflicting_logical_id}, but the predecessor's logical_id is {predecessor_logical_id}"
        )


class EntryDispatchIntegrityError(RuntimeError):
    """A persisted entry dispatch row does not name exactly its expected backing."""


class SaveProjection:
    """One-save projection cache shared by core identity and SQL encoding."""

    def __init__(self, *, store_timestamp: int | None = None) -> None:
        self.store_timestamp = store_timestamp
        self.values_by_source: dict[tuple[type, int], Mapping[str, object]] = {}
        self.validated: set[tuple[type, int]] = set()
        self.metadata_rows: dict[tuple[type, int], Mapping[str, Any]] = {}
        self.metadata_children: dict[tuple[type, int, str], Any] = {}
        self.metadata_content_ids: dict[tuple[type, int], str] = {}
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
        # SaveProjection only memoizes the standard deterministic projection;
        # use core's trusted route so the source-owned content-id cache is
        # shared across saves.  Arbitrary custom projectors remain uncached via
        # the public content_id path.
        return _trusted_content_id(source, as_record=record_type, projector=self.projector)


@dataclass(frozen=True)
class _MetadataPlan:
    skipped_specs: tuple[FieldSpec, ...]
    skipped_nested: tuple[FieldSpec, ...]
    descend_specs: tuple[FieldSpec, ...]


_MISSING_METADATA = object()


@functools.cache
def _metadata_reachable_types(record_type: type) -> frozenset[type]:
    reachable: set[type] = set()
    pending = [record_type]
    while pending:
        current = pending.pop()
        if current in reachable:
            continue
        reachable.add(current)
        pending.extend(
            spec.target for spec in resolve_schema(current).fields if not spec.derived and spec.target is not None
        )
    return frozenset(reachable)


def _metadata_has_plan(record_type: type) -> bool:
    for reachable_type in _metadata_reachable_types(record_type):
        hints = typing.get_type_hints(reachable_type, include_extras=True)
        if any(
            not spec.derived and _has_identity_skip(hints.get(spec.field))
            for spec in resolve_schema(reachable_type).fields
        ):
            return True
    return False


@functools.cache
def _metadata_plan(record_type: type) -> _MetadataPlan | None:
    schema = resolve_schema(record_type)
    hints = typing.get_type_hints(record_type, include_extras=True)
    skipped_specs: list[FieldSpec] = []
    skipped_nested: list[FieldSpec] = []
    descend_specs: list[FieldSpec] = []
    for spec in schema.fields:
        if spec.derived:
            continue
        identity_skipped = _has_identity_skip(hints.get(spec.field))
        if identity_skipped:
            if spec.role in {"scalar", "encoded", "fixed_array"}:
                skipped_specs.append(spec)
            else:
                skipped_nested.append(spec)
        elif spec.target is not None and _metadata_has_plan(spec.target):
            descend_specs.append(spec)
    if not skipped_specs and not skipped_nested and not descend_specs:
        return None
    return _MetadataPlan(tuple(skipped_specs), tuple(skipped_nested), tuple(descend_specs))


def _has_identity_skip(annotation: Any) -> bool:
    origin = typing.get_origin(annotation)
    if origin is Annotated:
        arguments = typing.get_args(annotation)
        return any(isinstance(marker, IdentitySkip) for marker in arguments[1:]) or _has_identity_skip(arguments[0])
    if origin in (typing.Union, types.UnionType):
        return any(_has_identity_skip(argument) for argument in typing.get_args(annotation))
    return False


def reject_cursor_proxy(obj: Any) -> None:
    """Reject a lazy cursor row before a backend attempts to save it."""
    if getattr(obj, "__httk_cursor_proxy__", False):
        raise TypeError("cursor rows cannot be saved; materialize the record first")


class IdentityCaches:
    """Weak identity caches shared by storage backends."""

    def __init__(self) -> None:
        self._instances: weakref.WeakValueDictionary[tuple[type, int], Any] = weakref.WeakValueDictionary()
        self._sids: weakref.WeakKeyDictionary[Any, dict[type, int]] = weakref.WeakKeyDictionary()
        self._sids_by_identity: dict[tuple[type, int], int] = {}
        """Reverse cache for instances that cannot be hashed (e.g. they hold a list).

        Keyed on ``id()``, with a finalizer dropping each entry when its
        instance dies, so a recycled id can never resolve to a stale sid.
        """

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
