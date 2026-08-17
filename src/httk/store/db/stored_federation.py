"""Bounded, durable federation of stored entry-family property plans.

Unlike :mod:`httk.store.federated_store`, which is the general portable query
protocol, this module joins only configured durable entry families.
It can therefore retain a stable backing inventory, push candidate filtering
and bounds into SQL, and delay record hydration until a global page is known.
"""

import heapq
import json
import warnings
from collections.abc import Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final, cast

from httk.core.optimade import FilterAst

from httk.store.db.stored_properties import (
    StoredPropertySqlCandidateStream,
    StoredPropertySqlPlan,
)
from httk.store.store_common import EntryStore

__all__ = [
    "DuplicateEntryIdError",
    "StoredEntryFederation",
    "StoredEntryOrigin",
    "StoredEntryPage",
    "StoredEntrySource",
]


_AUDIT_BATCH_SIZE: Final = 1_000
_CONTENT_ID_LENGTH: Final = 64
_CONTENT_ID_CHARACTERS: Final = frozenset("0123456789abcdef")


@dataclass(frozen=True)
class StoredEntrySource:
    """One named configured family in one durable entry store.

    ``public_id_prefix`` is concatenated with every backing's canonical
    content id.  It is intentionally not required to be unique: callers may
    retain a legacy unprefixed source, in which case collisions are detected
    when their visible ids are fetched or explicitly audited.

    :param store: The durable entry store containing the entry family.
    :param entry_family: The logical entry-family class to serve.
    :param name: The unique name used to identify this source.
    :param public_id_prefix: The prefix prepended to canonical content ids.
    """

    store: EntryStore
    entry_family: type
    name: str
    public_id_prefix: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.store, EntryStore):
            raise TypeError("StoredEntrySource.store must be an EntryStore")
        if not isinstance(self.entry_family, type):
            raise TypeError("StoredEntrySource.entry_family must be an entry-family class")
        if not isinstance(self.name, str) or not self.name or self.name != self.name.strip():
            raise ValueError("StoredEntrySource.name must be a non-empty stripped string")
        if not isinstance(self.public_id_prefix, str):
            raise TypeError("StoredEntrySource.public_id_prefix must be a string")


@dataclass(frozen=True)
class StoredEntryOrigin:
    """The durable source of one public entry id.

    :param source: The configured source name.
    :param source_index: The source's position in the federation.
    :param backing: The concrete backing name.
    :param content_id: The canonical content id claimed by the backing.
    """

    source: str
    source_index: int
    backing: str
    content_id: str


class DuplicateEntryIdError(RuntimeError):
    """Several durable origins claim the same public entry id.

    Call :meth:`StoredEntryFederation.audit_duplicate_ids` to perform the
    intentionally explicit complete audit; ordinary pages inspect only the
    candidates they would otherwise return.

    :param public_id: The public id claimed by multiple origins.
    :param origins: The durable origins claiming the id.
    """

    def __init__(self, public_id: str, origins: Sequence[StoredEntryOrigin]) -> None:
        self.public_id = public_id
        self.origins = tuple(origins)
        rendered = ", ".join(f"{item.source}/{item.backing}" for item in self.origins)
        super().__init__(
            f"duplicate public entry id {public_id!r} from {rendered}; "
            "call audit_duplicate_ids() to audit the complete federation"
        )


@dataclass(frozen=True)
class StoredEntryPage:
    """One immutable globally paginated response.

    ``total_count`` is the exact filtered count before global offset/limit.
    The sentinel establishing :attr:`more_data_available` is ID-only and is
    never present in :attr:`rows`.

    :param rows: The rows visible in this page.
    :param total_count: The exact filtered count before paging bounds.
    :param more_data_available: Whether another row exists after this page.
    """

    rows: tuple[Mapping[str, Any], ...]
    total_count: int
    more_data_available: bool

    def __post_init__(self) -> None:
        if not isinstance(self.rows, tuple):
            object.__setattr__(self, "rows", tuple(self.rows))
        if isinstance(self.total_count, bool) or not isinstance(self.total_count, int) or self.total_count < 0:
            raise ValueError("StoredEntryPage.total_count must be a non-negative integer")
        if not isinstance(self.more_data_available, bool):
            raise TypeError("StoredEntryPage.more_data_available must be bool")


@dataclass(frozen=True)
class _ResolvedSource:
    source: StoredEntrySource
    source_index: int
    plan: StoredPropertySqlPlan


@dataclass(frozen=True)
class _Stream:
    source: _ResolvedSource
    backing: type
    backing_name: str
    backing_index: int
    candidate_stream: StoredPropertySqlCandidateStream


@dataclass(frozen=True)
class _Candidate:
    stream: _Stream
    sid: int
    content_id: str
    sort_values: tuple[Any, ...]
    store_timestamp: int | None = None

    @property
    def public_id(self) -> str:
        return self.stream.source.source.public_id_prefix + self.content_id

    @property
    def origin(self) -> StoredEntryOrigin:
        return StoredEntryOrigin(
            self.stream.source.source.name,
            self.stream.source.source_index,
            self.stream.backing_name,
            self.content_id,
        )


class _Descending:
    """Heap key wrapper reversing only one comparable non-null value."""

    __slots__ = ("value",)

    def __init__(self, value: Any) -> None:
        self.value = value

    def __lt__(self, other: "_Descending") -> bool:
        return self.value > other.value

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _Descending) and self.value == other.value


class StoredEntryFederation:
    """Merge one or more configured durable entry-family sources.

    Sources preserve caller order.  Without a sort, rows remain in source,
    persisted-backing, and native database order and candidate SQL contains no
    ``ORDER BY``.  With a sort, each backing stream orders in SQL and this
    object performs a bounded heap merge with a deterministic public-id/source
    /backing tie-breaker.

    Pages probe all sibling backings in prefixes shared by multiple sources.
    Within-source corruption is otherwise audit-only; use
    :meth:`audit_duplicate_ids` to detect it.

    :param sources: The configured sources to merge in caller order.
    """

    def __init__(self, sources: Sequence[StoredEntrySource]) -> None:
        if isinstance(sources, (str, bytes)):
            raise TypeError("StoredEntryFederation.sources must be a sequence of StoredEntrySource values")
        values = tuple(sources)
        if not values:
            raise ValueError("StoredEntryFederation requires at least one source")
        if not all(isinstance(item, StoredEntrySource) for item in values):
            raise TypeError("StoredEntryFederation.sources must contain StoredEntrySource values")
        names = tuple(item.name for item in values)
        if len(set(names)) != len(names):
            raise ValueError("StoredEntryFederation source names must be unique")
        entry_family = values[0].entry_family
        if any(item.entry_family is not entry_family for item in values[1:]):
            raise ValueError("StoredEntryFederation sources must use one exact entry_family")
        if sum(item.public_id_prefix == "" for item in values) > 1:
            warnings.warn(
                "multiple StoredEntrySource values use an empty public_id_prefix; "
                "duplicate ids remain lazy until fetch(), a visible page, or audit_duplicate_ids()",
                RuntimeWarning,
                stacklevel=2,
            )
        resolved_sources = tuple(
            _ResolvedSource(
                source,
                index,
                source.store.stored_property_plan(source.entry_family),
            )
            for index, source in enumerate(values)
        )
        entry_type = resolved_sources[0].plan.entry_type
        definition_id = _definition_id(resolved_sources[0].plan)
        if any(
            item.plan.entry_type != entry_type or _definition_id(item.plan) != definition_id
            for item in resolved_sources[1:]
        ):
            raise ValueError("StoredEntryFederation sources must use equal entry type and definition")
        self._sources = resolved_sources
        stream_groups: dict[str, list[tuple[int, int]]] = {}
        for source in resolved_sources:
            for backing_index in range(len(source.plan.backings)):
                stream_groups.setdefault(source.source.public_id_prefix, []).append(
                    (source.source_index, backing_index)
                )
        self._page_colliding_streams = {
            prefix: frozenset(streams)
            for prefix, streams in stream_groups.items()
            if len({source_index for source_index, _backing_index in streams}) > 1
        }
        self._audit_streams = {
            prefix: frozenset(streams) for prefix, streams in stream_groups.items() if len(streams) > 1
        }

    @property
    def sources(self) -> tuple[StoredEntrySource, ...]:
        """Return the immutable declared source order.

        :return: The declared sources in caller order.
        """
        return tuple(item.source for item in self._sources)

    def snapshot_cutoff_ns(self, now_ns: int) -> int | None:
        """Return the latest instant strictly before every capable source's current bucket.

        Per-source floor conversion then selects each store's last completed
        timestamp unit, so future monotonic writes cannot enter the snapshot,
        even when sources use different timestamp resolutions. Timestamp-
        disabled sources are ignored; ``None`` means no source is capable.

        :param now_ns: Current time in nanoseconds.
        :return: A nanosecond cutoff, or ``None`` when no source stores timestamps.
        """
        completed_buckets = []
        for source in self._sources:
            store = cast(Any, source.source.store)
            if store.store_timestamps:
                resolution = store.store_timestamp_resolution
                completed_buckets.append((now_ns // resolution) * resolution)
        return None if not completed_buckets else min(completed_buckets) - 1

    @property
    def _colliding_streams(self) -> Mapping[str, frozenset[tuple[int, int]]]:
        """Cross-source stream groups requiring page-time duplicate probes.

        Page serving intentionally does not detect duplicates between record
        classes in one source: the store write path maintains dispatch
        consistency, and such duplicates require out-of-band modification.
        :meth:`audit_duplicate_ids` is the designed detector for that
        corruption class.
        """
        return self._page_colliding_streams

    def query(
        self,
        filter_string: str | FilterAst | None = None,
        *,
        sort: Sequence[tuple[str, bool]] = (),
        offset: int = 0,
        limit: int | None = None,
        as_of: object = None,
        fields: Collection[str] | None = None,
    ) -> StoredEntryPage:
        """Return one globally merged page with an exact filtered total.

        ``limit=0`` intentionally runs only count plus an ID-only sentinel:
        it is suitable for metadata initialization and never duplicate-probes
        or hydrates a candidate.

        :param filter_string: The OPTIMADE filter, parsed filter tree, or no filter.
        :param sort: The property sort keys and directions.
        :param offset: The number of matching rows to skip globally.
        :param limit: The maximum number of rows to return, or no maximum.
        :param as_of: Optional historic cutoff. Sources without store timestamps
            deliberately omit this cutoff and serve their current state.
            Dependencies of a visible row remain visible because references
            only point at earlier-or-equal rows from the same transaction.
        :param fields: The response property names to render, or ``None`` to render every configured property.
        :return: The globally merged page.
        :raises DuplicateEntryIdError: If a visible id has multiple cross-source origins.
        """
        _validate_page_bounds(offset, limit)
        ordered = _normalized_sort(sort)
        stream_sort = _stream_sort(ordered) if ordered else ()
        streams = self._streams(filter_string, stream_sort, as_of=as_of)
        counts = tuple(stream.candidate_stream.searcher.count() for stream in streams)
        total_count = sum(counts)
        if ordered:
            candidates = self._sorted_candidates(streams, ordered, offset, limit)
        else:
            candidates = self._unsorted_candidates(streams, counts, offset, limit)
        visible_count = len(candidates) if limit is None else min(limit, len(candidates))
        visible = candidates[:visible_count]
        more = False if limit is None else len(candidates) > visible_count
        # A public page may detect collisions only for ids it would expose.
        # The ID-only sentinel is deliberately excluded, especially for
        # limit=0 metadata calls.
        for candidate in visible:
            self._probe_candidate(candidate, as_of=as_of)
        rows = self._render_page(visible, fields)
        return StoredEntryPage(rows, total_count, more)

    def fetch(
        self, public_id: str, *, as_of: object = None, fields: Collection[str] | None = None
    ) -> Mapping[str, Any] | None:
        """Fetch one public id and detect a collision among its possible origins.

        :param public_id: The public id to fetch.
        :param as_of: Optional historic cutoff. Sources without store timestamps
            deliberately omit this cutoff and serve their current state.
        :param fields: The response property names to render, or ``None`` to render every configured property.
        :return: The fetched response row, or ``None`` when it is absent.
        :raises DuplicateEntryIdError: If the id has multiple origins.
        """
        if not isinstance(public_id, str):
            raise TypeError("StoredEntryFederation.fetch public_id must be a string")
        matches = self._probe_public_id(public_id, as_of=as_of)
        if not matches:
            return None
        return self._row(matches[0], fields)

    def audit_duplicate_ids(self, *, batch_size: int = _AUDIT_BATCH_SIZE) -> None:
        """Lazily scan sorted ID-only batches and raise on the first collision.

        The audit includes duplicate ids across backings within one source as
        well as duplicates across sources.

        :param batch_size: The maximum number of candidate ids read per batch.
        :return: None.
        :raises DuplicateEntryIdError: If any public id has multiple origins.
        """
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise TypeError("audit_duplicate_ids batch_size must be an integer")
        if batch_size < 1:
            raise ValueError("audit_duplicate_ids batch_size must be positive")
        stream_keys = frozenset(stream for group in self._audit_streams.values() for stream in group)
        if not stream_keys:
            return
        streams = self._streams(None, (("id", False),), stream_keys)
        iterators = [_BatchedCandidateIterator(stream, batch_size) for stream in streams]
        heap: list[tuple[tuple[Any, ...], int, _Candidate]] = []
        for index, iterator in enumerate(iterators):
            candidate = _next_or_none(iterator)
            if candidate is not None:
                heapq.heappush(heap, ((candidate.public_id,), index, candidate))
        while heap:
            _key, index, candidate = heapq.heappop(heap)
            group = [candidate]
            following = _next_or_none(iterators[index])
            if following is not None:
                heapq.heappush(heap, ((following.public_id,), index, following))
            while heap and heap[0][2].public_id == candidate.public_id:
                _same_key, same_index, same = heapq.heappop(heap)
                group.append(same)
                following = _next_or_none(iterators[same_index])
                if following is not None:
                    heapq.heappush(heap, ((following.public_id,), same_index, following))
            if len(group) > 1:
                raise DuplicateEntryIdError(candidate.public_id, tuple(item.origin for item in group))

    def _streams(
        self,
        filter_string: str | FilterAst | None,
        sort: Sequence[tuple[str, bool]],
        stream_keys: frozenset[tuple[int, int]] | None = None,
        *,
        as_of: object = None,
    ) -> tuple[_Stream, ...]:
        streams: list[_Stream] = []
        for source in self._sources:
            source_as_of = as_of if getattr(source.source.store, "store_timestamps", False) else None
            candidates = source.plan.candidate_searchers(
                filter_string,
                sort=sort,
                public_id_prefix=source.source.public_id_prefix,
                as_of=source_as_of,
            )
            for backing_index, candidate in enumerate(candidates):
                if stream_keys is not None and (source.source_index, backing_index) not in stream_keys:
                    continue
                streams.append(_Stream(source, candidate.backing, candidate.backing_name, backing_index, candidate))
        return tuple(streams)

    def _unsorted_candidates(
        self,
        streams: Sequence[_Stream],
        counts: Sequence[int],
        offset: int,
        limit: int | None,
    ) -> list[_Candidate]:
        remaining = None if limit is None else limit + 1
        skipped = offset
        result: list[_Candidate] = []
        for stream, count in zip(streams, counts, strict=True):
            if skipped >= count:
                skipped -= count
                continue
            take = count - skipped if remaining is None else min(count - skipped, remaining)
            if take <= 0:
                break
            searcher = stream.candidate_stream.searcher
            if skipped:
                searcher.add_offset(skipped)
            searcher.set_limit(take)
            fetched = tuple(_candidates(stream))
            result.extend(fetched)
            skipped = 0
            if remaining is not None:
                remaining -= len(fetched)
                if remaining == 0:
                    break
        return result

    def _sorted_candidates(
        self,
        streams: Sequence[_Stream],
        sort: Sequence[tuple[str, bool]],
        offset: int,
        limit: int | None,
    ) -> list[_Candidate]:
        needed = None if limit is None else offset + limit + 1
        iterators: list[Iterator[_Candidate]] = []
        heap: list[tuple[tuple[Any, ...], int, _Candidate]] = []
        for index, stream in enumerate(streams):
            if needed is not None:
                stream.candidate_stream.searcher.set_limit(needed)
            iterator = iter(_candidates(stream))
            iterators.append(iterator)
            candidate = _next_or_none(iterator)
            if candidate is not None:
                heapq.heappush(heap, (_sort_key(candidate, sort), index, candidate))
        result: list[_Candidate] = []
        target = None if limit is None else offset + limit + 1
        while heap and (target is None or len(result) < target):
            _key, index, candidate = heapq.heappop(heap)
            result.append(candidate)
            following = _next_or_none(iterators[index])
            if following is not None:
                heapq.heappush(heap, (_sort_key(following, sort), index, following))
        return result[offset:]

    def _probe_candidate(self, candidate: _Candidate, *, as_of: object = None) -> None:
        colliding = self._colliding_streams.get(candidate.stream.source.source.public_id_prefix)
        if colliding is None:
            return
        matches = [candidate]
        filter_string = "id = " + json.dumps(candidate.public_id)
        candidate_key = (candidate.stream.source.source_index, candidate.stream.backing_index)
        for stream in self._streams(filter_string, (), colliding - {candidate_key}, as_of=as_of):
            stream.candidate_stream.searcher.set_limit(1)
            matches.extend(_candidates(stream))
        if len(matches) > 1:
            raise DuplicateEntryIdError(candidate.public_id, tuple(item.origin for item in matches))

    def _probe_public_id(self, public_id: str, *, as_of: object = None) -> tuple[_Candidate, ...]:
        stream_keys = frozenset(
            (source.source_index, backing_index)
            for source in self._sources
            if _content_id_for_public_id(public_id, source.source.public_id_prefix) is not None
            for backing_index in range(len(source.plan.backings))
        )
        if not stream_keys:
            return ()
        filter_string = "id = " + json.dumps(public_id)
        matches: list[_Candidate] = []
        for stream in self._streams(filter_string, (), stream_keys, as_of=as_of):
            stream.candidate_stream.searcher.set_limit(1)
            matches.extend(_candidates(stream))
        if len(matches) > 1:
            raise DuplicateEntryIdError(public_id, tuple(item.origin for item in matches))
        return tuple(matches)

    @staticmethod
    def _render_page(visible: Sequence[_Candidate], fields: Collection[str] | None) -> tuple[Mapping[str, Any], ...]:
        """Render visible candidates, batching the record fetch per source backing.

        Candidates are grouped by their originating ``_Stream`` (one per
        ``(source, backing)`` per ``_streams()`` call, and every visible
        candidate comes from one such call) so each distinct backing table is
        hydrated in one batched ``fetch_many`` call instead of one ``fetch`` per
        row.  Grouping is by object identity; streams need not be hashable by
        value.  Rows are then rendered in the original ``visible`` order.

        :param visible: The page candidates in final output order.
        :param fields: The response property names to render, or ``None`` to render every configured property.
        :return: The rendered response rows in ``visible`` order.
        """
        groups: dict[int, list[_Candidate]] = {}
        for candidate in visible:
            groups.setdefault(id(candidate.stream), []).append(candidate)
        record_by_candidate: dict[int, object] = {}
        # A full render (fields is None) touches every configured property, so
        # eager hydration is kept to preserve the shipped batched profile. A
        # field subset renders lazily: rows.py chunk-batches child tables, so
        # untouched child tables are simply never SELECTed.
        eager = fields is None
        for group in groups.values():
            store = group[0].stream.source.source.store
            backing = group[0].stream.backing
            records: list[object] = store.fetch_many(backing, [candidate.sid for candidate in group], eager=eager)
            for candidate, record in zip(group, records, strict=True):
                record_by_candidate[id(candidate)] = record
        return tuple(
            StoredEntryFederation._render(candidate, record_by_candidate[id(candidate)], fields)
            for candidate in visible
        )

    @staticmethod
    def _render(candidate: _Candidate, record: object, fields: Collection[str] | None) -> Mapping[str, Any]:
        """Render one already-fetched record into its public response row.

        :param candidate: The visible candidate to render.
        :param record: The hydrated backing record for ``candidate``.
        :param fields: The response property names to render, or ``None`` to render every configured property.
        :return: The immutable response row.
        """
        source = candidate.stream.source
        row = source.plan.response_row(
            candidate.stream.backing,
            record,
            public_id=candidate.public_id,
            store_timestamp=candidate.store_timestamp,
            fields=fields,
        )
        return MappingProxyType(dict(row))

    @staticmethod
    def _row(candidate: _Candidate, fields: Collection[str] | None) -> Mapping[str, Any]:
        source = candidate.stream.source
        eager = fields is None
        record: object = source.source.store.fetch(candidate.stream.backing, candidate.sid, eager=eager)
        return StoredEntryFederation._render(candidate, record, fields)


class _BatchedCandidateIterator:
    """One sorted ID-only stream fetched in bounded SQL batches."""

    def __init__(self, stream: _Stream, batch_size: int) -> None:
        self._stream = stream
        self._batch_size = batch_size
        self._last_public_id: str | None = None
        self._rows: Iterator[_Candidate] = iter(())
        self._done = False

    def __iter__(self) -> "_BatchedCandidateIterator":
        return self

    def __next__(self) -> _Candidate:
        while True:
            candidate = _next_or_none(self._rows)
            if candidate is not None:
                return candidate
            if self._done:
                raise StopIteration
            filter_string = None if self._last_public_id is None else "id > " + json.dumps(self._last_public_id)
            fresh = self._stream.source.plan.candidate_searchers(
                filter_string,
                sort=(("id", False),),
                public_id_prefix=self._stream.source.source.public_id_prefix,
            )[self._stream.backing_index]
            fresh.searcher.set_limit(self._batch_size)
            batch_stream = _Stream(
                self._stream.source,
                fresh.backing,
                fresh.backing_name,
                self._stream.backing_index,
                fresh,
            )
            values = tuple(_candidates(batch_stream))
            if values:
                self._last_public_id = values[-1].public_id
            self._done = len(values) < self._batch_size
            self._rows = iter(values)


def _candidates(stream: _Stream) -> Iterator[_Candidate]:
    for values, _names in stream.candidate_stream.searcher:
        expected_width = 2 + stream.candidate_stream.sort_count + int(stream.candidate_stream.timestamp_output)
        if len(values) != expected_width:
            raise RuntimeError(
                f"candidate stream {stream.source.source.name}/{stream.backing_name} returned "
                f"{len(values)} values; expected {expected_width}"
            )
        sort_end = 2 + stream.candidate_stream.sort_count
        timestamp = values[sort_end] if stream.candidate_stream.timestamp_output else None
        yield _Candidate(stream, int(values[0]), str(values[1]), tuple(values[2:sort_end]), timestamp)


def _next_or_none(iterator: Iterator[_Candidate]) -> _Candidate | None:
    try:
        return next(iterator)
    except StopIteration:
        return None


def _content_id_for_public_id(public_id: str, prefix: str) -> str | None:
    """Return the canonical content-id suffix when ``public_id`` has ``prefix``."""
    if not public_id.startswith(prefix):
        return None
    value = public_id[len(prefix) :]
    if len(value) != _CONTENT_ID_LENGTH or not all(character in _CONTENT_ID_CHARACTERS for character in value):
        return None
    return value


def _definition_id(plan: StoredPropertySqlPlan) -> str | None:
    """The concrete definition id, including inherited standard definitions."""
    return plan.definition.definition_id or plan.definition.extends_id


def _validate_page_bounds(offset: int, limit: int | None) -> None:
    if isinstance(offset, bool) or not isinstance(offset, int):
        raise TypeError("StoredEntryFederation.query offset must be an integer")
    if offset < 0:
        raise ValueError("StoredEntryFederation.query offset must be non-negative")
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int)):
        raise TypeError("StoredEntryFederation.query limit must be an integer or None")
    if limit is not None and limit < 0:
        raise ValueError("StoredEntryFederation.query limit must be non-negative or None")


def _normalized_sort(sort: Sequence[tuple[str, bool]]) -> tuple[tuple[str, bool], ...]:
    if isinstance(sort, (str, bytes)):
        raise TypeError("StoredEntryFederation.query sort must be a sequence of (property, descending) pairs")
    result: list[tuple[str, bool]] = []
    for item in sort:
        if not isinstance(item, tuple) or len(item) != 2:
            raise TypeError("StoredEntryFederation.query sort entries must be (property, descending) pairs")
        name, descending = item
        if not isinstance(name, str) or not name:
            raise ValueError("StoredEntryFederation.query sort property names must be non-empty strings")
        if not isinstance(descending, bool):
            raise TypeError("StoredEntryFederation.query sort descending flags must be bool")
        if name in {existing for existing, _direction in result}:
            raise ValueError(f"StoredEntryFederation.query sort repeats {name!r}")
        result.append((name, descending))
    return tuple(result)


def _stream_sort(sort: Sequence[tuple[str, bool]]) -> tuple[tuple[str, bool], ...]:
    """Append the public-id stream tie-break only when a user key lacks it."""
    return tuple(sort) if any(name == "id" for name, _descending in sort) else (*sort, ("id", False))


def _sort_key(candidate: _Candidate, sort: Sequence[tuple[str, bool]]) -> tuple[Any, ...]:
    values: list[Any] = []
    for value, (_name, descending) in zip(candidate.sort_values[: len(sort)], sort, strict=True):
        values.append((1, None) if value is None else (0, _Descending(value) if descending else value))
    return (*values, candidate.public_id, candidate.stream.source.source_index, candidate.stream.backing_name)
