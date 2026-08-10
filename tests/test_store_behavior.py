"""Backend-neutral store semantics that every storage backend must provide.

The packet-2d census classifies table/column identifiers, raw SQL, SQLAlchemy
objects or exceptions, statement counting, dialect names, WHERE/HAVING and
compiled-SQL internals, and private cursor-token formats as SQL-physical.  The
query-result values, counts, ordering, set/reference/string outcomes, paging
traversal, cursor rejection behavior, and stored-property predicate outcomes
belong in the backend-neutral query and paging suites.
"""

import datetime
import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from fractions import Fraction
from typing import Annotated, ClassVar
from zoneinfo import ZoneInfo

import pytest
from httk.core import FileEntry, FileRecord, FracScalar, FracVector
from httk.core.storage import (
    IdentitySkip,
    Indexed,
    Shape,
    Skip,
    StorageInfo,
    StorageProjectionCycleError,
    Unique,
    content_id,
    stored_property,
)

from httk.store.store_common import EntryMetadataConflictError


@dataclass(frozen=True)
class Author:
    name: str
    year: int


@dataclass(frozen=True)
class AuthorTag:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(dedup="by_value")

    author: Author
    tag: str
    value: str


@dataclass(frozen=True)
class LogEvent:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(dedup="none")

    message: str


@dataclass(frozen=True)
class RollbackChild:
    name: str


@dataclass(frozen=True)
class RollbackParent:
    child: RollbackChild
    name: Annotated[str, Unique()]


@dataclass(frozen=True)
class OptionalChildRoundTrip:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(dedup="none")

    value: str
    notes: list[str] | None = None


@dataclass(frozen=True)
class RowVector:
    vec: Annotated[FracVector, Shape(1, 3)]


@dataclass(frozen=True)
class FloatRecord:
    scalar: float
    values: tuple[float, ...]


class CursorProxy:
    __httk_cursor_proxy__ = True


@dataclass(frozen=True)
class Sample:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(indexes=(("formula", "spacegroup"),))

    formula: Annotated[str, Indexed()]
    spacegroup: int
    energy: float
    stable: bool
    payload: bytes
    note: str | None
    ratio: Fraction
    scale: FracScalar
    created: datetime.datetime
    cell: Annotated[FracVector, Shape(3, 3)]
    coords: Annotated[FracVector, Shape(0, 3)]
    symbols: list[str]
    tags: tuple[str, ...]
    ratios: list[Fraction]
    authors: list[Author]
    reference: Author | None
    weight: float | None = None
    scratch: Annotated[str, Skip()] = "unstored"

    @stored_property
    def natoms(self) -> int:
        return len(self.symbols)


def make_sample(**overrides) -> Sample:
    sample = Sample(
        formula="CaTiO3",
        spacegroup=221,
        energy=-12.5,
        stable=True,
        payload=b"\x00\x01\xff",
        note=None,
        ratio=Fraction(1, 3),
        scale=FracScalar(2, denom=7),
        created=datetime.datetime(2026, 7, 24, 12, 30, 0),  # noqa: DTZ001
        cell=FracVector([[1, Fraction(1, 3), 0], [0, 1, 0], [0, 0, Fraction(2, 3)]]),
        coords=FracVector(
            [[0, 0, 0], [Fraction(1, 2), Fraction(1, 2), Fraction(1, 2)], [Fraction(1, 3), Fraction(2, 3), 1]]
        ),
        symbols=["Ca", "Ti", "O"],
        tags=("perovskite", "oxide"),
        ratios=[Fraction(1, 3), Fraction(-7, 5)],
        authors=[Author("Ada", 1852)],
        reference=Author("Boole", 1854),
    )
    return replace(sample, **overrides) if overrides else sample


@dataclass(frozen=True)
class LeafView:
    value: int
    note: str | None = None


@dataclass(frozen=True)
class RootView:
    name: str
    primary: LeafView
    related: list[LeafView]
    history: tuple[LeafView, ...]
    modified: datetime.datetime
    note: str | None = None


_calls: dict[tuple[str, int], int] = {}


@dataclass(frozen=True)
class LeafRecord:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(
        storage_name="behavior_leaf", identity_name="tests.behavior.LeafRecord"
    )
    __httk_canonical_source__: ClassVar[type] = LeafView

    value: int
    note: Annotated[str | None, IdentitySkip()] = None

    @classmethod
    def __httk_project__(cls, source: LeafView) -> Mapping[str, object]:
        key = ("leaf", id(source))
        _calls[key] = _calls.get(key, 0) + 1
        return {"value": source.value, "note": source.note}


@dataclass(frozen=True)
class RootRecord:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(
        storage_name="behavior_root", identity_name="tests.behavior.RootRecord"
    )
    __httk_canonical_source__: ClassVar[type] = RootView

    name: str
    primary: LeafRecord
    related: list[LeafRecord]
    history: tuple[LeafRecord, ...]
    modified: Annotated[datetime.datetime, IdentitySkip()]
    note: Annotated[str | None, IdentitySkip()] = None

    @classmethod
    def __httk_project__(cls, source: RootView) -> Mapping[str, object]:
        key = ("root", id(source))
        _calls[key] = _calls.get(key, 0) + 1
        return {
            "name": source.name,
            "primary": source.primary,
            "related": source.related,
            "history": source.history,
            "modified": source.modified,
            "note": source.note,
        }


@dataclass(frozen=True)
class SummaryRecord:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="behavior_summary")
    __httk_canonical_source__: ClassVar[type] = RootView

    name: str
    primary_value: int

    @classmethod
    def __httk_project__(cls, source: RootView) -> Mapping[str, object]:
        return {"name": source.name, "primary_value": source.primary.value}


@dataclass(frozen=True)
class SummaryReference:
    target: SummaryRecord
    value: str


@dataclass(frozen=True)
class CycleRecord:
    name: str
    link: Annotated["CycleRecord | None", IdentitySkip()] = None


@dataclass(frozen=True)
class RecursiveMetadataRecord:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="behavior_recursive_metadata")

    value: int
    child: "RecursiveMetadataRecord | None" = None
    note: Annotated[str, IdentitySkip()] = "stored"


@dataclass(frozen=True)
class RecursiveNoMetadataRecord:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="behavior_recursive_no_metadata")

    value: int
    child: "RecursiveNoMetadataRecord | None" = None


@dataclass(frozen=True)
class IdentityChildContainer:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="behavior_identity_child_container")

    children: Annotated[list[RecursiveNoMetadataRecord], IdentitySkip()]


@dataclass(frozen=True)
class FoldMetadata:
    value: str
    instants: Annotated[tuple[datetime.datetime, ...], IdentitySkip()]


@dataclass(frozen=True)
class MetadataProbeRecord:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="behavior_metadata_probe")

    value: int
    metadata: Annotated[str, IdentitySkip()]
    constructed: ClassVar[int] = 0

    def __post_init__(self) -> None:
        type(self).constructed += 1


@dataclass(frozen=True)
class ValidationProbeView:
    value: int


@dataclass(frozen=True)
class ValidationProbeRecord:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="behavior_validation_probe")
    __httk_canonical_source__: ClassVar[type] = ValidationProbeView

    value: int
    calls: ClassVar[list[int]] = []

    @classmethod
    def __httk_project__(cls, source: ValidationProbeView) -> Mapping[str, object]:
        return {"value": source.value}

    @classmethod
    def __httk_validate__(cls, source: "ValidationProbeRecord") -> None:
        cls.calls.append(source.value)


ValidationProbeView.__httk_storage_record__ = ValidationProbeRecord
LeafView.__httk_storage_record__ = LeafRecord
RootView.__httk_storage_record__ = RootRecord


def _root(name: str = "one") -> RootView:
    shared = LeafView(1, "leaf metadata")
    return RootView(
        name,
        shared,
        [shared, LeafView(2)],
        (LeafView(3),),
        datetime.datetime(2026, 8, 1, 12, tzinfo=datetime.timezone(datetime.timedelta(hours=2))),
    )


def _assert_named_exception(call, name: str) -> None:
    with pytest.raises(Exception) as caught:
        call()
    assert type(caught.value).__name__ == name


def test_round_trip_all_field_roles(store_factory):
    sample = make_sample()
    store = store_factory()
    sid = store.save(sample)
    fetched = store_factory.reopen(store).fetch(Sample, sid)
    assert fetched == sample
    assert isinstance(fetched.symbols, list)
    assert isinstance(fetched.tags, tuple)
    assert fetched.scratch == "unstored"
    assert fetched.natoms == 3


def test_round_trip_exact_rationals_and_arrays(store_factory):
    sample = make_sample()
    store = store_factory()
    sid = store.save(sample)
    fetched = store_factory.reopen(store).fetch(Sample, sid)
    assert fetched.ratio == Fraction(1, 3)
    assert fetched.scale.to_fraction() == Fraction(2, 7)
    assert fetched.cell == sample.cell
    assert fetched.coords == sample.coords
    assert fetched.ratios == [Fraction(1, 3), Fraction(-7, 5)]


def test_round_trip_optionals_and_optional_reference(store_factory):
    store = store_factory()
    sample = make_sample(note="a note", weight=1.25, reference=None)
    fetched = store_factory.reopen(store).fetch(Sample, store.save(sample))
    assert fetched.note == "a note"
    assert fetched.weight == 1.25
    assert fetched.reference is None
    assert fetched == sample


def test_float_round_trip_and_content_lookup_preserve_signed_zero(store_factory):
    record = FloatRecord(-0.0, (-0.0, 0.0, 1.25))
    key = content_id(record)
    store = store_factory()
    sid = store.save(record)
    reopened = store_factory.reopen(store)
    fetched = reopened.fetch(FloatRecord, sid)
    assert content_id(fetched) == key
    assert reopened.fetch_by_content_id(FloatRecord, key) == fetched
    assert math.copysign(1.0, fetched.scalar) == -1.0
    assert [math.copysign(1.0, value) for value in fetched.values[:2]] == [-1.0, 1.0]


def test_fixed_array_accepts_single_row_for_shape_1_n(store_factory):
    store = store_factory()
    sid = store.save(RowVector(FracVector([Fraction(1, 3), 1, 0])))
    assert store_factory.reopen(store).fetch(RowVector, sid).vec == FracVector([[Fraction(1, 3), 1, 0]])


def test_fixed_array_wrong_shape_raises_naming_field(store_factory):
    with pytest.raises(ValueError, match="cell"):
        store_factory().save(make_sample(cell=FracVector([[1, 0], [0, 1]])))


def test_dedup_content_id_reuses_identity(store_factory):
    store = store_factory()
    assert store.save(Author("Ada", 1852)) == store.save(Author("Ada", 1852))


def test_dedup_content_id_does_not_duplicate_record_graph(store_factory):
    store = store_factory()
    assert store.save(make_sample()) == store.save(make_sample())


def test_dedup_by_value_matches_parent_values(store_factory):
    store = store_factory()
    first = AuthorTag(Author("Ada", 1852), "role", "pioneer")
    assert store.save(first) == store.save(AuthorTag(Author("Ada", 1852), "role", "pioneer"))
    assert store.save(AuthorTag(Author("Ada", 1852), "role", "mathematician")) != store.sid_of(first)


def test_dedup_none_always_inserts(store_factory):
    store = store_factory()
    assert store.save(LogEvent("started")) != store.save(LogEvent("started"))


def test_save_is_visible_after_return(store_factory):
    store = store_factory()
    author = Author("C", 3)
    sid = store.save(author)
    assert store.fetch(Author, sid) == author


def test_implicit_rollback_clears_recursively_saved_child_caches(store_factory):
    store = store_factory()
    store.save(RollbackParent(RollbackChild("kept"), "unique"))
    rolled_back = RollbackChild("rolled back")
    with pytest.raises(Exception):
        store.save(RollbackParent(rolled_back, "unique"))
    assert store.sid_of(rolled_back) is None


def test_sid_of_unknown_object_is_none(store_factory):
    assert store_factory().sid_of(Author("New", 1900)) is None


def test_cursor_proxy_save_is_rejected(store_factory):
    with pytest.raises(TypeError, match="cursor rows cannot be saved"):
        store_factory().save(CursorProxy())


def test_optional_child_none_and_empty_round_trip(store_factory):
    store = store_factory()
    sids = [store.save(OptionalChildRoundTrip("value", notes)) for notes in (None, [], ["note"])]
    reopened = store_factory.reopen(store)
    assert [reopened.fetch(OptionalChildRoundTrip, sid).notes for sid in sids] == [None, [], ["note"]]


def test_fetch_missing_sid_raises_keyerror(store_factory):
    with pytest.raises(KeyError):
        store_factory().fetch(Author, 424242)


def test_referring_returns_matching_join_objects(store_factory):
    store = store_factory()
    ada = Author("Ada", 1852)
    boole = Author("Boole", 1854)
    for author in (ada, boole):
        store.save(author)
    tag1 = AuthorTag(ada, "role", "pioneer")
    tag2 = AuthorTag(ada, "field", "computing")
    tag3 = AuthorTag(boole, "field", "logic")
    for tag in (tag1, tag2, tag3):
        store.save(tag)
    assert store.referring(AuthorTag, field="author", to=ada) == [tag1, tag2]
    assert store.referring(AuthorTag, field="author", to=boole) == [tag3]


def test_referring_rejects_unknown_object(store_factory):
    with pytest.raises(ValueError, match="has not been stored"):
        store_factory().referring(AuthorTag, field="author", to=Author("New", 1900))


def test_referring_rejects_non_reference_field_and_wrong_target(store_factory):
    store = store_factory()
    ada = Author("Ada", 1852)
    store.save(ada)
    _assert_named_exception(lambda: store.referring(AuthorTag, field="tag", to=ada), "SchemaError")
    _assert_named_exception(lambda: store.referring(AuthorTag, field="author", to=LogEvent("x")), "SchemaError")


def test_fetch_by_content_id_found_and_missing(store_factory):
    store = store_factory()
    ada = Author("Ada", 1852)
    store.save(ada)
    assert store.fetch_by_content_id(Author, content_id(ada)) == ada
    assert store.fetch_by_content_id(Author, "0" * 64) is None


def test_fetch_by_content_id_rejects_other_policies(store_factory):
    _assert_named_exception(
        lambda: store_factory().fetch_by_content_id(LogEvent, "0" * 64),
        "SchemaError",
    )


def test_projected_nested_save_calls_each_projection_once(store_factory):
    source = _root()
    expected_content_id = content_id(source)
    _calls.clear()
    store = store_factory()
    sid = store.save(source)
    fetched = store.fetch(RootRecord, sid)
    assert fetched.name == source.name
    assert fetched.primary == LeafRecord(1, "leaf metadata")
    assert fetched.related == [LeafRecord(1, "leaf metadata"), LeafRecord(2)]
    assert fetched.history == (LeafRecord(3),)
    assert _calls == {
        ("root", id(source)): 1,
        ("leaf", id(source.primary)): 1,
        ("leaf", id(source.related[1])): 1,
        ("leaf", id(source.history[0])): 1,
    }
    assert content_id(fetched) == expected_content_id


def test_identity_skipped_metadata_is_stored_and_checked(store_factory):
    source = _root()
    store = store_factory()
    sid = store.save(source)
    same_instant = replace(source, modified=source.modified.astimezone(datetime.UTC))
    assert store.save(same_instant) == sid
    assert store.fetch(RootRecord, sid).modified == source.modified.astimezone(datetime.UTC)
    with pytest.raises(EntryMetadataConflictError, match="modified"):
        store.save(replace(source, modified=source.modified + datetime.timedelta(seconds=1)))
    with pytest.raises(EntryMetadataConflictError, match="note"):
        store.save(replace(source, note="new metadata"))
    with pytest.raises(EntryMetadataConflictError, match="primary.note"):
        store.save(replace(source, primary=replace(source.primary, note="changed")))


def test_recursive_metadata_plan_descends_a_finite_chain(store_factory):
    leaf = RecursiveMetadataRecord(3, note="stored")
    middle = RecursiveMetadataRecord(2, leaf)
    root = RecursiveMetadataRecord(1, middle)
    conflicting = replace(root, child=replace(middle, child=replace(leaf, note="changed")))
    store = store_factory()
    store.save(root)
    with pytest.raises(EntryMetadataConflictError, match="child.child.note"):
        store.save(conflicting)


def test_identity_skipped_child_length_conflict_keeps_parent_detail(store_factory):
    store = store_factory()
    store.save(IdentityChildContainer([RecursiveNoMetadataRecord(1)]))
    with pytest.raises(EntryMetadataConflictError, match=r"children: stored .*received \[\]"):
        store.save(IdentityChildContainer([]))


def test_dedup_hit_does_not_reconstruct_graph_for_metadata(store_factory):
    MetadataProbeRecord.constructed = 0
    store = store_factory()
    store.save(MetadataProbeRecord(1, "metadata"))
    second = MetadataProbeRecord(1, "metadata")
    second_sid = store.save(second)
    assert second_sid == store.sid_of(second)
    assert MetadataProbeRecord.constructed == 2


def test_validate_hook_runs_once_for_an_exact_record_save(store_factory):
    ValidationProbeRecord.calls.clear()
    store = store_factory()
    store.save(ValidationProbeRecord(1))
    assert ValidationProbeRecord.calls == [1]


def test_validate_hook_skips_domain_sources_and_fetches(store_factory):
    ValidationProbeRecord.calls.clear()
    store = store_factory()
    sid = store.save(ValidationProbeRecord(1))
    ValidationProbeRecord.calls.clear()
    store.fetch(ValidationProbeRecord, sid)
    store.save(ValidationProbeView(2))
    assert ValidationProbeRecord.calls == []


def test_explicit_alternate_record_projection(store_factory):
    source = _root()
    store = store_factory()
    sid = store.save(source, as_record=SummaryRecord)
    assert store.fetch(SummaryRecord, sid) == SummaryRecord("one", 1)
    assert store.sid_of(_root(), as_record=SummaryRecord) == sid


def test_reference_save_uses_declared_record_target(store_factory):
    source = _root()
    store = store_factory()
    default_sid = store.save(source)
    store.save(_root("other"), as_record=SummaryRecord)
    target_sid = store.save(source, as_record=SummaryRecord)
    reference_sid = store.save(SummaryReference(source, "match"))
    assert default_sid != target_sid
    assert store.fetch(SummaryReference, reference_sid).target == SummaryRecord("one", 1)


def test_non_weakrefable_projected_source_uses_database_sid_lookup(store_factory):
    source = (7,)
    store = store_factory()
    sid = store.save(source, as_record=TupleRecord)
    assert store.fetch(TupleRecord, sid) == TupleRecord(7)
    assert store.sid_of(source, as_record=TupleRecord) == sid


def test_identity_skipped_recursive_link_reports_traversal_path(store_factory):
    record = CycleRecord("cycle")
    object.__setattr__(record, "link", record)
    with pytest.raises(StorageProjectionCycleError) as caught:
        store_factory().save(record)
    assert caught.value.path == "link"


def test_datetime_metadata_containers_compare_utc_instants(store_factory):
    zone = ZoneInfo("America/New_York")
    first = datetime.datetime(2026, 11, 1, 1, 30, tzinfo=zone, fold=0)
    second = datetime.datetime(2026, 11, 1, 1, 30, tzinfo=zone, fold=1)
    store = store_factory()
    sid = store.save(FoldMetadata("fold", (first,)))
    assert store.save(FoldMetadata("fold", (first.astimezone(datetime.UTC),))) == sid
    with pytest.raises(EntryMetadataConflictError, match="instants"):
        store.save(FoldMetadata("fold", (second,)))


def test_concrete_record_save_round_trips(store_factory):
    record = RootRecord(
        "record",
        LeafRecord(4),
        [LeafRecord(5)],
        (LeafRecord(6),),
        datetime.datetime(2026, 8, 1, tzinfo=datetime.UTC),
    )
    store = store_factory()
    sid = store.save(record)
    assert store_factory.reopen(store).fetch(RootRecord, sid) == record


def test_file_entry_dispatch_round_trips(store_factory):
    store = store_factory(entry_records={FileEntry: FileRecord})
    record = FileRecord(
        url="https://example.org/files/data.json",
        name="data.json",
        size=1234,
        media_type="application/json",
        description="Example data",
        sha256="a" * 64,
        immutable_id="file-immutable",
        last_modified=datetime.datetime(2026, 8, 1, 12, 0, tzinfo=datetime.UTC),
        checksums={"sha256": "a" * 64},
    )

    class DomainFile(FileRecord):
        pass

    source = DomainFile(**record.__dict__)
    sid = store.save(source, as_record=FileRecord)
    fetched = store_factory.reopen(store).fetch_entry(FileEntry, record.id)
    assert fetched == replace(record, checksums=None)
    assert fetched.url == record.url
    assert fetched.name == record.name
    assert fetched.size == record.size
    assert fetched.media_type == record.media_type
    assert fetched.description == record.description
    assert fetched.immutable_id == record.immutable_id
    assert fetched.last_modified == record.last_modified
    assert fetched.sha256 == record.sha256
    assert fetched.checksums is None
    with pytest.raises(EntryMetadataConflictError):
        store.save(replace(record, immutable_id="different"))
    assert store.save(record) == sid
    other = replace(record, url="https://mirror.example.org/files/data.json")
    assert store.save(other) != sid
    assert other.id != record.id


# Keep this record declaration local to the behavior module: it is used only
# to exercise alternate projection identity and has no SQL-specific surface.
@dataclass(frozen=True)
class TupleRecord:
    __httk_canonical_source__: ClassVar[type] = tuple

    value: int

    @classmethod
    def __httk_project__(cls, source: tuple[int, ...]) -> Mapping[str, object]:
        return {"value": source[0]}
