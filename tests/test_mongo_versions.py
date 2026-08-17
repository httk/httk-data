"""Phase 7 versioned-mode parity coverage for the MongoDB backend.

Mirrors the essential SQL cases (``test_versions_replace.py`` /
``test_versions_query.py`` / ``test_versions_fsck.py``) for MongoStore: the
``replace`` verb and revive protection, scoped query filtering (family interval,
non-family role-or-reachability, forward exemption), ``scoped=False``, the
``ts_end`` pseudo-column, the OPTIMADE current-view/semi-join behavior, and the
fsck version invariants. Every test skips cleanly without a configured MongoDB
test server; ``replace`` and the versioned save-guards need multi-document
transactions, so those tests skip on a standalone (non-replica-set) server.
"""

from dataclasses import dataclass
from typing import Annotated, ClassVar

import pytest
from httk.core.storage import StorageInfo, Unique, content_id

from httk.store.db import LifecycleScopeError, RecordReviveError, RecordSupersededError
from httk.store.storage_layout import EntryFamilyDeclaration, EntryRecordDeclaration

pymongo = pytest.importorskip("pymongo")

# --------------------------------------------------------------------- records


@dataclass(frozen=True)
class Rec:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(dedup="content_id", storage_name="mv_rec")

    name: Annotated[str, Unique()]
    payload: str


class RecFamily:
    """Application-owned single-backing family."""


REC_LAYOUT = EntryFamilyDeclaration(
    name="mv-rec-family",
    family=RecFamily,
    records=(EntryRecordDeclaration(name="mv-rec", record=Rec),),
)


@dataclass(frozen=True)
class RecA:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(dedup="content_id", storage_name="mv_ma")

    value: str


@dataclass(frozen=True)
class RecB:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(dedup="content_id", storage_name="mv_mb")

    number: int


class MultiFamily:
    """Application-owned two-backing family."""


MULTI_LAYOUT = EntryFamilyDeclaration(
    name="mv-multi-family",
    family=MultiFamily,
    records=(
        EntryRecordDeclaration(name="mv-multi-a", record=RecA),
        EntryRecordDeclaration(name="mv-multi-b", record=RecB),
    ),
)


@dataclass(frozen=True)
class NonFamily:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(dedup="content_id", storage_name="mv_nonfam")

    x: str


@dataclass(frozen=True)
class Datablock:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(dedup="content_id", storage_name="mv_datablock")

    payload: str


@dataclass(frozen=True)
class Item:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(dedup="content_id", storage_name="mv_item")

    block: Datablock
    tag: str


@dataclass(frozen=True)
class Entry:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(dedup="content_id", storage_name="mv_entry")

    name: Annotated[str, Unique()]
    note: str
    block: Datablock
    items: list[Item]


class EntryFamily:
    """Application-owned single-backing family for the versioned entry."""


ENTRY_LAYOUT = EntryFamilyDeclaration(
    name="mv-entry-family",
    family=EntryFamily,
    records=(EntryRecordDeclaration(name="mv-entry", record=Entry),),
)


@dataclass(frozen=True)
class F2:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(dedup="content_id", storage_name="mv_f2")

    name: Annotated[str, Unique()]
    tag: str


@dataclass(frozen=True)
class F1:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(dedup="content_id", storage_name="mv_f1")

    name: Annotated[str, Unique()]
    other: F2


class Fam1:
    """Application-owned single-backing family for F1."""


class Fam2:
    """Application-owned single-backing family for F2."""


TWO_FAMILY = (
    EntryFamilyDeclaration(name="mv-f1-family", family=Fam1, records=(EntryRecordDeclaration(name="mv-f1", record=F1),)),
    EntryFamilyDeclaration(name="mv-f2-family", family=Fam2, records=(EntryRecordDeclaration(name="mv-f2", record=F2),)),
)


@dataclass(frozen=True)
class CycA:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(dedup="content_id", storage_name="mv_cyca")

    tag: str
    b: "CycB | None" = None


@dataclass(frozen=True)
class CycB:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(dedup="content_id", storage_name="mv_cycb")

    tag: str
    a: CycA | None = None


@dataclass(frozen=True)
class CycEntry:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(dedup="content_id", storage_name="mv_cycentry")

    name: Annotated[str, Unique()]
    a: CycA


class CycFamily:
    """Application-owned family anchoring the reference cycle."""


CYC_LAYOUT = EntryFamilyDeclaration(
    name="mv-cyc-family",
    family=CycFamily,
    records=(EntryRecordDeclaration(name="mv-cyc", record=CycEntry),),
)


# --------------------------------------------------------------------- helpers


def _store(mongo_test_database, families, *, mode="versioned"):
    from httk.store.mongo import MongoStore

    return MongoStore(mongo_test_database, entry_families=families, store_timestamps=mode)


def _versioned(mongo_test_database, families):
    """Build a versioned store, skipping when the server lacks transactions."""
    store = _store(mongo_test_database, families)
    if not store._database.supports_transactions:
        pytest.skip("versioned replace requires MongoDB multi-document transactions (a replica set)")
    return store


def _at(store, ns):
    store._clock = lambda: ns


def _doc(store, collection, sid):
    return store._database.database[collection].find_one({"_id": sid})


def _notes(store, *, as_of=None, scoped=True):
    searcher = store.searcher(as_of=as_of, scoped=scoped)
    variable = searcher.variable(Entry)
    searcher.output(variable, "e")
    return sorted(row[0][0].note for row in searcher)


def _payloads(store, *, as_of=None, scoped=True):
    searcher = store.searcher(as_of=as_of, scoped=scoped)
    variable = searcher.variable(Datablock)
    searcher.output(variable, "d")
    return sorted(row[0][0].payload for row in searcher)


# --------------------------------------------------------------------- replace: happy path


def test_replace_closes_old_and_opens_successor(mongo_test_database):
    store = _versioned(mongo_test_database, (REC_LAYOUT,))
    _at(store, 1_000_000)
    old_sid = store.save(Rec(name="x", payload="v1"))
    _at(store, 2_000_000)
    new_sid = store.replace(Rec(name="x", payload="v1"), Rec(name="x", payload="v2"))

    old_doc = _doc(store, "mv_rec", old_sid)
    new_doc = _doc(store, "mv_rec", new_sid)
    # Half-open intervals abut exactly: old.ts_end == new.ts_start.
    assert old_doc["ts_end"] == new_doc["ts_start"]
    assert old_doc["replaced_by_sid"] == new_sid
    assert new_doc["ts_end"] is None and new_doc["replaced_by_sid"] is None
    assert store.fetch(Rec, old_sid).payload == "v1"
    assert store.fetch(Rec, new_sid).payload == "v2"


def test_replace_keeps_same_unique_key(mongo_test_database):
    store = _versioned(mongo_test_database, (REC_LAYOUT,))
    _at(store, 1_000_000)
    store.save(Rec(name="same", payload="v1"))
    _at(store, 2_000_000)
    store.replace(Rec(name="same", payload="v1"), Rec(name="same", payload="v2"))
    collection = store._database.database["mv_rec"]
    ends = sorted((document["ts_start"], document["ts_end"]) for document in collection.find({}))
    assert ends == [(1000, 2000), (2000, None)]


# --------------------------------------------------------------------- target resolution


def test_replace_by_instance_content_id_and_sid_single_backing(mongo_test_database):
    store = _versioned(mongo_test_database, (REC_LAYOUT,))
    _at(store, 1_000_000)
    store.save(Rec(name="a", payload="v1"))
    _at(store, 2_000_000)
    store.replace(Rec(name="a", payload="v1"), Rec(name="a", payload="v2"))
    _at(store, 3_000_000)
    cid = content_id(Rec(name="a", payload="v2"), as_record=Rec)
    store.replace(cid, Rec(name="a", payload="v3"))
    _at(store, 4_000_000)
    current = store.sid_of(Rec(name="a", payload="v3"))
    assert current is not None
    store.replace(current, Rec(name="a", payload="v4"))
    ends = sorted(document["ts_end"] for document in store._database.database["mv_rec"].find({}))
    assert ends == [None, 2000, 3000, 4000]


def test_replace_by_sid_on_multi_backing_is_ambiguous(mongo_test_database):
    store = _versioned(mongo_test_database, (MULTI_LAYOUT,))
    _at(store, 1_000_000)
    sid = store.save(RecA(value="hi"))
    _at(store, 2_000_000)
    with pytest.raises(ValueError, match="ambiguous by sid"):
        store.replace(sid, RecB(number=3))


def test_replace_content_id_resolves_via_multi_backing_dispatch(mongo_test_database):
    store = _versioned(mongo_test_database, (MULTI_LAYOUT,))
    _at(store, 1_000_000)
    store.save(RecA(value="hello"))
    _at(store, 2_000_000)
    cid = content_id(RecA(value="hello"), as_record=RecA)
    store.replace(cid, RecB(number=7))
    document = store._database.database["mv_ma"].find_one({})
    assert document is not None and document["ts_end"] == 2000


def test_replace_unknown_target_raises(mongo_test_database):
    store = _versioned(mongo_test_database, (REC_LAYOUT,))
    _at(store, 1_000_000)
    store.save(Rec(name="a", payload="v1"))
    _at(store, 2_000_000)
    with pytest.raises(ValueError, match="not stored"):
        store.replace("no-such-content-id", Rec(name="a", payload="v2"))
    with pytest.raises(ValueError, match="not present"):
        store.replace(999, Rec(name="a", payload="v2"))


# --------------------------------------------------------------------- superseded / revive


def test_replacing_a_superseded_target_raises(mongo_test_database):
    store = _versioned(mongo_test_database, (REC_LAYOUT,))
    _at(store, 1_000_000)
    old_sid = store.save(Rec(name="x", payload="v1"))
    _at(store, 2_000_000)
    store.replace(Rec(name="x", payload="v1"), Rec(name="x", payload="v2"))
    _at(store, 3_000_000)
    with pytest.raises(RecordSupersededError):
        store.replace(old_sid, Rec(name="x", payload="v3"))


def test_double_replace_of_same_target_second_raises_superseded(mongo_test_database):
    store = _versioned(mongo_test_database, (REC_LAYOUT,))
    _at(store, 1_000_000)
    old_sid = store.save(Rec(name="x", payload="v1"))
    _at(store, 2_000_000)
    store.replace(old_sid, Rec(name="x", payload="v2"))
    _at(store, 3_000_000)
    with pytest.raises(RecordSupersededError):
        store.replace(old_sid, Rec(name="x", payload="v3"))


def test_replace_onto_superseded_content_raises_revive(mongo_test_database):
    store = _versioned(mongo_test_database, (REC_LAYOUT,))
    _at(store, 1_000_000)
    store.save(Rec(name="x", payload="v1"))
    _at(store, 2_000_000)
    store.replace(Rec(name="x", payload="v1"), Rec(name="x", payload="v2"))
    _at(store, 3_000_000)
    with pytest.raises(RecordReviveError):
        store.replace(Rec(name="x", payload="v2"), Rec(name="x", payload="v1"))


def test_plain_save_of_superseded_content_raises_revive(mongo_test_database):
    store = _versioned(mongo_test_database, (REC_LAYOUT,))
    _at(store, 1_000_000)
    store.save(Rec(name="x", payload="v1"))
    _at(store, 2_000_000)
    store.replace(Rec(name="x", payload="v1"), Rec(name="x", payload="v2"))
    _at(store, 3_000_000)
    with pytest.raises(RecordReviveError):
        store.save(Rec(name="x", payload="v1"))


def test_replace_with_content_identical_to_target_raises_revive(mongo_test_database):
    store = _versioned(mongo_test_database, (REC_LAYOUT,))
    _at(store, 1_000_000)
    store.save(Rec(name="x", payload="v1"))
    _at(store, 2_000_000)
    with pytest.raises(RecordReviveError):
        store.replace(Rec(name="x", payload="v1"), Rec(name="x", payload="v1"))


# --------------------------------------------------------------------- converge / transaction


def test_replace_converges_onto_existing_current_row(mongo_test_database):
    store = _versioned(mongo_test_database, (REC_LAYOUT,))
    _at(store, 1_000_000)
    a_sid = store.save(Rec(name="a", payload="pa"))
    c_sid = store.save(Rec(name="c", payload="pc"))
    _at(store, 2_000_000)
    result = store.replace(Rec(name="a", payload="pa"), Rec(name="c", payload="pc"))
    assert result == c_sid
    a_doc = _doc(store, "mv_rec", a_sid)
    c_doc = _doc(store, "mv_rec", c_sid)
    assert a_doc["ts_end"] == 2000 and a_doc["replaced_by_sid"] == c_sid
    assert c_doc["ts_end"] is None


def test_replace_shares_pinned_timestamp_inside_transaction(mongo_test_database):
    store = _versioned(mongo_test_database, (REC_LAYOUT,))
    _at(store, 1_000_000)
    old_sid = store.save(Rec(name="x", payload="v1"))
    _at(store, 2_000_000)
    with store.transaction():
        other = store.save(Rec(name="y", payload="w1"))
        new_sid = store.replace(Rec(name="x", payload="v1"), Rec(name="x", payload="v2"))
    other_doc = _doc(store, "mv_rec", other)
    new_doc = _doc(store, "mv_rec", new_sid)
    old_doc = _doc(store, "mv_rec", old_sid)
    assert other_doc["ts_start"] == new_doc["ts_start"] == old_doc["ts_end"]


def test_replace_rollback_leaves_target_current(mongo_test_database):
    store = _versioned(mongo_test_database, (REC_LAYOUT,))
    _at(store, 1_000_000)
    old_sid = store.save(Rec(name="x", payload="v1"))
    _at(store, 2_000_000)
    with pytest.raises(RuntimeError, match="boom"), store.transaction():
        store.replace(Rec(name="x", payload="v1"), Rec(name="x", payload="v2"))
        raise RuntimeError("boom")
    documents = list(store._database.database["mv_rec"].find({}))
    assert len(documents) == 1
    assert documents[0]["_id"] == old_sid and documents[0]["ts_end"] is None


def test_same_transaction_save_then_replace_is_zero_length(mongo_test_database):
    store = _versioned(mongo_test_database, (REC_LAYOUT,))
    _at(store, 1_000_000)
    with pytest.raises(ValueError, match="zero-length"), store.transaction():
        store.save(Rec(name="z", payload="v1"))
        store.replace(Rec(name="z", payload="v1"), Rec(name="z", payload="v2"))


# --------------------------------------------------------------------- rejections


def test_replace_refused_in_creation_mode(mongo_test_database):
    store = _store(mongo_test_database, (REC_LAYOUT,), mode="creation")
    with pytest.raises(ValueError, match="versioned"):
        store.replace(Rec(name="x", payload="v1"), Rec(name="x", payload="v2"))


def test_replace_refused_for_non_family_record(mongo_test_database):
    store = _versioned(mongo_test_database, (REC_LAYOUT,))
    with pytest.raises(ValueError, match="entry family"):
        store.replace("cid", NonFamily(x="q"))


# --------------------------------------------------------------------- unique-among-current


def test_save_second_current_unique_value_is_refused(mongo_test_database):
    store = _versioned(mongo_test_database, (REC_LAYOUT,))
    _at(store, 1_000_000)
    store.save(Rec(name="dup", payload="v1"))
    # A second CURRENT row with the same author-Unique name is refused in-tx.
    _at(store, 2_000_000)
    with pytest.raises(pymongo.errors.DuplicateKeyError):
        store.save(Rec(name="dup", payload="v2"))
    # After superseding the first, the same name is free again.
    store.replace(Rec(name="dup", payload="v1"), Rec(name="dup", payload="v2"))
    assert store.sid_of(Rec(name="dup", payload="v2")) is not None


# --------------------------------------------------------------------- scoping: family root


def test_family_root_current_and_as_of_boundary(mongo_test_database):
    store = _versioned(mongo_test_database, (ENTRY_LAYOUT,))
    block = Datablock("shared")
    _at(store, 1_000_000)
    store.save(Entry(name="e", note="v1", block=block, items=[]))
    _at(store, 2_000_000)
    store.replace(
        Entry(name="e", note="v1", block=block, items=[]), Entry(name="e", note="v2", block=block, items=[])
    )
    assert _notes(store) == ["v2"]
    assert _notes(store, as_of=1_500_000) == ["v1"]
    assert _notes(store, as_of=2_000_000) == ["v2"]  # half-open boundary
    assert _notes(store, as_of=3_000_000) == ["v2"]
    assert _notes(store, scoped=False) == ["v1", "v2"]
    assert _notes(store, scoped=False, as_of=1_500_000) == ["v1"]


# --------------------------------------------------------------------- scoping: non-family root


def test_non_family_root_ownership_and_role(mongo_test_database):
    store = _versioned(mongo_test_database, (ENTRY_LAYOUT,))
    _at(store, 1_000_000)
    store.save(Entry(name="e1", note="v1", block=Datablock("shared"), items=[Item(Datablock("ia_only"), "x")]))
    store.save(Entry(name="e2", note="c", block=Datablock("shared"), items=[Item(Datablock("ic"), "z")]))
    _at(store, 2_000_000)
    store.replace(
        Entry(name="e1", note="v1", block=Datablock("shared"), items=[Item(Datablock("ia_only"), "x")]),
        Entry(name="e1", note="v2", block=Datablock("db_only"), items=[Item(Datablock("ib"), "y")]),
    )
    # shared owned by current e2; db_only by current e1(v2); ib/ic via a current
    # entry's Item child (two-hop path); ia_only only via superseded e1 -> hidden.
    assert _payloads(store) == ["db_only", "ib", "ic", "shared"]

    _at(store, 3_000_000)
    store.save(Datablock("standalone"))
    assert "standalone" in _payloads(store)
    assert "standalone" not in _payloads(store, as_of=2_500_000)
    assert "ia_only" in _payloads(store, scoped=False)


# --------------------------------------------------------------------- forward exemption


def test_forward_traversal_is_never_lifecycle_filtered(mongo_test_database):
    store = _versioned(mongo_test_database, TWO_FAMILY)
    _at(store, 1_000_000)
    store.save(F2(name="k", tag="t1"))
    store.save(F1(name="m", other=F2(name="k", tag="t1")))
    _at(store, 2_000_000)
    store.replace(F2(name="k", tag="t1"), F2(name="k", tag="t2"))

    f2_searcher = store.searcher()
    f2_variable = f2_searcher.variable(F2)
    f2_searcher.output(f2_variable, "f")
    assert sorted(row[0][0].tag for row in f2_searcher) == ["t2"]

    hit = store.searcher()
    f1 = hit.variable(F1)
    hit.add(f1.other.tag == "t1")
    hit.output(f1, "f")
    assert [row[0][0].name for row in hit] == ["m"]

    miss = store.searcher()
    f1b = miss.variable(F1)
    miss.add(f1b.other.tag == "t2")
    miss.output(f1b, "f")
    assert [row[0][0].name for row in miss] == []


# --------------------------------------------------------------------- cyclic schema


def test_cyclic_non_family_scoped_raises_but_unscoped_works(mongo_test_database):
    store = _versioned(mongo_test_database, (CYC_LAYOUT,))
    store.save(CycEntry(name="c", a=CycA(tag="x")))

    searcher = store.searcher()
    with pytest.raises(LifecycleScopeError, match="scoped=False"):
        searcher.variable(CycA)

    unscoped = store.searcher(scoped=False)
    variable = unscoped.variable(CycA)
    unscoped.output(variable, "a")
    assert [row[0][0].tag for row in unscoped] == ["x"]


# --------------------------------------------------------------------- referring


def test_referring_current_view_default(mongo_test_database):
    store = _versioned(mongo_test_database, (ENTRY_LAYOUT,))
    block = Datablock("shared")
    _at(store, 1_000_000)
    store.save(Entry(name="e1", note="v1", block=block, items=[]))
    store.save(Entry(name="e2", note="c", block=block, items=[]))
    _at(store, 2_000_000)
    store.replace(
        Entry(name="e1", note="v1", block=block, items=[]), Entry(name="e1", note="v2", block=block, items=[])
    )
    current = sorted(e.note for e in store.referring(Entry, field="block", to=block))
    assert current == ["c", "v2"]  # superseded e1(v1) hidden


# --------------------------------------------------------------------- ts_end pseudo-column


def test_ts_end_pseudo_column(mongo_test_database):
    store = _versioned(mongo_test_database, (ENTRY_LAYOUT,))
    block = Datablock("shared")
    _at(store, 1_000_000)
    store.save(Entry(name="e", note="v1", block=block, items=[]))
    _at(store, 2_000_000)
    store.replace(
        Entry(name="e", note="v1", block=block, items=[]), Entry(name="e", note="v2", block=block, items=[])
    )

    current = store.searcher(scoped=False)
    cur = current.variable(Entry)
    current.add(cur.ts_end == None)  # noqa: E711
    current.output(cur, "e")
    assert sorted(row[0][0].note for row in current) == ["v2"]

    closed = store.searcher(scoped=False)
    clo = closed.variable(Entry)
    closed.add(clo.ts_end <= 2_000_000)
    closed.output(clo, "e")
    assert sorted(row[0][0].note for row in closed) == ["v1"]

    with pytest.raises(AttributeError, match="versioned family table"):
        _ = store.searcher(scoped=False).variable(Datablock).ts_end


def test_ts_end_pseudo_column_requires_versioned_mode(mongo_test_database):
    store = _store(mongo_test_database, (ENTRY_LAYOUT,), mode="creation")
    with pytest.raises(AttributeError, match="versioned family table"):
        _ = store.searcher().variable(Entry).ts_end


# --------------------------------------------------------------------- optimade


def test_optimade_family_current_view(mongo_test_database):
    from httk.store.mongo import optimade_filter_searcher

    store = _versioned(mongo_test_database, (ENTRY_LAYOUT,))
    block = Datablock("shared")
    _at(store, 1_000_000)
    store.save(Entry(name="e", note="v1", block=block, items=[]))
    _at(store, 2_000_000)
    store.replace(
        Entry(name="e", note="v1", block=block, items=[]), Entry(name="e", note="v2", block=block, items=[])
    )
    assert [item[0][0].note for item in optimade_filter_searcher(store, Entry, '_httk_custom_note = "v2"')] == ["v2"]
    assert list(optimade_filter_searcher(store, Entry, '_httk_custom_note = "v1"')) == []


def test_optimade_related_semijoin_uses_pinned_superseded_target(mongo_test_database):
    from httk.store.mongo import optimade_filter_searcher

    store = _versioned(mongo_test_database, TWO_FAMILY)
    _at(store, 1_000_000)
    store.save(F2(name="k", tag="t1"))
    store.save(F1(name="m", other=F2(name="k", tag="t1")))
    _at(store, 2_000_000)
    store.replace(F2(name="k", tag="t1"), F2(name="k", tag="t2"))

    searcher = optimade_filter_searcher(store, F1, 'related._httk_custom_tag = "t1"', related_classes={"related": F2})
    assert [item[0][0].name for item in searcher] == ["m"]
    assert list(optimade_filter_searcher(store, F2, '_httk_custom_tag = "t1"')) == []
    assert [item[0][0].tag for item in optimade_filter_searcher(store, F2, '_httk_custom_tag = "t2"')] == ["t2"]


def test_optimade_ts_end_unknown_and_known(mongo_test_database):
    from httk.store.mongo import optimade_filter_searcher

    store = _versioned(mongo_test_database, (ENTRY_LAYOUT,))
    block = Datablock("shared")
    _at(store, 1_000_000)
    store.save(Entry(name="e", note="v1", block=block, items=[]))
    _at(store, 2_000_000)
    store.replace(
        Entry(name="e", note="v1", block=block, items=[]), Entry(name="e", note="v2", block=block, items=[])
    )
    unknown = optimade_filter_searcher(store, Entry, "_httk_ts_end IS UNKNOWN")
    assert [row[0][0].note for row in unknown] == ["v2"]
    assert list(optimade_filter_searcher(store, Entry, "_httk_ts_end IS KNOWN")) == []


# --------------------------------------------------------------------- entry provider


def test_entry_provider_hides_superseded(mongo_test_database):
    from httk.store.mongo import StoreEntryProvider

    store = _versioned(mongo_test_database, (ENTRY_LAYOUT,))
    block = Datablock("shared")
    _at(store, 1_000_000)
    store.save(Entry(name="e1", note="v1", block=block, items=[]))
    store.save(Entry(name="e2", note="c", block=block, items=[]))
    _at(store, 2_000_000)
    store.replace(
        Entry(name="e1", note="v1", block=block, items=[]), Entry(name="e1", note="v2", block=block, items=[])
    )
    superseded = content_id(Entry(name="e1", note="v1", block=block, items=[]), as_record=Entry)
    current_b = content_id(Entry(name="e1", note="v2", block=block, items=[]), as_record=Entry)
    current_c = content_id(Entry(name="e2", note="c", block=block, items=[]), as_record=Entry)

    provider = StoreEntryProvider(store, {"entries": Entry, "blocks": Datablock})
    ids = {row["id"] for row in provider.records("entries")}
    assert ids == {current_b, current_c}
    assert superseded not in ids

    rels = provider.relationships("entries")
    assert superseded not in rels  # no edges FROM the superseded entry
    assert set(rels) == {current_b, current_c}


# --------------------------------------------------------------------- fsck invariants


def _replaced(store):
    old = store.save(Rec(name="x", payload="v1"))
    _at(store, 2_000_000)
    new = store.replace(Rec(name="x", payload="v1"), Rec(name="x", payload="v2"))
    return old, new


def test_clean_versioned_store_passes_fsck(mongo_test_database):
    store = _versioned(mongo_test_database, (REC_LAYOUT,))
    _at(store, 1_000_000)
    _replaced(store)
    assert store.fsck(repair=False, collect_garbage=False).violations == ()


def test_ts_end_without_replaced_by_is_reported(mongo_test_database):
    store = _versioned(mongo_test_database, (REC_LAYOUT,))
    _at(store, 1_000_000)
    old, _new = _replaced(store)
    store._database.database["mv_rec"].update_one({"_id": old}, {"$set": {"replaced_by_sid": None}})
    violations = store.fsck(repair=False, collect_garbage=False).violations
    assert violations == (f"collection 'mv_rec' sid {old} has ts_end 2000 but no replaced_by_sid",)


def test_replaced_by_without_ts_end_is_reported(mongo_test_database):
    store = _versioned(mongo_test_database, (REC_LAYOUT,))
    _at(store, 1_000_000)
    old, new = _replaced(store)
    store._database.database["mv_rec"].update_one({"_id": new}, {"$set": {"replaced_by_sid": old, "ts_end": None}})
    violations = store.fsck(repair=False, collect_garbage=False).violations
    assert violations == (f"collection 'mv_rec' sid {new} has replaced_by_sid {old} but no ts_end",)


def test_zero_length_interval_is_reported(mongo_test_database):
    store = _versioned(mongo_test_database, (REC_LAYOUT,))
    _at(store, 1_000_000)
    old, _new = _replaced(store)
    document = store._database.database["mv_rec"].find_one({"_id": old})
    store._database.database["mv_rec"].update_one({"_id": old}, {"$set": {"ts_end": document["ts_start"]}})
    violations = store.fsck(repair=False, collect_garbage=False).violations
    assert any("is not after ts_start" in violation for violation in violations)


def test_replaced_by_missing_sid_is_reported(mongo_test_database):
    store = _versioned(mongo_test_database, (REC_LAYOUT,))
    _at(store, 1_000_000)
    old, _new = _replaced(store)
    store._database.database["mv_rec"].update_one({"_id": old}, {"$set": {"replaced_by_sid": 999999}})
    violations = store.fsck(repair=False, collect_garbage=False).violations
    assert violations == (f"collection 'mv_rec' sid {old} replaced_by_sid 999999 references a missing row",)


def test_successor_starting_after_predecessor_closed_is_reported(mongo_test_database):
    store = _versioned(mongo_test_database, (REC_LAYOUT,))
    _at(store, 1_000_000)
    old, new = _replaced(store)
    store._database.database["mv_rec"].update_one({"_id": old}, {"$set": {"ts_end": 1500}})
    violations = store.fsck(repair=False, collect_garbage=False).violations
    assert violations == (
        f"collection 'mv_rec' sid {old} successor {new} ts_start 2000 is after ts_end 1500",
    )


def test_future_ts_end_is_reported_and_clamped(mongo_test_database):
    store = _versioned(mongo_test_database, (REC_LAYOUT,))
    store._clock = lambda: 1_000_000_000
    old, _new = _replaced(store)
    store._database.database["mv_rec"].update_one({"_id": old}, {"$set": {"ts_end": 10_000_000_000}})
    store._clock = lambda: 1_500_000_000
    reported = store.fsck(repair=False, collect_garbage=False).violations
    assert any("ts_end" in violation and "exceeds" in violation for violation in reported)
    clamped = store.fsck(repair=True, collect_garbage=False, clamp_future_timestamps=True).violations
    assert any("ts_end" in violation and "clamped" in violation for violation in clamped)
    document = store._database.database["mv_rec"].find_one({"_id": old})
    assert document["ts_end"] == 1_500_000


def test_duplicate_current_unique_value_reported(mongo_test_database):
    store = _versioned(mongo_test_database, (REC_LAYOUT,))
    _at(store, 1_000_000)
    orig = store.save(Rec(name="dup", payload="v1"))
    store._database.database["mv_rec"].insert_one(
        {
            "_id": 999,
            "_httk_role": "main",
            "ts_start": 500,
            "ts_end": None,
            "replaced_by_sid": None,
            "content_id": "0" * 64,
            "f": {"name": "dup", "payload": "v2"},
        }
    )
    violations = store.fsck(repair=False, collect_garbage=False).violations
    assert violations == ("collection 'mv_rec' field 'name' value 'dup' appears in 2 current rows",)
    # Close the injected duplicate: it no longer occupies a current slot.
    store._database.database["mv_rec"].update_one(
        {"_id": 999}, {"$set": {"ts_end": 1000, "replaced_by_sid": orig}}
    )
    assert store.fsck(repair=False, collect_garbage=False).violations == ()
