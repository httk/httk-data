"""Phase 4 versioned-mode query filtering: current view by default, forward exemption.

A scoped searcher (the default) shows one consistent time slice entered from the
top: the current view, or the half-open ``[ts_start, ts_end)`` slice named by
``as_of``. Family-table roots get an interval predicate; non-family
(deduplicated dependency/content) roots get the derived
``role = 1 OR EXISTS(ownership chain to a live family row)`` form. Forward
reference traversals and child joins are never lifecycle-filtered, so a pinned
reference to a row later superseded in its own entry role stays a correct member
of the aggregate. ``scoped=False`` disables all injection (an ``as_of`` cutoff on
the root still applies).
"""

import contextlib
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Annotated, ClassVar

import pytest
from httk.core.storage import StorageInfo, Unique

from httk.store.db import (
    Database,
    LifecycleScopeError,
    SqlStore,
    StoreEntryProvider,
    StoredEntryFederation,
    StoredEntrySource,
    optimade_filter_searcher,
)
from httk.store import PageOrder, PaginationCursorError
from httk.store.storage_layout import EntryFamilyDeclaration, EntryRecordDeclaration

# --------------------------------------------------------------------- records


@dataclass(frozen=True)
class Datablock:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(dedup="content_id", storage_name="q_datablock")

    payload: str


@dataclass(frozen=True)
class Item:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(dedup="content_id", storage_name="q_item")

    block: Datablock
    tag: str


@dataclass(frozen=True)
class Entry:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(dedup="content_id", storage_name="q_entry")

    name: Annotated[str, Unique()]
    note: str
    block: Datablock
    items: list[Item]


class EntryFamily:
    """Application-owned single-backing family for the versioned entry."""


ENTRY_LAYOUT = EntryFamilyDeclaration(
    name="q-entry-family",
    family=EntryFamily,
    records=(EntryRecordDeclaration(name="q-entry", record=Entry),),
)


# Two-family schema: F1 references F2, both lifecycle-bearing.
@dataclass(frozen=True)
class F2:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(dedup="content_id", storage_name="q_f2")

    name: Annotated[str, Unique()]
    tag: str


@dataclass(frozen=True)
class F1:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(dedup="content_id", storage_name="q_f1")

    name: Annotated[str, Unique()]
    other: F2


class Fam1:
    """Application-owned single-backing family for F1."""


class Fam2:
    """Application-owned single-backing family for F2."""


TWO_FAMILY = (
    EntryFamilyDeclaration(name="q-f1-family", family=Fam1, records=(EntryRecordDeclaration(name="q-f1", record=F1),)),
    EntryFamilyDeclaration(name="q-f2-family", family=Fam2, records=(EntryRecordDeclaration(name="q-f2", record=F2),)),
)


# Cyclic non-family schema reachable from a family.
@dataclass(frozen=True)
class CycA:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(dedup="content_id", storage_name="q_cyca")

    tag: str
    b: "CycB | None" = None


@dataclass(frozen=True)
class CycB:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(dedup="content_id", storage_name="q_cycb")

    tag: str
    a: CycA | None = None


@dataclass(frozen=True)
class CycEntry:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(dedup="content_id", storage_name="q_cycentry")

    name: Annotated[str, Unique()]
    a: CycA


class CycFamily:
    """Application-owned family anchoring the reference cycle."""


CYC_LAYOUT = EntryFamilyDeclaration(
    name="q-cyc-family",
    family=CycFamily,
    records=(EntryRecordDeclaration(name="q-cyc", record=CycEntry),),
)


# --------------------------------------------------------------------- backends


@contextlib.contextmanager
def _database(backend: str) -> Iterator[Database]:
    if backend == "sqlite":
        with Database.sqlite() as database:
            yield database
    else:
        with Database.duckdb() as database:
            yield database


_BACKENDS = ["sqlite", "duckdb"]


def _at(store: SqlStore, ns: int) -> None:
    store._clock = lambda: ns


def _entry_store(database: Database) -> SqlStore:
    return SqlStore(database, entry_families=(ENTRY_LAYOUT,), store_timestamps="versioned")


def _payloads(store: SqlStore, *, as_of: object = None, scoped: bool = True) -> list[str]:
    searcher = store.searcher(as_of=as_of, scoped=scoped)
    variable = searcher.variable(Datablock)
    searcher.output(variable, "d")
    return sorted(row[0][0].payload for row in searcher)


def _notes(store: SqlStore, *, as_of: object = None, scoped: bool = True) -> list[str]:
    searcher = store.searcher(as_of=as_of, scoped=scoped)
    variable = searcher.variable(Entry)
    searcher.output(variable, "e")
    return sorted(row[0][0].note for row in searcher)


# --------------------------------------------------------------------- family root


@pytest.mark.parametrize("backend", _BACKENDS)
def test_family_root_current_and_as_of_boundary(backend: str) -> None:
    with _database(backend) as database:
        store = _entry_store(database)
        block = Datablock("shared")
        _at(store, 1_000_000)
        store.save(Entry(name="e", note="v1", block=block, items=[]))
        _at(store, 2_000_000)
        store.replace(
            Entry(name="e", note="v1", block=block, items=[]), Entry(name="e", note="v2", block=block, items=[])
        )

        assert _notes(store) == ["v2"]  # default: current only
        assert _notes(store, as_of=1_500_000) == ["v1"]  # before replace
        assert _notes(store, as_of=2_000_000) == ["v2"]  # half-open boundary: successor visible
        assert _notes(store, as_of=3_000_000) == ["v2"]  # after replace
        assert _notes(store, scoped=False) == ["v1", "v2"]  # escape hatch: both
        assert _notes(store, scoped=False, as_of=1_500_000) == ["v1"]  # cutoff still applies


# --------------------------------------------------------------------- non-family root


@pytest.mark.parametrize("backend", _BACKENDS)
def test_non_family_root_ownership_and_role(backend: str) -> None:
    with _database(backend) as database:
        store = _entry_store(database)
        _at(store, 1_000_000)
        a_sid = store.save(
            Entry(name="e1", note="v1", block=Datablock("shared"), items=[Item(Datablock("ia_only"), "x")])
        )
        store.save(Entry(name="e2", note="c", block=Datablock("shared"), items=[Item(Datablock("ic"), "z")]))
        _at(store, 2_000_000)
        store.replace(
            Entry(name="e1", note="v1", block=Datablock("shared"), items=[Item(Datablock("ia_only"), "x")]),
            Entry(name="e1", note="v2", block=Datablock("db_only"), items=[Item(Datablock("ib"), "y")]),
        )
        assert a_sid  # the superseded row still exists

        # shared: owned by current e2 (and superseded e1) -> visible.
        # db_only: owned by current e1(v2) -> visible.
        # ib/ic: owned via a current entry's Item child -> visible (two-hop path).
        # ia_only: owned only via superseded e1's Item child -> hidden.
        assert _payloads(store) == ["db_only", "ib", "ic", "shared"]

        # Standalone save promotes the role: visible with no live owner.
        _at(store, 3_000_000)
        store.save(Datablock("standalone"))
        assert "standalone" in _payloads(store)
        # ... but hidden as_of before its standalone ts_start (role term cutoff).
        assert "standalone" not in _payloads(store, as_of=2_500_000)

        # scoped=False shows every dependency row, superseded owners included.
        assert "ia_only" in _payloads(store, scoped=False)


# --------------------------------------------------------------------- forward exemption


@pytest.mark.parametrize("backend", _BACKENDS)
def test_forward_traversal_is_never_lifecycle_filtered(backend: str) -> None:
    with _database(backend) as database:
        store = SqlStore(database, entry_families=TWO_FAMILY, store_timestamps="versioned")
        _at(store, 1_000_000)
        store.save(F2(name="k", tag="t1"))
        store.save(F1(name="m", other=F2(name="k", tag="t1")))
        _at(store, 2_000_000)
        store.replace(F2(name="k", tag="t1"), F2(name="k", tag="t2"))

        # F2 root scoped: only the current successor.
        f2_searcher = store.searcher()
        f2_variable = f2_searcher.variable(F2)
        f2_searcher.output(f2_variable, "f")
        assert sorted(row[0][0].tag for row in f2_searcher) == ["t2"]

        # F1 scoped, traversing forward to the SUPERSEDED F2 by its old tag: still matches.
        hit = store.searcher()
        f1 = hit.variable(F1)
        hit.add(f1.other.tag == "t1")
        hit.output(f1, "f")
        assert [row[0][0].name for row in hit] == ["m"]

        # Sanity: F1 pins the old F2 row, so the current tag does not match.
        miss = store.searcher()
        f1b = miss.variable(F1)
        miss.add(f1b.other.tag == "t2")
        miss.output(f1b, "f")
        assert [row[0][0].name for row in miss] == []


# --------------------------------------------------------------------- cyclic schema


def test_cyclic_non_family_scoped_raises_but_unscoped_works() -> None:
    with Database.sqlite() as database:
        store = SqlStore(database, entry_families=(CYC_LAYOUT,), store_timestamps="versioned")
        store.save(CycEntry(name="c", a=CycA(tag="x")))

        searcher = store.searcher()
        with pytest.raises(LifecycleScopeError, match="scoped=False"):
            searcher.variable(CycA)

        unscoped = store.searcher(scoped=False)
        variable = unscoped.variable(CycA)
        unscoped.output(variable, "a")
        assert [row[0][0].tag for row in unscoped] == ["x"]


# --------------------------------------------------------------------- referring


@pytest.mark.parametrize("backend", _BACKENDS)
def test_referring_current_view_and_as_of(backend: str) -> None:
    with _database(backend) as database:
        store = _entry_store(database)
        block = Datablock("shared")
        _at(store, 1_000_000)
        store.save(Entry(name="e1", note="v1", block=block, items=[]))
        store.save(Entry(name="e2", note="c", block=block, items=[]))
        _at(store, 2_000_000)
        store.replace(
            Entry(name="e1", note="v1", block=block, items=[]),
            Entry(name="e1", note="v2", block=block, items=[]),
        )

        current = sorted(e.note for e in store.referring(Entry, field="block", to=block))
        assert current == ["c", "v2"]  # superseded e1(v1) hidden
        historic = sorted(e.note for e in store.referring(Entry, field="block", to=block, as_of=1_500_000))
        assert historic == ["c", "v1"]
        every = sorted(e.note for e in store.referring(Entry, field="block", to=block, scoped=False))
        assert every == ["c", "v1", "v2"]


# --------------------------------------------------------------------- entry provider


def test_entry_provider_hides_superseded() -> None:
    with Database.sqlite() as database:
        store = _entry_store(database)
        block = Datablock("shared")
        _at(store, 1_000_000)
        a_sid = store.save(Entry(name="e1", note="v1", block=block, items=[]))
        c_sid = store.save(Entry(name="e2", note="c", block=block, items=[]))
        _at(store, 2_000_000)
        b_sid = store.replace(
            Entry(name="e1", note="v1", block=block, items=[]),
            Entry(name="e1", note="v2", block=block, items=[]),
        )

        provider = StoreEntryProvider(store, {"entries": Entry, "blocks": Datablock})
        ids = {row["__id"] for row in provider.records("entries")}
        assert ids == {f"entries-{b_sid}", f"entries-{c_sid}"}
        assert f"entries-{a_sid}" not in ids

        rels = provider.relationships("entries")
        assert f"entries-{a_sid}" not in rels  # no edges FROM the superseded entry
        assert set(rels) == {f"entries-{b_sid}", f"entries-{c_sid}"}


# --------------------------------------------------------------------- optimade


def test_optimade_family_current_view() -> None:
    with Database.sqlite() as database:
        store = _entry_store(database)
        block = Datablock("shared")
        _at(store, 1_000_000)
        store.save(Entry(name="e", note="v1", block=block, items=[]))
        _at(store, 2_000_000)
        store.replace(
            Entry(name="e", note="v1", block=block, items=[]),
            Entry(name="e", note="v2", block=block, items=[]),
        )
        assert [item[0][0].note for item in optimade_filter_searcher(store, Entry, '_httk_custom_note = "v2"')] == [
            "v2"
        ]
        assert list(optimade_filter_searcher(store, Entry, '_httk_custom_note = "v1"')) == []


def test_optimade_related_semijoin_uses_pinned_superseded_target() -> None:
    with Database.sqlite() as database:
        store = SqlStore(database, entry_families=TWO_FAMILY, store_timestamps="versioned")
        _at(store, 1_000_000)
        store.save(F2(name="k", tag="t1"))
        store.save(F1(name="m", other=F2(name="k", tag="t1")))
        _at(store, 2_000_000)
        store.replace(F2(name="k", tag="t1"), F2(name="k", tag="t2"))

        # The related F2 row F1 pins was superseded in its own entry role; the
        # nested semi-join (scoped=False) still finds it, so F1 still matches.
        searcher = optimade_filter_searcher(
            store, F1, 'related._httk_custom_tag = "t1"', related_classes={"related": F2}
        )
        assert [item[0][0].name for item in searcher] == ["m"]

        # A direct F2 filter is current-view: the superseded tag matches nothing.
        assert list(optimade_filter_searcher(store, F2, '_httk_custom_tag = "t1"')) == []
        assert [item[0][0].tag for item in optimade_filter_searcher(store, F2, '_httk_custom_tag = "t2"')] == ["t2"]


# --------------------------------------------------------------------- ts_end column


@pytest.mark.parametrize("backend", _BACKENDS)
def test_ts_end_pseudo_column(backend: str) -> None:
    with _database(backend) as database:
        store = _entry_store(database)
        block = Datablock("shared")
        _at(store, 1_000_000)
        store.save(Entry(name="e", note="v1", block=block, items=[]))
        _at(store, 2_000_000)
        store.replace(
            Entry(name="e", note="v1", block=block, items=[]),
            Entry(name="e", note="v2", block=block, items=[]),
        )

        # ts_end IS NULL selects the current row of the lineage.
        current = store.searcher(scoped=False)
        cur = current.variable(Entry)
        current.add(cur.ts_end == None)  # noqa: E711
        current.output(cur, "e")
        assert sorted(row[0][0].note for row in current) == ["v2"]

        # An ordering comparison works on the superseded (closed) row.
        closed = store.searcher(scoped=False)
        clo = closed.variable(Entry)
        closed.add(clo.ts_end <= 2_000_000)
        closed.output(clo, "e")
        assert sorted(row[0][0].note for row in closed) == ["v1"]

        # A non-family variable in versioned mode has no ts_end.
        with pytest.raises(AttributeError, match="versioned family table"):
            _ = store.searcher(scoped=False).variable(Datablock).ts_end


def test_ts_end_pseudo_column_requires_versioned_mode() -> None:
    with Database.sqlite() as database:
        store = SqlStore(database, entry_families=(ENTRY_LAYOUT,), store_timestamps="creation")
        with pytest.raises(AttributeError, match="versioned family table"):
            _ = store.searcher().variable(Entry).ts_end


@pytest.mark.parametrize("backend", _BACKENDS)
def test_optimade_ts_end_unknown_and_ts_start_bound(backend: str) -> None:
    with _database(backend) as database:
        store = _entry_store(database)
        block = Datablock("shared")
        _at(store, 1_000_000)
        store.save(Entry(name="e", note="v1", block=block, items=[]))
        _at(store, 2_000_000)
        store.replace(
            Entry(name="e", note="v1", block=block, items=[]),
            Entry(name="e", note="v2", block=block, items=[]),
        )

        # IS UNKNOWN (ts_end IS NULL) matches the current view; scoped default
        # already hides the superseded row, so the current one is the only hit.
        unknown = optimade_filter_searcher(store, Entry, "_httk_ts_end IS UNKNOWN")
        assert [row[0][0].note for row in unknown] == ["v2"]

        # IS KNOWN (ts_end IS NOT NULL) matches nothing in the current view; the
        # only closed row is hidden by default scoping.
        assert list(optimade_filter_searcher(store, Entry, "_httk_ts_end IS KNOWN")) == []

        # A ts_start bound resolves against the current row.
        bound = optimade_filter_searcher(store, Entry, "_httk_ts_start <= 2000000")
        assert [row[0][0].note for row in bound] == ["v2"]


# --------------------------------------------------------------------- as_of forms


def test_as_of_forms_and_interval_semantics() -> None:
    from datetime import UTC, datetime

    with Database.sqlite() as database:
        store = _entry_store(database)
        block = Datablock("shared")
        _at(store, 1_000_000)
        store.save(Entry(name="e", note="v1", block=block, items=[]))
        _at(store, 3_000_000)
        store.replace(
            Entry(name="e", note="v1", block=block, items=[]),
            Entry(name="e", note="v2", block=block, items=[]),
        )
        for cutoff in (2_000_000, datetime(1970, 1, 1, 0, 0, 0, 2000, tzinfo=UTC), "1970-01-01T00:00:00.002000Z"):
            assert _notes(store, as_of=cutoff) == ["v1"]
        assert _notes(store, as_of=3_000_000) == ["v2"]  # boundary
        assert _notes(store, as_of=2_999_000) == ["v1"]


# --------------------------------------------------------------------- paging


def test_scoped_and_unscoped_plans_have_distinct_fingerprints() -> None:
    with Database.sqlite() as database:
        store = _entry_store(database)
        block = Datablock("shared")
        _at(store, 1_000_000)
        for name in ("a", "b", "c", "d"):
            store.save(Entry(name=name, note="v1", block=block, items=[]))
        _at(store, 2_000_000)
        store.replace(
            Entry(name="a", note="v1", block=block, items=[]),
            Entry(name="a", note="v2", block=block, items=[]),
        )

        order = (PageOrder("name"),)
        scoped = store.searcher()
        scoped_variable = scoped.variable(Entry)
        scoped_page = scoped.results(entry=scoped_variable, name=scoped_variable.name).page(size=2, order_by=order)
        assert [row.name for row in scoped_page.rows] == ["a", "b"]  # current 'a' (v2) still current
        token = scoped_page.next
        assert token is not None

        unscoped = store.searcher(scoped=False)
        unscoped_variable = unscoped.variable(Entry)
        unscoped_result = unscoped.results(entry=unscoped_variable, name=unscoped_variable.name)
        with pytest.raises(PaginationCursorError, match="different query"):
            unscoped_result.page(size=2, order_by=order, cursor=token)


# --------------------------------------------------------------------- federation


def test_stored_entry_federation_hides_superseded_and_honors_as_of() -> None:
    from test_db_stored_federation import FederatedCalculation, FederationFirst

    with Database.sqlite() as database:
        source = SqlStore(database, entry_records={FederatedCalculation: FederationFirst}, store_timestamps="versioned")
        source._clock = lambda: 1_000_000
        source.save(FederationFirst("old", None))
        source._clock = lambda: 2_000_000
        source.replace(FederationFirst("old", None), FederationFirst("new", None))

        federation = StoredEntryFederation((StoredEntrySource(source, FederatedCalculation, "src", "src:"),))
        assert [row["immutable_id"] for row in federation.query(sort=(("immutable_id", False),)).rows] == ["new"]
        assert [
            row["immutable_id"] for row in federation.query(as_of=1_500_000, sort=(("immutable_id", False),)).rows
        ] == ["old"]
        assert [
            row["immutable_id"] for row in federation.query(as_of=2_000_000, sort=(("immutable_id", False),)).rows
        ] == ["new"]


def test_stored_entry_federation_serves_ts_end() -> None:
    from test_db_stored_federation import FederatedCalculation, FederationFirst

    with Database.sqlite() as database:
        source = SqlStore(database, entry_records={FederatedCalculation: FederationFirst}, store_timestamps="versioned")
        source._clock = lambda: 1_000_000
        source.save(FederationFirst("old", None))
        source._clock = lambda: 2_000_000
        source.replace(FederationFirst("old", None), FederationFirst("new", None))

        federation = StoredEntryFederation((StoredEntrySource(source, FederatedCalculation, "src", "src:"),))
        # Declared _httk_ts_end: null on the current row (served response path).
        current = federation.query(sort=(("immutable_id", False),)).rows
        assert [(row["immutable_id"], row["_httk_ts_end"]) for row in current] == [("new", None)]
        # Its historic close time (ns-scaled) on the superseded row via as_of.
        historic = federation.query(as_of=1_500_000, sort=(("immutable_id", False),)).rows
        assert [(row["immutable_id"], row["_httk_ts_end"]) for row in historic] == [("old", 2_000_000)]


# --------------------------------------------------------------------- smoke / e2e


def test_cod_like_shared_datablock_flow() -> None:
    with Database.sqlite() as database:
        store = _entry_store(database)
        shared = Datablock("cif-shared")
        _at(store, 1_000_000)
        store.save(Entry(name="cod-1", note="rev1", block=shared, items=[Item(Datablock("aux"), "old")]))
        _at(store, 2_000_000)
        store.replace(
            Entry(name="cod-1", note="rev1", block=shared, items=[Item(Datablock("aux"), "old")]),
            Entry(name="cod-1", note="rev2", block=shared, items=[Item(Datablock("aux2"), "new")]),
        )

        assert _notes(store) == ["rev2"]
        assert _notes(store, as_of=1_500_000) == ["rev1"]
        assert _notes(store, as_of=2_000_000) == ["rev2"]  # boundary
        assert _notes(store, scoped=False) == ["rev1", "rev2"]

        # The shared datablock survives (still owned by the current revision);
        # the auxiliary block dropped by the new revision is gone from the view.
        payloads = _payloads(store)
        assert "cif-shared" in payloads and "aux2" in payloads
        assert "aux" not in payloads
        assert "aux" in _payloads(store, scoped=False)
