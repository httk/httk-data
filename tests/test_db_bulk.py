"""``bulk_ingest`` equivalence and contracts across SQLite and DuckDB.

Every test proves the bulk fast path against an ordinary ``save()`` loop or a
reopened store — both for a fresh empty-store build and for an incremental
append into a pre-populated store — so the buffered/pre-assigned-sid pipeline
stays observationally identical to the per-record path. MongoStore has no
``bulk_ingest`` and is skipped through the shared ``store_factory`` fixture.
"""

import datetime
from dataclasses import dataclass, replace
from fractions import Fraction
from typing import Annotated, ClassVar

import pytest
import sqlalchemy
from httk.core import FracScalar, FracVector
from httk.core.register import register_entry_family, register_entry_record
from httk.core.storage import IdentitySkip, Indexed, Shape, StorageInfo, content_id

from httk.data.db.mapping import CONTENT_ID_COLUMN
from httk.data.store_common import EntryDispatchIntegrityError, EntryMetadataConflictError

# --------------------------------------------------------------------- record classes


@dataclass(frozen=True)
class Author:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="bulk_author")

    name: str
    year: int


@dataclass(frozen=True)
class AuthorTag:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="bulk_author_tag", dedup="by_value")

    author: Author
    tag: str
    value: str


@dataclass(frozen=True)
class LogEvent:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="bulk_log_event", dedup="none")

    message: str


@dataclass(frozen=True)
class OptionalChildRoundTrip:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="bulk_optional_child", dedup="none")

    value: str
    notes: list[str] | None = None


@dataclass(frozen=True)
class Sample:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(
        storage_name="bulk_sample", indexes=(("formula", "spacegroup"),)
    )

    formula: Annotated[str, Indexed()]
    spacegroup: int
    energy: float
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


@dataclass(frozen=True)
class Leaf:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="bulk_leaf", identity_name="tests.bulk.Leaf")

    value: int
    note: Annotated[str | None, IdentitySkip()] = None


@dataclass(frozen=True)
class Root:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="bulk_root", identity_name="tests.bulk.Root")

    name: str
    primary: Leaf
    related: list[Leaf]
    modified: Annotated[datetime.datetime, IdentitySkip()]


@dataclass(frozen=True)
class ValidatedRecord:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="bulk_validated", dedup="none")

    value: int

    @classmethod
    def __httk_validate__(cls, source: "ValidatedRecord") -> None:
        if source.value < 0:
            raise ValueError(f"validator rejected value {source.value}")


class BulkCalcFamily:
    type = "bulk-calculations"


@dataclass(frozen=True)
class BulkCalcA:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="bulk_calc_a")

    label: str
    value: int


@dataclass(frozen=True)
class BulkCalcB:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="bulk_calc_b")

    label: str
    kind: str


register_entry_family(name="test-db-bulk-calculations", family=f"{__name__}:BulkCalcFamily")
register_entry_record(name="test-db-bulk-calc-a", family="test-db-bulk-calculations", record=f"{__name__}:BulkCalcA")
register_entry_record(name="test-db-bulk-calc-b", family="test-db-bulk-calculations", record=f"{__name__}:BulkCalcB")


@dataclass(frozen=True)
class Elem:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="bulk_elem")

    text: str


@dataclass(frozen=True)
class ByValParent:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="bulk_byval_parent", dedup="by_value")

    a: int
    elems: list[Elem]


@dataclass(frozen=True)
class NoneRec:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="bulk_none_rec", dedup="none")

    text: str


@dataclass(frozen=True)
class ContentParent:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="bulk_content_parent")

    name: str
    log: NoneRec


@dataclass(frozen=True)
class Node:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="bulk_node", dedup="by_value")

    val: int
    ref: "Node | None" = None


@dataclass(frozen=True)
class MetaScalar:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="bulk_meta_scalar")

    key: str
    note: Annotated[str, IdentitySkip()]


def make_sample(**overrides) -> Sample:
    sample = Sample(
        formula="CaTiO3",
        spacegroup=221,
        energy=-12.5,
        ratio=Fraction(1, 3),
        scale=FracScalar(2, 7),
        created=datetime.datetime(2026, 7, 24, 12, 30, 0),  # noqa: DTZ001
        cell=FracVector.create([[1, Fraction(1, 3), 0], [0, 1, 0], [0, 0, Fraction(2, 3)]]),
        coords=FracVector.create(
            [[0, 0, 0], [Fraction(1, 2), Fraction(1, 2), Fraction(1, 2)], [Fraction(1, 3), Fraction(2, 3), 1]]
        ),
        symbols=["Ca", "Ti", "O"],
        tags=("perovskite", "oxide"),
        ratios=[Fraction(1, 3), Fraction(-7, 5)],
        authors=[Author("Ada", 1852)],
        reference=Author("Boole", 1854),
    )
    return replace(sample, **overrides) if overrides else sample


def _root(name: str = "one", *, note: str | None = "leaf metadata") -> Root:
    shared = Leaf(1, note)
    return Root(
        name,
        shared,
        [shared, Leaf(2)],
        datetime.datetime(2026, 8, 1, 12, tzinfo=datetime.timezone(datetime.timedelta(hours=2))),
    )


def _stream() -> list[object]:
    """A mixed object stream: policies, references, children, optionals, family."""
    return [
        Author("Ada", 1852),
        Author("Ada", 1852),  # content_id collapse
        Author("Grace", 1906),
        AuthorTag(Author("Ada", 1852), "role", "pioneer"),
        AuthorTag(Author("Ada", 1852), "role", "pioneer"),  # by_value collapse
        AuthorTag(Author("Ada", 1852), "role", "mathematician"),
        LogEvent("started"),
        LogEvent("started"),  # none: two rows
        make_sample(),
        make_sample(),  # content_id collapse (children not duplicated)
        make_sample(formula="NaCl", weight=1.25, reference=None),
        OptionalChildRoundTrip("a", None),
        OptionalChildRoundTrip("b", []),
        OptionalChildRoundTrip("c", ["note"]),
        _root("one"),
        _root("one"),  # content_id collapse; identity-skipped metadata equal
        _root("two"),
        BulkCalcA("alpha", 1),
        BulkCalcB("beta", "kind-b"),
        BulkCalcA("alpha", 1),  # dispatch/content_id collapse
    ]


# --------------------------------------------------------------------- helpers


def _require_bulk(store) -> None:
    if not hasattr(store, "bulk_ingest"):
        pytest.skip("backend does not provide bulk_ingest")


def _app_tables(store) -> dict[str, sqlalchemy.Table]:
    """Registered non-marker record/dispatch tables of a store, by physical name."""
    return {name: table for name, table in store._metadata.tables.items() if not name.startswith("_httk_")}


def _table_stats(store, database) -> dict[str, tuple[int, frozenset[str] | None]]:
    """Per-table row counts and (for content-addressed tables) content_id sets."""
    stats: dict[str, tuple[int, frozenset[str] | None]] = {}
    with database.engine.connect() as connection:
        for name, table in _app_tables(store).items():
            count = connection.execute(sqlalchemy.text(f'SELECT count(*) FROM "{name}"')).scalar_one()
            content_ids: frozenset[str] | None = None
            if CONTENT_ID_COLUMN in table.c:
                content_ids = frozenset(
                    str(row[0]) for row in connection.execute(sqlalchemy.select(table.c[CONTENT_ID_COLUMN])).all()
                )
            stats[name] = (int(count), content_ids)
    return stats


def _database_of(store):
    return store._database


# --------------------------------------------------------------------- tests


@pytest.mark.parametrize("chunk_size", [100_000, 3])
def test_bulk_matches_save_loop(store_factory, chunk_size):
    """Bulk-built and save()-built stores agree on counts, content ids, and records."""
    save_store = store_factory(entry_records={BulkCalcFamily: (BulkCalcA, BulkCalcB)})
    _require_bulk(save_store)
    bulk_store = store_factory(entry_records={BulkCalcFamily: (BulkCalcA, BulkCalcB)})

    save_sids = [save_store.save(obj) for obj in _stream()]
    bulk_sids: list[int] = []
    with bulk_store.bulk_ingest(chunk_size=chunk_size) as bulk:
        for obj in _stream():
            bulk_sids.append(bulk.save(obj))

    assert bulk_sids == save_sids
    assert _table_stats(bulk_store, _database_of(bulk_store)) == _table_stats(save_store, _database_of(save_store))

    # Reconstructed-record equality through reopened stores.
    reopened = store_factory.reopen(bulk_store)
    assert reopened.fetch(Author, bulk_sids[0]) == Author("Ada", 1852)
    assert reopened.fetch(Sample, bulk_sids[8]) == make_sample()
    assert reopened.fetch(Sample, bulk_sids[10]) == make_sample(formula="NaCl", weight=1.25, reference=None)
    assert reopened.fetch(OptionalChildRoundTrip, bulk_sids[11]).notes is None
    assert reopened.fetch(OptionalChildRoundTrip, bulk_sids[12]).notes == []
    assert reopened.fetch(OptionalChildRoundTrip, bulk_sids[13]).notes == ["note"]
    assert reopened.fetch(Root, bulk_sids[14]) == Root(
        "one", Leaf(1, "leaf metadata"), [Leaf(1, "leaf metadata"), Leaf(2)], _root().modified
    )
    # fetch_entry resolves the multi-record family through its dispatch table.
    assert reopened.fetch_entry(BulkCalcFamily, content_id(BulkCalcA("alpha", 1))) == BulkCalcA("alpha", 1)
    assert reopened.fetch_entry(BulkCalcFamily, content_id(BulkCalcB("beta", "kind-b"))) == BulkCalcB("beta", "kind-b")


def test_bulk_returned_sids_match_fetch(store_factory):
    """Each sid returned by bulk .save reconstructs the object it was assigned to."""
    store = store_factory()
    _require_bulk(store)
    objects = [Author("Ada", 1852), Author("Grace", 1906), make_sample()]
    with store.bulk_ingest() as bulk:
        sids = [bulk.save(obj) for obj in objects]
    reopened = store_factory.reopen(store)
    assert reopened.fetch(Author, sids[0]) == Author("Ada", 1852)
    assert reopened.fetch(Author, sids[1]) == Author("Grace", 1906)
    assert reopened.fetch(Sample, sids[2]) == make_sample()


def test_bulk_intra_ingest_content_and_value_dedup(store_factory):
    """Equal content buffers one row; by_value collapses on parent columns only."""
    store = store_factory()
    _require_bulk(store)
    with store.bulk_ingest() as bulk:
        first = bulk.save(Author("Ada", 1852))
        again = bulk.save(Author("Ada", 1852))
        tag_a = bulk.save(AuthorTag(Author("Ada", 1852), "role", "pioneer"))
        tag_b = bulk.save(AuthorTag(Author("Ada", 1852), "role", "pioneer"))
        tag_c = bulk.save(AuthorTag(Author("Ada", 1852), "role", "mathematician"))
    assert first == again
    assert tag_a == tag_b != tag_c
    stats = _table_stats(store, _database_of(store))
    assert stats["bulk_author"][0] == 1
    assert stats["bulk_author_tag"][0] == 2


def test_bulk_content_id_metadata_conflict_raises(store_factory):
    """A content-id hit with mismatched identity-excluded metadata raises like save()."""
    store = store_factory()
    _require_bulk(store)
    with pytest.raises(EntryMetadataConflictError, match="modified"):
        with store.bulk_ingest() as bulk:
            bulk.save(_root("one"))
            bulk.save(replace(_root("one"), modified=_root().modified + datetime.timedelta(seconds=1)))
    # The failed ingest left nothing behind, so a fresh ingest proceeds cleanly.
    with store.bulk_ingest() as bulk:
        bulk.save(_root("one"))
        again = bulk.save(_root("one"))  # equal metadata: accepted and deduplicated
    assert store.fetch is not None  # store remains usable
    assert isinstance(again, int)


def test_successful_empty_bulk_clears_under_construction_marker(store_factory):
    """A successful empty-store ingest leaves only the required metadata keys."""
    store = store_factory()
    _require_bulk(store)
    with store.bulk_ingest() as bulk:
        bulk.save(Author("marker", 2026))
    database = _database_of(store)
    with database.engine.connect() as connection:
        keys = set(connection.execute(sqlalchemy.text("SELECT key FROM _httk_store_metadata")).scalars())
    assert keys == {"protocol", "entry_declaration"}


def test_bulk_nested_metadata_conflict_raises(store_factory):
    """The in-memory metadata comparison descends into references like save()."""
    store = store_factory()
    _require_bulk(store)
    with pytest.raises(EntryMetadataConflictError, match="primary.note"):
        with store.bulk_ingest() as bulk:
            bulk.save(_root("one", note="leaf metadata"))
            bulk.save(_root("one", note="changed"))


def test_bulk_metadata_conflict_ignored_when_disabled(store_factory):
    """verify_metadata=False accepts a metadata mismatch without raising."""
    store = store_factory()
    _require_bulk(store)
    with store.bulk_ingest(verify_metadata=False) as bulk:
        first = bulk.save(_root("one"))
        again = bulk.save(replace(_root("one"), modified=_root().modified + datetime.timedelta(seconds=1)))
    assert first == again
    assert _table_stats(store, _database_of(store))["bulk_root"][0] == 1


def test_bulk_dispatch_resolves_and_detects_conflict(store_factory):
    """Two backings of one family dispatch correctly; a conflicting backing raises."""
    store = store_factory(entry_records={BulkCalcFamily: (BulkCalcA, BulkCalcB)})
    _require_bulk(store)
    with store.bulk_ingest() as bulk:
        bulk.save(BulkCalcA("alpha", 1))
        bulk.save(BulkCalcB("beta", "kind-b"))
        bulk.save(BulkCalcA("alpha", 1))  # same content id, same backing: fine
    reopened = store_factory.reopen(store)
    assert reopened.fetch_entry(BulkCalcFamily, content_id(BulkCalcA("alpha", 1))) == BulkCalcA("alpha", 1)
    assert reopened.fetch_entry(BulkCalcFamily, content_id(BulkCalcB("beta", "kind-b"))) == BulkCalcB("beta", "kind-b")


def test_bulk_dispatch_conflicting_backing_raises(store_factory, monkeypatch):
    """Two backings colliding on one dispatch content id raise EntryDispatchIntegrityError."""
    store = store_factory(entry_records={BulkCalcFamily: (BulkCalcA, BulkCalcB)})
    _require_bulk(store)
    # Force both backings to share a dispatch content id to provoke the conflict.
    shared = content_id(BulkCalcA("alpha", 1))
    monkeypatch.setattr("httk.data.store_common.SaveProjection.content_id", lambda self, rt, src: shared)
    with pytest.raises(EntryDispatchIntegrityError, match="conflicting backing"):
        with store.bulk_ingest() as bulk:
            bulk.save(BulkCalcA("alpha", 1))
            bulk.save(BulkCalcB("beta", "kind-b"))


def test_bulk_context_owns_the_write_path(store_factory):
    """Ordinary writes are refused while a bulk context is open, and resume afterwards."""
    store = store_factory()
    _require_bulk(store)
    with store.bulk_ingest() as bulk:
        bulk.save(Author("Ada", 1852))
        with pytest.raises(RuntimeError, match="bulk_ingest"):
            store.save(Author("Grace", 1906))
        with pytest.raises(RuntimeError, match="bulk_ingest"):
            store.ensure_tables(Author)
        with pytest.raises(RuntimeError, match="bulk_ingest"):
            with store.transaction():
                pass
    # The store is usable again after the context exits.
    sid = store.save(Author("Grace", 1906))
    assert store.fetch(Author, sid) == Author("Grace", 1906)


def test_bulk_failure_leaves_no_tables_or_rows(store_factory):
    """A validator raising mid-ingest drops every created table; a later save works."""
    store = store_factory()
    _require_bulk(store)
    with pytest.raises(ValueError, match="validator rejected"):
        with store.bulk_ingest() as bulk:
            bulk.save(ValidatedRecord(1))
            bulk.save(Author("Ada", 1852))
            bulk.save(ValidatedRecord(-1))  # validator raises here

    with _database_of(store).engine.connect() as connection:
        from httk.data.db.layout import actual_table_names

        present = {name for name in actual_table_names(connection) if not name.startswith("_httk_")}
    assert present == set()

    # A subsequent ordinary save proceeds on the cleaned store.
    sid = store.save(Author("Ada", 1852))
    assert store_factory.reopen(store).fetch(Author, sid) == Author("Ada", 1852)


def test_bulk_then_ordinary_save_continues_sids(store_factory):
    """After a bulk ingest, ordinary saves continue without a primary-key collision."""
    store = store_factory()
    _require_bulk(store)
    with store.bulk_ingest() as bulk:
        bulk_sids = [bulk.save(Author(name, year)) for name, year in (("Ada", 1852), ("Grace", 1906))]
    new_sid = store.save(Author("Boole", 1854))
    assert new_sid not in bulk_sids
    assert store.fetch(Author, new_sid) == Author("Boole", 1854)
    # And existing rows remain fetchable alongside the new one.
    assert store.fetch(Author, bulk_sids[0]) == Author("Ada", 1852)


# --------------------------------------------------------------------- incremental (populated store) tests


def _physical_counts(database) -> dict[str, int]:
    """Row counts of the physically present, non-marker tables of a database."""
    from httk.data.db.layout import actual_table_names

    counts: dict[str, int] = {}
    with database.engine.connect() as connection:
        for name in actual_table_names(connection):
            if name.startswith("_httk_"):
                continue
            counts[name] = int(connection.execute(sqlalchemy.text(f'SELECT count(*) FROM "{name}"')).scalar_one())
    return counts


def _reference_sid(database, table_name: str, sid: int, column: str):
    """The stored foreign-key value of one row, read straight from the database."""
    with database.engine.connect() as connection:
        return connection.execute(
            sqlalchemy.text(f'SELECT "{column}" FROM "{table_name}" WHERE sid = :sid'), {"sid": sid}
        ).scalar_one()


def _stream_a() -> list[object]:
    """A first stream saved into the store before the incremental append."""
    return [
        Author("Ada", 1852),
        Author("Grace", 1906),
        AuthorTag(Author("Ada", 1852), "role", "pioneer"),
        make_sample(),
        _root("one"),
        BulkCalcA("alpha", 1),
        BulkCalcB("beta", "kind-b"),
    ]


def _stream_b() -> list[object]:
    """A second stream that overlaps A, overlaps itself, and references into A's records."""
    return [
        Author("Ada", 1852),  # existing content hit -> A's sid
        Author("Zoe", 2000),  # new
        Author("Zoe", 2000),  # intra-B content collapse
        AuthorTag(Author("Ada", 1852), "role", "pioneer"),  # existing by_value hit, references into A
        AuthorTag(Author("Zoe", 2000), "role", "new"),  # new, references a new author
        AuthorTag(Author("Zoe", 2000), "role", "new"),  # intra-B by_value collapse
        make_sample(),  # existing content hit (children not duplicated)
        make_sample(formula="NaCl", weight=1.25, reference=None),  # new
        _root("one"),  # existing content hit, identity-skipped metadata equal
        _root("two"),  # new root referencing an existing leaf and a new one
        BulkCalcA("alpha", 1),  # existing entry, same backing
        BulkCalcA("gamma", 3),  # new entry
    ]


@pytest.mark.parametrize("chunk_size", [100_000, 2])
def test_bulk_incremental_matches_save_loop(store_factory, chunk_size):
    """Appending stream B by bulk equals appending it by save() into the same pre-populated store."""
    entry = {BulkCalcFamily: (BulkCalcA, BulkCalcB)}
    save_store = store_factory(entry_records=entry)
    _require_bulk(save_store)
    bulk_store = store_factory(entry_records=entry)

    for obj in _stream_a():
        save_store.save(obj)
        bulk_store.save(obj)

    # A's leaf sid, captured before B, is the sid B's references must resolve to.
    leaf = Leaf(1, "leaf metadata")
    leaf_sid = bulk_store.sid_of(leaf)
    assert leaf_sid is not None

    for obj in _stream_b():
        save_store.save(obj)
    bulk_sids: list[int] = []
    with bulk_store.bulk_ingest(chunk_size=chunk_size) as bulk:
        for obj in _stream_b():
            bulk_sids.append(bulk.save(obj))

    assert _table_stats(bulk_store, _database_of(bulk_store)) == _table_stats(save_store, _database_of(save_store))

    reopened = store_factory.reopen(bulk_store)
    # The new records' returned sids are final and reconstruct their objects.
    root_two_sid = bulk_sids[9]
    assert reopened.fetch(Root, root_two_sid) == Root(
        "two", Leaf(1, "leaf metadata"), [Leaf(1, "leaf metadata"), Leaf(2)], _root().modified
    )
    assert reopened.fetch(Author, bulk_sids[1]) == Author("Zoe", 2000)
    assert reopened.fetch(Sample, bulk_sids[7]) == make_sample(formula="NaCl", weight=1.25, reference=None)
    # Remap correctness: B's new Root("two") references A's original leaf row.
    assert _reference_sid(_database_of(bulk_store), "bulk_root", root_two_sid, "primary_sid") == leaf_sid
    assert reopened.fetch_entry(BulkCalcFamily, content_id(BulkCalcA("gamma", 3))) == BulkCalcA("gamma", 3)


def test_bulk_incremental_metadata_conflict_against_existing(store_factory):
    """A content hit against an existing row verifies metadata like save(), and can be disabled."""
    store = store_factory()
    _require_bulk(store)
    store.save(_root("one"))
    conflicting = replace(_root("one"), modified=_root().modified + datetime.timedelta(seconds=1))
    before = _physical_counts(_database_of(store))
    with pytest.raises(EntryMetadataConflictError, match="modified"), store.bulk_ingest() as bulk:
        bulk.save(conflicting)
    assert _physical_counts(_database_of(store)) == before  # rolled back

    # verify_metadata=False accepts the mismatch and deduplicates against the existing row.
    with store.bulk_ingest(verify_metadata=False) as bulk:
        bulk.save(conflicting)
    assert _physical_counts(_database_of(store))["bulk_root"] == 1


def test_bulk_incremental_nested_metadata_conflict_against_existing(store_factory):
    """A nested identity-excluded metadata mismatch against existing rows raises and rolls back."""
    store = store_factory()
    _require_bulk(store)
    store.save(_root("one", note="leaf metadata"))
    before = _physical_counts(_database_of(store))
    with pytest.raises(EntryMetadataConflictError, match="note"), store.bulk_ingest() as bulk:
        bulk.save(_root("one", note="changed"))
    assert _physical_counts(_database_of(store)) == before


def test_bulk_incremental_by_value_hit_needs_no_metadata(store_factory):
    """A by_value hit against an existing row deduplicates on parent columns with no metadata check."""
    store = store_factory()
    _require_bulk(store)
    store.save(AuthorTag(Author("Ada", 1852), "role", "pioneer"))
    with store.bulk_ingest() as bulk:
        sid = bulk.save(AuthorTag(Author("Ada", 1852), "role", "pioneer"))
    assert isinstance(sid, int)
    assert _physical_counts(_database_of(store))["bulk_author_tag"] == 1


@pytest.mark.parametrize("index_strategy", ["auto", "keep", "rebuild"])
def test_bulk_incremental_index_strategy_equivalent(store_factory, index_strategy):
    """Every index strategy yields the same final state as a save() loop, indexed probe included."""
    reference = store_factory()
    _require_bulk(reference)
    bulk_store = store_factory()

    base = [make_sample(), make_sample(formula="NaCl", weight=1.25, reference=None), Author("Ada", 1852)]
    extra = [
        make_sample(),  # existing hit
        make_sample(formula="MgO", spacegroup=225),  # new
        make_sample(formula="NaCl", weight=1.25, reference=None),  # existing hit
        Author("Zoe", 2000),  # new
    ]
    for obj in base:
        reference.save(obj)
        bulk_store.save(obj)
    for obj in extra:
        reference.save(obj)
    with bulk_store.bulk_ingest(index_strategy=index_strategy) as bulk:
        for obj in extra:
            bulk.save(obj)

    assert _table_stats(bulk_store, _database_of(bulk_store)) == _table_stats(reference, _database_of(reference))
    # A probe through the indexed (formula, spacegroup) columns must agree.
    table = bulk_store._table("bulk_sample")
    with _database_of(bulk_store).engine.connect() as connection:
        probe = connection.execute(
            sqlalchemy.select(sqlalchemy.func.count()).select_from(table).where(table.c["formula"] == "NaCl")
        ).scalar_one()
    assert probe == 1


def test_bulk_incremental_rebuild_duplicate_rolls_back(store_factory):
    """A rebuild whose uniqueness check finds a duplicate aborts and leaves the store unchanged."""
    store = store_factory()
    _require_bulk(store)
    store.save(Author("Ada", 1852))
    database = _database_of(store)
    table = store._table("bulk_author")
    unique_index = next(index for index in table.indexes if index.unique)
    # Corrupt the store out of band: drop the unique index, then (in a separate
    # committed transaction, so DuckDB has applied the drop) plant a duplicate.
    with database.engine.begin() as connection:
        connection.execute(sqlalchemy.schema.DropIndex(unique_index, if_exists=True))
    with database.engine.begin() as connection:
        row = dict(connection.execute(sqlalchemy.select(table)).mappings().one())
        row["sid"] = 9999
        connection.execute(sqlalchemy.insert(table), row)
    before = _physical_counts(database)
    # SQLite fails when the unique index is recreated; DuckDB (which keeps the
    # index) fails in the finalize duplicate scan with a RuntimeError.
    with (
        pytest.raises((sqlalchemy.exc.IntegrityError, RuntimeError)),
        store.bulk_ingest(index_strategy="rebuild") as bulk,
    ):
        bulk.save(Author("Zoe", 2000))
    assert _physical_counts(database) == before  # the new row was rolled back


def test_bulk_incremental_multichunk_cross_chunk_references(store_factory):
    """With a tiny chunk size, references crossing chunk boundaries still remap to existing sids."""
    store = store_factory()
    _require_bulk(store)
    store.save(Author("Ada", 1852))
    reference = store_factory()
    reference.save(Author("Ada", 1852))

    # Each chunk of two references the existing Author("Ada"); the referenced author is
    # resolved and (if a hit) remapped before any referring row is flushed.
    tags = [AuthorTag(Author("Ada", 1852), "role", f"tag{i}") for i in range(5)] + [
        AuthorTag(Author("Ada", 1852), "role", "tag2")
    ]  # a repeat of an earlier chunk's row
    for tag in tags:
        reference.save(tag)
    with store.bulk_ingest(chunk_size=2) as bulk:
        for tag in tags:
            bulk.save(tag)

    assert _table_stats(store, _database_of(store)) == _table_stats(reference, _database_of(reference))
    ada_sid = store.sid_of(Author("Ada", 1852))
    with _database_of(store).engine.connect() as connection:
        author_sids = {
            int(row[0])
            for row in connection.execute(sqlalchemy.text('SELECT DISTINCT author_sid FROM "bulk_author_tag"')).all()
        }
    assert author_sids == {ada_sid}


def test_bulk_incremental_dispatch(store_factory):
    """An incremental dispatch keeps the existing backing, adds new entries, and detects conflicts."""
    entry = {BulkCalcFamily: (BulkCalcA, BulkCalcB)}
    store = store_factory(entry_records=entry)
    _require_bulk(store)
    store.save(BulkCalcA("alpha", 1))
    store.save(BulkCalcB("beta", "kind-b"))
    with store.bulk_ingest() as bulk:
        bulk.save(BulkCalcA("alpha", 1))  # existing entry, same backing: fine
        bulk.save(BulkCalcA("gamma", 3))  # a genuinely new entry
    reopened = store_factory.reopen(store)
    assert reopened.fetch_entry(BulkCalcFamily, content_id(BulkCalcA("alpha", 1))) == BulkCalcA("alpha", 1)
    assert reopened.fetch_entry(BulkCalcFamily, content_id(BulkCalcA("gamma", 3))) == BulkCalcA("gamma", 3)


def test_bulk_incremental_dispatch_conflicting_backing_raises(store_factory, monkeypatch):
    """Re-saving a known dispatch content id under a different backing raises and rolls back."""
    entry = {BulkCalcFamily: (BulkCalcA, BulkCalcB)}
    store = store_factory(entry_records=entry)
    _require_bulk(store)
    store.save(BulkCalcA("alpha", 1))
    before = _physical_counts(_database_of(store))
    shared = content_id(BulkCalcA("alpha", 1))
    monkeypatch.setattr("httk.data.store_common.SaveProjection.content_id", lambda self, rt, src: shared)
    with pytest.raises(EntryDispatchIntegrityError, match="conflicting backing"), store.bulk_ingest() as bulk:
        bulk.save(BulkCalcB("beta", "kind-b"))  # same (forced) content id, different backing
    assert _physical_counts(_database_of(store)) == before


def test_bulk_incremental_failure_atomicity_and_recovery(store_factory):
    """A mid-ingest failure on a populated store leaves counts and indexes intact; later saves work."""
    store = store_factory()
    _require_bulk(store)
    store.save(Author("Ada", 1852))
    store.save(Author("Grace", 1906))
    database = _database_of(store)
    before = _physical_counts(database)

    with pytest.raises(ValueError, match="validator rejected"), store.bulk_ingest(index_strategy="rebuild") as bulk:
        bulk.save(Author("Zoe", 2000))
        bulk.save(ValidatedRecord(-1))  # validator raises here
    assert _physical_counts(database) == before

    # The unique content-id index survived the failed rebuild: a duplicate save deduplicates.
    again = store.save(Author("Ada", 1852))
    assert again == store.sid_of(Author("Ada", 1852))
    # And an ordinary save proceeds and reopens cleanly.
    new_sid = store.save(Author("Boole", 1854))
    assert store_factory.reopen(store).fetch(Author, new_sid) == Author("Boole", 1854)
    assert _physical_counts(database)["bulk_author"] == 3


def test_bulk_resolved_sid_maps_hit_to_existing(store_factory):
    """A populated-mode hit's provisional sid resolves to the pre-existing sid and fetches correctly."""
    store = store_factory()
    _require_bulk(store)
    existing = store.save(Author("Ada", 1852))
    with store.bulk_ingest() as bulk:
        provisional = bulk.save(Author("Ada", 1852))  # deduplicates against the existing row
        new_sid = bulk.save(Author("Zoe", 2000))  # a genuinely new record
        with pytest.raises(RuntimeError, match="resolved_sid"):
            bulk.resolved_sid(Author, provisional)  # resolution is incomplete while the context is open
    assert provisional != existing
    assert bulk.resolved_sid(Author, provisional) == existing  # remapped to the pre-existing sid
    assert bulk.resolved_sid(Author, new_sid) == new_sid  # a survivor resolves to itself
    reopened = store_factory.reopen(store)
    assert reopened.fetch(Author, bulk.resolved_sid(Author, provisional)) == Author("Ada", 1852)
    with pytest.raises(KeyError):
        bulk.resolved_sid(Author, 987654321)  # a sid this ingest never returned


def test_bulk_resolved_sid_identity_on_empty_store(store_factory):
    """Every sid returned by an empty-store ingest resolves to itself."""
    store = store_factory()
    _require_bulk(store)
    with store.bulk_ingest() as bulk:
        sids = [bulk.save(Author("Ada", 1852)), bulk.save(Author("Grace", 1906))]
        sample_sid = bulk.save(make_sample())
    for record_type, sid in ((Author, sids[0]), (Author, sids[1]), (Sample, sample_sid)):
        assert bulk.resolved_sid(record_type, sid) == sid


def test_bulk_incremental_no_orphan_byvalue_child_elements(store_factory):
    """A by_value parent hit leaves no trace of its eagerly-buffered child element records (F1)."""
    save_store = store_factory()
    _require_bulk(save_store)
    bulk_store = store_factory()
    for store in (save_store, bulk_store):
        store.save(ByValParent(1, [Elem("first")]))
    save_store.save(ByValParent(1, [Elem("second")]))  # by_value hit on parent columns
    with bulk_store.bulk_ingest() as bulk:
        bulk.save(ByValParent(1, [Elem("second")]))
    assert _physical_counts(_database_of(bulk_store)) == _physical_counts(_database_of(save_store))
    assert _physical_counts(_database_of(bulk_store))["bulk_elem"] == 1


def test_bulk_incremental_no_orphan_none_descendant(store_factory):
    """A content parent hit does not insert a fresh dedup='none' descendant (F1)."""
    save_store = store_factory()
    _require_bulk(save_store)
    bulk_store = store_factory()
    for store in (save_store, bulk_store):
        store.save(ContentParent("x", NoneRec("evt")))
    save_store.save(ContentParent("x", NoneRec("evt")))  # content hit
    with bulk_store.bulk_ingest() as bulk:
        bulk.save(ContentParent("x", NoneRec("evt")))
    assert _physical_counts(_database_of(bulk_store)) == _physical_counts(_database_of(save_store))
    assert _physical_counts(_database_of(bulk_store))["bulk_none_rec"] == 1


def test_bulk_incremental_orphan_kept_when_referenced_by_survivor(store_factory):
    """A descendant buffered under a hit parent survives if a surviving record also references it (F1)."""
    save_store = store_factory()
    _require_bulk(save_store)
    bulk_store = store_factory()
    for store in (save_store, bulk_store):
        store.save(ByValParent(1, [Elem("first")]))
    batch = [ByValParent(1, [Elem("shared")]), ByValParent(2, [Elem("shared")])]  # first hits, second survives
    for obj in batch:
        save_store.save(obj)
    with bulk_store.bulk_ingest() as bulk:
        for obj in batch:
            bulk.save(obj)
    assert _physical_counts(_database_of(bulk_store)) == _physical_counts(_database_of(save_store))
    # Elem("shared") is kept once — reached through the surviving ByValParent(2, ...).
    assert _physical_counts(_database_of(bulk_store))["bulk_elem"] == 2


def test_bulk_incremental_self_referential_by_value_fixpoint(store_factory):
    """A self-referential by_value chain, both rows already present, deduplicates without a duplicate (F3)."""
    save_store = store_factory()
    _require_bulk(save_store)
    bulk_store = store_factory()
    for store in (save_store, bulk_store):
        store.save(Node(2, Node(1)))  # populates Node(1) and Node(2, ref=Node(1))
    save_store.save(Node(2, Node(1)))
    with bulk_store.bulk_ingest() as bulk:
        bulk.save(Node(2, Node(1)))  # identical chain, fresh objects
    assert _physical_counts(_database_of(bulk_store)) == _physical_counts(_database_of(save_store))
    assert _physical_counts(_database_of(bulk_store))["bulk_node"] == 2


def test_bulk_incremental_resolved_sid_distinguishes_tables(store_factory):
    """A fresh record's returned sid does not resolve through another table's remap entry (F2).

    The fresh Sample's sid is made to numerically coincide with the Author's
    remapped provisional sid, so a regression to bare-int keying would route the
    Sample sid through the Author remap and return the wrong value.
    """
    store = store_factory()
    _require_bulk(store)
    store.save(Author("seed", 1))  # Author sid 1 pre-exists; bulk allocates Author sids from 2
    with store.bulk_ingest() as bulk:
        author_provisional = bulk.save(Author("seed", 1))  # provisional 2 -> remaps to existing Author sid 1
        bulk.save(make_sample())  # fresh Sample sid 1
        sample_collision = bulk.save(make_sample(formula="NaCl", weight=1.25, reference=None))  # fresh Sample sid 2
    # The fresh Sample's sid equals the Author's remapped provisional sid.
    assert sample_collision == author_provisional
    author_resolved = bulk.resolved_sid(Author, author_provisional)
    sample_resolved = bulk.resolved_sid(Sample, sample_collision)
    assert author_resolved == store.sid_of(Author("seed", 1))  # 1, the pre-existing row
    assert sample_resolved == sample_collision  # itself, not routed through the Author remap
    assert author_resolved != sample_resolved
    reopened = store_factory.reopen(store)
    assert reopened.fetch(Sample, sample_resolved) == make_sample(formula="NaCl", weight=1.25, reference=None)


def test_bulk_ingest_refused_inside_open_transaction(store_factory):
    """Opening a bulk context inside an open store transaction fails fast rather than deadlocking (F4)."""
    store = store_factory()
    _require_bulk(store)
    with store.transaction(), pytest.raises(RuntimeError, match="store.transaction"), store.bulk_ingest():
        pass


def test_bulk_incremental_cross_chunk_metadata_verified_against_stored(store_factory):
    """A content hit in a later chunk verifies identity-excluded metadata against the flushed row (F5)."""
    store = store_factory()
    _require_bulk(store)
    with store.bulk_ingest(chunk_size=1) as bulk:  # flush between the two saves
        bulk.save(MetaScalar("k", "note1"))
        bulk.save(MetaScalar("k", "note1"))  # equal metadata across the chunk boundary: deduplicated
    assert _physical_counts(_database_of(store))["bulk_meta_scalar"] == 1

    conflict_store = store_factory()
    with (
        pytest.raises(EntryMetadataConflictError, match="note"),
        conflict_store.bulk_ingest(chunk_size=1) as bulk,
    ):
        bulk.save(MetaScalar("k", "note1"))
        bulk.save(MetaScalar("k", "note2"))  # conflicting metadata against the stored row
