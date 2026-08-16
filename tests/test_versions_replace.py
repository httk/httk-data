"""Phase 3 versioned-mode ``replace`` and revive-protection coverage.

``replace(old, new_obj)`` supersedes a current family entry: it saves
``new_obj`` at the pinned transaction timestamp ``T`` and closes ``old`` by
setting ``ts_end = T`` (so the half-open intervals ``[ts_start, ts_end)`` abut
exactly) and ``replaced_by_sid`` to the successor. Reviving a superseded row —
saving content that deduplicates onto one — is refused on the save, bulk, and
replace paths. Query filtering stays out of scope until Phase 4, so ``fetch``
still returns superseded rows by sid.
"""

import contextlib
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Annotated, ClassVar

import pytest
import sqlalchemy
from httk.core.storage import StorageInfo, Unique, content_id

from httk.store.db import (
    Database,
    RecordReviveError,
    RecordSupersededError,
    ReplaceConflictError,
    SqlStore,
)
from httk.store.db.mapping import REPLACED_BY_COLUMN, TS_END_COLUMN, TS_START_COLUMN
from httk.store.storage_layout import EntryFamilyDeclaration, EntryRecordDeclaration

# --------------------------------------------------------------------- records


@dataclass(frozen=True)
class Rec:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(dedup="content_id", storage_name="repl_rec")

    name: Annotated[str, Unique()]
    payload: str


class RecFamily:
    """Application-owned single-backing family (deliberately unregistered)."""


SINGLE_LAYOUT = EntryFamilyDeclaration(
    name="test-replace-family",
    family=RecFamily,
    records=(EntryRecordDeclaration(name="test-replace-rec", record=Rec),),
)
_REC_TABLE = "repl_rec"


@dataclass(frozen=True)
class RecA:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(dedup="content_id", storage_name="repl_ma")

    value: str


@dataclass(frozen=True)
class RecB:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(dedup="content_id", storage_name="repl_mb")

    number: int


class MultiFamily:
    """Application-owned two-backing family."""


MULTI_LAYOUT = EntryFamilyDeclaration(
    name="test-replace-multi-family",
    family=MultiFamily,
    records=(
        EntryRecordDeclaration(name="test-replace-multi-a", record=RecA),
        EntryRecordDeclaration(name="test-replace-multi-b", record=RecB),
    ),
)


@dataclass(frozen=True)
class NonFamily:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(dedup="content_id", storage_name="repl_nonfam")

    x: str


# --------------------------------------------------------------------- backends


@contextlib.contextmanager
def _database(backend: str) -> Iterator[Database]:
    if backend == "sqlite":
        with Database.sqlite() as database:
            yield database
    elif backend == "duckdb":
        with Database.duckdb() as database:
            yield database
    else:
        from postgres_support import postgres_database

        with postgres_database() as database:
            yield database


_BACKENDS = [
    "sqlite",
    "duckdb",
    pytest.param("postgresql", marks=pytest.mark.xdist_group("postgres")),
]


def _store(database: Database, *, mode: str = "versioned") -> SqlStore:
    return SqlStore(database, entry_families=(SINGLE_LAYOUT,), store_timestamps=mode)


def _at(store: SqlStore, ns: int) -> None:
    store._clock = lambda: ns


def _rows(database: Database, table: str) -> list[tuple[int, int, int | None, int | None]]:
    with database.engine.connect() as connection:
        return [
            (int(sid), int(start), None if end is None else int(end), None if repl is None else int(repl))
            for sid, start, end, repl in connection.execute(
                sqlalchemy.text(
                    f"SELECT sid, {TS_START_COLUMN}, {TS_END_COLUMN}, {REPLACED_BY_COLUMN} "
                    f"FROM {table} ORDER BY sid"
                )
            ).all()
        ]


# --------------------------------------------------------------------- happy path


@pytest.mark.parametrize("backend", _BACKENDS)
def test_replace_closes_old_and_opens_successor(backend: str) -> None:
    with _database(backend) as database:
        store = _store(database)
        _at(store, 1_000_000)
        old_sid = store.save(Rec(name="x", payload="v1"))
        _at(store, 2_000_000)
        new_sid = store.replace(Rec(name="x", payload="v1"), Rec(name="x", payload="v2"))

        rows = _rows(database, _REC_TABLE)
        old_row = next(row for row in rows if row[0] == old_sid)
        new_row = next(row for row in rows if row[0] == new_sid)
        # Half-open intervals abut exactly: old.ts_end == new.ts_start.
        assert old_row[2] == new_row[1]
        assert old_row[3] == new_sid  # replaced_by_sid
        assert new_row[2] is None and new_row[3] is None  # successor is current
        assert store._store_timestamp_mark == new_row[1]
        # Both rows present; the superseded one is still fetchable by sid (no
        # query filtering until Phase 4).
        assert store.fetch(Rec, old_sid).payload == "v1"
        assert store.fetch(Rec, new_sid).payload == "v2"


def test_replace_keeps_same_unique_key() -> None:
    # The successor keeps the author-Unique key of the old row; closing the old
    # row before the insert frees the unique-among-current slot.
    with Database.sqlite() as database:
        store = _store(database)
        _at(store, 1_000_000)
        store.save(Rec(name="same", payload="v1"))
        _at(store, 2_000_000)
        store.replace(Rec(name="same", payload="v1"), Rec(name="same", payload="v2"))
        assert [row[2] for row in _rows(database, _REC_TABLE)] == [2000, None]


# --------------------------------------------------------------------- target resolution


def test_replace_by_instance_content_id_and_sid_single_backing() -> None:
    with Database.sqlite() as database:
        store = _store(database)
        # by instance
        _at(store, 1_000_000)
        store.save(Rec(name="a", payload="v1"))
        _at(store, 2_000_000)
        store.replace(Rec(name="a", payload="v1"), Rec(name="a", payload="v2"))
        # by content_id string
        _at(store, 3_000_000)
        cid = content_id(Rec(name="a", payload="v2"), as_record=Rec)
        store.replace(cid, Rec(name="a", payload="v3"))
        # by int sid
        _at(store, 4_000_000)
        current = store.sid_of(Rec(name="a", payload="v3"))
        assert current is not None
        store.replace(current, Rec(name="a", payload="v4"))
        # Four rows, only the last current.
        ends = [row[2] for row in _rows(database, _REC_TABLE)]
        assert ends == [2000, 3000, 4000, None]


def test_replace_content_id_resolves_via_multi_backing_dispatch() -> None:
    with Database.sqlite() as database:
        store = SqlStore(database, entry_families=(MULTI_LAYOUT,), store_timestamps="versioned")
        _at(store, 1_000_000)
        store.save(RecA(value="hello"))
        _at(store, 2_000_000)
        cid = content_id(RecA(value="hello"), as_record=RecA)
        # old is a RecA content id, new is a different backing (RecB) of the same family.
        store.replace(cid, RecB(number=7))
        with database.engine.connect() as connection:
            end = connection.execute(sqlalchemy.text(f"SELECT {TS_END_COLUMN} FROM repl_ma")).scalar_one()
        assert end == 2000


def test_replace_by_instance_multi_backing() -> None:
    with Database.sqlite() as database:
        store = SqlStore(database, entry_families=(MULTI_LAYOUT,), store_timestamps="versioned")
        _at(store, 1_000_000)
        store.save(RecA(value="hi"))
        _at(store, 2_000_000)
        store.replace(RecA(value="hi"), RecB(number=3))
        with database.engine.connect() as connection:
            end = connection.execute(sqlalchemy.text(f"SELECT {TS_END_COLUMN} FROM repl_ma")).scalar_one()
        assert end == 2000


def test_replace_by_sid_on_multi_backing_is_ambiguous() -> None:
    with Database.sqlite() as database:
        store = SqlStore(database, entry_families=(MULTI_LAYOUT,), store_timestamps="versioned")
        _at(store, 1_000_000)
        sid = store.save(RecA(value="hi"))
        _at(store, 2_000_000)
        with pytest.raises(ValueError, match="ambiguous by sid"):
            store.replace(sid, RecB(number=3))


def test_replace_unknown_target_raises() -> None:
    with Database.sqlite() as database:
        store = _store(database)
        _at(store, 1_000_000)
        store.save(Rec(name="a", payload="v1"))
        _at(store, 2_000_000)
        with pytest.raises(ValueError, match="not stored"):
            store.replace("no-such-content-id", Rec(name="a", payload="v2"))
        with pytest.raises(ValueError, match="not present"):
            store.replace(999, Rec(name="a", payload="v2"))


# --------------------------------------------------------------------- superseded / conflict


def test_replacing_a_superseded_target_raises() -> None:
    with Database.sqlite() as database:
        store = _store(database)
        _at(store, 1_000_000)
        old_sid = store.save(Rec(name="x", payload="v1"))
        _at(store, 2_000_000)
        store.replace(Rec(name="x", payload="v1"), Rec(name="x", payload="v2"))
        _at(store, 3_000_000)
        # old_sid is now superseded; replacing it again (by sid) is refused.
        with pytest.raises(RecordSupersededError):
            store.replace(old_sid, Rec(name="x", payload="v3"))


def test_double_replace_of_same_target_second_raises_superseded() -> None:
    # Serial code: the first replace supersedes the target, so the pre-read of
    # the second replace deterministically raises RecordSupersededError.
    with Database.sqlite() as database:
        store = _store(database)
        _at(store, 1_000_000)
        old_sid = store.save(Rec(name="x", payload="v1"))
        _at(store, 2_000_000)
        store.replace(old_sid, Rec(name="x", payload="v2"))
        _at(store, 3_000_000)
        with pytest.raises(RecordSupersededError):
            store.replace(old_sid, Rec(name="x", payload="v3"))


# --------------------------------------------------------------------- revive protection


@pytest.mark.parametrize("backend", _BACKENDS)
def test_replace_onto_superseded_content_raises_revive(backend: str) -> None:
    with _database(backend) as database:
        store = _store(database)
        _at(store, 1_000_000)
        store.save(Rec(name="x", payload="v1"))
        _at(store, 2_000_000)
        store.replace(Rec(name="x", payload="v1"), Rec(name="x", payload="v2"))
        _at(store, 3_000_000)
        # Replacing the current row with content equal to the superseded row.
        with pytest.raises(RecordReviveError):
            store.replace(Rec(name="x", payload="v2"), Rec(name="x", payload="v1"))


def test_plain_save_of_superseded_content_raises_revive() -> None:
    with Database.sqlite() as database:
        store = _store(database)
        _at(store, 1_000_000)
        store.save(Rec(name="x", payload="v1"))
        _at(store, 2_000_000)
        store.replace(Rec(name="x", payload="v1"), Rec(name="x", payload="v2"))
        _at(store, 3_000_000)
        with pytest.raises(RecordReviveError):
            store.save(Rec(name="x", payload="v1"))


def test_replace_with_content_identical_to_target_raises_revive() -> None:
    with Database.sqlite() as database:
        store = _store(database)
        _at(store, 1_000_000)
        store.save(Rec(name="x", payload="v1"))
        _at(store, 2_000_000)
        with pytest.raises(RecordReviveError):
            store.replace(Rec(name="x", payload="v1"), Rec(name="x", payload="v1"))


def test_bulk_ingest_of_superseded_content_fails() -> None:
    with Database.sqlite() as database:
        store = _store(database)
        _at(store, 1_000_000)
        store.save(Rec(name="x", payload="v1"))
        _at(store, 2_000_000)
        store.replace(Rec(name="x", payload="v1"), Rec(name="x", payload="v2"))
        _at(store, 3_000_000)
        with pytest.raises(RecordReviveError):
            with store.bulk_ingest() as bulk:
                bulk.save(Rec(name="x", payload="v1"))


# --------------------------------------------------------------------- converging lineage


def test_replace_converges_onto_existing_current_row() -> None:
    with Database.sqlite() as database:
        store = _store(database)
        _at(store, 1_000_000)
        a_sid = store.save(Rec(name="a", payload="pa"))
        c_sid = store.save(Rec(name="c", payload="pc"))
        _at(store, 2_000_000)
        # new content deduplicates onto the existing current row C.
        result = store.replace(Rec(name="a", payload="pa"), Rec(name="c", payload="pc"))
        assert result == c_sid
        rows = {row[0]: row for row in _rows(database, _REC_TABLE)}
        assert rows[a_sid][2] == 2000  # A superseded
        assert rows[a_sid][3] == c_sid  # replaced_by == C
        assert rows[c_sid][2] is None  # C stays current


# --------------------------------------------------------------------- transaction semantics


def test_replace_shares_pinned_timestamp_inside_transaction() -> None:
    with Database.sqlite() as database:
        store = _store(database)
        _at(store, 1_000_000)
        old_sid = store.save(Rec(name="x", payload="v1"))
        _at(store, 2_000_000)
        with store.transaction():
            other = store.save(Rec(name="y", payload="w1"))
            new_sid = store.replace(Rec(name="x", payload="v1"), Rec(name="x", payload="v2"))
        rows = {row[0]: row for row in _rows(database, _REC_TABLE)}
        # The save and the replace share one pinned T for the whole transaction.
        assert rows[other][1] == rows[new_sid][1] == rows[old_sid][2]


def test_replace_rollback_leaves_target_current() -> None:
    with Database.sqlite() as database:
        store = _store(database)
        _at(store, 1_000_000)
        old_sid = store.save(Rec(name="x", payload="v1"))
        _at(store, 2_000_000)
        with pytest.raises(RuntimeError, match="boom"):
            with store.transaction():
                store.replace(Rec(name="x", payload="v1"), Rec(name="x", payload="v2"))
                raise RuntimeError("boom")
        rows = _rows(database, _REC_TABLE)
        assert rows == [(old_sid, 1000, None, None)]


def test_same_transaction_save_then_replace_is_zero_length() -> None:
    with Database.sqlite() as database:
        store = _store(database)
        _at(store, 1_000_000)
        with pytest.raises(ValueError, match="zero-length"):
            with store.transaction():
                store.save(Rec(name="z", payload="v1"))
                store.replace(Rec(name="z", payload="v1"), Rec(name="z", payload="v2"))


# --------------------------------------------------------------------- rejections


def test_replace_refused_in_creation_mode() -> None:
    with Database.sqlite() as database:
        store = _store(database, mode="creation")
        with pytest.raises(ValueError, match="versioned"):
            store.replace(Rec(name="x", payload="v1"), Rec(name="x", payload="v2"))


def test_replace_refused_on_degraded_profile(tmp_path) -> None:
    path = tmp_path / "degraded.sqlite"
    database = Database.sqlite(path, degraded=True)
    try:
        store = _store(database)
        assert store.write_profile == "degraded"
        with pytest.raises(RuntimeError, match="transactional store profile"):
            store.replace(Rec(name="x", payload="v1"), Rec(name="x", payload="v2"))
    finally:
        database.engine.dispose()


def test_replace_refused_during_bulk_context() -> None:
    with Database.sqlite() as database:
        store = _store(database)
        with pytest.raises(RuntimeError, match="bulk_ingest"):
            with store.bulk_ingest():
                store.replace(Rec(name="x", payload="v1"), Rec(name="x", payload="v2"))


def test_replace_refused_for_non_family_record() -> None:
    with Database.sqlite() as database:
        store = _store(database)
        with pytest.raises(ValueError, match="entry family"):
            store.replace("cid", NonFamily(x="q"))
