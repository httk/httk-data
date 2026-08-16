"""Phase 2 versioned-mode schema and save coverage for the SQL backend.

Versioned mode adds two lifecycle columns to the parent tables of every family
backing record (``ts_end``, NULL = current, indexed; ``replaced_by_sid``, no
index).  Dependency and child tables never gain lifecycle columns, but every
reference ``*_sid`` column is reverse-indexed store-wide.  Author-``Unique``
fields become unique-among-current: a partial unique index on SQLite and
PostgreSQL, and an in-transaction save-side check on DuckDB (no partial indexes).
"""

import contextlib
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Annotated, ClassVar

import pytest
import sqlalchemy
from httk.core.storage import StorageInfo, Unique

from httk.store.db import Database, SqlStore, StorageLayoutUpgradeRequiredError
from httk.store.db.mapping import REPLACED_BY_COLUMN, TS_END_COLUMN, TS_START_COLUMN
from httk.store.storage_layout import EntryFamilyDeclaration, EntryRecordDeclaration

# --------------------------------------------------------------------- records


@dataclass(frozen=True)
class Dep:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(dedup="content_id", storage_name="versions_dep")

    tag: str


@dataclass(frozen=True)
class Tag:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(dedup="content_id", storage_name="versions_tag")

    label: str


@dataclass(frozen=True)
class Rec:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(dedup="content_id", storage_name="versions_rec")

    name: Annotated[str, Unique()]
    dep: Dep
    tags: tuple[Tag, ...]
    notes: tuple[str, ...]


class RecFamily:
    """Application-owned family holder (deliberately unregistered)."""


LAYOUT = EntryFamilyDeclaration(
    name="test-versions-family",
    family=RecFamily,
    records=(EntryRecordDeclaration(name="test-versions-rec", record=Rec),),
)

_REC_TABLE = "versions_rec"
_DEP_TABLE = "versions_dep"
_TAGS_CHILD_TABLE = "versions_rec_tags"


def _rec(name: str, tag: str = "t", label: str = "l") -> Rec:
    return Rec(name=name, dep=Dep(tag=tag), tags=(Tag(label=label),), notes=("n1", "n2"))


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


def _versioned_store(database: Database) -> SqlStore:
    return SqlStore(database, entry_families=(LAYOUT,), store_timestamps="versioned")


def _index_names(store: SqlStore, table_name: str) -> set[str]:
    return {index.name for index in store._table(table_name).indexes if index.name is not None}


# --------------------------------------------------------------------- tests


@pytest.mark.parametrize("backend", _BACKENDS)
def test_versioned_family_table_has_lifecycle_columns(backend: str) -> None:
    with _database(backend) as database:
        store = _versioned_store(database)
        sid = store.save(_rec("a"))

        rec = store._table(_REC_TABLE)
        assert isinstance(rec.c[TS_END_COLUMN].type, sqlalchemy.BigInteger)
        assert rec.c[TS_END_COLUMN].nullable
        assert rec.c[REPLACED_BY_COLUMN].nullable
        assert not rec.c[TS_START_COLUMN].nullable
        assert f"ix_{_REC_TABLE}_{TS_END_COLUMN}" in _index_names(store, _REC_TABLE)
        # replaced_by_sid is fsck-only: it must carry no index of its own.
        assert not any(list(index.columns) == [rec.c[REPLACED_BY_COLUMN]] for index in rec.indexes)

        dep = store._table(_DEP_TABLE)
        assert TS_START_COLUMN in dep.c
        assert TS_END_COLUMN not in dep.c
        assert REPLACED_BY_COLUMN not in dep.c

        assert store.store_timestamp_mode == "versioned"
        assert store.fetch(Rec, sid).name == "a"
        with database.engine.connect() as connection:
            ends = connection.execute(sqlalchemy.text(f"SELECT {TS_END_COLUMN} FROM {_REC_TABLE}")).all()
        assert ends == [(None,)]


def test_marker_records_versioned_mode() -> None:
    with Database.sqlite() as database:
        _versioned_store(database)
        with database.engine.connect() as connection:
            marker = connection.execute(
                sqlalchemy.text("SELECT value FROM _httk_store_metadata WHERE key = 'store_timestamps'")
            ).scalar_one()
        assert marker == "v2:versioned:1000"


def test_creation_mode_has_no_lifecycle_columns_or_reverse_indexes() -> None:
    with Database.sqlite() as database:
        store = SqlStore(database, entry_families=(LAYOUT,), store_timestamps="creation")
        store.save(_rec("a"))
        rec = store._table(_REC_TABLE)
        assert TS_END_COLUMN not in rec.c
        assert REPLACED_BY_COLUMN not in rec.c
        # No un-annotated reference reverse index outside versioned mode.
        assert f"ix_{_REC_TABLE}_dep_sid" not in _index_names(store, _REC_TABLE)
        assert f"ix_{_TAGS_CHILD_TABLE}_tags_sid" not in _index_names(store, _TAGS_CHILD_TABLE)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_reverse_indexes_cover_reference_columns(backend: str) -> None:
    with _database(backend) as database:
        store = _versioned_store(database)
        store.save(_rec("a"))
        # Un-annotated parent reference column and storable child-element sid
        # column are both indexed in versioned mode.
        assert f"ix_{_REC_TABLE}_dep_sid" in _index_names(store, _REC_TABLE)
        assert f"ix_{_TAGS_CHILD_TABLE}_tags_sid" in _index_names(store, _TAGS_CHILD_TABLE)


def test_sqlite_unique_field_uses_partial_index() -> None:
    with Database.sqlite() as database:
        store = _versioned_store(database)
        store.save(_rec("a"))
        with database.engine.connect() as connection:
            sql = connection.execute(
                sqlalchemy.text("SELECT sql FROM sqlite_master WHERE type = 'index' AND name = :name"),
                {"name": f"uq_{_REC_TABLE}_name"},
            ).scalar_one()
        assert "UNIQUE" in sql.upper()
        assert "WHERE ts_end IS NULL" in sql


@pytest.mark.parametrize("backend", ["sqlite", pytest.param("postgresql", marks=pytest.mark.xdist_group("postgres"))])
def test_partial_unique_index_rejects_duplicate_current(backend: str) -> None:
    with _database(backend) as database:
        store = _versioned_store(database)
        store.save(_rec("a", tag="x"))
        # Same unique name, different content id -> the partial unique index fires.
        with pytest.raises(sqlalchemy.exc.IntegrityError):
            store.save(_rec("a", tag="y"))
        store.save(_rec("b", tag="z"))


def test_duckdb_unique_field_has_no_unique_index_but_save_enforces() -> None:
    with Database.duckdb() as database:
        store = _versioned_store(database)
        store.save(_rec("a", tag="x"))
        rec = store._table(_REC_TABLE)
        name_indexes = [index for index in rec.indexes if list(index.columns) == [rec.c["name"]]]
        assert name_indexes, "expected a plain lookup index on the unique column"
        assert all(not index.unique for index in name_indexes)
        # The save transaction enforces unique-among-current in place of the index.
        with pytest.raises(sqlalchemy.exc.IntegrityError):
            store.save(_rec("a", tag="y"))
        # A distinct current value still inserts.
        store.save(_rec("b", tag="z"))


@pytest.mark.parametrize("backend", _BACKENDS)
def test_versioned_store_reopens_cleanly(backend: str) -> None:
    if backend == "sqlite":
        import os
        import tempfile

        path = os.path.join(tempfile.mkdtemp(), "store.sqlite")
        with Database.sqlite(path) as database:
            _versioned_store(database).save(_rec("a"))
        with Database.sqlite(path) as database:
            reopened = _versioned_store(database)
            assert reopened.store_timestamp_mode == "versioned"
            assert reopened.fetch(Rec, 1).name == "a"
        return
    # DuckDB and PostgreSQL keep the same live Database across reopen.
    with _database(backend) as database:
        _versioned_store(database).save(_rec("a"))
        reopened = _versioned_store(database)
        assert reopened.store_timestamp_mode == "versioned"


def test_reopen_with_mismatched_mode_raises() -> None:
    import os
    import tempfile

    path = os.path.join(tempfile.mkdtemp(), "store.sqlite")
    with Database.sqlite(path) as database:
        _versioned_store(database).save(_rec("a"))
    with Database.sqlite(path) as database, pytest.raises(StorageLayoutUpgradeRequiredError, match="store_timestamps"):
        SqlStore(database, entry_families=(LAYOUT,), store_timestamps="creation")

    other = os.path.join(tempfile.mkdtemp(), "creation.sqlite")
    with Database.sqlite(other) as database:
        SqlStore(database, entry_families=(LAYOUT,), store_timestamps="creation").save(_rec("a"))
    with Database.sqlite(other) as database, pytest.raises(StorageLayoutUpgradeRequiredError, match="store_timestamps"):
        SqlStore(database, entry_families=(LAYOUT,), store_timestamps="versioned")
