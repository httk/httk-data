"""SQL-physical store mechanics: DDL, SQL layout, transactions, and identity behavior."""

import datetime
import gc
import os
import pathlib
import subprocess
import sys
from dataclasses import dataclass, replace
from fractions import Fraction
from typing import Annotated, ClassVar

import pytest
import sqlalchemy
from httk.core import FracScalar, FracVector
from httk.core.storage import Indexed, Shape, Skip, StorageInfo, Unique, stored_property

from httk.store.db import Database, SqlStore, resolve_schema
from httk.store.db.mapping import sqlalchemy_metadata, table_for


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
class UniqueName:
    name: Annotated[str, Unique()]


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


@pytest.fixture(params=["sqlite", "duckdb"])
def database(request):
    """An in-memory database per supported dialect (duckdb skips where not installed)."""
    if request.param == "duckdb":
        pytest.importorskip("duckdb_engine")
        with Database.duckdb() as db:
            yield db
    else:
        with Database.sqlite() as db:
            yield db


def _count(db: Database, table_name: str) -> int:
    with db.engine.connect() as connection:
        return connection.execute(sqlalchemy.text(f"SELECT COUNT(*) FROM {table_name}")).scalar_one()


# --------------------------------------------------------------------- DDL / mapping


def test_metadata_holds_parent_child_and_referenced_tables():
    metadata = sqlalchemy_metadata([resolve_schema(Sample)])
    assert {
        "sample",
        "sample_coords",
        "sample_symbols",
        "sample_tags",
        "sample_ratios",
        "sample_authors",
        "author",
    } <= set(metadata.tables)


def test_optional_child_presence_column_is_non_nullable():
    table = sqlalchemy_metadata([resolve_schema(OptionalChildRoundTrip)]).tables["optional_child_round_trip"]
    assert isinstance(table.c["notes_present"].type, sqlalchemy.Boolean)
    assert not table.c["notes_present"].nullable


def test_parent_table_columns_and_types():
    metadata = sqlalchemy_metadata([resolve_schema(Sample)])
    table = metadata.tables["sample"]
    assert table.c["sid"].primary_key
    assert isinstance(table.c["sid"].type, sqlalchemy.Integer)
    assert isinstance(table.c["formula"].type, sqlalchemy.Text) and not table.c["formula"].nullable
    assert isinstance(table.c["spacegroup"].type, sqlalchemy.Integer)
    assert isinstance(table.c["energy"].type, sqlalchemy.Float)
    assert isinstance(table.c["energy_exact"].type, sqlalchemy.Text)
    assert isinstance(table.c["stable"].type, sqlalchemy.Boolean)
    assert isinstance(table.c["payload"].type, sqlalchemy.LargeBinary)
    assert table.c["note"].nullable and table.c["weight"].nullable
    assert isinstance(table.c["ratio"].type, sqlalchemy.Float)
    assert isinstance(table.c["ratio_exact"].type, sqlalchemy.Text)
    assert isinstance(table.c["created"].type, sqlalchemy.Text)
    for i in range(9):
        assert isinstance(table.c[f"cell_{i}"].type, sqlalchemy.Float)
    assert isinstance(table.c["cell_exact"].type, sqlalchemy.Text)
    assert isinstance(table.c["natoms"].type, sqlalchemy.Integer)
    assert "coords" not in table.c and "symbols" not in table.c
    reference = table.c["reference_sid"]
    assert reference.nullable
    assert not reference.foreign_keys


def test_content_id_column_only_under_content_id_policy():
    metadata = sqlalchemy_metadata([resolve_schema(Sample), resolve_schema(AuthorTag), resolve_schema(LogEvent)])
    assert "content_id" in metadata.tables["sample"].c
    assert "content_id" in metadata.tables["author"].c
    assert "content_id" not in metadata.tables["author_tag"].c
    assert "content_id" not in metadata.tables["log_event"].c
    unique_indexes = {index.name for index in metadata.tables["sample"].indexes if index.unique}
    assert "uq_sample_content_id" in unique_indexes


def test_single_composite_and_unique_indexes():
    metadata = sqlalchemy_metadata([resolve_schema(Sample), resolve_schema(UniqueName)])
    sample_indexes = {index.name: index for index in metadata.tables["sample"].indexes}
    assert [column.name for column in sample_indexes["ix_sample_formula"].columns] == ["formula"]
    assert not sample_indexes["ix_sample_formula"].unique
    assert [column.name for column in sample_indexes["ix_sample_formula_spacegroup"].columns] == [
        "formula",
        "spacegroup",
    ]
    unique_indexes = {index.name: index for index in metadata.tables["unique_name"].indexes}
    assert unique_indexes["uq_unique_name_name"].unique
    assert [column.name for column in unique_indexes["uq_unique_name_name"].columns] == ["name"]


def test_child_tables_have_parent_fk_index_and_element_columns():
    metadata = sqlalchemy_metadata([resolve_schema(Sample)])
    symbols = metadata.tables["sample_symbols"]
    assert not symbols.c["sample_sid"].nullable
    assert not symbols.c["sample_sid"].foreign_keys
    assert not symbols.c["symbols_index"].nullable
    assert isinstance(symbols.c["symbols"].type, sqlalchemy.Text)
    assert {index.name for index in symbols.indexes} == {"ix_sample_symbols_sample_sid"}
    coords = metadata.tables["sample_coords"]
    for i in range(3):
        assert isinstance(coords.c[f"coords_{i}"].type, sqlalchemy.Float)
    assert isinstance(coords.c["coords_exact"].type, sqlalchemy.Text)
    authors = metadata.tables["sample_authors"]
    assert not authors.c["authors_sid"].foreign_keys


def test_table_for_is_idempotent_per_metadata():
    metadata = sqlalchemy.MetaData()
    schema = resolve_schema(Sample)
    assert table_for(schema, metadata) is table_for(schema, metadata)


# --------------------------------------------------------------------- round trips


def test_reopened_sql_store_does_not_reuse_live_identity(database):
    sample = make_sample()
    sid = SqlStore(database, entry_records={}).save(sample)
    fetched = SqlStore(database).fetch(Sample, sid)  # a reopened store: no identity cache involved
    assert fetched is not sample


def test_derived_property_is_stored_in_parent_table(database):
    sid = SqlStore(database, entry_records={}).save(make_sample())
    with database.engine.connect() as connection:
        stored = connection.execute(sqlalchemy.text(f"SELECT natoms FROM sample WHERE sid = {sid}")).scalar_one()
    assert stored == 3


# --------------------------------------------------------------------- dedup


def test_dedup_content_id_reuses_row(database):
    store = SqlStore(database, entry_records={})
    store.save(Author("Ada", 1852))
    store.save(Author("Ada", 1852))
    assert _count(database, "author") == 1


def test_dedup_content_id_does_not_duplicate_children(database):
    store = SqlStore(database, entry_records={})
    store.save(make_sample())
    store.save(make_sample())
    assert _count(database, "sample") == 1
    assert _count(database, "sample_symbols") == 3
    assert _count(database, "sample_authors") == 1
    assert _count(database, "author") == 2  # Ada (child element) + Boole (reference)


def test_dedup_by_value_matches_parent_columns(database):
    store = SqlStore(database, entry_records={})
    store.save(AuthorTag(Author("Ada", 1852), "role", "pioneer"))
    store.save(AuthorTag(Author("Ada", 1852), "role", "pioneer"))
    assert _count(database, "author_tag") == 1
    store.save(AuthorTag(Author("Ada", 1852), "role", "mathematician"))
    assert _count(database, "author_tag") == 2


def test_dedup_none_always_inserts(database):
    store = SqlStore(database, entry_records={})
    store.save(LogEvent("started"))
    store.save(LogEvent("started"))
    assert _count(database, "log_event") == 2


# --------------------------------------------------------------------- transactions


def test_transaction_rolls_back_on_exception(database):
    store = SqlStore(database, entry_records={})
    store.ensure_tables(Author)
    with pytest.raises(RuntimeError, match="boom"), store.transaction():
        store.save(Author("X", 1))
        raise RuntimeError("boom")
    assert _count(database, "author") == 0


def test_transaction_shares_connection_and_commits(database):
    store = SqlStore(database, entry_records={})
    with store.transaction():
        sid1 = store.save(Author("A", 1))
        with store.transaction():  # nesting is flat: joins the outer transaction
            sid2 = store.save(Author("B", 2))
        assert store.fetch(Author, sid1).name == "A"  # reads inside see uncommitted writes
    assert sid1 != sid2
    assert _count(database, "author") == 2


# --------------------------------------------------------------------- identity cache


def test_fetch_returns_same_object_while_alive(database):
    store = SqlStore(database, entry_records={})
    sid = store.save(Author("Ada", 1852))
    assert store.fetch(Author, sid) is store.fetch(Author, sid)


def test_save_then_fetch_returns_saved_object(database):
    store = SqlStore(database, entry_records={})
    author = Author("Ada", 1852)
    sid = store.save(author)
    assert store.fetch(Author, sid) is author
    assert store.sid_of(author) == sid


def test_sid_of_tracks_unhashable_instances(database):
    """A storable class holding a list is unhashable, and must still be tracked.

    The reverse cache is keyed by equality, which such an instance cannot
    support; without an identity-keyed fallback ``sid_of`` reported a
    just-saved instance as never stored, and ``referring`` then raised for it.
    """
    store = SqlStore(database, entry_records={})
    sample = make_sample()  # its `symbols: list[str]` makes it unhashable
    with pytest.raises(TypeError):
        hash(sample)

    sid = store.save(sample)
    assert store.sid_of(sample) == sid
    assert store.sid_of(make_sample()) == sid  # content identity is resolved through the database

    # The identity entry must not outlive the instance, or a recycled id()
    # could resolve to a stale sid.
    throwaway = make_sample()
    store.save(throwaway)
    tracked_with_throwaway = len(store._sids_by_identity)
    del throwaway
    gc.collect()
    assert len(store._sids_by_identity) < tracked_with_throwaway


def test_sql_unique_violation_rolls_back_recursive_save(database):
    store = SqlStore(database, entry_records={})
    store.save(RollbackParent(RollbackChild("kept"), "unique"))
    rolled_back = RollbackChild("rolled back")

    with pytest.raises(sqlalchemy.exc.IntegrityError):
        store.save(RollbackParent(rolled_back, "unique"))

    with pytest.raises(KeyError):
        store.fetch(RollbackChild, 2)


def test_sql_optional_child_presence_query(database):
    store = SqlStore(database, entry_records={})
    sids = [store.save(OptionalChildRoundTrip("value", notes)) for notes in (None, [], ["note"])]
    fresh = SqlStore(database)

    searcher = fresh.searcher()
    variable = searcher.variable(OptionalChildRoundTrip)
    searcher.add(variable.notes_present == True)
    searcher.output(variable, "record")
    assert [result[0][0] for result in searcher] == [fresh.fetch(OptionalChildRoundTrip, sid) for sid in sids[1:]]


# --------------------------------------------------------------------- referring


# --------------------------------------------------------------------- fetch_by_content_id


# --------------------------------------------------------------------- database lifecycle


def test_in_memory_database_is_shared_across_operations():
    with Database.sqlite() as database:
        sid = SqlStore(database, entry_records={}).save(Author("Ada", 1852))
        assert SqlStore(database).fetch(Author, sid) == Author("Ada", 1852)


def test_file_backed_database_persists_across_instances(tmp_path):
    path = tmp_path / "authors.sqlite"
    database = Database.sqlite(path)
    sid = SqlStore(database, entry_records={}).save(Author("Ada", 1852))
    database.dispose()
    with Database.sqlite(path) as reopened:
        assert SqlStore(reopened).fetch(Author, sid) == Author("Ada", 1852)


def test_file_backed_duckdb_persists_and_continues_sids(tmp_path):
    pytest.importorskip("duckdb_engine")
    path = tmp_path / "authors.duckdb"
    database = Database.duckdb(path)
    sid = SqlStore(database, entry_records={}).save(Author("Ada", 1852))
    database.dispose()
    with Database.duckdb(path) as reopened:
        store = SqlStore(reopened)
        assert store.fetch(Author, sid) == Author("Ada", 1852)
        # The sid sequence lives in the file: new saves do not collide with old sids.
        assert store.save(Author("Boole", 1854)) != sid


# --------------------------------------------------------------------- lazy import behavior


def _subprocess_env() -> dict[str, str]:
    env = dict(os.environ)
    src = str(pathlib.Path(__file__).resolve().parent.parent / "src")
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    return env


def test_plain_import_does_not_import_sqlalchemy():
    code = "import httk.store.db\nimport sys\nassert 'sqlalchemy' not in sys.modules, 'sqlalchemy was imported'"
    subprocess.run([sys.executable, "-c", code], check=True, env=_subprocess_env())


def test_missing_sqlalchemy_raises_import_error_naming_extra():
    code = (
        "import sys\n"
        "import importlib.abc\n"
        "class Block(importlib.abc.MetaPathFinder):\n"
        "    def find_spec(self, fullname, path=None, target=None):\n"
        "        if fullname.partition('.')[0] == 'sqlalchemy':\n"
        "            raise ModuleNotFoundError(f'No module named {fullname!r}', name=fullname)\n"
        "        return None\n"
        "sys.meta_path.insert(0, Block())\n"
        "import httk.store.db\n"
        "try:\n"
        "    httk.store.db.Database\n"
        "except ImportError as error:\n"
        "    assert 'httk-store[db]' in str(error), error\n"
        "else:\n"
        "    raise SystemExit('expected an ImportError')\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True, env=_subprocess_env())
