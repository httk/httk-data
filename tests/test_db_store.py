"""Tests for the SQL layer (httk.data.db.engine/mapping/store): DDL, round-trips, dedup, transactions."""

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
from httk.core import FracScalar, FracVector, IdentitySkip, Indexed, Shape, Skip, StorageInfo, Unique, stored_property

from httk.data.db import Database, EntryMetadataConflictError, SchemaError, SqlStore, content_id, resolve_schema
from httk.data.db.mapping import sqlalchemy_metadata, table_for


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
class OptionalChildMetadata:
    value: str
    notes: Annotated[list[str] | None, IdentitySkip()] = None


@dataclass(frozen=True)
class RowVector:
    vec: Annotated[FracVector, Shape(1, 3)]


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
        scale=FracScalar(2, 7),
        created=datetime.datetime(2026, 7, 24, 12, 30, 0),
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
    assert {"sample", "sample_coords", "sample_symbols", "sample_tags", "sample_ratios", "sample_authors", "author"} \
        <= set(metadata.tables)


def test_parent_table_columns_and_types():
    metadata = sqlalchemy_metadata([resolve_schema(Sample)])
    table = metadata.tables["sample"]
    assert table.c["sid"].primary_key
    assert isinstance(table.c["sid"].type, sqlalchemy.Integer)
    assert isinstance(table.c["formula"].type, sqlalchemy.Text) and not table.c["formula"].nullable
    assert isinstance(table.c["spacegroup"].type, sqlalchemy.Integer)
    assert isinstance(table.c["energy"].type, sqlalchemy.Float)
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
    assert {fk.target_fullname for fk in reference.foreign_keys} == {"author.sid"}


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
    assert {fk.target_fullname for fk in symbols.c["sample_sid"].foreign_keys} == {"sample.sid"}
    assert not symbols.c["symbols_index"].nullable
    assert isinstance(symbols.c["symbols"].type, sqlalchemy.Text)
    assert {index.name for index in symbols.indexes} == {"ix_sample_symbols_sample_sid"}
    coords = metadata.tables["sample_coords"]
    for i in range(3):
        assert isinstance(coords.c[f"coords_{i}"].type, sqlalchemy.Float)
    assert isinstance(coords.c["coords_exact"].type, sqlalchemy.Text)
    authors = metadata.tables["sample_authors"]
    assert {fk.target_fullname for fk in authors.c["authors_sid"].foreign_keys} == {"author.sid"}


def test_table_for_is_idempotent_per_metadata():
    metadata = sqlalchemy.MetaData()
    schema = resolve_schema(Sample)
    assert table_for(schema, metadata) is table_for(schema, metadata)


# --------------------------------------------------------------------- round trips


def test_round_trip_equality(database):
    sample = make_sample()
    sid = SqlStore(database).save(sample)
    fetched = SqlStore(database).fetch(Sample, sid)  # a fresh store: no identity cache involved
    assert fetched is not sample
    assert fetched == sample
    assert isinstance(fetched.symbols, list)
    assert isinstance(fetched.tags, tuple)
    assert fetched.scratch == "unstored"
    assert fetched.natoms == 3


def test_round_trip_exact_rationals(database):
    sample = make_sample()
    fetched = SqlStore(database).fetch(Sample, SqlStore(database).save(sample))
    assert fetched.ratio == Fraction(1, 3)
    assert fetched.scale.to_fraction() == Fraction(2, 7)
    assert fetched.cell == sample.cell
    assert fetched.coords == sample.coords
    assert fetched.ratios == [Fraction(1, 3), Fraction(-7, 5)]


def test_round_trip_optionals_present(database):
    sample = make_sample(note="a note", weight=1.25)
    fetched = SqlStore(database).fetch(Sample, SqlStore(database).save(sample))
    assert fetched.note == "a note"
    assert fetched.weight == 1.25


def test_round_trip_optional_reference_none(database):
    sample = make_sample(reference=None)
    fetched = SqlStore(database).fetch(Sample, SqlStore(database).save(sample))
    assert fetched.reference is None
    assert fetched == sample


def test_derived_property_is_stored_in_parent_table(database):
    sid = SqlStore(database).save(make_sample())
    with database.engine.connect() as connection:
        stored = connection.execute(sqlalchemy.text(f"SELECT natoms FROM sample WHERE sid = {sid}")).scalar_one()
    assert stored == 3


def test_fixed_array_accepts_single_row_for_shape_1_n(database):
    store = SqlStore(database)
    sid = store.save(RowVector(FracVector.create([Fraction(1, 3), 1, 0])))
    fetched = SqlStore(database).fetch(RowVector, sid)
    assert fetched.vec == FracVector.create([[Fraction(1, 3), 1, 0]])


def test_fixed_array_wrong_shape_raises_naming_field(database):
    with pytest.raises(ValueError, match="cell"):
        SqlStore(database).save(make_sample(cell=FracVector.create([[1, 0], [0, 1]])))


# --------------------------------------------------------------------- dedup


def test_dedup_content_id_reuses_row(database):
    store = SqlStore(database)
    sid1 = store.save(Author("Ada", 1852))
    sid2 = store.save(Author("Ada", 1852))
    assert sid1 == sid2
    assert _count(database, "author") == 1


def test_dedup_content_id_does_not_duplicate_children(database):
    store = SqlStore(database)
    sid1 = store.save(make_sample())
    sid2 = store.save(make_sample())
    assert sid1 == sid2
    assert _count(database, "sample") == 1
    assert _count(database, "sample_symbols") == 3
    assert _count(database, "sample_authors") == 1
    assert _count(database, "author") == 2  # Ada (child element) + Boole (reference)


def test_dedup_by_value_matches_parent_columns(database):
    store = SqlStore(database)
    sid1 = store.save(AuthorTag(Author("Ada", 1852), "role", "pioneer"))
    sid2 = store.save(AuthorTag(Author("Ada", 1852), "role", "pioneer"))
    assert sid1 == sid2
    assert _count(database, "author_tag") == 1
    sid3 = store.save(AuthorTag(Author("Ada", 1852), "role", "mathematician"))
    assert sid3 != sid1
    assert _count(database, "author_tag") == 2


def test_dedup_none_always_inserts(database):
    store = SqlStore(database)
    sid1 = store.save(LogEvent("started"))
    sid2 = store.save(LogEvent("started"))
    assert sid1 != sid2
    assert _count(database, "log_event") == 2


# --------------------------------------------------------------------- transactions


def test_transaction_rolls_back_on_exception(database):
    store = SqlStore(database)
    store.ensure_tables(Author)
    with pytest.raises(RuntimeError, match="boom"):
        with store.transaction():
            store.save(Author("X", 1))
            raise RuntimeError("boom")
    assert _count(database, "author") == 0


def test_transaction_shares_connection_and_commits(database):
    store = SqlStore(database)
    with store.transaction():
        sid1 = store.save(Author("A", 1))
        with store.transaction():  # nesting is flat: joins the outer transaction
            sid2 = store.save(Author("B", 2))
        assert store.fetch(Author, sid1).name == "A"  # reads inside see uncommitted writes
    assert sid1 != sid2
    assert _count(database, "author") == 2


def test_save_outside_transaction_autocommits(database):
    sid = SqlStore(database).save(Author("C", 3))
    assert SqlStore(database).fetch(Author, sid) == Author("C", 3)


# --------------------------------------------------------------------- identity cache


def test_fetch_returns_same_object_while_alive(database):
    store = SqlStore(database)
    sid = store.save(Author("Ada", 1852))
    assert store.fetch(Author, sid) is store.fetch(Author, sid)


def test_save_then_fetch_returns_saved_object(database):
    store = SqlStore(database)
    author = Author("Ada", 1852)
    sid = store.save(author)
    assert store.fetch(Author, sid) is author
    assert store.sid_of(author) == sid


def test_sid_of_unknown_object_is_none(database):
    assert SqlStore(database).sid_of(Author("New", 1900)) is None


def test_sid_of_tracks_unhashable_instances(database):
    """A storable class holding a list is unhashable, and must still be tracked.

    The reverse cache is keyed by equality, which such an instance cannot
    support; without an identity-keyed fallback ``sid_of`` reported a
    just-saved instance as never stored, and ``referring`` then raised for it.
    """
    store = SqlStore(database)
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


def test_implicit_rollback_clears_recursively_saved_child_caches(database):
    store = SqlStore(database)
    store.save(RollbackParent(RollbackChild("kept"), "unique"))
    rolled_back = RollbackChild("rolled back")

    with pytest.raises(sqlalchemy.exc.IntegrityError):
        store.save(RollbackParent(rolled_back, "unique"))

    assert store.sid_of(rolled_back) is None
    with pytest.raises(KeyError):
        store.fetch(RollbackChild, 2)


def test_optional_identity_metadata_none_matches_freshly_loaded_empty_child(database):
    source = OptionalChildMetadata("same")
    sid = SqlStore(database).save(source)
    fresh = SqlStore(database)

    assert fresh.fetch(OptionalChildMetadata, sid).notes == []
    assert fresh.save(source) == sid
    with pytest.raises(EntryMetadataConflictError, match="notes"):
        fresh.save(OptionalChildMetadata("same", ["different"]))


def test_fetch_missing_sid_raises_keyerror(database):
    store = SqlStore(database)
    store.ensure_tables(Author)
    with pytest.raises(KeyError):
        store.fetch(Author, 424242)


# --------------------------------------------------------------------- referring


def test_referring_returns_matching_join_objects(database):
    store = SqlStore(database)
    ada = Author("Ada", 1852)
    boole = Author("Boole", 1854)
    store.save(ada)
    store.save(boole)
    tag1 = AuthorTag(ada, "role", "pioneer")
    tag2 = AuthorTag(ada, "field", "computing")
    tag3 = AuthorTag(boole, "field", "logic")
    for tag in (tag1, tag2, tag3):
        store.save(tag)
    assert store.referring(AuthorTag, field="author", to=ada) == [tag1, tag2]
    assert store.referring(AuthorTag, field="author", to=boole) == [tag3]


def test_referring_rejects_unknown_object(database):
    store = SqlStore(database)
    store.ensure_tables(AuthorTag)
    with pytest.raises(ValueError, match="has not been stored"):
        store.referring(AuthorTag, field="author", to=Author("New", 1900))


def test_referring_rejects_non_reference_field_and_wrong_target(database):
    store = SqlStore(database)
    ada = Author("Ada", 1852)
    store.save(ada)
    with pytest.raises(SchemaError):
        store.referring(AuthorTag, field="tag", to=ada)
    with pytest.raises(SchemaError):
        store.referring(AuthorTag, field="author", to=LogEvent("x"))


# --------------------------------------------------------------------- fetch_by_content_id


def test_fetch_by_content_id_found_and_missing(database):
    store = SqlStore(database)
    ada = Author("Ada", 1852)
    store.save(ada)
    assert store.fetch_by_content_id(Author, content_id(ada)) is ada
    assert store.fetch_by_content_id(Author, "0" * 64) is None


def test_fetch_by_content_id_rejects_other_policies(database):
    with pytest.raises(SchemaError, match="content_id"):
        SqlStore(database).fetch_by_content_id(LogEvent, "0" * 64)


# --------------------------------------------------------------------- database lifecycle


def test_in_memory_database_is_shared_across_operations():
    with Database.sqlite() as database:
        sid = SqlStore(database).save(Author("Ada", 1852))
        assert SqlStore(database).fetch(Author, sid) == Author("Ada", 1852)


def test_file_backed_database_persists_across_instances(tmp_path):
    path = tmp_path / "authors.sqlite"
    database = Database.sqlite(path)
    sid = SqlStore(database).save(Author("Ada", 1852))
    database.dispose()
    with Database.sqlite(path) as reopened:
        assert SqlStore(reopened).fetch(Author, sid) == Author("Ada", 1852)


def test_file_backed_duckdb_persists_and_continues_sids(tmp_path):
    pytest.importorskip("duckdb_engine")
    path = tmp_path / "authors.duckdb"
    database = Database.duckdb(path)
    sid = SqlStore(database).save(Author("Ada", 1852))
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
    code = "import httk.data.db\nimport sys\nassert 'sqlalchemy' not in sys.modules, 'sqlalchemy was imported'"
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
        "import httk.data.db\n"
        "try:\n"
        "    httk.data.db.Database\n"
        "except ImportError as error:\n"
        "    assert 'httk-data[db]' in str(error), error\n"
        "else:\n"
        "    raise SystemExit('expected an ImportError')\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True, env=_subprocess_env())
