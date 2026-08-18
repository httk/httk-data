"""Storing frozen dataclasses in a database, exactly

`httk.store.backend.sql` stores **plain frozen dataclasses**. There is no base class to
inherit, no metaclass, no ORM session: any frozen dataclass whose field types
resolve is storable, and the vocabulary that fine-tunes how it is stored lives
in *httk-core* (`Indexed`, `Unique`, `Skip`, `Shape`, `StorageInfo`,
`stored_property`) so that a domain module can declare storable classes without
depending on httk-store at all.

This example declares one such class and then does, in order, the five things
you actually do with a store.

## Declaring

The markers below are `typing.Annotated` metadata plus one class variable:

- `Annotated[str, Indexed()]` — put a single-column index on this field.
- `__httk_storage__: ClassVar[StorageInfo]` — table-level settings: composite
  indexes, and the deduplication policy (`"content_id"` by default).
- `Annotated[FracVector, Shape(3, 3)]` — a fixed-shape tensor, flattened into
  nine columns of the parent table.
- `Annotated[FracVector, Shape(0, 3)]` — a *variable* number of 3-vectors; `0`
  means "any length", so this becomes a child table with one row per vector.
- `list[str]` — likewise a child table. `Author | None` becomes a nullable
  foreign key, and the referenced `Author` is saved recursively first.
- `Annotated[str, Skip()]` — not stored at all; it comes back as the field's
  default.
- `@stored_property` — a *derived* value that is computed once, written into
  its own column (so it is queryable and indexable), and recomputed rather than
  read back when the instance is reconstructed.

## Round-tripping, exactly

`fetch` gives back an equal instance — a lazy row by default, decoded field by
field on access (`eager=True` materializes it up front) — and for rational
values it gives back an *exactly* equal one. `Fraction(1, 3)` is not representable as a float, so the
example prints both the reconstructed value and what a float round-trip would
have cost. This is not an accident of the encoding: every rational field is
stored as a canonical exact text column (the round-trip source of truth)
alongside float companion columns that exist only so that SQL can compare and
sort. Comparisons on rational fields are therefore documented-approximate,
while the values themselves are lossless.

## Content identity and deduplication

The default policy is `dedup="content_id"`: before inserting, the store hashes
the instance's canonical form and looks for a row with that hash. Saving an
*equal* instance therefore returns the sid of the existing row instead of
writing a second one — which the row count below shows. `content_id(obj)` is a
pure function you can compute without a database, and `fetch_by_content_id`
looks a row up by it. (The alternatives are `dedup="by_value"`, matching on the
parent columns only, and `dedup="none"`, which always inserts.)

## Join classes and `referring`

A "tag" or "link" class is just another storable dataclass holding a reference.
`store.referring(Tag, field="measurement", to=measurement)` returns every
stored `Tag` whose `measurement` field points at that instance — the reverse of
the foreign key, without writing a query.

## Persistence

`Backend.sqlite(path)` is a file; `Backend.sqlite()` is in memory. The last
section saves into a file, disposes the database entirely, reopens it in a
second `Backend`/`SqlStore` pair, and fetches the same sid back.
"""

import tempfile
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Annotated, ClassVar

from httk.core import FracVector
from httk.core.storage import Indexed, Shape, Skip, StorageInfo, content_id, stored_property

from httk.store.backend.schema import resolve_schema
from httk.store.backend.sql import Backend, SqlStore

HTTK_EXAMPLE_REQUIRES = ["sqlalchemy"]


@dataclass(frozen=True)
class Author:
    """A referenced class: saved recursively, deduplicated on its own."""

    name: str
    year: int


@dataclass(frozen=True)
class Measurement:
    """A storable record exercising every kind of field the mapper knows."""

    #: Table-level storage settings: here one composite index.
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(indexes=(("formula", "spacegroup"),))

    formula: Annotated[str, Indexed()]  # indexed column
    spacegroup: int  # plain column
    energy: Fraction  # exact text column + float companion
    cell: Annotated[FracVector, Shape(3, 3)]  # fixed tensor, flattened inline
    coords: Annotated[FracVector, Shape(0, 3)]  # variable rows -> child table
    symbols: list[str]  # child table
    author: Author | None = None  # nullable foreign key
    scratch: Annotated[str, Skip()] = "not stored"  # never written

    @stored_property
    def natoms(self) -> int:
        """Derived, stored in its own column, recomputed on load."""
        return len(self.symbols)


@dataclass(frozen=True)
class Tag:
    """A join class: a reference plus the metadata being attached to it."""

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(dedup="by_value")

    measurement: Measurement
    name: str
    value: str


ADA = Author("Lovelace", 1843)

PEROVSKITE = Measurement(
    formula="CaTiO3",
    spacegroup=221,
    energy=Fraction(-1, 3),
    cell=FracVector([[1, Fraction(1, 3), 0], [0, 1, 0], [0, 0, Fraction(2, 3)]]),
    coords=FracVector([[0, 0, 0], [Fraction(1, 2), Fraction(1, 2), Fraction(1, 2)]]),
    symbols=["Ca", "Ti", "O"],
    author=ADA,
)

ROCKSALT = Measurement(
    formula="NaCl",
    spacegroup=225,
    energy=Fraction(-5, 4),
    cell=FracVector([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
    coords=FracVector([[0, 0, 0]]),
    symbols=["Na", "Cl"],
)


def row_count(store: SqlStore, cls: type) -> int:
    """The number of stored rows of ``cls``, via the query DSL (never raw SQL)."""
    searcher = store.searcher()
    searcher.output(searcher.variable(cls), "obj")
    return searcher.count()


def show_schema() -> None:
    """`resolve_schema` reads a storable class into the mapper's own description."""
    schema = resolve_schema(Measurement)
    print("== How Measurement maps onto tables ==")
    print(f"  parent table: {schema.table_name} (dedup policy: {schema.dedup})")
    for spec in schema.fields:
        columns = ", ".join(column.name for column in spec.columns)
        if spec.child is not None:
            print(f"  {spec.field:<11} role={spec.role:<9} -> child table {spec.child.table_name}")
        else:
            print(f"  {spec.field:<11} role={spec.role:<9} -> columns: {columns}")
    print(f"  composite indexes: {schema.composite_indexes}")
    print()


def show_round_trip(store: SqlStore, database: Backend) -> int:
    """Save, then fetch through a *fresh* store so no identity cache is involved."""
    print("== Save and fetch back ==")
    sid = store.save(PEROVSKITE)
    print(f"  saved as sid {sid}")

    fetched = SqlStore(database).fetch(Measurement, sid)
    print(f"  fetched a different object: {fetched is not PEROVSKITE}")
    print(f"  ... that compares equal:    {fetched == PEROVSKITE}")
    print(f"  list stays a list:          {fetched.symbols!r}")
    print(f"  derived property recomputed: natoms = {fetched.natoms}")
    print(f"  skipped field is the default:  {fetched.scratch!r}")
    print(f"  reference reconstructed:    {fetched.author}")
    print()

    print("== Rationals round-trip exactly ==")
    print(
        f"  energy stored: {PEROVSKITE.energy}   fetched: {fetched.energy}   equal: "
        f"{fetched.energy == PEROVSKITE.energy}"
    )
    print(f"  cell[0][1]:    {fetched.cell[0][1].to_fraction()}  (exact, not {float(Fraction(1, 3))!r})")
    print(f"  a float round-trip would instead have given: {Fraction(float(Fraction(1, 3)))}")
    print(
        f"  whole tensors equal: cell {fetched.cell == PEROVSKITE.cell}, coords {fetched.coords == PEROVSKITE.coords}"
    )
    print()
    return sid


def show_dedup(store: SqlStore, sid: int) -> None:
    """An equal instance is not stored twice: the content id resolves to the same row."""
    print("== Content identity and deduplication ==")
    print(f"  content_id(PEROVSKITE) = {content_id(PEROVSKITE)[:16]}...")

    twin = Measurement(
        formula="CaTiO3",
        spacegroup=221,
        energy=Fraction(-1, 3),
        cell=FracVector([[1, Fraction(1, 3), 0], [0, 1, 0], [0, 0, Fraction(2, 3)]]),
        coords=FracVector([[0, 0, 0], [Fraction(1, 2), Fraction(1, 2), Fraction(1, 2)]]),
        symbols=["Ca", "Ti", "O"],
        author=Author("Lovelace", 1843),
        scratch="a different scratch value: not stored, so not part of the identity",
    )
    twin_sid = store.save(twin)
    print(f"  saving an equal instance returned sid {twin_sid} (the same row: {twin_sid == sid})")
    print(f"  measurement rows: {row_count(store, Measurement)}, author rows: {row_count(store, Author)}")

    by_content = store.fetch_by_content_id(Measurement, content_id(PEROVSKITE))
    print(f"  fetch_by_content_id found it: {by_content is not None}")
    print(f"  an unknown content id gives None: {store.fetch_by_content_id(Measurement, '0' * 64) is None}")
    print()


def show_referring(store: SqlStore) -> None:
    """`referring` walks a foreign key backwards, from the target to the join rows."""
    print("== Join classes and referring() ==")
    store.save(ROCKSALT)
    tags = [
        Tag(PEROVSKITE, "quality", "good"),
        Tag(PEROVSKITE, "source", "experiment"),
        Tag(ROCKSALT, "quality", "poor"),
    ]
    for tag in tags:
        store.save(tag)

    for measurement in (PEROVSKITE, ROCKSALT):
        found = store.referring(Tag, field="measurement", to=measurement)
        rendered = ", ".join(f"{tag.name}={tag.value}" for tag in found)
        print(f"  tags referring to {measurement.formula}: {rendered}")
    print()


def show_persistence() -> None:
    """A file-backed database outlives the process objects that wrote it."""
    print("== Persistence to a file ==")
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "measurements.sqlite"

        database = Backend.sqlite(path)
        sid = SqlStore(database, entry_records={}).save(ROCKSALT)
        database.dispose()  # close the engine entirely: nothing is left in memory
        print(f"  wrote sid {sid} to {path.name} ({path.stat().st_size} bytes on disk)")

        with Backend.sqlite(path) as reopened:
            store = SqlStore(reopened)
            fetched = store.fetch(Measurement, sid)
            print(f"  reopened in a second store, fetched: {fetched.formula}, energy {fetched.energy}")
            print(f"  equal to what was written: {fetched == ROCKSALT}")


def main() -> None:
    show_schema()
    with Backend.sqlite() as database:  # in memory; Backend.sqlite(path) for a file
        store = SqlStore(database, entry_records={})
        sid = show_round_trip(store, database)
        show_dedup(store, sid)
        show_referring(store)
    show_persistence()


if __name__ == "__main__":
    main()
