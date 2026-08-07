# Storing frozen dataclasses in a database, exactly

`httk.data.db` stores **plain frozen dataclasses**. There is no base class to
inherit, no metaclass, no ORM session: any frozen dataclass whose field types
resolve is storable, and the vocabulary that fine-tunes how it is stored lives
in *httk-core* (`Indexed`, `Unique`, `Skip`, `Shape`, `StorageInfo`,
`stored_property`) so that a domain module can declare storable classes without
depending on httk-data at all.

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

`fetch` gives back an equal instance, and for rational values it gives back an
*exactly* equal one. `Fraction(1, 3)` is not representable as a float, so the
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

`Database.sqlite(path)` is a file; `Database.sqlite()` is in memory. The last
section saves into a file, disposes the database entirely, reopens it in a
second `Database`/`SqlStore` pair, and fetches the same sid back.

```{literalinclude} ../../examples/storable_records.py
:language: python
:lines: 66-
```
