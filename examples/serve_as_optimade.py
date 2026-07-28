"""Serving a database as an OPTIMADE API

The last step of the chain: a store full of frozen dataclasses becomes an
OPTIMADE API without either module knowing about the other. `StoreEntryProvider`
implements httk-core's neutral `httk.core.EntryProvider` contract over an
`SqlStore`; *httk-optimade*'s `adapter_from_providers` consumes that contract.
*httk-data* never imports *httk-optimade*, and *httk-optimade* never imports
*httk-data* — they meet only at the contract that httk-core owns.

## What the provider derives from the schema

`StoreEntryProvider(store, {"books": Book, "writers": Writer})` needs nothing
but the mapping from entry-type name to storable class. From each class's
storage schema it derives:

- an `EntryTypeDefinition` with one OPTIMADE property per servable stored
  field, named with a registered database-specific prefix (`_httk_` by
  default), plus the mandatory `id` and `type`. You may pass your own
  `definitions=` instead, as long as they describe everything served.
- JSON-able records: rationals are served as their nearest floats (the exact
  values stay in the database), datetimes as ISO text, fixed-shape tensors as
  nested lists, and `stored_property` values as ordinary properties.
- relationships, for reference fields and `list[Storable]` fields **whose
  target class is also served**. A `Related` marker on the field adds the
  OPTIMADE `role`/`description` metadata; `StorageInfo(links=...)` lets a
  separate join class contribute relationships without being served itself.

Fields with no OPTIMADE value representation — `bytes`, custom codecs — are
simply not served, and a reference field whose target class is not served
becomes neither a property nor a relationship.

## Running a query

`adapter_from_providers([provider])` builds a backend adapter: it reads the
definitions, loads the records, and wires the filter handlers and the
relationship id fields. `execute_query(adapter, entries, response_fields, ...)`
is then the whole OPTIMADE query pipeline minus HTTP — the same call an
OPTIMADE web endpoint makes after parsing a request. It takes the filter as an
already-parsed AST from `parse_optimade_filter`, so this example goes
end to end from a filter *string* to served records.

Note the difference from the *OPTIMADE filters* example: there the filter runs
as SQL against the store and yields dataclass instances; here it runs through
the serving layer and yields **records** — id, type and the selected response
fields — which is what an API actually returns.

## Ids

By default an entry is identified as `"<entry type>-<sid>"`, using the row's
integer sid: `books-1`, `writers-3`. An `id_of` callback replaces that scheme
everywhere at once, including inside relationships.
"""

import datetime
from dataclasses import dataclass
from fractions import Fraction
from typing import Annotated, Any

from httk.core import Related, parse_optimade_filter, stored_property

from httk.data.db import Database, SqlStore, StoreEntryProvider

HTTK_EXAMPLE_REQUIRES = ["sqlalchemy", "httk.optimade"]


@dataclass(frozen=True)
class Writer:
    name: str
    born: int


@dataclass(frozen=True)
class Book:
    title: str
    pages: int
    price: Fraction  # exact in the database, served as a float
    in_print: bool
    published: datetime.datetime  # served as ISO text
    cover: bytes  # no OPTIMADE representation: not served
    keywords: list[str]
    author: Annotated[Writer | None, Related(role="author", description="Wrote the book")] = None

    @stored_property
    def nkeywords(self) -> int:
        return len(self.keywords)


ADA = Writer("Ada", 1815)
BOOLE = Writer("Boole", 1815)
CARA = Writer("Cara", 1820)

BOOKS = [
    Book(
        title="Analytical Engines",
        pages=350,
        price=Fraction(1, 3),
        in_print=True,
        published=datetime.datetime(2026, 7, 24, 12, 30, 0),  # noqa: DTZ001 (naive datetime is the storage contract)
        cover=b"\x00\xff",
        keywords=["computing", "history"],
        author=ADA,
    ),
    Book(
        title="Laws of Thought",
        pages=420,
        price=Fraction(7, 2),
        in_print=True,
        published=datetime.datetime(2024, 3, 1, 9, 0, 0),  # noqa: DTZ001 (naive datetime is the storage contract)
        cover=b"",
        keywords=["logic"],
        author=BOOLE,
    ),
    Book(
        title="Silence",
        pages=120,
        price=Fraction(-7, 5),
        in_print=False,
        published=datetime.datetime(2020, 1, 1, 0, 0, 0),  # noqa: DTZ001 (naive datetime is the storage contract)
        cover=b"",
        keywords=[],
    ),
]

#: OPTIMADE filter strings to run through the serving layer, with the fields to return.
QUERIES: list[tuple[str, list[str], str]] = [
    ("books", ["id", "_httk_title", "_httk_pages"], "_httk_pages > 200"),
    ("books", ["id", "_httk_price"], "_httk_price < 0"),
    ("books", ["id", "_httk_keywords"], '_httk_keywords HAS "history"'),
    ("books", ["id", "_httk_title"], '_httk_in_print = TRUE AND _httk_title CONTAINS "o"'),
    ("writers", ["id", "_httk_name"], "_httk_born = 1820"),
]


def populate() -> SqlStore:
    store = SqlStore(Database.sqlite())
    with store.transaction():
        for writer in (ADA, BOOLE, CARA):  # saved first, so their sids are 1, 2, 3
            store.save(writer)
        for book in BOOKS:
            store.save(book)
    return store


def show_provider(provider: StoreEntryProvider) -> None:
    """The provider's answers to the entry-provider contract, derived from the schema."""
    print("== Auto-generated entry-type definitions ==")
    for entry_type, definition in sorted(provider.entry_types().items()):
        print(f"  {entry_type}: {', '.join(sorted(definition.properties))}")
    books = provider.entry_types()["books"].properties
    print(f"  _httk_price is served as: {books['_httk_price'].optimade_type} (a Fraction in the database)")
    print(f"  _httk_published is served as: {books['_httk_published'].optimade_type}")
    print(f"  _httk_nkeywords is served as: {books['_httk_nkeywords'].optimade_type} (a stored_property)")
    print("  '_httk_cover' (bytes) is absent: no OPTIMADE value representation.")
    print("  '_httk_author' is absent as a property: references surface as relationships.")
    print()

    print("== Records, keyed through property_keys() ==")
    property_keys = provider.property_keys("books")
    for record in provider.records("books"):
        served = {name: record[key] for name, key in property_keys.items()}
        print(
            f"  {served['id']}: title={served['_httk_title']!r}, price={served['_httk_price']}, "
            f"published={served['_httk_published']!r}, keywords={served['_httk_keywords']}"
        )
    print()

    print("== Relationships ==")
    for entry_id, related in provider.relationships("books").items():
        for entry in related:
            print(f"  {entry_id} -> {entry.id} (type {entry.entry_type}, role={entry.role}, {entry.description})")
    print("  'Silence' has no author, so it declares no relationships.")
    print()


def show_queries(provider: StoreEntryProvider) -> None:
    """Hand the provider to httk-optimade and run OPTIMADE queries against it."""
    from httk.optimade import adapter_from_providers
    from httk.optimade.backend import execute_query

    adapter = adapter_from_providers([provider])
    print("== The adapter serves ==")
    print(f"  {sorted(adapter.schema.all_entries)}")
    print()

    print("== OPTIMADE queries, executed against the database ==")
    for entry_type, response_fields, filter_string in QUERIES:
        results = list(
            execute_query(
                adapter,
                [entry_type],
                response_fields,
                [],  # unknown response fields: none
                100,  # response limit
                0,  # response offset
                parse_optimade_filter(filter_string),
            )
        )
        print(f"  {entry_type}?filter={filter_string}")
        for result in results:
            rendered: dict[str, Any] = {name: result.values[name] for name in response_fields}
            print(f"    {rendered}")
        if not results:
            print("    (no matches)")
    print()


def show_custom_ids(store: SqlStore) -> None:
    """`id_of` replaces the default '<entry type>-<sid>' scheme everywhere."""

    def id_of(entry_type: str, sid: int, obj: Any) -> str:
        label = getattr(obj, "title", None) or obj.name
        return f"{entry_type}/{label.lower().replace(' ', '-')}"

    provider = StoreEntryProvider(store, {"books": Book, "writers": Writer}, id_of=id_of)
    print("== Custom ids ==")
    property_keys = provider.property_keys("books")
    for record in provider.records("books"):
        print(f"  {record[property_keys['id']]}")
    for entry_id, related in provider.relationships("books").items():
        for entry in related:
            print(f"  {entry_id} -> {entry.id} (type {entry.entry_type})")


def main() -> None:
    store = populate()
    provider = StoreEntryProvider(store, {"books": Book, "writers": Writer})
    show_provider(provider)
    show_queries(provider)
    show_custom_ids(store)


if __name__ == "__main__":
    main()
