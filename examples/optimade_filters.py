"""Querying stored dataclasses with OPTIMADE filter strings

`optimade_filter_searcher(store, cls, filter_string)` takes an **OPTIMADE
filter string** — the query language of the OPTIMADE API, as it arrives in a
URL — and returns a searcher over the stored rows of `cls`. It is the bridge
between the text a user (or a remote client) writes and the search DSL of the
*searching* example: the filter is parsed by httk-core's OPTIMADE grammar,
translated by `httk.data.optimade_query` against a handler table derived from
the class's storage schema, and executed as an ordinary search. Each match
is available as the friendly `searcher.results().scalars()` stream.

This is useful well before you serve anything: it lets a filter string be
evaluated against a local database with no HTTP server, no adapter and no
OPTIMADE endpoint in sight. (For the full serving path, see the
*serve as OPTIMADE* example.)

## Property names are the *served* names

An OPTIMADE filter may only mention properties the API declares, and a
database-specific property must carry a registered provider prefix. The stored
fields of `cls` are therefore filterable as `_httk_<field>` — `_httk_name`,
`_httk_x`, `_httk_symbols` — and **not** as the bare field name. This holds
inside dotted relationship paths too: `refs._httk_doi`, not `refs.doi`.

The rules for names that do not resolve follow the OPTIMADE specification
rather than being uniformly strict:

- an unknown name *carrying* the prefix is an error
  (`FilterTranslationError`, category `"unrecognized-property"`), because a
  client asking for `_httk_bananas` has made a mistake worth reporting;
- an unknown name *without* a prefix matches nothing, silently, because it may
  be a standard property this implementation does not support.

## Operators

Everything the grammar offers is available. Comparisons on numeric and string
properties; `STARTS WITH`, `ENDS WITH` and `CONTAINS` for substrings; and the
list operators `HAS`, `HAS ALL`, `HAS ANY`, `HAS ONLY` on variable-length
fields, which map onto the set semantics described in the *searching* example.
`NOT` negates, with the same set reading: `NOT _httk_symbols HAS "O"` selects
the records where *no* symbol is O.

Rational fields (`Fraction` and friends) are stored exactly but compared on
their float companion columns, so a numeric filter on one is
documented-approximate.

## Relationship filters

`related_classes={"refs": Publication}` declares that the relationship type
`refs` is served from the `Publication` field of `cls` — either a reference
field or a `list[Publication]` child field. That unlocks depth-1 dotted
filters, resolved as a **two-phase semi-join**: a nested filter search over
`Publication` collects the sids of the matching publications, and those are
then matched against the relationship's own column. Two consequences worth
knowing:

- `NOT refs._httk_year >= 2000` is the complement over the *parent* rows, so it
  includes rows with no related publication at all.
- A related filter matching nothing yields an empty result, not an error.

Relationship *ids* are filterable as `refs.id HAS "refs-<sid>"`, using the same
`"<entry type>-<sid>"` ids `StoreEntryProvider` mints by default. Nesting
deeper than one level (`refs.other._httk_x`) raises `FilterTranslationError`
with category `"not-implemented"`.
"""

from dataclasses import dataclass
from fractions import Fraction

from httk.data.db import Database, SqlStore, optimade_filter_searcher
from httk.data.optimade_query import FilterTranslationError
from httk.data.query import Searcher

HTTK_EXAMPLE_REQUIRES = ["sqlalchemy"]


@dataclass(frozen=True)
class Publication:
    doi: str
    year: int


@dataclass(frozen=True)
class Material:
    name: str
    x: Fraction
    symbols: list[str]
    ref: Publication | None = None


PUB_A = Publication("10.1000/alpha", 1999)
PUB_B = Publication("10.2000/beta", 2005)

MATERIALS = [
    Material("alpha oxide", Fraction(1, 2), ["O", "H"], PUB_A),
    Material("beta metal", Fraction(5, 2), ["Fe"], PUB_B),
    Material("gamma oxide", Fraction(7, 2), ["O"], None),
]

#: Plain filters over the material's own (prefixed) properties.
PLAIN_FILTERS = [
    '_httk_name = "gamma oxide"',
    "_httk_x > 1",
    "_httk_x <= 2.5 AND _httk_x > 0",
    '_httk_name STARTS WITH "beta"',
    '_httk_name CONTAINS "oxide"',
    '_httk_name ENDS WITH "metal"',
    '_httk_symbols HAS "O"',
    '_httk_symbols HAS ALL "O","H"',
    '_httk_symbols HAS ANY "H","Fe"',
    '_httk_symbols HAS ONLY "O","H"',
    'NOT _httk_symbols HAS "O"',
    '_httk_symbols HAS "O" AND _httk_x > 1',
    'bananas = 3',
]

#: Depth-1 filters through the 'refs' relationship, needing related_classes.
RELATED_FILTERS = [
    'refs._httk_doi CONTAINS "10."',
    'refs._httk_doi CONTAINS "10.2"',
    "refs._httk_year >= 2000",
    "NOT refs._httk_year >= 2000",
    '_httk_x > 1 AND refs._httk_doi CONTAINS "10.2"',
    'refs._httk_doi CONTAINS "nomatch"',
]

#: Filters that cannot be translated, and the category each reports.
REJECTED_FILTERS = [
    "_httk_bananas = 3",
    'id = "materials-1"',
    "refs.other._httk_x = 1",
]


def populate() -> SqlStore:
    store = SqlStore(Database.sqlite())
    with store.transaction():
        for material in MATERIALS:
            store.save(material)
    return store


def names(searcher: Searcher) -> list[str]:
    """The `name` of every matched material."""
    return [material.name for material in searcher.results().scalars()]


def show_plain_filters(store: SqlStore) -> None:
    print("== Filters over the material's own properties ==")
    for filter_string in PLAIN_FILTERS:
        searcher = optimade_filter_searcher(store, Material, filter_string)
        print(f"  {filter_string:<38} -> {names(searcher)}")
    print("  ('bananas' carries no provider prefix, so it is an unsupported standard")
    print("   property and matches nothing rather than raising.)")
    print()


def show_related_filters(store: SqlStore) -> None:
    print("== Depth-1 relationship filters (two-phase semi-join) ==")
    for filter_string in RELATED_FILTERS:
        searcher = optimade_filter_searcher(store, Material, filter_string, related_classes={"refs": Publication})
        print(f"  {filter_string:<45} -> {names(searcher)}")
    print("  'NOT refs._httk_year >= 2000' also returns 'gamma oxide', which has no")
    print("  related publication at all: the negation is over the material rows.")
    print()

    print("== Filtering on a relationship id ==")
    sid = store.sid_of(PUB_A)
    filter_string = f'refs.id HAS "refs-{sid}"'
    searcher = optimade_filter_searcher(store, Material, filter_string, related_classes={"refs": Publication})
    print(f"  {filter_string:<45} -> {names(searcher)}")
    print("  The id format is the one StoreEntryProvider mints: '<entry type>-<sid>'.")
    print()


def show_rejections(store: SqlStore) -> None:
    print("== Filters that cannot be translated ==")
    for filter_string in REJECTED_FILTERS:
        try:
            optimade_filter_searcher(store, Material, filter_string, related_classes={"refs": Publication})
        except FilterTranslationError as error:
            print(f"  {filter_string:<28} -> {error.category}: {error}")
        else:
            raise AssertionError(f"expected {filter_string!r} to be rejected")


def main() -> None:
    store = populate()
    show_plain_filters(store)
    show_related_filters(store)
    show_rejections(store)


if __name__ == "__main__":
    main()
