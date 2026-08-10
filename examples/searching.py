"""Querying a store with the search DSL

Once frozen dataclasses are stored (see the *storable records* example), you
query them with `store.searcher()`. The DSL is deliberately not SQL and not an
ORM: it is a small backend-agnostic protocol (`httk.store.query`) that the SQL
layer implements and that other stores — including *httk-serve*'s in-memory
reference store — implement identically, so the same query program runs
unchanged against either.

A search is built, not written. Four calls do everything:

`variable(cls)`
: Bind a class to a query variable. Two variables of the same class self-join,
  so "another record with the same spacegroup" is expressible.

`add(expression)`
: Add a condition. Expressions come from comparing a variable's fields;
  `&`, `|` and `~` combine them. Several `add()` calls are ANDed.

`output(variable_or_field, name)`
: Declare what a match yields — the whole reconstructed object, or one field.
  Declaration order is result order.

iteration / `count()`
: Run it. Iterating yields one `SearchResult` per match; `count()` returns how
  many there are, ignoring `set_limit`/`add_offset`.

## What the fields give you

Comparisons (`==`, `!=`, `<`, `<=`, `>`, `>=`) work on scalar fields. String
fields add `contains`, `startswith` and `endswith`, and these always match
**literal** text: `%` and `_` are ordinary characters, not wildcards. `is_in`
is membership on a root field.

A reference field *chains*: `v.ref.doi == "10.1/a"` joins the referenced table
automatically, and every condition written through the same reference field
shares one join. `v.ref == None` is the IS NULL test, and `v.ref == some_object`
compares against an already-stored instance.

## Variable-length fields are sets

A `list[...]` field lives in a child table, so a condition on it is a statement
about a *set* of rows, and the DSL says which one you mean:

- `v.symbols.has_any("O", "Ca")` — at least one element is in the set.
- `v.symbols.has_only("O", "Ca", "Ti")` — every element is in the set (a
  record with no elements at all satisfies this: the empty set is a subset of
  anything). `is_in` on a child field means exactly the same thing.
- `~` negates the *set* statement rather than the row: `~v.symbols.has_any("O")`
  is "no element is O". For the same reason `~(v.symbols == "O")` also means
  "no element is O", not "some element is not O".

`has_any(a) & has_any(b)` is the "HAS ALL" pattern, and works because each
access to a child field mints an independent join.

## Results

The low-level protocol yields `SearchResult`, a named 2-tuple of `values` (one
per `output()`, in declaration order) and `names`. The canonical `results()`
API yields named lazy rows; use the portable protocol form
`for (structure,), _names in search:` when needed. Result rows bypass the
identity cache; `fetch()` retains the same-object-while-alive guarantee.

Sorting on a rational field runs on its float companion column, so it is
documented-approximate; the values themselves are still exact.
"""

from dataclasses import dataclass
from fractions import Fraction
from typing import Any, ClassVar

from httk.core.storage import StorageInfo

from httk.store.db import Database, SqlStore
from httk.store.query import Searcher, SearchVariable

HTTK_EXAMPLE_REQUIRES = ["sqlalchemy"]


@dataclass(frozen=True)
class Publication:
    doi: str
    title: str


@dataclass(frozen=True)
class Structure:
    formula: str
    spacegroup: int
    energy: Fraction
    symbols: list[str]
    ref: Publication | None = None


@dataclass(frozen=True)
class Tag:
    """A join class attaching key/value metadata to a structure."""

    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(dedup="by_value")

    structure: Structure
    name: str
    value: str


ALPHA = Publication("10.1/alpha", "Alpha")
BETA = Publication("10.1/beta", "Beta")

STRUCTURES = [
    Structure("CaTiO3", 221, Fraction(-1, 3), ["O", "Ca", "Ti"], ALPHA),
    Structure("NaCl", 225, Fraction(1, 2), ["Na", "Cl"], ALPHA),
    Structure("MgO", 225, Fraction(-5, 4), ["Mg", "O"], BETA),
    Structure("CaO", 225, Fraction(0), ["Ca", "O"], None),
    Structure("SrCaTiO", 62, Fraction(3, 2), ["O", "Ca", "Ti", "Sr"], BETA),
    Structure("Vacuum", 1, Fraction(7, 8), [], None),
]

TAGS = [
    Tag(STRUCTURES[0], "quality", "good"),
    Tag(STRUCTURES[2], "quality", "poor"),
    Tag(STRUCTURES[4], "quality", "good"),
]


def populate() -> SqlStore:
    """An in-memory store holding the structures and their tags."""
    store = SqlStore(Database.sqlite(), entry_records={})
    with store.transaction():  # one transaction for the whole load
        for structure in STRUCTURES:
            store.save(structure)
        for tag in TAGS:
            store.save(tag)
    return store


def structure_search(store: SqlStore) -> tuple[Searcher, SearchVariable]:
    """A searcher over `Structure`, outputting the whole object."""
    searcher = store.searcher()
    variable = searcher.variable(Structure)
    searcher.output(variable, "structure")
    return searcher, variable


def matched(searcher: Searcher) -> list[str]:
    """The formulas of the matched structures, in the order the search yields them."""
    return [structure.formula for structure in searcher.results().scalars()]


def show(store: SqlStore, label: str, build: Any) -> None:
    """Run one condition against `Structure` and print what it matched."""
    searcher, variable = structure_search(store)
    searcher.add(build(variable))
    print(f"  {label:<48} -> {sorted(matched(searcher))}")


def show_comparisons(store: SqlStore) -> None:
    print("== Comparisons and combinators ==")
    show(store, 'formula == "NaCl"', lambda v: v.formula == "NaCl")
    show(store, "spacegroup >= 225", lambda v: v.spacegroup >= 225)
    show(store, "energy < 0", lambda v: v.energy < Fraction(0))
    show(store, 'formula.startswith("Ca")', lambda v: v.formula.startswith("Ca"))
    show(store, 'formula.contains("aTi")', lambda v: v.formula.contains("aTi"))
    show(store, 'formula.endswith("O")', lambda v: v.formula.endswith("O"))
    show(store, 'formula.is_in("NaCl", "MgO")', lambda v: v.formula.is_in("NaCl", "MgO"))
    show(
        store,
        'spacegroup == 225 & formula.startswith("M")',
        lambda v: (v.spacegroup == 225) & v.formula.startswith("M"),
    )
    show(store, 'formula == "NaCl" | formula == "MgO"', lambda v: (v.formula == "NaCl") | (v.formula == "MgO"))
    show(store, "~(spacegroup == 225)", lambda v: ~(v.spacegroup == 225))
    print()


def show_references(store: SqlStore) -> None:
    print("== Reference chains (an automatic join) ==")
    show(store, 'ref.doi == "10.1/alpha"', lambda v: v.ref.doi == "10.1/alpha")
    show(store, 'ref.title == "Beta"', lambda v: v.ref.title == "Beta")
    show(store, "ref is NULL", lambda v: v.ref == None)
    show(store, "ref == the stored ALPHA object", lambda v: v.ref == ALPHA)
    print("  Both conditions below are written through the same reference field,")
    print("  so they share a single join rather than joining the table twice:")
    searcher, v = structure_search(store)
    searcher.add(v.ref.doi == "10.1/beta")
    searcher.add(v.ref.title == "Beta")
    print(f"  ref.doi == '10.1/beta' AND ref.title == 'Beta'      -> {sorted(matched(searcher))}")
    print()

    print("== Joining two variables ==")
    searcher = store.searcher()
    tag = searcher.variable(Tag)
    structure = searcher.variable(Structure)
    searcher.add(tag.structure == structure)  # the join condition
    searcher.add(tag.name == "quality")
    searcher.add(tag.value == "good")
    searcher.output(structure, "structure")
    print(f"  structures tagged quality=good                   -> {sorted(matched(searcher))}")

    print("  ... and a self-join: other structures sharing NaCl's spacegroup")
    searcher = store.searcher()
    a = searcher.variable(Structure)
    b = searcher.variable(Structure)
    searcher.add(a.formula == "NaCl")
    searcher.add(a.spacegroup == b.spacegroup)
    searcher.add(b.formula != "NaCl")
    searcher.output(b, "structure")
    print(f"                                                   -> {sorted(matched(searcher))}")
    print()


def show_set_operations(store: SqlStore) -> None:
    print("== Set operations on the variable-length 'symbols' field ==")
    show(store, 'symbols.has_any("O")', lambda v: v.symbols.has_any("O"))
    show(store, 'symbols.has_only("O", "Ca", "Ti")', lambda v: v.symbols.has_only("O", "Ca", "Ti"))
    show(store, 'symbols.is_in("O", "Ca", "Ti")   [same as has_only]', lambda v: v.symbols.is_in("O", "Ca", "Ti"))
    show(store, '~symbols.has_any("Ca", "Ti")', lambda v: ~v.symbols.has_any("Ca", "Ti"))
    show(store, '~symbols.has_only("O", "Ca", "Ti")', lambda v: ~v.symbols.has_only("O", "Ca", "Ti"))
    show(store, '"HAS ALL": has_any("Ca") & has_any("Ti")', lambda v: v.symbols.has_any("Ca") & v.symbols.has_any("Ti"))
    show(store, 'symbols == "O"    [some element is O]', lambda v: v.symbols == "O")
    show(store, '~(symbols == "O") [no element is O]', lambda v: ~(v.symbols == "O"))
    show(
        store, 'spacegroup == 225 & ~symbols.has_any("Ca")', lambda v: (v.spacegroup == 225) & ~v.symbols.has_any("Ca")
    )
    print("  Note that 'Vacuum' (no symbols at all) satisfies has_only but never has_any.")
    print()


def show_ordering_and_paging(store: SqlStore) -> None:
    print("== Sorting, limit, offset, count ==")
    searcher, v = structure_search(store)
    searcher.add_sort(v.energy, False)  # False = ascending
    print(f"  by energy ascending:            {matched(searcher)}")

    searcher, v = structure_search(store)
    searcher.add_sort(v.spacegroup, True)  # True = descending
    searcher.add_sort(v.formula, False)  # secondary key
    print(f"  by spacegroup desc, formula asc: {matched(searcher)}")

    searcher, v = structure_search(store)
    searcher.add_sort(v.formula, False)
    searcher.set_limit(2)
    print(f"  first two by formula:            {matched(searcher)}")
    searcher.set_limit(-1)  # -1 means "no bound"
    searcher.add_offset(4)
    print(f"  skipping the first four:         {matched(searcher)}")

    searcher, v = structure_search(store)
    searcher.add(v.spacegroup == 225)
    searcher.set_limit(1)
    print(f"  count() ignores limit/offset:    {searcher.count()} match, {len(list(searcher))} returned")
    print()


def show_results(store: SqlStore) -> None:
    print("== Lazy ResultSet rows ==")
    searcher = store.searcher()
    v = searcher.variable(Structure)
    searcher.add(v.symbols.has_any("Ti"))
    searcher.add_sort(v.formula, False)
    results = searcher.results(structure=v, formula=v.formula, energy=v.energy)

    print(f"  names = {results.names}; matches = {len(results)}")
    for row in results:
        print(f"  {row.structure.formula}: energy {row.energy} (exact), symbols {row.structure.symbols}")
    print(f"  scalars: {[structure.formula for structure in results.scalars('structure')]}")
    energies = results.column("energy")
    print(f"  exact energies = {list(energies)}")
    print(f"  float energies = {list(energies.floats())}")
    print(f"  as FracVector = {energies.to_fracvector()}")

    cursor = results.cursor()
    current = next(cursor)
    print(f"  cursor row = {current.structure.formula}")
    # Cursor rows are valid until the cursor advances; copy the value if needed.
    next(cursor)

    print("  Low-level portable protocol form:")
    searcher, v = structure_search(store)
    searcher.add(v.formula == "MgO")
    for (structure,), names in searcher:
        print(f"    {names} -> {structure.formula}")
        print(f"    equal to the saved instance: {structure == STRUCTURES[2]}")


def main() -> None:
    store = populate()
    show_comparisons(store)
    show_references(store)
    show_set_operations(store)
    show_ordering_and_paging(store)
    show_results(store)


if __name__ == "__main__":
    main()
