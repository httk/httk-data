# Database storage

`httk.data.db` is the database storage layer of *httk₂*: it stores **plain
frozen dataclasses** in a relational database, makes them queryable through a
backend-agnostic search DSL, and serves them through the neutral
`httk.core.EntryProvider` contract (e.g. as an OPTIMADE API via
*httk-optimade*). SQL generation and dialect handling run on SQLAlchemy Core
internally; the public API exposes no SQLAlchemy types.

## Installing

The SQL layer is an optional extra (plain `import httk.data` works without it):

```bash
python -m pip install "httk-data[db]"      # SQLite (built into Python) via sqlalchemy
python -m pip install "httk-data[duckdb]"  # additionally the DuckDB backend
```

Touching a SQL-backed name (such as `httk.data.db.Database`) without the extra
installed raises an `ImportError` naming it.

## Declaring a storable class

Storability is non-intrusive: any frozen dataclass whose fields resolve is
storable — there is no base class. The stdlib-only marker vocabulary lives in
*httk-core* (`Indexed`, `Unique`, `Skip`, `Shape`, `StorageInfo`,
`stored_property`), so domain modules can declare storable classes without
depending on httk-data:

```python
from dataclasses import dataclass
from fractions import Fraction
from typing import Annotated, ClassVar

from httk.core import FracVector, Indexed, Shape, StorageInfo, stored_property


@dataclass(frozen=True)
class Author:
    name: str
    year: int


@dataclass(frozen=True)
class StructureRecord:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(indexes=(("spacegroup", "formula"),))

    formula: Annotated[str, Indexed()]
    spacegroup: int
    energy: Fraction                                      # stored exactly (see below)
    cell_basis: Annotated[FracVector, Shape(3, 3)]        # fixed-shape tensor, stored inline
    reduced_coords: Annotated[FracVector, Shape(0, 3)]    # variable rows, child table
    symbols: list[str]                                    # child table
    reference: Author | None = None                       # foreign key, saved recursively

    @stored_property
    def natoms(self) -> int:                              # stored & queryable; recomputed on load
        return len(self.symbols)
```

Scalars (`int`/`float`/`str`/`bool`/`bytes`) become columns, `X | None` makes
them nullable, rationals and datetimes are encoded by value codecs, lists and
tuples become child tables, and nested storable dataclasses become foreign keys
(saved recursively first). Classes you cannot modify can be described
externally with `register_schema_override`.

## Storing and fetching

`Database` names where data lives; `SqlStore` saves and reconstructs
instances. Saving deduplicates per the class's `StorageInfo.dedup` policy
(by content identity by default), and `transaction()` scopes several
operations into one database transaction:

```python
from httk.data.db import Database, SqlStore

db = Database.sqlite("example.sqlite")   # or Database.sqlite() in memory,
store = SqlStore(db)                     # or Database.duckdb("example.duckdb")

with store.transaction():
    sid = store.save(record)             # returns the integer sid; dedups; recurses

same_record = store.fetch(StructureRecord, sid)   # reconstructed exactly
```

While a saved or fetched instance is alive, fetching its sid again returns the
very same object. Join-objects pointing at a stored instance are found with
`store.referring(TagClass, field="structure", to=record)`.

## Searching

`store.searcher()` opens a query through the backend-agnostic protocols in
`httk.data.query`: bind classes to variables, add conditions, declare outputs,
iterate. Variables of the same class self-join; reference fields chain
(`v.reference.name`); variable-length fields support the set operations
(`has_any`, `has_only`), and `~` negates them as sets. String matching
(`contains`, `startswith`, `endswith`) always takes **literal** text — `%` and
`_` match themselves:

```python
search = store.searcher()
s = search.variable(StructureRecord)
search.add(s.spacegroup == 225)
search.add(s.reference.name == "Ada")            # auto-joins the author table
search.add(s.symbols.has_only("O", "Ca", "Ti"))  # for-all over the child rows
search.add(~s.symbols.has_any("Fe"))             # no child row is iron
search.output(s, "structure")
for values, _names in search:                    # one SearchResult per match
    print(values[0].formula)                     # a fully reconstructed instance
```

A plain comparison on a child field is existential un-negated and set-negating
under `~`: `s.symbols == "O"` means "some symbol is O", `~(s.symbols == "O")`
means "no symbol is O" (not "some symbol is not O"), agreeing with
`~s.symbols.has_any("O")`. `is_in` reads by field kind: on a root field
`s.formula.is_in("CaTiO3", "NaCl")` is plain membership, while on a child field
`s.symbols.is_in("O", "Ca", "Ti")` is the for-all reading — every element must
be in the set — exactly the same as `has_only`.

`s.always_true()` and `s.always_false()` are constant conditions on a search
variable. They are reserved method names that never resolve to a stored field,
and they matter mainly to code that builds filters programmatically: the
obvious alternative, a `field == field` probe, is not NULL-safe — it yields
NULL rather than true for a row whose field is NULL, and so silently drops
rows.

Iteration yields a `SearchResult`: a named 2-tuple of `values` (one entry per
`output()` call, in declaration order) and `names`. It unpacks and indexes like
the plain tuple it always was, so `for (structure,), _names in search:` and
`result[0][0]` keep working.

## Exact rationals, approximate comparisons

Rational values (`fractions.Fraction`, `FracScalar`, `SurdScalar`,
`FracVector` tensors) are stored **losslessly**: a canonical exact text column
is the round-trip source of truth, alongside float companion columns used for
querying and indexing. Stored values therefore reconstruct exactly at
arbitrary precision — but SQL comparisons (and sorting) on rational fields run
on the float companions and are **documented approximate**. Content identity
and deduplication always use the exact form.

## Serving through OPTIMADE

`StoreEntryProvider` bridges a store to the `httk.core.EntryProvider`
contract: it auto-generates an OPTIMADE entry-type definition per served class
from its schema (every schema-derived property named with a registered
database-specific prefix, `_httk_` by default), yields JSON-able records, and
declares relationships for reference fields whose target class is also served:

```python
from httk.data.db import StoreEntryProvider

provider = StoreEntryProvider(store, {"structures": StructureRecord, "authors": Author})
```

Handing the provider to *httk-optimade*'s `adapter_from_providers` serves the
database as an OPTIMADE API; neither module imports the other. Fields with no
OPTIMADE value representation (`bytes`, custom codecs) are not served, and
rationals are served as their nearest floats. The provider is also registered
(as `data-db-store`) for discovery through the `httk.core` registry.
