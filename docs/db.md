# Database storage

`httk.data.db` is the database storage layer of *httk₂*: it stores **plain
frozen dataclasses** in a relational database, makes them queryable through a
backend-agnostic search DSL, and serves them through the neutral
`httk.core.EntryProvider` contract (e.g. as an OPTIMADE API via
*httk-serve*). SQL generation and dialect handling run on SQLAlchemy Core
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

A source object with an exact `__httk_storage_binding__` can be saved directly;
`save(source, as_record=OtherRecord)` selects another declared projection.
Nested record fields are projected recursively. A projected source must expose
any derived `stored_property` declared by its target record because storage
does not construct an intermediate record merely to evaluate that property.

While a saved or fetched instance is alive, fetching its sid again returns the
very same object. Join-objects pointing at a stored instance are found with
`store.referring(TagClass, field="structure", to=record)`.

## Searching

`store.searcher()` opens a query through the backend-agnostic protocols in
`httk.data.query`: bind classes to variables and add conditions. Freeze the
query into the user-facing lazy result set with `results()`. Variables of the
same class self-join; reference fields chain (`v.reference.name`),
variable-length fields support the set operations (`has_any`, `has_only`), and
`~` negates them as sets. String matching (`contains`, `startswith`,
`endswith`) always takes **literal** text — `%` and `_` match themselves:

```python
search = store.searcher()
s = search.variable(StructureRecord)
search.add(s.spacegroup == 225)
search.add(s.reference.name == "Ada")            # auto-joins the author table
search.add(s.symbols.has_only("O", "Ca", "Ti"))  # for-all over the child rows
search.add(~s.symbols.has_any("Fe"))             # no child row is iron
results = search.results(structure=s, energy=s.energy)
for row in results:                               # lazy ResultRow values
    print(row.structure.formula, row.energy)      # exact rational energy
```

`ResultRow` supports names, attributes, and positions. `scalars()` is the
short form for a one-column result (or takes a column name), and `first()` and
`one()` return one row; `one()` raises `NoResultError` or
`MultipleResultsError` unless there is exactly one. Results are reusable:
`len(results)`, `results[1:3]`, and re-iteration are all supported. A slice is
a view over its own positions: iteration, `len()`, indexing, `first()`,
`one()`, and `column()` are all scoped to that slice without re-querying.

Scalar columns stay exact by default. `column()` returns a `ResultColumn` with
an explicit approximate view through `.floats()` and an exact rational tensor
through `.to_fracvector()` for integer, fraction, and fracscalar columns:

```python
energies = results.column("energy")
exact = list(energies)
approximate = list(energies.floats())
as_vector = energies.to_fracvector()
```

`.to_fracvector()` rejects floats, surds, strings, datetimes, and other
non-rational projections. Variable-length CHILD-role projections are rejected
when `results()` is declared; reference-path projections are supported.

### Continuation pages

`SqlResultSet.page()` is an optional capability (described neutrally by
`httk.data.PageableResultSetLike`), separate from the required `ResultSetLike`
contract. It uses a stable keyset/seek order over named **root scalar result
projections** and returns an immutable `ResultPage`:

```python
from httk.data import PageOrder

page = results.page(
    size=100,
    order_by=(PageOrder("spacegroup"), PageOrder("energy", descending=True)),
)
for row in page.rows:
    print(row.structure.formula, row.energy)

if page.next is not None:
    later = results.page(
        size=100,
        order_by=(PageOrder("spacegroup"), PageOrder("energy", descending=True)),
        cursor=page.next,
    )
```

`PageOrder.name` refers to the output name (`"energy"` above), not a
SQLAlchemy column. Ordering accepts root scalar and encoded-scalar projections;
object outputs, child/reference-derived keys, duplicate names, an existing
`add_sort()`, a nonzero query offset, and a query limit are rejected. The SQL
implementation always appends the root `sid` as an internal ascending
tie-breaker, so duplicate user-order values do not duplicate or skip rows.
`nulls="first"`/`"last"` is explicit and portable across SQLite and DuckDB.
An empty order tuple is valid when storage order by root `sid` is sufficient.

The opaque URL-safe `ContinuationToken` contains only a version, tagged anchor
values, root sid, direction, and a digest binding the frozen query/output/order
schema and dialect. It contains no SQL and decoded values are always bound
parameters. It is deliberately not authenticated: a web boundary can wrap its
string value in an HMAC or another authenticated envelope. Corrupt, oversized,
non-canonical, or mismatched tokens raise `PaginationCursorError`; do not
construct application tokens yourself.

Pages are **live**: every call uses a fresh read connection and does not keep a
driver cursor or transaction open. On an unchanged store, following `next` and
then `previous` recovers the original page. Concurrent direct database changes
can move, insert, or remove matches between calls, so continuation paging does
not promise snapshot consistency.

A page fetches at most `size + 1` root match rows and uses a lexicographic seek
predicate rather than a query offset; the library hard-caps `size` at 10,000.
`include_total=False` (the default) does not count. `include_total=True` runs
the normal exact SQL count separately. This bounds application memory and match
row transfer, not database CPU for arbitrary filters or sorts; indexed root
order fields benefit from the indexes declared in the schema.

`cursor()` bounds the number of hydrated record/proxy objects held by a
row-by-row consumer, but not the raw values pinned by the result set. The
object value in each cursor `ResultRow` is an instance of the record class, so
views can be built on it. Each object output uses an explicitly unhashable,
reused proxy that expires when the cursor advances. Equality on an expired
cursor row raises; copying and pickling cursor rows are rejected even before
expiry. Components already filled into a view before advancing remain
readable on that view, but later component fills raise
`ExpiredCursorRowError`.

### Low-level portable protocol

The backend-neutral protocol form remains useful for code that must run on
any `Searcher` implementation. Declare outputs and iterate its plain
`SearchResult` values directly:

```python
search.output(s, "structure")
for (structure,), names in search:
    print(names, structure.formula)
```

This is the low-level/portable layer; SQL consumers should generally use
`results()`.

### Neutral portable Store profile

`httk.data.Store` is intentionally a small, backend-neutral contract:
`store.searcher()` returns a one-query `Searcher`, which binds one or more
backend-defined targets with `variable()`, receives expressions through
`add()`, and exposes `count()`, limit/offset, sorting, iteration, and
`results()`. A portable result supports iteration, `len()`, `first()`, `one()`,
and `scalars()`; `one()` uses the shared `NoResultError` and
`MultipleResultsError` exceptions. `UnsupportedQueryError` means that a
requested expression is outside a particular backend's portable subset.

This profile is deliberately what a remote, read-only OPTIMADE store can
implement too: it supports a single root endpoint, portable scalar/flat-list
filters, named outputs, and result cardinality without making the caller
depend on SQLAlchemy or a database dialect. Query code that only needs this
profile should depend on `httk.data.Store`, not `SqlStore`.

The following are SQL-specific extensions, not portable Store requirements:
persisting/fetching frozen dataclasses with `save()` and `fetch()`, schema and
transaction management, recursive reference storage, lazy SQL rows,
`ResultColumn.floats()`/`to_fracvector()`, cursor rows, child/reference joins,
continuation pages, and SQL's approximate comparisons for exact rationals. Do not assume those
operations exist on a remote or in-memory Store.

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

### Result and identity semantics

Search rows are lazy subclasses of the storable class, so `isinstance(row,
StructureRecord)` is true. The parent row is loaded when the row is first
used, and fields decode independently as they are accessed. They provide the
dataclass-generated compare/hash/repr behavior, honoring each field's
`compare`, `hash`, and `repr` flags, plus the same content-id and `save()`
behavior as the eager record. Classes with custom `__eq__` or `__hash__` are
rejected for lazy rows. Each row also exposes its database `sid`.

`dataclasses.replace(row, ...)` creates a new ordinary dataclass instance of
the lazy row class and runs validation. Lazy rows intentionally reject
`copy.copy`, `copy.deepcopy`, and pickling. Search rows bypass the store's
identity cache: two result rows for one sid are not an identity guarantee.
`fetch()` retains identity-while-alive, so repeated fetches of a sid return
the same live object.

An object output from an outer join can be `None`; the result row is retained,
not dropped. If a matched sid is deleted before an object output's lazy row is
hydrated, hydration raises `StaleResultError`; exact scalar projections cannot
become stale because their values and `_exact` companion texts came from the
outer SELECT.

### Memory and statement cost

The result set materializes a match index that pins the matched sids, raw
scalar-output values, and, for exact projections, their `_exact` companion
texts. All of those arrive in the one outer SELECT; there is no second
per-chunk exact-value fetch. Hydrated records live in a weak chunk cache of
500 parent sids; live rows pin their own chunk, while re-iteration may
rehydrate chunks that are no longer pinned. `cursor()` limits hydrated
record/proxy objects, not this pinned raw result data. A typical full-object
pass costs one outer match query plus about one parent query per 500 rows and
one batch per touched child-field group per 500 rows — not one SELECT per row.
The public store API is append-only, so rows do not disappear during normal
use; direct database edits can produce `StaleResultError` for object outputs.

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

Handing the provider to *httk-serve*'s `adapter_from_providers` serves the
database as an OPTIMADE API. *httk-data* does not depend on *httk-serve*: the
provider handoff uses the httk-core contract, while *httk-serve* also consumes
*httk-data*'s neutral query and store APIs. Fields with no OPTIMADE value
representation (`bytes`, custom codecs) are not served, and rationals are
served as their nearest floats. The provider is also registered (as
`data-db-store`) for discovery through the `httk.core` registry.
