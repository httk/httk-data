# Database storage in detail

`httk.store.db` is the database storage layer of *httk₂*: it stores **plain
frozen dataclasses** in a relational database, makes them queryable through a
backend-agnostic search DSL, and serves them through the neutral
`httk.core.EntryProvider` contract (e.g. as an OPTIMADE API via
*httk-serve*). SQL generation and dialect handling run on SQLAlchemy Core
internally; the public API exposes no SQLAlchemy types.

## Installing

The SQL layer is an optional extra (plain `import httk.store` works without it):

```bash
python -m pip install "httk-store[db]"      # SQLite (built into Python) via sqlalchemy
python -m pip install "httk-store[duckdb]"  # additionally the DuckDB backend
python -m pip install "httk-store[clickhouse]"  # ClickHouse backend
```

Touching a SQL-backed name (such as `httk.store.db.Database`) without the extra
installed raises an `ImportError` naming it.

## Declaring a storable class

Storability is non-intrusive: any frozen dataclass whose fields resolve is
storable — there is no base class. The stdlib-only marker vocabulary lives in
*httk-core* (`Indexed`, `Unique`, `Skip`, `Shape`, `StorageInfo`,
`stored_property`), so domain modules can declare storable classes without
depending on httk-store:

```python
from dataclasses import dataclass
from fractions import Fraction
from typing import Annotated, ClassVar

from httk.core import FracVector
from httk.core.storage import Indexed, Shape, StorageInfo, stored_property


@dataclass(frozen=True)
class Author:
    name: str
    year: int


@dataclass(frozen=True)
class StructureRecord:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(indexes=(("spacegroup", "formula"),))

    formula: Annotated[str, Indexed()]
    spacegroup: int
    energy: Fraction  # stored exactly (see below)
    cell_basis: Annotated[FracVector, Shape(3, 3)]  # fixed-shape tensor, stored inline
    reduced_coords: Annotated[FracVector, Shape(0, 3)]  # variable rows, child table
    symbols: list[str]  # child table
    reference: Author | None = None  # foreign key, saved recursively

    @stored_property
    def natoms(self) -> int:  # stored & queryable; recomputed on load
        return len(self.symbols)
```

Scalars (`int`/`str`/`bool`/`bytes`) become columns, while `float` gets a query
`DOUBLE` plus an exact hexadecimal text companion so signed zero and every
other finite binary64 value round-trip unchanged. `X | None` makes fields
nullable, rationals and datetimes are encoded by value codecs, lists and tuples
become child tables, and nested storable dataclasses become foreign keys (saved
recursively first). Classes you cannot modify can be described externally with
`register_schema_override`.

## Storing and fetching

`Database` names where data lives; `SqlStore` saves and reconstructs
instances. Saving deduplicates per the class's `StorageInfo.dedup` policy
(by content identity by default), and `transaction()` scopes several
operations into one database transaction:

```python
from httk.store.db import Database, SqlStore

db = Database.sqlite("example.sqlite")  # or Database.sqlite() in memory,
store = SqlStore(db, entry_records={})  # first-time custom-record store
# Reopen an initialized database with: SqlStore(db)

with store.transaction():
    sid = store.save(record)  # returns the integer sid; dedups; recurses

same_record = store.fetch(StructureRecord, sid)  # reconstructed exactly
```

### Vocabulary

An entry family is a logical key such as `StructureEntry`.
A record is a durable frozen-dataclass representation; a family may have several.
Backend/View is the representation pattern: a backend owns data, and a view presents it.
A content id identifies the record's content across stores; a SID is only a local row id.

Every database starts with a persisted, versioned layout declaration. Passing
`entry_records={}` says that this is a private/custom-record store with no
queryable entry families. An entry store instead maps each registered logical
family to the exact durable Record representation or representations it may
contain:

```python
store = SqlStore(
    db,
    entry_records={StructureEntry: UnitcellStructureRecord},
)
```

A single record is queried directly. A tuple of two or more records creates
a small family dispatch table, while the representation-specific data remains
in its normalized Record tables. Saving an exact configured record (including
saving a naturally bound domain object) makes it discoverable through
`fetch_entry(StructureEntry, content_id)`; that method returns the actual
concrete Record.

Later `SqlStore(db)` calls trust the persisted declaration; there is no layout
mode or schema diffing. Missing or edited record tables fail with the database's
own errors when used. Tables are created lazily on the first write; reads never
issue DDL. Old, unversioned, or incompatible layouts raise
`StorageLayoutUpgradeRequiredError`; this redesign does not migrate old stores,
so rebuild them explicitly.

A source object with an exact `__httk_storage_record__` can be saved directly;
`save(source, as_record=OtherRecord)` selects another declared projection.
Nested record fields are projected recursively. A projected source must expose
any derived `stored_property` declared by its target record because storage
does not construct an intermediate record merely to evaluate that property.
Record validation runs at this storage boundary through `__httk_validate__`.
Optional child fields use presence columns, so `None` remains distinct from an
empty child value.

While a saved or fetched instance is alive, fetching its sid again returns the
very same object. Join-objects pointing at a stored instance are found with
`store.referring(TagClass, field="structure", to=record)`.

## Permanentization, degraded writes, and fsck

SQL stores use a storage-only `_httk_role` parent column: `1` marks a record
saved at the public top level and `0` marks a recursively saved dependency.
The column is not part of content identity, by-value matching, canonical
encoding, hydrated records, or query results. Saving a dependency again at the
top level promotes its existing row to main; bulk canonicalization likewise
keeps the maximum role of all collapsed occurrences.

The usual `Database.sqlite(...)` and `Database.duckdb(...)` stores have the
persisted `transactional` write profile (the absent metadata value means the
same thing). SQLite additionally exposes an explicitly opt-in, artificial
transactionless conformance vehicle:

```python
db = Database.sqlite("recovery-test.sqlite", degraded=True)
store = SqlStore(db, entry_records={})
```

That construction stamps the `degraded` profile and can only reopen through a
similarly configured database. Opening validates the live SQLite DB-API
autocommit state, not just the construction flag; a transactional profile also
rejects an autocommit engine. It is SQLite-only in this release: it uses DB-API
autocommit to model SQL-like backends that cannot provide transaction rollback.
The profile is deliberately single-writer. A database-visible writer lease is
acquired on mutation and held until `Database.dispose()`; another instance can
inspect the holder/age and explicitly call `store.steal_lease()` when recovery
authority is clear.

Degraded saves permanently write dependencies first, then child-element rows
under a preallocated monotonic sid, then the parent sid row last. Thus a visible
parent means its subtree is complete; a failed write may leave only dependency
or child residue. No compensation deletion is attempted. Per-operation dirty
markers cost one lookup, one upsert, and one conditional delete per touched
table; a leftover marker arranges a targeted ownerless-child sweep before the
next write to that table. Sid counters are created and initialized lazily at
the first allocation for each parent table. `bulk_ingest()` is intentionally unavailable for degraded stores in
v2.3.0; use ordered `save()` calls.

Run `store.fsck(known_types=(...))` after a failed degraded writer (or for an
integrity audit). It repairs missing dispatch rows for main entries, sweeps
ownerless child rows, marks from main and dispatch roots, removes unreachable
dependency rows, and reports dangling logical references. It refuses garbage
collection if it finds an ordinary application table it cannot attribute to
the declared layout or `known_types`; no unrelated table is guessed or swept.
SQLite transactional fsck uses `BEGIN IMMEDIATE`. DuckDB callers must pass
`exclusive=True`, which is an explicit acknowledgement that the database is
offline from all writers for the entire fsck; DuckDB cannot otherwise enforce
the necessary read/delete exclusion. Invalid role values are violations; with
`repair=True` fsck normalizes them to dependency role `0` rather than inventing
a new root.

### ClickHouse bulk-fenced writes

For local/CI server setup and the required `_httk_bootstrap` KeeperMap DDL,
see the [ClickHouse testing guide](../clickhouse-testing.md).

ClickHouse uses KeeperMap metadata and the persisted `bulk-fenced` profile.
Reads do not acquire a lease. A bulk writer acquires a fresh, never-reused
token with a strict insert, verifies that exact value during the P2 bulk-entry
and marker operations, and releases it with an exact-value delete when
`Database.dispose()` runs. P3 adds verification around its durable phases. The
`ingest_state` marker is also a strict insert and carries the lease token plus
a fresh per-ingest nonce; it is cleared only by an exact-value delete after a
successful ingest.
`steal_lease()` is intentionally unavailable.

If a writer dies with only a lease residue, inspect `_httk_store_metadata`,
verify that the writer is no longer alive, and delete only the observed lease
value with a ClickHouse client:

```sql
SELECT key, value FROM _httk_store_metadata WHERE key = 'lease';
SET keeper_map_strict_mode = 1;
DELETE FROM _httk_store_metadata
WHERE key = 'lease' AND value = '<observed lease JSON>';
```

Never clear `ingest_state` merely because its lease was removed. Its presence
means the store may contain partial or inconsistent physical state, so the
default remedy is `DROP DATABASE`, recreate the bootstrap table, and re-ingest.
Only after a verified cleanup/rebuild has restored the declared empty-store
invariant may an operator clear the exact observed marker value. Use the same
strict setting:

```sql
SET keeper_map_strict_mode = 1;
DELETE FROM _httk_store_metadata
WHERE key = 'ingest_state' AND value = '<observed marker JSON>';
```

Do not delete values belonging to a live writer or use broad key-only deletes.

## Bulk ingestion

For SQLite and DuckDB, `store.bulk_ingest()` is a faster path than a `save()`
loop for **building a store from scratch or appending a large increment** to
one. It returns a
`httk.store.db.bulk.BulkIngest` context manager that mirrors `save()` but buffers
encoded rows with pre-assigned sids and appends them in `executemany` batches
inside one transaction, instead of one statement round-trip and an in-database
deduplication protocol per record. It is a near drop-in for the save loop:

ClickHouse bulk ingestion is currently fresh-store-only and stops at the P2
lease-plus-marker boundary until P3 supplies its nontransactional loader and
finalizer. It does not provide rollback or exact restoration; marker residue
fails closed and the default recovery is drop-and-reingest.

```python
# Per-record save loop
with store.transaction():
    for structure in structures:
        store.save(structure)
```

```python
# Bulk-ingest drop-in
with store.bulk_ingest() as bulk:
    for structure in structures:
        bulk.save(structure)
```

Reach for it when the increment is large; for a handful of records the ordinary
`save()` path is simpler and the round-trips it saves are negligible.

### Contract

**Exclusive write ownership.** While a `bulk_ingest()` context is open the
store's ordinary write path belongs to it: `save()`, `ensure_tables()`, and
`transaction()` on the same `httk.store.db.SqlStore` raise `RuntimeError`, and a
second `bulk_ingest()` context on the same store is refused. Reads from an
already-open store remain available; a new open is rejected while an
empty-store ingest marker is present.

**SQLite/DuckDB transaction and restoration.** On SQLite and DuckDB, the whole
ingest runs in a single transaction that commits only on clean exit. Any exception — a metadata
conflict, a uniqueness violation, or one you raise inside the block — rolls the
transaction back, drops every table the context created, restores any index it
dropped, removes its staging tables, and clears the store's identity caches,
leaving the store exactly as it was before the context opened. For an
empty-store ingest, cleanup verifies that only the metadata table remains and
then clears its marker, so retrying is safe. A hard crash can leave the marker
behind; subsequent opens reject that store and require dropping and re-ingesting
it.

These transaction and restoration guarantees do not apply to ClickHouse. Its
nontransactional P3 ingest will use the marker as a fail-closed recovery gate;
an interrupted marker defaults to drop-and-re-ingest.

**Deduplication and uniqueness are post-conditions, not per-row checks.** Within
the stream, records deduplicate set-wise in memory by the class's
`StorageInfo.dedup` policy (content identity by default, `by_value`, or `none`),
exactly as `save()` would. Global uniqueness against what is already stored is
enforced at the boundaries rather than per row: on a physically empty store the
record tables are created index-less and their separable indexes (content-id
uniqueness, `Indexed`/`Unique`, composite, and child parent-sid) are built once
the stream completes — building the unique index *is* the verification, and a
duplicate aborts the ingest. On a populated store each flushed chunk is staged
into an ordinary `bulkstage_<table>` table and resolved set-wise against the
target: a content-id anti-join, a `by_value` whole-parent-column anti-join with
null-safe equality, and a sid remap that rewrites every still-buffered reference
to the deduplicated existing sid.

**Returned sids are provisional.** `bulk.save()` returns an integer sid like
`save()`, but it is provisional while the context is open: a record that
deduplicates against a row the store already held is remapped to that existing
sid at flush. After the context exits cleanly,
`httk.store.db.bulk.BulkIngest.resolved_sid` maps any returned sid — provisional
or final — to its durable stored sid. It keys on the bare sid value, so resolve
a returned sid against the type it was saved as (sids are allocated per table,
and one value can recur across tables).

**`verify_metadata`** (default `True`, a plain `bool`) controls whether a
content-id hit compares its identity-excluded metadata against the first
in-memory occurrence — or against the stored row for a hit against existing
data — reproducing `save()` and raising
`httk.store.store_common.EntryMetadataConflictError` on a conflict. Pass
`verify_metadata=False` to skip the comparison when the stream is
known-consistent.

**`index_strategy`** (`"auto"`, `"keep"`, or `"rebuild"`, default `"auto"`)
governs only how an *existing* table's separable indexes are handled during an
append: `"keep"` appends through them, `"rebuild"` drops and recreates them at
the end (where the unique-index creation re-verifies global uniqueness), and
`"auto"` chooses per table by the staged-to-existing row ratio. On DuckDB, which
reserves a dropped index's name until commit, a `"rebuild"` decision instead
keeps the indexes in place — relying on their incremental maintenance — and
verifies content-id uniqueness with a duplicate scan at finalize; the final
indexes are identical either way.

**`finalize`** (`"auto"`, `"parity"`, or `"deferred"`, default `"auto"`)
chooses the finalization profile. `"deferred"` is an explicit fresh-store
profile at any worker count; `"parity"` is the historical in-database path.
`"auto"` selects deferred only for a physically empty, supported serial ingest;
it selects parity for every other case, including `workers>1`. At current batch
scales the parallel in-database merge is faster, while serial deferred gains
about 36%.

**Nested conflict paths differ by prefix.** Because the bulk encoder resolves
referenced and child records eagerly and only discovers their existing-row hits
at flush, an `httk.store.store_common.EntryMetadataConflictError` reached through
a `descend` field (a non-skipped reference whose target itself carries skipped
metadata) is reported at the descendant record's own path (`"Leaf.note"`) rather
than the ancestor field path `save()` would use (`"Root.primary.note"`). The
exception type, message template, and roll-back are identical; only the path
prefix differs.

**`chunk_size`** (default `100_000`) is the number of top-level `save()` calls
buffered before a flush. Buffered rows and the in-memory dedup indexes are held
until the next flush, so peak memory scales with the chunk size and each
record's fan-out into child and reference rows: lower it for very wide records
or a tight memory budget, raise it to amortize the staging round-trips over more
rows. Identity caches are deliberately not populated by bulk ingestion.

**`on_progress`** is an optional `(records_buffered_total, rows_flushed_total)`
callback invoked after each flush, for progress reporting over a long build.

### Performance

Bulk ingestion gains most on flat records with little fan-out: measured against
the per-record `save()` loop it is roughly **30x** faster on DuckDB and **13x**
on SQLite for flat rows, easing to about **5x** (DuckDB) and **4x** (SQLite) for
structure-shaped records whose child and reference tables dominate the row
count. These figures come from single-threaded runs against a tmpfs database, so
the per-record baseline they improve on is already I/O-favorable; both the
speed-up and the absolute throughput will differ on slower storage.

### Parallel ingestion

For the *offline build* of a store from a large stream, `bulk_ingest(workers=N)`
with `N > 1` encodes the stream in a pool of forked worker processes and merges
their per-table shards set-wise. Encoding — the bottleneck for structure-shaped
records — runs across cores; the merge (loading shards, collapsing cross-worker
duplicates, renumbering to compact sids, and building the indexes) runs once in
the main process inside the ingest's single transaction.

```python
with store.bulk_ingest(workers=12) as bulk:
    bulk.save(layout_record)
    for material in materials:
        bulk.save(material)
```

On DuckDB workers hand rows off as Parquet shards, so parallel mode there needs
`pyarrow`; install it with the combined extra:

```console
$ pip install "httk-store[duckdb,parallel]"
```

SQLite workers write one native shard database each and need no extra dependency.

**Empty target only.** Parallel mode is for building a fresh store, not for
appending: opening `workers>1` on a store that already holds application rows is
refused (use `workers=1` for incremental appends). On DuckDB the restriction is
stronger — *any* pre-existing application table is refused, because the merge
renumbers and deletes rows in place and DuckDB will not do that through a live
foreign-key constraint.

**Physical schema is foreign-key free.** SQLite and DuckDB use the same FK-free
physical DDL for serial and parallel builds. Logical reference, ownership,
child-element, and dispatch edges remain available to the storage algorithms,
while column types, keys, checks, and indexes are unchanged.

**Provisional tokens.** Because a worker encodes each object asynchronously, the
sid is not known when `save` returns; in parallel mode `save` returns an opaque
token instead. After the context exits cleanly,
`httk.store.db.bulk.BulkIngest.resolved_sid` maps each returned token to its
durable stored sid, exactly as it maps a provisional sid on the serial path. A
lost task (an unpicklable object, or a worker that crashed or was killed) aborts
the ingest rather than committing a partial store, and `on_progress` is rejected
up front because per-flush counts are not observable across processes.

**Identity-excluded metadata restriction.** The merge verifies identity-excluded
(`IdentitySkip`) metadata with a grouped column scan rather than by reconstructing
every duplicate record. That covers scalar skip columns and skipped references to
content-addressed or by_value records, and it reports a conflict against the
schema field. A few shapes fall outside it and are rejected up front (naming
`workers=1`): an identity-excluded child *sequence*, an identity-excluded
reference to — or `descend` into — a non-deduplicated (`dedup="none"`) record,
and a self-referential identity-excluded reference. Opening with
`verify_metadata=False` lifts the restriction.

**Measured speed-up.** Building the ~9,000-material altermagnets store into a
file-backed DuckDB database, parallel mode reaches about **6.6x** at 24 workers
when replicas share substructure (the realistic case, where the merge collapses
many cross-worker duplicates) and about **11x** at 24 workers with distinct roots
and shared atomic descendants (each material and its structure distinct, their
cells/sites/species still shared, so the merge collapses much less). The encode
phase scales with the worker count; the merge is a small fixed fraction of the total.
The benefit is real only for large builds — the pool fork, the shard round-trip,
and the merge are pure overhead on a small stream — so `workers` defaults to `1`.
Reproduce with `benchmarks/bench50_parallel.py`.

## Searching

`store.searcher()` opens a query through the backend-agnostic protocols in
`httk.store.query`: bind classes to variables and add conditions. Freeze the
query into the user-facing lazy result set with `results()`. Variables of the
same class self-join; reference fields chain (`v.reference.name`),
variable-length fields support the set operations (`has_any`, `has_only`), and
`~` negates them as sets. String matching (`contains`, `startswith`,
`endswith`) always takes **literal** text — `%` and `_` match themselves:

```python
search = store.searcher()
s = search.variable(StructureRecord)
search.add(s.spacegroup == 225)
search.add(s.reference.name == "Ada")  # auto-joins the author table
search.add(s.symbols.has_only("O", "Ca", "Ti"))  # for-all over the child rows
search.add(~s.symbols.has_any("Fe"))  # no child row is iron
results = search.results(structure=s, energy=s.energy)
for row in results:  # lazy ResultRow values
    print(row.structure.formula, row.energy)  # exact rational energy
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
`httk.store.PageableResultSetLike`), separate from the required `ResultSetLike`
contract. It uses a stable keyset/seek order over named **root scalar result
projections** and returns an immutable `ResultPage`:

```python
from httk.store import PageOrder

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

`httk.store.Store` is intentionally a small, backend-neutral contract:
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
profile should depend on `httk.store.Store`, not `SqlStore`.

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
from httk.store.db import StoreEntryProvider

provider = StoreEntryProvider(store, {"structures": StructureRecord, "authors": Author})
```

Handing the provider to *httk-serve*'s `adapter_from_providers` serves the
database as an OPTIMADE API. *httk-store* does not depend on *httk-serve*: the
provider handoff uses the httk-core contract, while *httk-serve* also consumes
*httk-store*'s neutral query and store APIs. Fields with no OPTIMADE value
representation (`bytes`, custom codecs) are not served, and rationals are
served as their nearest floats. The provider is also registered (as
`store-db-store`) for discovery through the `httk.core` registry.
