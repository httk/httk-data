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
python -m pip install "httk-store[postgresql]"  # PostgreSQL backend (psycopg 3)
python -m pip install "httk-store[clickhouse]"  # ClickHouse backend
```

`Database.postgres(url)` opens a PostgreSQL store from a `postgresql://` URL.
It is fully transactional and rides the ordinary `transactional` write profile
with no special-casing, and it supports bulk ingestion (`store.bulk_ingest()`)
with the same parity/deferred/parallel behavior as SQLite and DuckDB. Only the
psycopg 3 driver is supported: a bare `postgresql://` URL is normalized to
`postgresql+psycopg://` and any other explicit driver is rejected. See the
[PostgreSQL testing guide](../postgres-testing.md) for local setup.

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

same_record = store.fetch(StructureRecord, sid)  # a lazy row, decoded on access
eager = store.fetch(StructureRecord, sid, eager=True)  # fully materialized now
```

### Lazy records

`fetch()`, `fetch_many()`, `fetch_by_content_id()`, `fetch_entry()`, and
`referring()` return **lazy rows** by default: the parent row is read now, but
every child, reference, and derived field decodes only when first accessed, and
recursively — a lazy record's children are lazy rows too. Pass `eager=True` to
any of them to fully materialize the base dataclass up front. A lazy row is a
subclass of the storable class (`isinstance(row, StructureRecord)` holds, and
`type(row)` is the row subclass), exposes its database `sid`, and reuses the
memoized value on repeated access to one field.

**Identity.** Repeated default fetches of one live `(class, sid)` return the
same object; a live materialized instance takes precedence over creating a
proxy, so an `eager=True` fetch after a lazy one hands back a fresh instance and
replaces the cached slot (materialized-wins). Mixing eager and lazy access can
therefore yield two distinct but equal objects for one sid while a caller still
holds the older one. Eager materialization reuses live *materialized* nested
objects but not lazily fetched ones (the exact-type guard skips a proxy).
Internal cache maintenance — a failed write, dedup compensation after a save —
may re-materialize a later fetch. Strict `is` identity across arbitrary call
sequences is **not** promised, and never truly was: these cache clears already
qualified the older "very same object" wording.

**Lifetime.** A lazy row's deferred reads open a *fresh* connection outside the
originating transaction, so a record that must outlive its transaction,
connection, or engine should be fetched with `eager=True`. Accessing a lazy row
whose originating transaction rolled back raises `ExpiredLazyRecordError`
(naming the class and sid) — including on a field that was read before the
rollback, and including a child sequence whose deferred read executed inside the
rolled-back transaction. Committed-transaction rows and rows fetched outside any
transaction are unaffected. Engine disposal does **not** invalidate proxies: the
pool silently reconnects for file-backed databases, so the lifetime rule is
correctness discipline, not an enforced guard.

**Memory and cost.** A single live lazy row pins not just its own chunk (up to
500 parent rows plus any child blocks already read) but the transitive closure
of chunks and hydrators reached through its shared hydration context —
reference targets are pinned wholesale when first read, so one held record can
retain a broad slice of the graph until the cyclic garbage collector reclaims
it (proxies sit in reference cycles with their pinned chunk, so `gc.collect()`
is what frees them). Reading a field is a descriptor call plus the liveness
guard, so a repeated read of an already-decoded field costs a few hundred
nanoseconds rather than a plain-attribute access — bind a field to a local in
hot loops.

**Append-only framing.** The public store API is append-only, so rows a caller
fetched are not deleted or mutated in normal use. Under the lazy default, two
abnormal-deletion cases are honest caveats rather than guarded errors: abnormal
external deletion of a *referenced* row surfaces at attribute access as
`StaleResultError` naming the referenced class and sid, while abnormally deleted
*child* rows are indistinguishable from a legitimately empty sequence (child
rows carry no tombstones) and cannot raise. `KeyError` remains the at-call-time
contract for an absent parent sid, in both modes. A reference *field* memoizes
its target proxy without recording a per-field rollback token (unlike a child
sequence, which does): correctness still holds because the target's own chunk
carries the token and raises `ExpiredLazyRecordError` when the target's fields
are touched.

**Other behaviors.** A lazy row exposes stored/codec values without re-running
`__post_init__` (record authors keep `__post_init__` idempotent with respect to
stored representations); `eager=True` runs it. `dataclasses.replace(row, ...)`
returns a plain instance of the row class and does run `__post_init__`. Lazy
rows reject `copy.copy`, `copy.deepcopy`, and pickling (materialize with
`eager=True` first). Reference cycles are tolerated lazily — each hop is a fresh
proxy — whereas `eager=True` on a cyclic graph raises `SchemaError`. Storable
records must not declare `@dataclass(order=True)` — ordering comparisons
between a lazy row and a materialized instance raise `TypeError` (the generated
`__lt__` requires the exact class) — and none do in the workspace today.

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

Applications may keep a private entry family out of global plugin discovery.
Supply its stable persistence names and classes directly with
`EntryFamilyDeclaration` and `EntryRecordDeclaration`:

```python
from httk.store import EntryFamilyDeclaration, EntryRecordDeclaration

private_entries = EntryFamilyDeclaration(
    name="my-application-publications",
    family=PublicationEntry,
    records=(
        EntryRecordDeclaration(
            name="my-application-publication",
            record=PublicationRecord,
        ),
    ),
)
store = SqlStore(db, entry_families=(private_entries,))
```

This is a store-local binding, not a registry operation. The store persists
the stable names and optional entry-definition IRIs but never persists or
imports arbitrary Python paths. Consequently, every reopen of a store with
application-owned declarations must supply the same `entry_families` value.
Omitting it raises `EntryLayoutBindingError`. Installed reusable modules should
continue to use registry-backed `entry_records`, which permits automatic
resolution on `SqlStore(db)`.
Both arguments may be supplied together when one store combines reusable
module families with application-private families; name or class collisions
are rejected while constructing the combined layout.

A single record is queried directly. A tuple of two or more records creates
a small family dispatch table, while the representation-specific data remains
in its normalized Record tables. Saving an exact configured record (including
saving a naturally bound domain object) makes it discoverable through
`fetch_entry(StructureEntry, content_id)`; that method returns the actual
concrete Record.

Later registry-backed `SqlStore(db)` calls trust the persisted declaration; there is no layout
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

Repeated default fetches of one live `(class, sid)` return the same object (see
[Lazy records](#lazy-records) for the full identity contract). Join-objects
pointing at a stored instance are found with
`store.referring(TagClass, field="structure", to=record)`, which returns lazy
rows by default and accepts `eager=True`.

## Store timestamps

`SqlStore` enables store-managed timestamps by default:

```python
store = SqlStore(
    db,
    entry_records={},
    store_timestamps="creation",
    store_timestamp_resolution=1_000,  # nanoseconds per stored unit; default: 1,000 (microseconds)
)
```

The stored value is `time.time_ns() // store_timestamp_resolution`. The public
query API accepts a canonical nanosecond integer or an RFC3339/ISO-8601
timezone-aware value and converts it to the store's units. For example, this
historic query returns rows present at `T`:

```python
searcher = store.searcher()
record = searcher.variable(StructureRecord)
searcher.output(record, "record")
searcher.add(record.ts_start <= "2026-01-01T00:00:00Z")
rows = searcher.results(record=record)
```

The equivalent OPTIMADE filter is:

```python
from httk.store.db import optimade_filter_searcher

rows = optimade_filter_searcher(
    store, StructureRecord, '_httk_ts_start <= "2026-01-01T00:00:00Z"'
)
```

`present at time T` means exactly `ts_start <= T`. FIRST-STORED-WINS
applies: a deduplication re-save does not replace the original timestamp, and
promoting a dependency to a main row does not replace it. One timestamp is
captured per save transaction and one per bulk batch, so all rows written by
that unit share its value.

Before capture, the writer checks a process-local high-water mark. A clock
regression smaller than 1 ms waits briefly when `clock_regression_grace=True`
(the default); larger regressions, or a failed grace wait, raise
`StoreClockRegressionError`. Set `clock_regression_grace=False` to skip the
wait, or `allow_clock_regression=True` to disable the guard. The mark is
per-process: reopening seeds it from stored rows, but it is not a cross-process
clock-coordination protocol.

`store.fsck()` checks for timestamps beyond the current clock plus the allowed
future slack. An administrative repair can clamp them:

```python
store.fsck(repair=True, clamp_future_timestamps=True, known_types=(StructureRecord,))
```

Clamping is destructive to historic-query fidelity; inspect a non-repair fsck
report and confirm the skew before using it.

The query stack also exposes `as_of=T` on stored-property federation
`query()`/`fetch()` and on the general `FederatedStore.searcher()`. The serving
layer accepts `_httk_as_of` and includes it in stable pagination plans. Stored
federation is availability-first: a source with `store_timestamps="off"`
deliberately ignores the cutoff and serves that source's current state; sources
with timestamps enabled apply their own-resolution cutoff. Existing layouts do
not require an enable/disable migration for reading this capability.

### Versioned stores: current view by default

Under `store_timestamps="versioned"`, `replace(old, new)` supersedes a family
entry (see the `replace` API for lifetime intervals). Every query entered from
the top then sees one consistent time slice by default: `store.searcher()` and
`store.referring()` return only the current view (`as_of=T` returns the
half-open `[ts_start, ts_end)` slice — at `T == successor.ts_start` the successor
is visible and the closed predecessor is not). A non-family dependency row is
current when it was saved standalone or is still owned by a live family entry;
forward reference traversals and child joins inside a match are never
lifecycle-filtered, so a pinned reference to a row later superseded in its own
role stays a correct member of that aggregate. Pass `scoped=False` to
`searcher()`/`referring()` to disable lifecycle filtering entirely (an `as_of`
cutoff still applies). One documented approximation: role promotion (a
dependency later saved top-level) is not timestamped, so a non-family row's
`as_of` visibility reflects its current role.

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

For SQLite, DuckDB, and PostgreSQL, `store.bulk_ingest()` is a faster path than
a `save()` loop for **building a store from scratch or appending a large
increment** to one. It returns a
`httk.store.db.bulk.BulkIngest` context manager that mirrors `save()` but buffers
encoded rows with pre-assigned sids and appends them in `executemany` batches
inside one transaction, instead of one statement round-trip and an in-database
deduplication protocol per record. It is a near drop-in for the save loop:

ClickHouse bulk ingestion is currently fresh-store-only and stops at the P2
lease-plus-marker boundary until P3 supplies its nontransactional loader and
finalizer. It does not provide rollback or exact restoration; marker residue
fails closed and the default recovery is drop-and-reingest.

**Known limitation — PostgreSQL bulk `NaN` in a list-of-floats field.** Under
PostgreSQL bulk ingest, a `NaN` value inside a stored **list-of-floats (child)
field** is not preserved: it reads back as `NULL`. Bulk ingest stages rows
through SQLite shards, which cannot represent `NaN`, so the value is lost in the
list column. A **scalar** float `NaN` IS preserved under bulk ingest, and the
serial `save()` path preserves `NaN` in both scalar and list-of-floats fields on
every backend.

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

**Nested entry promotion.** `bulk.save(envelope, promote=StructureRecord)`
makes every nested `StructureRecord` occurrence a top-level entry while keeping
the envelope as the returned root. Pass an iterable of classes to promote more
than one record type. Each class must be reachable from the envelope's stored
schema; projection, role marking, and entry dispatch all remain inside the same
worker task.

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
`fetch()` returns lazy rows too (see [Lazy records](#lazy-records)); repeated
default fetches of one live sid return the same object.

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
