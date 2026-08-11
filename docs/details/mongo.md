# MongoDB storage in detail

`httk.store.mongo` stores the same plain frozen dataclasses as the SQL layer,
but uses MongoDB's document model: one document per record, embedded child
arrays, and links to referenced records in their own collections. It exposes
the same neutral `Store`/`Searcher` protocols, entry-family dispatch, stored
properties, continuation paging, and entry-provider surface as the SQL
backend.

Choose `MongoStore` when MongoDB is already the operational data service, when
document-shaped records and embedded children are a useful fit, or when the
same store must be shared by applications that already speak MongoDB. Choose
`SqlStore` when a relational deployment, SQL tooling, or SQL's stronger
transaction and constraint model is the better operational fit. The two
backends share the storage concepts and neutral query vocabulary, but they do
not hide the MongoDB-specific limits documented in
[Differences from the SQL backend](#differences-from-the-sql-backend).

## Installing

The backend is optional:

```bash
python -m pip install "httk-store[mongodb]"
```

This installs `pymongo>=4.6`. Importing `httk.store` itself does not require
PyMongo; importing a MongoDB backend name without the extra raises an
`ImportError` naming `httk-store[mongodb]`.

## Connecting and choosing a deployment

Create a `MongoDatabase` from a MongoDB URI and give it a database name. A
replica set is the recommended deployment, including a single-node replica
set for a development or CI database:

```python
from httk.store.mongo import MongoDatabase, MongoStore

uri = "mongodb://127.0.0.1:27017/?replicaSet=httk2rs"
with MongoDatabase.connect(uri, database="materials", transactions="require") as database:
    store = MongoStore(database, entry_records={})
    # Use store.save(), store.fetch(), and store.searcher() here.
```

The `transactions` option has three values:

- `"auto"` (the default) probes the server. A replica set enables
  multi-document transactions; a standalone server opens in degraded mode.
- `"require"` refuses to open unless the `hello` response identifies a
  replica set. Use this when a torn multi-document write is unacceptable.
- `"never"` explicitly selects degraded mode, even when the server is a
  replica set. This is useful for tests that exercise the no-transaction
  behavior.

In degraded mode MongoStore emits a warning and does not provide
multi-document transaction atomicity. Writes are crash-safe at the individual
document level and proceed bottom-up, so a crash can leave complete but
unreachable orphan documents. A record cannot point to a missing referenced
sid, but a record-family dispatch write can be left for a later repair. The
`transaction()` context manager raises `TransactionsUnavailableError` in this
mode; use a replica set and `transactions="require"` when explicit
transactions are needed.

`MongoDatabase.connect()` configures PyMongo with `w="majority"`,
`journal=True`, and `readConcernLevel="majority"`. Explicit store
transactions also use majority read and write concern with journaling. These
defaults provide the intended durability behavior on a properly configured
MongoDB deployment; they do not turn a standalone server into a
multi-document transactional deployment.

## Declaring records and opening a store

Record declarations are the same non-intrusive frozen-dataclass declarations
described in [Database storage](db.md#declaring-a-storable-class). The Mongo
backend persists the logical entry-family declaration in its metadata
collection. On the first open, pass `entry_records`; later opens validate the
persisted declaration rather than silently changing it:

```python
store = MongoStore(
    database,
    entry_records={StructureEntry: StructureRecord},
)
```

The document layout is backend-specific, while the vocabulary of entry
families, records, content ids, sids, projections, and stored properties is
shared with the SQL layer. See [Vocabulary](db.md#vocabulary) for those
concepts and [Declaring a storable class](db.md#declaring-a-storable-class)
for the marker and schema rules. MongoDB sids are integers allocated from a
reserved counters collection; they are local to a store and are never reused.

## Storing and fetching

`save()` recursively stores a record graph and returns its integer sid.
`fetch()` reconstructs the record, while `fetch_by_content_id()` and
`fetch_entry()` provide content-addressed and entry-family access. The three
`StorageInfo.dedup` policies are supported with the same content, value, and
non-deduplicating meanings as `SqlStore`; identity-excluded metadata conflicts
are still checked. Nested non-storable children are embedded in the owning
document. Nested storable records are stored in their own collection and
referenced by sid. For the shared save/fetch, projection, validation, and
dedup semantics, see [Storing and fetching](db.md#storing-and-fetching).

An entry family with several backing record classes has a separate dispatch
collection. Saving a configured backing makes its content identity discoverable
through `fetch_entry()`, which returns the concrete backing record. In
transaction mode the backing and dispatch writes share one transaction. In
degraded mode the backing is written first, so a crash can temporarily make
`fetch_entry()` report dispatch integrity failure; re-saving the record or
running fsck repairs the main-role case.

## Store timestamps

`MongoStore` enables store-managed timestamps by default:

```python
store = MongoStore(
    database,
    entry_records={},
    store_timestamps=True,
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
searcher.add(record.store_timestamp <= "2026-01-01T00:00:00Z")
rows = searcher.results(record=record)
```

The equivalent OPTIMADE filter is:

```python
from httk.store.mongo import optimade_filter_searcher

rows = optimade_filter_searcher(
    store, StructureRecord, '_httk_store_timestamp <= "2026-01-01T00:00:00Z"'
)
```

`present at time T` means exactly `store_timestamp <= T`. FIRST-STORED-WINS
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

Follow-ups are a serve-level as-of parameter, a per-source federation cutoff,
and an enable/disable migration for existing layouts.

## Roles, leases, and fsck

Every Mongo record document has a store-managed role:

- `main` marks a top-level record or a record addressed by an entry dispatch.
- `dep` marks a record reachable only as a dependency of another record.

This distinction lets MongoStore retain crash residue safely until an explicit
integrity pass. `store.fsck()` takes an exclusive fsck lease, blocks writers
for its duration, checks entry dispatches, repairs missing dispatches for
main-role family records, marks records reachable from roots, and deletes
unmarked dependency-role documents. It never creates a dispatch for a
dependency-role backing. The return value is an immutable `FsckSummary` with
per-collection examined, repaired, conflict, and deleted counts plus reported
violations.

After reopening a store, pass record classes that were not discoverable from
the current store declaration or session through `known_types` so fsck can
attribute their ordinary collections safely:

```python
summary = store.fsck(
    repair=True,
    collect_garbage=True,
    known_types=(StructureRecord, Author),
)
```

`repair_conflicts=True` allows fsck to delete invalid dispatch documents after
reporting them. It is a repair choice, not the default. `force=True` is an
administrative stale-lock override: use it only after verifying that the
previous owner is dead. The lease protocol has no fencing. The
`clear_stale_lock()` operation has the same administrative requirement.

Running fsck while other store processes remain open is discouraged. A live
process can retain identity-cached instances of records that fsck swept; its
next write observes the generation bump and clears those caches, but a cached
read before then can be a silent stale read. Uncached fetches of swept sids
raise `KeyError`, and sids are never reused.

## Querying and paging

`MongoStore.searcher()` follows the same neutral query protocols and expression
vocabulary as `SqlStore`: bind variables with `variable()`, add conditions
with `add()`, declare outputs, and consume either portable `SearchResult`
values or a named `results()` set. Reference paths, child set operations,
stored-property plans, scalar projections, sorting, offsets, limits, and
OPTIMADE filter wiring use the shared concepts documented in
[Searching](db.md#searching) and [Neutral portable Store profile](db.md#neutral-portable-store-profile).
Disconnected cartesian variables are outside MongoStore's supported query
profile.

```python
search = store.searcher()
s = search.variable(StructureRecord)
search.add(s.spacegroup == 225)
search.add(s.symbols.has_only("O", "Ca", "Ti"))
results = search.results(structure=s, energy=s.energy)

for row in results:
    print(row.structure.formula, row.energy)
```

Mongo result sets provide `len()`, iteration, `first()`, `one()`,
`scalars()`, and scalar `column()` access with the shared result exceptions.
`results()` materializes the result rows needed by its consumer; a query with
a client-verified predicate applies verification before count, offset, limit,
or output consumption.

`MongoResultSet.page()` is the optional keyset-paging capability described by
the neutral `PageableResultSetLike` protocol. It uses a live aggregation and
an opaque continuation token, with an internal sid tie-breaker and explicit
null ordering. The normal restrictions apply: one root variable, scalar root
outputs for order keys, no `add_sort()`, nonzero offset, or query limit, and a
page size of at most 10,000. Pages do not promise snapshot consistency across
calls. See [Continuation pages](db.md#continuation-pages) for the token and
consumer contract.

Stored properties that use `scaled_exact_equal()` (or another predicate that
needs exact client verification) have an important Mongo-specific cost. The
client-side evaluator is authoritative over hydrated records. In the Phase 5
implementation, the server prefilter is the degenerate, empty, trivially
conservative prefilter: every candidate backing is transferred to the client
for exact evaluation. The result iterator over-fetches candidates and applies
offsets, limits, counts, and page assembly only after verification. This has
the same per-row evaluation asymptotics as SQL's UDF full scan, plus candidate
transfer cost; the approved epsilon-window prefilter remains a follow-up
optimization.

## Entry providers and federation

`httk.store.mongo.StoreEntryProvider` serves configured entry families or
concrete backing records through the neutral `httk.core.EntryProvider`
contract. Family entries use the Mongo stored-property plan, and relationship
links can be declared with the same provider-facing concepts as the SQL
surface. Stored-federation membership uses the Mongo entry-store protocol and
content identities; it does not require converting a Mongo store into a SQL
store.

## Differences from the SQL backend

The following are accepted residual divergences of the MongoDB design. They
are operational behavior, not guarantees to be inferred from SQL parity.

1. **Degraded dispatch crash window.** Degraded mode admits a crash window
   where an entry record exists without its dispatch document;
   `fetch_entry()` raises, re-save or fsck repairs, and the mode is announced.

2. **Non-transactional index and validator creation.** Index and validator
   creation is not transactional. It is idempotent, additive, and synchronous
   before the first insert.

3. **No transactions in degraded mode.** `transaction()` raises in degraded
   mode.

4. **Orphan documents.** Unreachable orphan documents, for any dedup policy,
   can exist between a degraded-mode crash or dedup-discard and the next fsck.
   Degraded-mode compensation deletes nothing; fsck is the collector.
   SQL's v2.3.0 degraded SQLite profile follows the same main/dependency role
   and fsck model; see [the SQL permanentization section](db.md#permanentization-degraded-writes-and-fsck).

5. **Client-verified exact predicates.** `scaled_exact_equal()` and any
   client-verified predicate cost client-side verification and over-fetch. If
   the packet-level non-pageable fallback is taken, it requires separate
   maintainer sign-off. In the Phase 5 state, the server prefilter is the
   degenerate empty, trivially conservative one: such plans transfer every
   candidate backing to the client for exact evaluation. They have the same
   per-row evaluation asymptotics as SQL's UDF full scan, but with candidate
   transfer cost. The epsilon window remains an approved follow-up
   optimization.

6. **Fsck exclusion and stale-lock administration.** fsck blocks all writes for
   its duration. Cross-process exclusion is an advisory lease handshake.
   `force=True` stale-lock override is an administrative assertion with no
   fencing.

7. **Post-fsck cached instances.** After an fsck in one process, another live
   process may serve identity-cached instances of swept orphans until its next
   write observes the generation bump. Cached-instance reads are potentially
   silent stale reads, with no signal to the caller. Uncached fetches of swept
   sids raise `KeyError`; aliasing never occurs because sids are not reused.
   Running fsck while other store processes are open is therefore discouraged.

8. **BSON size ceiling.** Records whose embedded document exceeds MongoDB's
   BSON size ceiling (16 MB) are rejected with `RecordTooLargeError`; SQL has
   no such ceiling.

9. **Fetched-object identity.** Fetched-object identity-while-alive is a
   per-store implementation property, not a portable protocol guarantee.

10. **String matching case.** String matching is canonically case-sensitive;
    SQLite's ASCII case-insensitive `LIKE` is the divergent backend.

11. **Sharded clusters.** Sharded clusters are untested and unsupported for
    now.

## Testing profiles

Mongo tests are opt-in: set `HTTK_TEST_MONGODB_URI` to a reachable MongoDB
deployment. Without it, Mongo-specific tests and the Mongo parameter of the
neutral backend suite are skipped. A replica set exercises transaction mode;
tests also explicitly select `transactions="never"` where degraded behavior
is under test.

The default test profile excludes tests marked `extended` and uses the normal
fast coverage. The extended profile includes those tests and increases the
seeded randomized fsck graph rounds. Run both tiers with the Mongo URI:

```bash
export HTTK_TEST_MONGODB_URI='mongodb://127.0.0.1:27127/?replicaSet=httk2rs'
python -m pytest tests/ -q
HTTK_TEST_PROFILE=extended python -m pytest tests/ -q -m ""
```

The extended knob is intentionally independent of the connection URI. It
changes test depth, not MongoStore semantics. The repository's default `make
ci` remains Mongo-free; the dedicated CI job runs the live Mongo suite in both
tiers.
