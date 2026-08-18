# Backend storage

`httk.store.backend.sql` stores **plain frozen dataclasses** in a relational database
(SQLite, DuckDB, or PostgreSQL), makes them queryable through a
backend-agnostic search DSL, and serves them through the neutral `httk.core.EntryProvider` contract —
no SQLAlchemy types in the public API, no base class to inherit:

```python
from httk.store.backend.sql import Backend, SqlStore

db = Backend.sqlite("results.sqlite")
store = SqlStore(db, entry_records={})   # first open declares the store
# reopen later with just: SqlStore(db)

with store.transaction():
    sid = store.save(record)             # dedups and recurses automatically

same_record = store.fetch(type(record), sid)   # a lazy row; add eager=True to materialize
```

Records are content-addressed (`content_id`) as well as locally numbered
(`sid`), and identical content saves to one row however many times it arrives.

Store timestamps are enabled by default. They support historic predicates such
as `store_timestamp <= T`; configure their unit size with
`store_timestamp_resolution` (default: microseconds, `time_ns() // 1000`).
The [detailed guide](details/db.md#store-timestamps) covers the query syntax,
deduplication semantics, clock guard, and fsck repair behavior.

Append-only record replacement is available too: `store.replace(predecessor,
obj)` saves a logical successor sharing the predecessor's lineage, `store.history()`
walks a lineage, and `store.searcher(only_latest=True)` restricts root variables
to each lineage's latest row.

The full guide, {doc}`details/db`, covers declaring storable classes with the
httk-core marker vocabulary, entry families and multi-record dispatch, the
search DSL and stored properties, record replacement lineages, bulk ingestion
(including `bulk_ingest(workers=N)` and the crash-safe `finalize="deferred"`
fresh-store profile), the permanentization role model with `store.fsck()`
recovery, OPTIMADE serving, and store-layout versioning.
