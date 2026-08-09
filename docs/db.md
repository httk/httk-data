# Database storage

`httk.data.db` stores **plain frozen dataclasses** in a relational database
(SQLite or DuckDB), makes them queryable through a backend-agnostic search
DSL, and serves them through the neutral `httk.core.EntryProvider` contract —
no SQLAlchemy types in the public API, no base class to inherit:

```python
from httk.data.db import Database, SqlStore

db = Database.sqlite("results.sqlite")
store = SqlStore(db, entry_records={})   # first open declares the store
# reopen later with just: SqlStore(db)

with store.transaction():
    sid = store.save(record)             # dedups and recurses automatically

same_record = store.fetch(type(record), sid)   # reconstructed exactly
```

Records are content-addressed (`content_id`) as well as locally numbered
(`sid`), and identical content saves to one row however many times it arrives.

The full guide, {doc}`details/db`, covers declaring storable classes with the
httk-core marker vocabulary, entry families and multi-record dispatch, the
search DSL and stored properties, bulk ingestion (including
`bulk_ingest(workers=N)` and the crash-safe `finalize="deferred"` fresh-store
profile), the permanentization role model with `store.fsck()` recovery,
OPTIMADE serving, and store-layout versioning.
