# MongoDB storage

`httk.store.mongo` stores the same plain frozen dataclasses as the SQL layer in
MongoDB's document model: one document per record, embedded child arrays, and
the same neutral `Store`/`Searcher` protocols, entry-family dispatch, and
entry-provider surface as `SqlStore`.

```bash
python -m pip install "httk-store[mongodb]"
```

```python
from httk.store.mongo import MongoDatabase, MongoStore

uri = "mongodb://127.0.0.1:27017/?replicaSet=httk2rs"
with MongoDatabase.connect(uri, database="materials") as database:
    store = MongoStore(database, entry_records={})
    # store.save(), store.fetch(), store.searcher() work as with SqlStore
```

Choose `MongoStore` when MongoDB is already your operational data service or
document-shaped records fit; choose `SqlStore` for relational deployments and
SQL's stronger transaction model.

Store timestamps are enabled by default. They support historic predicates such
as `ts_start <= T`; configure their unit size with
`store_timestamp_resolution` (default: microseconds, `time_ns() // 1000`).
The [detailed guide](details/mongo.md#store-timestamps) covers the query
syntax, deduplication semantics, clock guard, and fsck repair behavior.

`store_timestamps="versioned"` adds record versioning: `store.replace(old, new)`
supersedes a family entry, queries return the current view by default (with
`as_of=T` slicing and a `scoped=False` escape), and each entry serves
`_httk_ts_start` / `_httk_ts_end` lifetime bounds. `replace` requires a
replica-set deployment (multi-document transactions). See
[Versioned stores](details/mongo.md#versioned-stores).

The full guide, {doc}`details/mongo`, covers the document mapping, query
translation, continuation paging, replica-set requirements, and the documented
differences from the SQL backend.
