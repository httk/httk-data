"""Storage backends for *httk-store*.

Each subpackage is a self-contained storage backend:

- :mod:`httk.store.backend.sql` — the SQLAlchemy-backed relational layer
  (:class:`~httk.store.backend.sql.engine.Backend`,
  :class:`~httk.store.backend.sql.store.SqlStore`) shared by every SQL dialect;
- :mod:`httk.store.backend.sqlite`, :mod:`httk.store.backend.duckdb`,
  :mod:`httk.store.backend.clickhouse`, :mod:`httk.store.backend.postgresql` —
  the per-dialect constructor and quirk code the SQL layer delegates to; and
- :mod:`httk.store.backend.mongo` — the MongoDB-backed layer
  (:class:`~httk.store.backend.mongo.store.MongoStore`).

Importing this package pulls in neither ``sqlalchemy`` nor ``pymongo``; the
driver-backed names load lazily on first use of the relevant subpackage.
"""
