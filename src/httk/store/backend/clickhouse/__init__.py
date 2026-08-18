"""ClickHouse-specific support for the SQL storage backend.

The dialect-agnostic engine wrapper and store live in
:mod:`httk.store.backend.sql`; this package holds the ClickHouse constructor
(:func:`httk.store.backend.clickhouse.engine.database`) and the Keeper-backed
schema, lease, and connection-guard machinery in
:mod:`httk.store.backend.clickhouse.support`.
"""
