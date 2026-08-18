"""DuckDB-specific construction for the SQL storage backend.

The dialect-agnostic engine wrapper and store live in
:mod:`httk.store.backend.sql`; this package holds only the DuckDB constructor
body (and its pandas-probe sentinel) that
:meth:`httk.store.backend.sql.engine.Backend.duckdb` delegates to.
"""
