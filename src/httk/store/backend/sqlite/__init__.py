"""SQLite-specific construction for the SQL storage backend.

The dialect-agnostic engine wrapper and store live in
:mod:`httk.store.backend.sql`; this package holds only the SQLite constructor
body that :meth:`httk.store.backend.sql.engine.Backend.sqlite` delegates to.
"""
