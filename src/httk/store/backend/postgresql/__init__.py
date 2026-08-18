"""PostgreSQL-specific support for the SQL storage backend.

The dialect-agnostic engine wrapper and store live in
:mod:`httk.store.backend.sql`; this package holds the PostgreSQL constructor
(:func:`httk.store.backend.postgresql.engine.database`) and the ``@compiles``
hook rewriting ``httk_fraction_scaled_equal`` to inline SQL in
:mod:`httk.store.backend.postgresql.compiler`.
"""
