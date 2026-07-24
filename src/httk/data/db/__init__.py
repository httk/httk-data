"""The SQL storage layer of httk-data: store frozen dataclasses in relational databases.

This subpackage turns plain frozen dataclasses — declared storable with the
stdlib-only marker vocabulary in httk-core (``Indexed``, ``Unique``, ``Skip``,
``Shape``, ``StorageInfo``, ``stored_property``) — into relational storage.
The pure-Python foundation lives here:

- :mod:`httk.data.db.schema` — :func:`resolve_schema` reads a storable class
  into a :class:`TableSchema`, the single source of truth for DDL, inserts,
  selects, and reconstruction;
- :mod:`httk.data.db.codecs` — the :class:`ValueCodec` registry with exact,
  round-trippable encodings for rationals, surds, and datetimes;
- :mod:`httk.data.db.identity` — :func:`canonical_form` and :func:`content_id`,
  the content identity used for deduplication.

These modules import cleanly without sqlalchemy. The SQL layer proper builds
on them and requires the ``httk-data[db]`` extra (sqlalchemy):

- :class:`~httk.data.db.engine.Database` — the engine wrapper naming where
  data lives (``Database.sqlite(...)``, ``Database.duckdb(...)``);
- :class:`~httk.data.db.store.SqlStore` — save/fetch/dedup/transactions for
  storable instances, on top of the schema-to-table mapping in
  :mod:`httk.data.db.mapping`;
- :class:`~httk.data.db.searcher.SqlSearcher` (from
  :meth:`~httk.data.db.store.SqlStore.searcher`) — the query DSL implementing
  the :mod:`httk.data.query` search protocols, with
  :class:`~httk.data.db.searcher.SqlVariable`,
  :class:`~httk.data.db.searcher.SqlColumn` and
  :class:`~httk.data.db.searcher.SqlExpression`;
- :class:`~httk.data.db.entry_provider.StoreEntryProvider` — the bridge that
  serves stored classes through the neutral :class:`~httk.core.EntryProvider`
  contract (e.g. as an OPTIMADE API via *httk-optimade*);
- :func:`~httk.data.db.optimade.optimade_filter_searcher` — OPTIMADE-filter
  querying over storable classes, tying the generic filter translation in
  :mod:`httk.data.optimade_query` to the SQL layer.

The sqlalchemy-backed names are imported lazily on first attribute access, so
``import httk.data.db`` keeps working without sqlalchemy; touching them
without sqlalchemy installed raises :class:`ImportError` naming the extra.
"""

import importlib
from typing import Any

from .codecs import (
    FRACTION_EXACT_FORMAT,
    FRACVECTOR_EXACT_FORMAT,
    SURD_EXACT_FORMAT,
    ScalarKind,
    ValueCodec,
    codec_for,
    codec_named,
    decode_fraction_exact,
    decode_fracvector_exact,
    decode_surdscalar_exact,
    encode_fraction_exact,
    encode_fracvector_exact,
    encode_fracvector_floats,
    encode_surdscalar_exact,
    known_value_codecs,
    register_value_codec,
)
from .identity import canonical_form, content_id
from .schema import (
    ChildTableSpec,
    ColumnSpec,
    FieldRole,
    FieldSpec,
    SchemaError,
    TableSchema,
    register_schema_override,
    resolve_schema,
    snake_case,
)

__all__ = [
    "ScalarKind",
    "FieldRole",
    "SchemaError",
    "ColumnSpec",
    "ChildTableSpec",
    "FieldSpec",
    "TableSchema",
    "resolve_schema",
    "register_schema_override",
    "snake_case",
    "ValueCodec",
    "register_value_codec",
    "known_value_codecs",
    "codec_for",
    "codec_named",
    "FRACTION_EXACT_FORMAT",
    "SURD_EXACT_FORMAT",
    "FRACVECTOR_EXACT_FORMAT",
    "encode_fraction_exact",
    "decode_fraction_exact",
    "encode_surdscalar_exact",
    "decode_surdscalar_exact",
    "encode_fracvector_exact",
    "decode_fracvector_exact",
    "encode_fracvector_floats",
    "canonical_form",
    "content_id",
    "Database",  # pyright: ignore[reportUnsupportedDunderAll]  (provided lazily via __getattr__)
    "SqlStore",  # pyright: ignore[reportUnsupportedDunderAll]  (provided lazily via __getattr__)
    "SqlSearcher",  # pyright: ignore[reportUnsupportedDunderAll]  (provided lazily via __getattr__)
    "SqlVariable",  # pyright: ignore[reportUnsupportedDunderAll]  (provided lazily via __getattr__)
    "SqlColumn",  # pyright: ignore[reportUnsupportedDunderAll]  (provided lazily via __getattr__)
    "SqlExpression",  # pyright: ignore[reportUnsupportedDunderAll]  (provided lazily via __getattr__)
    "StoreEntryProvider",  # pyright: ignore[reportUnsupportedDunderAll]  (provided lazily via __getattr__)
    "optimade_filter_searcher",  # pyright: ignore[reportUnsupportedDunderAll]  (provided lazily via __getattr__)
]

_SQL_EXPORTS = {
    "Database": ".engine",
    "SqlStore": ".store",
    "SqlSearcher": ".searcher",
    "SqlVariable": ".searcher",
    "SqlColumn": ".searcher",
    "SqlExpression": ".searcher",
    "StoreEntryProvider": ".entry_provider",
    "optimade_filter_searcher": ".optimade",
}


def __getattr__(name: str) -> Any:
    """Import the sqlalchemy-backed exports lazily (PEP 562), naming the extra when absent."""
    module_name = _SQL_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        module = importlib.import_module(module_name, __name__)
    except ModuleNotFoundError as error:
        if error.name is not None and error.name.partition(".")[0] == "sqlalchemy":
            raise ImportError(
                f"{__name__}.{name} needs sqlalchemy; install the 'httk-data[db]' extra to use the SQL layer"
            ) from error
        raise
    return getattr(module, name)
