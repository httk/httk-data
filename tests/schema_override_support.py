"""Shared schema-override contextmanager for the layout fingerprint tests."""

from collections.abc import Iterator
from contextlib import contextmanager

from httk.core.storage import StorageInfo

from httk.store.backend.sql import schema as schema_module
from httk.store.backend.sql.schema import register_schema_override


@contextmanager
def schema_override(cls: type, info: StorageInfo) -> Iterator[None]:
    """Register a schema override and evict it (and its cache entries) afterwards.

    :param cls: The storable class whose resolved schema is overridden.
    :param info: The replacement storage info.
    :return: A context active for the override's lifetime.
    """
    register_schema_override(cls, info)
    try:
        yield
    finally:
        schema_module._schema_overrides.pop(cls, None)
        for key in [key for key in schema_module._schema_cache if key[0] is cls]:
            del schema_module._schema_cache[key]
