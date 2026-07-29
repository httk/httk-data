"""Register entry providers implemented by :mod:`httk.data`."""

from httk.core.register import register_entry_provider

register_entry_provider(name="data-references", factory="httk.data.entry_providers:ReferenceEntryProvider")
register_entry_provider(name="data-files", factory="httk.data.entry_providers:FileEntryProvider")
register_entry_provider(name="data-calculations", factory="httk.data.entry_providers:CalculationEntryProvider")

# The database-backed provider (requires the httk-data[db] extra); the factory
# reference is lazy, so registration itself never imports sqlalchemy.
register_entry_provider(name="data-db-store", factory="httk.data.db.entry_provider:StoreEntryProvider")
