"""Register entry providers implemented by :mod:`httk.store`."""

from httk.core.register import register_entry_provider

register_entry_provider(name="store-references", factory="httk.store.entry_providers:ReferenceEntryProvider")
register_entry_provider(name="store-files", factory="httk.store.entry_providers:FileEntryProvider")
register_entry_provider(name="store-calculations", factory="httk.store.entry_providers:CalculationEntryProvider")
register_entry_provider(name="store-runs", factory="httk.store.entry_providers:RunEntryProvider")
register_entry_provider(name="store-records", factory="httk.store.entry_providers:DataRecordEntryProvider")

# The database-backed provider (requires the httk-store[db] extra); the factory
# reference is lazy, so registration itself never imports sqlalchemy.
register_entry_provider(name="store-db-store", factory="httk.store.db.entry_provider:StoreEntryProvider")
