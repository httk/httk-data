"""Minimal usage example for httk-data.

Serves a bibliographic reference through the entry-provider contract and
validates a record against the vendored OPTIMADE ``references`` definition.
"""

from httk.core import standard_entry_type

from httk.data import ReferenceEntryProvider, validate_record

provider = ReferenceEntryProvider({"ref-1": {"title": "A study", "year": "2021"}})
print(list(provider.entry_types()))

validate_record(
    standard_entry_type("references"),
    {"id": "ref-1", "type": "references", "title": "A study", "year": "2021"},
)
print("record is valid")
