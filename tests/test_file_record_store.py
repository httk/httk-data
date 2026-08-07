"""Coverage for SQL storage of OPTIMADE ``files`` records."""

import datetime
from dataclasses import replace

import pytest
from httk.core import FileEntry, FileRecord

from httk.data.db import Database, EntryMetadataConflictError, SqlStore


def test_sql_store_round_trips_file_record() -> None:
    with Database.sqlite() as database:
        store = SqlStore(database, entry_records={FileEntry: FileRecord})
        record = FileRecord(
            url="https://example.org/files/data.json",
            name="data.json",
            size=1234,
            media_type="application/json",
            description="Example data",
            sha256="a" * 64,
            immutable_id="file-immutable",
            last_modified=datetime.datetime(2026, 8, 1, 12, 0, tzinfo=datetime.UTC),
            checksums={"sha256": "a" * 64},
        )

        sid = store.save(record)
        fetched = SqlStore(database, entry_records={FileEntry: FileRecord}).fetch_entry(FileEntry, record.id)
        assert fetched == replace(record, checksums=None)
        assert fetched.url == record.url
        assert fetched.name == record.name
        assert fetched.size == record.size
        assert fetched.media_type == record.media_type
        assert fetched.description == record.description
        assert fetched.immutable_id == record.immutable_id
        assert fetched.last_modified == record.last_modified
        assert fetched.sha256 == record.sha256
        assert fetched.checksums is None

        with pytest.raises(EntryMetadataConflictError):
            store.save(replace(record, immutable_id="different"))

        assert store.save(record) == sid

        other = replace(record, url="https://mirror.example.org/files/data.json")
        assert store.save(other) != sid
        assert other.id != record.id
