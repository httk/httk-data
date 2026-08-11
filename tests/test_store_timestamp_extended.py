"""Extended bulk and deferred-finalize coverage for store timestamps."""

from dataclasses import dataclass
from typing import ClassVar

import pytest
import sqlalchemy
from httk.core.storage import StorageInfo

from httk.store.db import Database, SqlStore
from httk.store.db.bulk_deferred import DeferredFinalizer
from httk.store.db.mapping import SID_COLUMN, STORE_TIMESTAMP_COLUMN

TABLE_NAME = "extended_timestamp_record"


@dataclass(frozen=True)
class ExtendedTimestampRecord:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(dedup="by_value", storage_name=TABLE_NAME)

    value: int


@pytest.mark.extended
def test_parallel_workers_use_one_batch_timestamp_and_first_by_value_row() -> None:
    with Database.sqlite() as database:
        store = SqlStore(database, entry_records={})
        store._clock = lambda: 7_654_321
        with store.bulk_ingest(workers=2, finalize="parity") as bulk:
            first = bulk.save(ExtendedTimestampRecord(1))
            duplicate = bulk.save(ExtendedTimestampRecord(1))
            bulk.save(ExtendedTimestampRecord(2))

        assert bulk.resolved_sid(ExtendedTimestampRecord, first) == bulk.resolved_sid(
            ExtendedTimestampRecord, duplicate
        )
        with database.engine.connect() as connection:
            assert connection.execute(
                sqlalchemy.text(f'SELECT value, "{STORE_TIMESTAMP_COLUMN}" FROM {TABLE_NAME} ORDER BY value')
            ).all() == [(1, 7_654), (2, 7_654)]


@pytest.mark.extended
def test_deferred_by_value_collapse_ignores_timestamp_and_sqlite_validation_includes_it(monkeypatch) -> None:
    original_make_stage_views = DeferredFinalizer._make_stage_views

    def make_stage_views_with_timestamp_variant(finalizer: DeferredFinalizer) -> None:
        original_make_stage_views(finalizer)
        table_name = TABLE_NAME
        original_view = finalizer.stage_views[table_name]
        table = finalizer.store._table(table_name)
        columns = []
        for column in table.columns:
            if column.name == STORE_TIMESTAMP_COLUMN:
                expression = (
                    f'CASE WHEN "value" = 1 AND "{SID_COLUMN}" != '
                    f'(SELECT MIN("{SID_COLUMN}") FROM "{original_view}" WHERE "value" = 1) '
                    f'THEN "{STORE_TIMESTAMP_COLUMN}" + 1 ELSE "{STORE_TIMESTAMP_COLUMN}" END'
                )
            else:
                expression = f'"{column.name}"'
            columns.append(f'{expression} AS "{column.name}"')
        variant = finalizer._temp_name("timestamp_variant", table_name)
        finalizer._create_view(variant, f'SELECT {", ".join(columns)} FROM "{original_view}"')
        finalizer.stage_views[table_name] = variant

    monkeypatch.setattr(DeferredFinalizer, "_make_stage_views", make_stage_views_with_timestamp_variant)
    with Database.sqlite() as database:
        store = SqlStore(database, entry_records={})
        store._clock = lambda: 8_765_432
        with store.bulk_ingest(workers=2, finalize="deferred") as bulk:
            bulk.save(ExtendedTimestampRecord(1))
            bulk.save(ExtendedTimestampRecord(1))
            bulk.save(ExtendedTimestampRecord(2))

        with database.engine.connect() as connection:
            rows = connection.execute(
                sqlalchemy.text(f'SELECT value, "{STORE_TIMESTAMP_COLUMN}" FROM {TABLE_NAME} ORDER BY value')
            ).all()
        assert rows == [(1, 8_765), (2, 8_765)]
