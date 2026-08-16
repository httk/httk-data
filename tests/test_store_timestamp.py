"""Store-managed timestamp coverage for the SQL backend."""

import datetime
from dataclasses import dataclass

import pytest
import sqlalchemy
from httk.core.storage import StorageInfo

from httk.store.db import Database, SqlStore, StorageLayoutUpgradeRequiredError, StoreClockRegressionError
from httk.store.db.mapping import TS_START_COLUMN, sqlalchemy_metadata
from httk.store.db.optimade import optimade_filter_searcher
from httk.store.db.schema import resolve_schema
from httk.store.store_timestamp import (
    encode_store_timestamp_state,
    ns_operand_to_store_units,
    parse_store_timestamp_state,
)


@dataclass(frozen=True)
class TimestampRecord:
    value: int


@dataclass(frozen=True)
class TimestampValueRecord:
    __httk_storage__ = StorageInfo(dedup="by_value")
    value: int


@dataclass(frozen=True)
class TimestampNoneRecord:
    __httk_storage__ = StorageInfo(dedup="none")
    value: int


def test_parent_column_index_and_off_mapping():
    enabled = sqlalchemy_metadata([resolve_schema(TimestampRecord)]).tables["timestamp_record"]
    disabled = sqlalchemy_metadata([resolve_schema(TimestampRecord)], store_timestamps=False).tables["timestamp_record"]
    assert isinstance(enabled.c[TS_START_COLUMN].type, sqlalchemy.BigInteger)
    assert not enabled.c[TS_START_COLUMN].nullable
    assert "ts_start" not in disabled.c
    assert any(index.name == "ix_timestamp_record_ts_start" for index in enabled.indexes)


def test_one_save_uses_one_pinned_timestamp_and_dedup_does_not_touch_it():
    with Database.sqlite() as database:
        store = SqlStore(database, entry_records={})
        store._clock = lambda: 1_234_567
        first = store.save(TimestampRecord(1))
        with database.engine.connect() as connection:
            before = connection.execute(sqlalchemy.text("SELECT ts_start FROM timestamp_record")).all()
        assert before == [(1234,)]
        store._clock = lambda: 9_999_999
        assert store.save(TimestampRecord(1)) == first
        with database.engine.connect() as connection:
            assert connection.execute(sqlalchemy.text("SELECT ts_start FROM timestamp_record")).all() == before


def test_reopen_requires_exact_timestamp_configuration():
    with Database.sqlite() as database:
        SqlStore(database, entry_records={}, store_timestamp_resolution=1000)
        with pytest.raises(StorageLayoutUpgradeRequiredError, match="store_timestamps"):
            SqlStore(database, store_timestamps="off")
        with pytest.raises(StorageLayoutUpgradeRequiredError, match="store_timestamps"):
            SqlStore(database, store_timestamp_resolution=1)
        with database.engine.begin() as connection:
            connection.execute(
                sqlalchemy.text(
                    "UPDATE _httk_store_metadata SET value = 'v2:creation:01000' WHERE key = 'store_timestamps'"
                )
            )
        with pytest.raises(StorageLayoutUpgradeRequiredError, match="store_timestamps"):
            SqlStore(database, store_timestamp_resolution=1000)
        with database.engine.begin() as connection:
            connection.execute(
                sqlalchemy.text(
                    "UPDATE _httk_store_metadata SET value = 'v2:creation:1000' WHERE key = 'store_timestamps'"
                )
            )
        assert SqlStore(database, store_timestamp_resolution=1000).store_timestamps


def test_explicit_transaction_pins_timestamp_and_rollback_keeps_mark() -> None:
    with Database.sqlite() as database:
        store = SqlStore(database, entry_records={})
        store._clock = lambda: 1_000_000
        with store.transaction():
            store.save(TimestampNoneRecord(1))
            store._clock = lambda: 2_000_000
            with store.transaction():
                store.save(TimestampNoneRecord(2))
        with database.engine.connect() as connection:
            assert connection.execute(
                sqlalchemy.text("SELECT ts_start FROM timestamp_none_record ORDER BY value")
            ).all() == [(1000,), (1000,)]

        store._clock = lambda: 3_000_000
        with pytest.raises(RuntimeError), store.transaction():
            store.save(TimestampNoneRecord(3))
            raise RuntimeError("rollback")
        store._clock = lambda: 1_500_000
        store.save(TimestampNoneRecord(4))


def test_clock_regression_guard_and_opt_out():
    with Database.sqlite() as database:
        store = SqlStore(database, entry_records={})
        store._clock = lambda: 10_000
        store.save(TimestampRecord(1))
        reopened = SqlStore(database)
        reopened._clock = lambda: 9_000
        with pytest.raises(StoreClockRegressionError, match="ns"):
            reopened.save(TimestampRecord(2))
        allowed = SqlStore(database, allow_clock_regression=True)
        allowed._clock = lambda: 9_000
        allowed.save(TimestampRecord(2))


def test_fsck_future_timestamp_and_clamp():
    with Database.sqlite() as database:
        store = SqlStore(database, entry_records={})
        store._clock = lambda: 1_000_000_000
        sid = store.save(TimestampRecord(1))
        with database.engine.begin() as connection:
            connection.execute(
                sqlalchemy.text("UPDATE timestamp_record SET ts_start = :value WHERE sid = :sid"),
                {"value": 10_000_000, "sid": sid},
            )
        store = SqlStore(database, entry_records={})
        store._clock = lambda: 1_000_000_000
        report = store.fsck(repair=False, collect_garbage=False, known_types=(TimestampRecord,))
        assert report.violations and "ts_start" in report.violations[0]
        repaired = store.fsck(
            repair=True,
            collect_garbage=False,
            clamp_future_timestamps=True,
            known_types=(TimestampRecord,),
        )
        assert "clamped" in repaired.violations[0]
        assert store.fsck(repair=False, collect_garbage=False, known_types=(TimestampRecord,)).violations == ()
        store._clock = lambda: 1_500_000_000
        store.save(TimestampRecord(2))


def test_fsck_future_timestamp_nondivisor_boundary():
    with Database.sqlite() as database:
        store = SqlStore(database, entry_records={}, store_timestamp_resolution=3)
        store._clock = lambda: 1
        sid = store.save(TimestampRecord(1))
        with database.engine.begin() as connection:
            connection.execute(
                sqlalchemy.text("UPDATE timestamp_record SET ts_start = :value WHERE sid = :sid"),
                {"value": 666_666_667, "sid": sid},
            )
        assert store.fsck(repair=False, collect_garbage=False, known_types=(TimestampRecord,)).violations == ()
        with database.engine.begin() as connection:
            connection.execute(
                sqlalchemy.text("UPDATE timestamp_record SET ts_start = :value WHERE sid = :sid"),
                {"value": 666_666_668, "sid": sid},
            )
        report = store.fsck(repair=False, collect_garbage=False, known_types=(TimestampRecord,))
        assert report.violations and "ts_start" in report.violations[0]


def test_degraded_transaction_exception_recomputes_mark():
    with Database.sqlite(degraded=True) as database:
        store = SqlStore(database, entry_records={})
        store._clock = lambda: 3_000_000
        with pytest.raises(RuntimeError), store.transaction():
            store.save(TimestampNoneRecord(1))
            raise RuntimeError("durable degraded rollback")
        store._clock = lambda: 2_000_000
        with pytest.raises(StoreClockRegressionError):
            store.save(TimestampNoneRecord(2))


def test_fsck_clamp_mark_is_not_published_before_commit(monkeypatch):
    with Database.sqlite() as database:
        store = SqlStore(database, entry_records={})
        store._clock = lambda: 1_000_000_000
        sid = store.save(TimestampRecord(1))
        with database.engine.begin() as connection:
            connection.execute(
                sqlalchemy.text("UPDATE timestamp_record SET ts_start = :value WHERE sid = :sid"),
                {"value": 10_000_000, "sid": sid},
            )
        store = SqlStore(database, entry_records={})
        store._clock = lambda: 1_000_000_000

        def fail_after_clamp(*_args, **_kwargs):
            raise RuntimeError("fsck commit failure")

        monkeypatch.setattr("httk.store.db.fsck._repair_dispatches", fail_after_clamp)
        with pytest.raises(RuntimeError, match="fsck commit failure"):
            store.fsck(
                repair=True,
                collect_garbage=False,
                clamp_future_timestamps=True,
                known_types=(TimestampRecord,),
            )
        store._clock = lambda: 2_000_000_000
        with pytest.raises(StoreClockRegressionError):
            store.save(TimestampRecord(2))


def test_timestamp_mark_is_not_published_when_commit_fails(monkeypatch):
    with Database.sqlite() as database:
        store = SqlStore(database, entry_records={})
        store.ensure_tables(TimestampNoneRecord)
        real_commit = database.engine.dialect.do_commit

        def failing_commit(connection):
            real_commit(connection)
            raise RuntimeError("commit failed after DBAPI commit")

        monkeypatch.setattr(database.engine.dialect, "do_commit", failing_commit)
        store._clock = lambda: 1_000_000
        with pytest.raises(RuntimeError, match="commit failed after DBAPI commit"):
            store.save(TimestampNoneRecord(1))
        assert store._store_timestamp_mark is None


def test_query_operands_floor_sort_and_optimade_integer_path():
    with Database.sqlite() as database:
        store = SqlStore(database, entry_records={}, store_timestamp_resolution=1000)
        store._clock = lambda: 1_000_000
        store.save(TimestampNoneRecord(1))
        store._clock = lambda: 1_000_500
        store.save(TimestampNoneRecord(2))

        searcher = store.searcher()
        variable = searcher.variable(TimestampNoneRecord)
        searcher.output(variable, "record")
        searcher.add(variable.ts_start <= 1_000_499)
        assert {result[0][0].value for result in searcher} == {1, 2}

        for operand in (
            1_000_499,
            datetime.datetime(1970, 1, 1, 0, 0, 0, 1000, tzinfo=datetime.UTC),
            "1970-01-01T00:00:00.001000Z",
            "1970-01-01T01:00:00.001000+01:00",
        ):
            query = store.searcher()
            candidate = query.variable(TimestampNoneRecord)
            query.output(candidate, "record")
            query.add(candidate.ts_start <= operand)
            assert list(query)
        with pytest.raises(ValueError, match="timezone-aware"):
            _ = variable.ts_start <= datetime.datetime(1970, 1, 1)  # noqa: DTZ001

        ascending = store.searcher()
        asc = ascending.variable(TimestampNoneRecord)
        ascending.output(asc, "record")
        ascending.add_sort(asc.ts_start)
        assert [item[0][0].value for item in ascending] == [1, 2]
        descending = store.searcher()
        desc = descending.variable(TimestampNoneRecord)
        descending.output(desc, "record")
        descending.add_sort(desc.ts_start, descending=True)
        assert [item[0][0].value for item in descending] == [2, 1]

        exposed = store.searcher()
        exposed_variable = exposed.variable(TimestampNoneRecord)
        exposed.output(exposed_variable.ts_start, "stamp")
        assert [row[0][0] for row in exposed] == [1_000_000, 1_000_000]

        optimade = optimade_filter_searcher(store, TimestampNoneRecord, "_httk_ts_start <= 1000499")
        assert {item[0][0].value for item in optimade} == {1, 2}


def test_bulk_uses_one_batch_timestamp_and_keeps_existing_rows():
    with Database.sqlite() as database:
        store = SqlStore(database, entry_records={})
        store._clock = lambda: 1_000_000
        store.save(TimestampValueRecord(1))
        store._clock = lambda: 2_000_000
        with store.bulk_ingest(finalize="parity") as bulk:
            bulk.save(TimestampValueRecord(2))
            bulk.save(TimestampValueRecord(2))
            bulk.save(TimestampValueRecord(3))
        with database.engine.connect() as connection:
            rows = connection.execute(
                sqlalchemy.text("SELECT value, ts_start FROM timestamp_value_record ORDER BY value")
            ).all()
        assert rows == [(1, 1000), (2, 2000), (3, 2000)]


def test_disabled_query_and_resolution_one():
    with Database.sqlite() as database:
        store = SqlStore(database, entry_records={}, store_timestamps="off")
        query = store.searcher()
        variable = query.variable(TimestampRecord)
        with pytest.raises(AttributeError, match="store_timestamps"):
            _ = variable.ts_start
    with Database.sqlite() as database:
        store = SqlStore(database, entry_records={}, store_timestamp_resolution=1)
        store._clock = lambda: 1_234_567_890
        store.save(TimestampRecord(1))
        with database.engine.connect() as connection:
            assert (
                connection.execute(sqlalchemy.text("SELECT ts_start FROM timestamp_record")).scalar_one()
                == 1_234_567_890
            )


def test_shared_timestamp_state_and_operand_helpers():
    assert parse_store_timestamp_state(encode_store_timestamp_state("creation", 1000)) == ("creation", 1000)
    assert parse_store_timestamp_state(encode_store_timestamp_state("off", 1000)) == ("off", None)
    assert parse_store_timestamp_state("v2:creation:bad") is None
    assert parse_store_timestamp_state("v2:creation:01000") is None
    assert parse_store_timestamp_state("v2:creation:+1000") is None
    assert parse_store_timestamp_state("v2:creation:1_000") is None
    # Obsolete v1 markers no longer parse.
    assert parse_store_timestamp_state("v1:1000") is None
    assert ns_operand_to_store_units(1_000_499, 1000) == 1000
    assert ns_operand_to_store_units("1970-01-01T00:00:00.001000Z", 1000) == 1000
    with pytest.raises(ValueError, match="timezone-aware"):
        ns_operand_to_store_units(datetime.datetime(1970, 1, 1), 1000)  # noqa: DTZ001


def test_versioned_mode_not_implemented_and_invalid_mode_rejected():
    with Database.sqlite() as database:
        with pytest.raises(NotImplementedError, match="versioned stores are not implemented yet"):
            SqlStore(database, entry_records={}, store_timestamps="versioned")
        with pytest.raises(ValueError, match="store_timestamps must be one of"):
            SqlStore(database, entry_records={}, store_timestamps="bogus")  # type: ignore[arg-type]
