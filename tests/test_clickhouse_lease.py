"""P2 ClickHouse KeeperMap lease, marker, and crash-residue coverage."""

import json
import threading
import uuid

import pytest
import sqlalchemy
from conftest import clickhouse_test_uri
from sqlalchemy import text

from httk.store.db import Database, SqlStore
from httk.store.db import clickhouse as clickhouse_adapter
from httk.store.db.clickhouse import (
    acquire_lease,
    clear_ingest_marker,
    release_lease,
    verify_lease,
    write_ingest_marker,
)
from httk.store.db.layout import StoreUnderConstructionError


class _P2InjectedCrash(BaseException):
    """Test-only hard stop used to model an interrupted client process."""


@pytest.fixture
def clickhouse_p2_database():
    uri = clickhouse_test_uri()
    source_url = sqlalchemy.engine.make_url(uri)
    database_name = f"httk_p2_lease_{uuid.uuid4().hex}"
    admin = sqlalchemy.create_engine(source_url.set(database="default"))
    database = None
    try:
        with admin.begin() as connection:
            present = connection.execute(
                text("SELECT count(*) FROM system.tables WHERE database = 'default' AND name = '_httk_bootstrap'")
            ).scalar_one()
            if not present:
                pytest.skip("ClickHouse deployment table _httk_bootstrap is absent")
            connection.execute(text(f"CREATE DATABASE {database_name}"))
        target_admin = sqlalchemy.create_engine(source_url.set(database=database_name))
        try:
            with target_admin.begin() as connection:
                connection.execute(
                    text(
                        "CREATE TABLE _httk_bootstrap (key String, value String) "
                        "ENGINE=KeeperMap('/_httk_bootstrap') PRIMARY KEY key"
                    )
                )
        finally:
            target_admin.dispose()
        database = Database.clickhouse(source_url, database=database_name)
        yield database
    finally:
        if database is not None:
            database.dispose()
        with admin.begin() as connection:
            connection.execute(text(f"DROP DATABASE IF EXISTS {database_name}"))
        admin.dispose()


def _metadata(database: Database) -> dict[str, str]:
    with database.engine.connect() as connection:
        return dict(connection.execute(text("SELECT key, value FROM _httk_store_metadata")).all())


def _fresh_database(source_url: sqlalchemy.URL, name: str) -> Database:
    return Database.clickhouse(source_url, database=name)


def test_clickhouse_lease_and_marker_values_are_token_carrying(clickhouse_p2_database: Database) -> None:
    store = SqlStore(clickhouse_p2_database, entry_records={})
    with clickhouse_p2_database.engine.begin() as connection:
        lease = acquire_lease(connection, store._lease_owner)
        marker = write_ingest_marker(connection, lease)
        lease_json = json.loads(lease)
        marker_json = json.loads(marker)
        assert set(lease_json) == {"owner", "token", "acquired_at"}
        assert marker_json["acquired_at"] == lease_json["acquired_at"]
        assert marker_json["state"] == "bulk-ingest"
        assert marker_json["token"] == lease_json["token"]
        assert isinstance(marker_json["nonce"], str)
        verify_lease(connection, lease)
        clear_ingest_marker(connection, marker + "-stale")
        assert _metadata(clickhouse_p2_database)["ingest_state"] == marker
        clear_ingest_marker(connection, marker)
        next_marker = write_ingest_marker(connection, lease)
        assert json.loads(next_marker)["nonce"] != marker_json["nonce"]
        clear_ingest_marker(connection, next_marker)
        release_lease(connection, lease)
    assert "lease" not in _metadata(clickhouse_p2_database)


def test_clickhouse_lease_is_acquired_on_first_write_and_reads_do_not_touch_it(
    clickhouse_p2_database: Database,
) -> None:
    store = SqlStore(clickhouse_p2_database, entry_records={})
    with clickhouse_p2_database.engine.begin() as connection:
        store._ensure_degraded_lease(connection)
    before = _metadata(clickhouse_p2_database)
    source_url = sqlalchemy.engine.make_url(clickhouse_test_uri())
    fresh_database = _fresh_database(source_url, clickhouse_p2_database.engine.url.database)
    try:
        reopened = SqlStore(fresh_database)
        assert _metadata(clickhouse_p2_database) == before
        with (
            pytest.raises(RuntimeError, match="ClickHouse lease is held.*manual recovery"),
            reopened._database.engine.begin() as connection,
        ):
            reopened._ensure_degraded_lease(connection)
    finally:
        fresh_database.dispose()


def test_clickhouse_dispose_releases_exact_lease(clickhouse_p2_database: Database) -> None:
    store = SqlStore(clickhouse_p2_database, entry_records={})
    with clickhouse_p2_database.engine.begin() as connection:
        store._ensure_degraded_lease(connection)
    assert "lease" in _metadata(clickhouse_p2_database)
    clickhouse_p2_database.dispose()
    assert "lease" not in _metadata(clickhouse_p2_database)


def test_clickhouse_concurrent_bulk_admission_refuses_second_context(
    clickhouse_p2_database: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    from httk.store.db.bulk import BulkIngest

    store = SqlStore(clickhouse_p2_database, entry_records={})
    first_claimed = threading.Event()
    release_first = threading.Event()
    errors: list[BaseException] = []
    first_error: list[BaseException] = []

    def hold_first_claim(_: BulkIngest) -> None:
        if first_claimed.is_set():
            return
        first_claimed.set()
        if not release_first.wait(timeout=10):
            raise AssertionError("timed out waiting to release the first bulk admission")
        raise _P2InjectedCrash("release first bulk admission")

    monkeypatch.setattr(BulkIngest, "_after_bulk_context_claim", hold_first_claim)

    def enter_first() -> None:
        try:
            store.bulk_ingest(finalize="deferred").__enter__()
        except BaseException as error:
            first_error.append(error)

    first = threading.Thread(target=enter_first)
    first.start()
    try:
        assert first_claimed.wait(timeout=10)
        with pytest.raises(RuntimeError, match="already has an open") as error:
            store.bulk_ingest(finalize="deferred").__enter__()
        errors.append(error.value)
        assert len(errors) == 1
        assert isinstance(errors[0], RuntimeError)
        assert "already has an open" in str(errors[0])
    finally:
        release_first.set()
        first.join(timeout=20)
        assert not first.is_alive()
    assert len(first_error) == 1
    assert isinstance(first_error[0], _P2InjectedCrash)


def test_clickhouse_bulk_entry_cleanup_failure_releases_admission_and_mutex(
    clickhouse_p2_database: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A secondary entry-cleanup failure cannot strand the bulk ownership."""
    from httk.store.db.bulk import BulkIngest

    store = SqlStore(clickhouse_p2_database, entry_records={})

    def fail_scan(_: BulkIngest, __) -> set[str]:
        raise _P2InjectedCrash("scan failed")

    def fail_cleanup(_: BulkIngest) -> None:
        raise _P2InjectedCrash("cleanup failed")

    monkeypatch.setattr(BulkIngest, "_scan_store", fail_scan)
    monkeypatch.setattr(BulkIngest, "_clean_up_after_failure", fail_cleanup)
    with pytest.raises(_P2InjectedCrash, match="cleanup failed"):
        store.bulk_ingest(finalize="deferred").__enter__()
    assert not store._bulk_active
    assert store._mutation_lock.acquire(blocking=False)
    store._mutation_lock.release()


def test_clickhouse_bulk_boundary_leaves_lease_and_marker_for_crash_recovery(
    clickhouse_p2_database: Database,
) -> None:
    store = SqlStore(clickhouse_p2_database, entry_records={})
    bulk = store.bulk_ingest(finalize="deferred")
    bulk.__enter__()
    try:
        values = _metadata(clickhouse_p2_database)
        assert json.loads(values["lease"])["token"] == json.loads(values["ingest_state"])["token"]
        source_url = sqlalchemy.engine.make_url(clickhouse_test_uri())
        fresh_database = _fresh_database(source_url, clickhouse_p2_database.engine.url.database)
        try:
            with pytest.raises(StoreUnderConstructionError):
                SqlStore(fresh_database)
        finally:
            fresh_database.dispose()
        with clickhouse_p2_database.engine.begin() as connection:
            clear_ingest_marker(connection, values["ingest_state"])
    finally:
        bulk._release_connection(bulk._connection)
        bulk._connection = None
        bulk._release_bulk_ownership()
        store._release_bulk_context()


def test_clickhouse_preexisting_marker_survives_new_bulk_attempt(
    clickhouse_p2_database: Database,
) -> None:
    store = SqlStore(clickhouse_p2_database, entry_records={})
    bulk = store.bulk_ingest(finalize="deferred")
    bulk.__enter__()
    try:
        prior_marker = _metadata(clickhouse_p2_database)["ingest_state"]
        with pytest.raises(RuntimeError, match="already has an open"):
            store.bulk_ingest(finalize="deferred").__enter__()
        assert _metadata(clickhouse_p2_database)["ingest_state"] == prior_marker
    finally:
        bulk._release_connection(bulk._connection)
        bulk._connection = None
        bulk._release_bulk_ownership()
        store._release_bulk_context()


def test_clickhouse_crash_after_lease_before_marker_keeps_lease_until_dispose(
    clickhouse_p2_database: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    from httk.store.db.bulk import BulkIngest

    def crash(_: BulkIngest) -> None:
        raise _P2InjectedCrash("after lease")

    monkeypatch.setattr(BulkIngest, "_after_clickhouse_lease_acquired", crash)
    store = SqlStore(clickhouse_p2_database, entry_records={})
    with pytest.raises(_P2InjectedCrash):
        store.bulk_ingest(finalize="deferred").__enter__()
    values = _metadata(clickhouse_p2_database)
    assert "lease" in values
    assert "ingest_state" not in values
    source_url = sqlalchemy.engine.make_url(clickhouse_test_uri())
    fresh_database = _fresh_database(source_url, clickhouse_p2_database.engine.url.database)
    try:
        fresh_store = SqlStore(fresh_database)
        with (
            pytest.raises(RuntimeError, match="ClickHouse lease is held.*manual recovery"),
            fresh_database.engine.begin() as connection,
        ):
            fresh_store._ensure_degraded_lease(connection)
    finally:
        fresh_database.dispose()
    clickhouse_p2_database.dispose()
    assert "lease" not in _metadata(clickhouse_p2_database)


def test_clickhouse_uncertain_marker_insert_cleans_observed_token(
    clickhouse_p2_database: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_insert = clickhouse_adapter._strict_insert

    def uncertain_insert(connection, key: str, value: str) -> None:
        original_insert(connection, key, value)
        if key == "ingest_state":
            raise _P2InjectedCrash("ambiguous marker response")

    monkeypatch.setattr(clickhouse_adapter, "_strict_insert", uncertain_insert)
    store = SqlStore(clickhouse_p2_database, entry_records={})
    with pytest.raises(_P2InjectedCrash):
        store.bulk_ingest(finalize="deferred").__enter__()
    values = _metadata(clickhouse_p2_database)
    assert "lease" in values
    assert "ingest_state" not in values
    clickhouse_p2_database.dispose()
    assert "lease" not in _metadata(clickhouse_p2_database)


def test_clickhouse_marker_residue_rejects_fresh_open_after_marker_write(
    clickhouse_p2_database: Database,
) -> None:
    store = SqlStore(clickhouse_p2_database, entry_records={})
    with clickhouse_p2_database.engine.begin() as connection:
        lease = acquire_lease(connection, store._lease_owner)
        marker = write_ingest_marker(connection, lease)
    source_url = sqlalchemy.engine.make_url(clickhouse_test_uri())
    fresh_database = _fresh_database(source_url, clickhouse_p2_database.engine.url.database)
    try:
        with pytest.raises(StoreUnderConstructionError):
            SqlStore(fresh_database)
    finally:
        fresh_database.dispose()
    with clickhouse_p2_database.engine.begin() as connection:
        clear_ingest_marker(connection, marker)
        release_lease(connection, lease)


def test_clickhouse_interrupted_marker_clear_leaves_fail_closed_residue(
    clickhouse_p2_database: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    from httk.store.db.bulk import BulkIngest

    monkeypatch.setattr(BulkIngest, "_clickhouse_p3_boundary", lambda _: None)
    monkeypatch.setattr(BulkIngest, "_deferred_finalize", lambda _: None)
    monkeypatch.setattr(
        BulkIngest,
        "_before_clickhouse_marker_clear",
        lambda _: (_ for _ in ()).throw(_P2InjectedCrash("before marker clear")),
    )
    store = SqlStore(clickhouse_p2_database, entry_records={})
    with pytest.raises(_P2InjectedCrash), store.bulk_ingest(finalize="deferred"):
        pass
    values = _metadata(clickhouse_p2_database)
    assert "ingest_state" in values
    source_url = sqlalchemy.engine.make_url(clickhouse_test_uri())
    fresh_database = _fresh_database(source_url, clickhouse_p2_database.engine.url.database)
    try:
        with pytest.raises(StoreUnderConstructionError, match="dropped and re-ingested"):
            SqlStore(fresh_database)
    finally:
        fresh_database.dispose()
    with clickhouse_p2_database.engine.begin() as connection:
        clear_ingest_marker(connection, values["ingest_state"])
        release_lease(connection, values["lease"])


def test_clickhouse_p2_clean_exit_glue_clears_marker_but_keeps_lifecycle_lease(
    clickhouse_p2_database: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Prove P2 marker/lease glue; P3 must prove a real durable clean exit."""
    from httk.store.db.bulk import BulkIngest

    monkeypatch.setattr(BulkIngest, "_clickhouse_p3_boundary", lambda _: None)
    monkeypatch.setattr(BulkIngest, "_deferred_finalize", lambda _: None)
    store = SqlStore(clickhouse_p2_database, entry_records={})
    with store.bulk_ingest(finalize="deferred"):
        assert "ingest_state" in _metadata(clickhouse_p2_database)
    values = _metadata(clickhouse_p2_database)
    assert "ingest_state" not in values
    assert "lease" in values


def test_clickhouse_teardown_keeps_bulk_admission_closed_until_ownership_release(
    clickhouse_p2_database: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    from httk.store.db.bulk import BulkIngest

    teardown_started = threading.Event()
    second_finished = threading.Event()
    second_errors: list[BaseException] = []
    monkeypatch.setattr(BulkIngest, "_clickhouse_p3_boundary", lambda _: None)
    monkeypatch.setattr(BulkIngest, "_deferred_finalize", lambda _: None)

    def before_admission_release(_: BulkIngest) -> None:
        teardown_started.set()
        if not second_finished.wait(timeout=10):
            raise AssertionError("timed out waiting for the concurrent bulk admission")

    monkeypatch.setattr(BulkIngest, "_before_bulk_context_release", before_admission_release)
    store = SqlStore(clickhouse_p2_database, entry_records={})

    def enter_second() -> None:
        assert teardown_started.wait(timeout=10)
        try:
            store.bulk_ingest(finalize="deferred").__enter__()
        except BaseException as error:
            second_errors.append(error)
        finally:
            second_finished.set()

    thread = threading.Thread(target=enter_second)
    thread.start()
    with store.bulk_ingest(finalize="deferred"):
        pass
    thread.join(timeout=20)
    assert not thread.is_alive()
    assert len(second_errors) == 1
    assert isinstance(second_errors[0], RuntimeError)
    assert "already has an open" in str(second_errors[0])
    store._claim_bulk_context()
    store._release_bulk_context()


def test_clickhouse_interrupted_mutex_acquire_unwinds_bulk_admission(
    clickhouse_p2_database: Database,
) -> None:
    class InterruptingLock:
        def acquire(self) -> None:
            raise _P2InjectedCrash("interrupted mutex acquire")

    store = SqlStore(clickhouse_p2_database, entry_records={})
    store._mutation_lock = InterruptingLock()  # type: ignore[assignment]
    with pytest.raises(_P2InjectedCrash):
        store.bulk_ingest(finalize="deferred").__enter__()
    store._claim_bulk_context()
    store._release_bulk_context()


def test_clickhouse_dispose_waits_for_inflight_bulk_lifecycle_guard(
    clickhouse_p2_database: Database, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from httk.store.db import engine as engine_module
    from httk.store.db.bulk import BulkIngest

    entered = threading.Event()
    finalizer_started = threading.Event()
    release_finalizer = threading.Event()
    dispose_started = threading.Event()
    dispose_done = threading.Event()
    errors: list[BaseException] = []

    monkeypatch.setattr(engine_module, "_DISPOSE_WAIT_WARNING_SECONDS", 0.05)
    monkeypatch.setattr(BulkIngest, "_clickhouse_p3_boundary", lambda _: None)

    def finalizer(_: BulkIngest) -> None:
        finalizer_started.set()
        if not release_finalizer.wait(timeout=10):
            raise AssertionError("timed out waiting for the in-flight bulk release")

    monkeypatch.setattr(BulkIngest, "_deferred_finalize", finalizer)
    store = SqlStore(clickhouse_p2_database, entry_records={})

    def run_bulk() -> None:
        try:
            with store.bulk_ingest(finalize="deferred"):
                entered.set()
        except BaseException as error:
            errors.append(error)

    bulk_thread = threading.Thread(target=run_bulk)
    bulk_thread.start()
    assert entered.wait(timeout=10)
    assert finalizer_started.wait(timeout=10)

    def dispose() -> None:
        dispose_started.set()
        clickhouse_p2_database.dispose()
        dispose_done.set()

    dispose_thread = threading.Thread(target=dispose)
    dispose_thread.start()
    assert dispose_started.wait(timeout=10)
    assert not dispose_done.wait(timeout=0.2)
    assert any("in-flight lifecycle guard holder" in record.message for record in caplog.records)
    release_finalizer.set()
    bulk_thread.join(timeout=20)
    dispose_thread.join(timeout=20)
    assert not bulk_thread.is_alive()
    assert not dispose_thread.is_alive()
    assert not errors
    assert dispose_done.is_set()
    assert "lease" not in _metadata(clickhouse_p2_database)


def test_clickhouse_dispose_from_lifecycle_owner_fails_without_waiting(clickhouse_p2_database: Database) -> None:
    """Disposal from an owned lifecycle guard fails instead of self-deadlocking."""
    done = threading.Event()
    errors: list[BaseException] = []

    def dispose_from_guard() -> None:
        try:
            with clickhouse_p2_database.lifecycle_guard(
                clickhouse_p2_database.lifecycle_generation,
                holder="self-dispose probe",
            ):
                clickhouse_p2_database.dispose()
        except BaseException as error:
            errors.append(error)
        finally:
            done.set()

    probe = threading.Thread(target=dispose_from_guard, daemon=True)
    probe.start()
    assert done.wait(timeout=10)
    probe.join(timeout=1)
    assert not probe.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], RuntimeError)
    assert "cannot dispose from within an active bulk context" in str(errors[0])


def test_clickhouse_interrupted_release_is_exact_and_idempotent(
    clickhouse_p2_database: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = SqlStore(clickhouse_p2_database, entry_records={})
    with clickhouse_p2_database.engine.begin() as connection:
        store._ensure_degraded_lease(connection)
    lease = store._lease_value
    assert lease is not None
    original_release = clickhouse_adapter.release_lease

    def interrupted_release(connection, value: str | None) -> None:
        original_release(connection, value)
        raise RuntimeError("ambiguous release response")

    monkeypatch.setattr(clickhouse_adapter, "release_lease", interrupted_release)
    generation = store._lease_lifecycle_generation
    assert generation is not None
    with pytest.raises(RuntimeError, match="ambiguous release"):
        store._release_degraded_lease(generation)
    monkeypatch.undo()
    with clickhouse_p2_database.engine.begin() as connection:
        release_lease(connection, lease)
        release_lease(connection, lease)
    assert "lease" not in _metadata(clickhouse_p2_database)


def test_clickhouse_code_999_classification_uses_structured_exception_code() -> None:
    class CodedError(RuntimeError):
        code = 999

    wrapper = RuntimeError("reformatted server detail")
    wrapper.orig = CodedError("not node exists text")  # type: ignore[attr-defined]
    assert clickhouse_adapter._is_keeper_node_exists(wrapper)

    class OtherCodedError(RuntimeError):
        code = 998

    other = OtherCodedError("Node exists text must not override structured code")
    assert not clickhouse_adapter._is_keeper_node_exists(other)

    assert clickhouse_adapter._is_keeper_node_exists(RuntimeError("Node exists"))


def test_clickhouse_stale_release_cannot_delete_new_holder(clickhouse_p2_database: Database) -> None:
    store = SqlStore(clickhouse_p2_database, entry_records={})
    with clickhouse_p2_database.engine.begin() as connection:
        old = acquire_lease(connection, store._lease_owner)
        release_lease(connection, old)
        new = acquire_lease(connection, uuid.uuid4().hex)
        release_lease(connection, old)
        assert _metadata(clickhouse_p2_database)["lease"] == new
        release_lease(connection, new)
        release_lease(connection, new)
