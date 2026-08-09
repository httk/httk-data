"""P2 ClickHouse KeeperMap lease, marker, and crash-residue coverage."""

import json
import os
import threading
import uuid

import pytest
import sqlalchemy
from sqlalchemy import text

from httk.data.db import Database, SqlStore
from httk.data.db import clickhouse as clickhouse_adapter
from httk.data.db.clickhouse import (
    acquire_lease,
    clear_ingest_marker,
    release_lease,
    verify_lease,
    write_ingest_marker,
)
from httk.data.db.layout import StoreUnderConstructionError


class _P2InjectedCrash(BaseException):
    """Test-only hard stop used to model an interrupted client process."""


@pytest.fixture
def clickhouse_p2_database():
    uri = os.environ.get("HTTK_TEST_CLICKHOUSE_URI")
    if not uri:
        pytest.skip("HTTK_TEST_CLICKHOUSE_URI is not set")
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
    source_url = sqlalchemy.engine.make_url(os.environ["HTTK_TEST_CLICKHOUSE_URI"])
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
    clickhouse_p2_database: Database,
) -> None:
    store = SqlStore(clickhouse_p2_database, entry_records={})
    barrier = threading.Barrier(2)
    errors: list[BaseException] = []
    successes = []

    def enter_bulk() -> None:
        barrier.wait(timeout=10)
        bulk = store.bulk_ingest(finalize="deferred")
        try:
            bulk.__enter__()
        except BaseException as error:
            errors.append(error)
        else:
            successes.append(bulk)

    threads = [threading.Thread(target=enter_bulk) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=20)
        assert not thread.is_alive()
    for bulk in successes:
        bulk.__exit__(None, None, None)
    assert not successes
    assert sum(isinstance(error, NotImplementedError) for error in errors) == 1
    assert sum(isinstance(error, RuntimeError) and "already has an open" in str(error) for error in errors) == 1


def test_clickhouse_bulk_boundary_leaves_lease_and_marker_for_crash_recovery(
    clickhouse_p2_database: Database,
) -> None:
    store = SqlStore(clickhouse_p2_database, entry_records={})
    with pytest.raises(NotImplementedError, match="implemented in P3"):
        store.bulk_ingest(finalize="deferred").__enter__()
    values = _metadata(clickhouse_p2_database)
    assert json.loads(values["lease"])["token"] == json.loads(values["ingest_state"])["token"]
    source_url = sqlalchemy.engine.make_url(os.environ["HTTK_TEST_CLICKHOUSE_URI"])
    fresh_database = _fresh_database(source_url, clickhouse_p2_database.engine.url.database)
    try:
        with pytest.raises(StoreUnderConstructionError):
            SqlStore(fresh_database)
    finally:
        fresh_database.dispose()
    with clickhouse_p2_database.engine.begin() as connection:
        clear_ingest_marker(connection, values["ingest_state"])


def test_clickhouse_preexisting_marker_survives_new_bulk_attempt(
    clickhouse_p2_database: Database,
) -> None:
    store = SqlStore(clickhouse_p2_database, entry_records={})
    with pytest.raises(NotImplementedError, match="implemented in P3"):
        store.bulk_ingest(finalize="deferred").__enter__()
    prior_marker = _metadata(clickhouse_p2_database)["ingest_state"]
    with pytest.raises(Exception, match="Node exists"):
        store.bulk_ingest(finalize="deferred").__enter__()
    assert _metadata(clickhouse_p2_database)["ingest_state"] == prior_marker


def test_clickhouse_crash_after_lease_before_marker_keeps_lease_until_dispose(
    clickhouse_p2_database: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    from httk.data.db.bulk import BulkIngest

    def crash(_: BulkIngest) -> None:
        raise _P2InjectedCrash("after lease")

    monkeypatch.setattr(BulkIngest, "_after_clickhouse_lease_acquired", crash)
    store = SqlStore(clickhouse_p2_database, entry_records={})
    with pytest.raises(_P2InjectedCrash):
        store.bulk_ingest(finalize="deferred").__enter__()
    values = _metadata(clickhouse_p2_database)
    assert "lease" in values
    assert "ingest_state" not in values
    source_url = sqlalchemy.engine.make_url(os.environ["HTTK_TEST_CLICKHOUSE_URI"])
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
    source_url = sqlalchemy.engine.make_url(os.environ["HTTK_TEST_CLICKHOUSE_URI"])
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
    from httk.data.db.bulk import BulkIngest

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
    source_url = sqlalchemy.engine.make_url(os.environ["HTTK_TEST_CLICKHOUSE_URI"])
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
    from httk.data.db.bulk import BulkIngest

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
    from httk.data.db.bulk import BulkIngest

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
    clickhouse_p2_database: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    from httk.data.db.bulk import BulkIngest

    entered = threading.Event()
    finalizer_started = threading.Event()
    release_finalizer = threading.Event()
    dispose_started = threading.Event()
    dispose_done = threading.Event()
    errors: list[BaseException] = []

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
    release_finalizer.set()
    bulk_thread.join(timeout=20)
    dispose_thread.join(timeout=20)
    assert not bulk_thread.is_alive()
    assert not dispose_thread.is_alive()
    assert not errors
    assert dispose_done.is_set()
    assert "lease" not in _metadata(clickhouse_p2_database)


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
