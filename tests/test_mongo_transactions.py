"""Live MongoDB transaction, cache, and retry coverage."""

import os
import threading
from dataclasses import dataclass

import pytest
from httk.core.register import register_entry_family, register_entry_record
from pymongo import MongoClient, monitoring

from httk.data.db.schema import resolve_schema
from httk.data.mongo import MongoDatabase, MongoStore, TransactionsUnavailableError
from httk.data.mongo.mapping import collection_name_for, entry_dispatch_table_name


@dataclass(frozen=True)
class TxChild:
    value: str


class TxFamily:
    """Test-only multi-record entry family."""


@dataclass(frozen=True)
class TxEntry:
    value: str
    children: list[TxChild]


@dataclass(frozen=True)
class TxOther:
    value: str


register_entry_family(name="test-mongo-txn-family", family=f"{__name__}:TxFamily")
register_entry_record(name="test-mongo-txn-entry", family="test-mongo-txn-family", record=f"{__name__}:TxEntry")
register_entry_record(name="test-mongo-txn-other", family="test-mongo-txn-family", record=f"{__name__}:TxOther")


def _store(database) -> MongoStore:
    return MongoStore(database, entry_records={TxFamily: (TxEntry, TxOther)})


class _CommandCounter(monitoring.CommandListener):
    """Count commands sent by the dedicated retry-test client."""

    def __init__(self) -> None:
        self.commands: list[tuple[str, str | None]] = []

    def started(self, event) -> None:
        collection = event.command.get(event.command_name.lower())
        self.commands.append((event.command_name, collection if isinstance(collection, str) else None))

    def succeeded(self, event) -> None:
        pass

    def failed(self, event) -> None:
        pass

    def count(self, name: str, collection: str | None = None) -> int:
        """Return commands matching an optional collection name."""
        return sum(command == name and (collection is None or target == collection) for command, target in self.commands)


def _failpoint(client, command: dict, *, times: int) -> None:
    try:
        client.admin.command({"configureFailPoint": "failCommand", "mode": {"times": times}, "data": command})
    except Exception as error:
        if getattr(error, "code", None) == 59:
            pytest.skip("the configured MongoDB server does not expose configureFailPoint")
        raise


def _disable_failpoint(client) -> None:
    try:
        client.admin.command({"configureFailPoint": "failCommand", "mode": "off"})
    except Exception as error:
        if getattr(error, "code", None) != 59:
            raise


def test_transaction_commit_visibility_rollback_and_flat_nesting(mongo_test_database) -> None:
    store = _store(mongo_test_database)
    other_database = MongoDatabase.connect(
        os.environ["HTTK_TEST_MONGODB_URI"], database=mongo_test_database.database.name
    )
    other = _store(other_database)
    try:
        record = TxEntry("visible", [TxChild("child")])
        with store.transaction():
            sid = store.save(record)
            assert store.fetch(TxEntry, sid) == record
            with store.transaction():
                assert store.sid_of(record) == sid
            with pytest.raises(KeyError):
                other.fetch(TxEntry, sid)
        assert other.fetch(TxEntry, sid) == record
        with pytest.raises(ValueError), store.transaction():
            store.save(TxEntry("rolled-back", [TxChild("gone")]))
            raise ValueError("abort")
        database = mongo_test_database.database
        assert database[collection_name_for(resolve_schema(TxEntry))].count_documents({}) == 1
        assert database[collection_name_for(resolve_schema(TxChild))].count_documents({}) == 1
        dispatch = entry_dispatch_table_name(store.entry_layout[0].name)
        assert database[dispatch].count_documents({}) == 1
    finally:
        other_database.dispose()


def test_degraded_transaction_is_unavailable_and_rollback_does_not_cache(mongo_test_database) -> None:
    database = MongoDatabase.connect(
        os.environ["HTTK_TEST_MONGODB_URI"], database=mongo_test_database.database.name, transactions="never"
    )
    try:
        store = _store(database)
        with pytest.raises(TransactionsUnavailableError, match="replica-set"), store.transaction():
            pass
    finally:
        database.dispose()

    store = _store(mongo_test_database)
    record = TxEntry("cache", [])
    with pytest.raises(RuntimeError), store.transaction():
        sid = store.save(record)
        assert store.fetch(TxEntry, sid) == record
        raise RuntimeError("abort")
    assert store.sid_of(record) is None


def test_save_race_against_an_open_transaction_converges(mongo_test_database) -> None:
    first = _store(mongo_test_database)
    second_database = MongoDatabase.connect(
        os.environ["HTTK_TEST_MONGODB_URI"], database=mongo_test_database.database.name
    )
    second = _store(second_database)
    record = TxEntry("race", [])
    started = threading.Event()
    finished = threading.Event()
    result: list[int] = []
    errors: list[BaseException] = []

    def concurrent_save() -> None:
        started.set()
        try:
            result.append(second.save(record))
        except BaseException as error:  # pragma: no cover - assertion below reports it.
            errors.append(error)
        finally:
            finished.set()

    try:
        with first.transaction():
            winner = first.save(record)
            thread = threading.Thread(target=concurrent_save)
            thread.start()
            assert started.wait(1)
        assert finished.wait(5)
        thread.join(timeout=1)
        assert not errors
        assert result == [winner]
        assert mongo_test_database.database[collection_name_for(resolve_schema(TxEntry))].count_documents({}) == 1
    finally:
        second_database.dispose()


def test_save_retries_transient_and_unknown_commit_result(mongo_test_database) -> None:
    counter = _CommandCounter()
    client = MongoClient(os.environ["HTTK_TEST_MONGODB_URI"], event_listeners=[counter])
    database = MongoDatabase(client, mongo_test_database.database.name)
    store = _store(database)
    database_name = mongo_test_database.database.name
    entry_namespace = f"{database_name}.{collection_name_for(resolve_schema(TxEntry))}"
    try:
        try:
            _failpoint(
                mongo_test_database.client,
                {"failCommands": ["insert"], "namespace": entry_namespace, "errorCode": 251,
                 "errorLabels": ["TransientTransactionError"]},
                times=1,
            )
            assert store.save(TxEntry("transient", [])) > 0
            assert counter.count("insert", collection_name_for(resolve_schema(TxEntry))) >= 2
        finally:
            _disable_failpoint(mongo_test_database.client)
        try:
            _failpoint(
                mongo_test_database.client,
                {"failCommands": ["commitTransaction"], "errorCode": 91,
                 "errorLabels": ["UnknownTransactionCommitResult"]},
                times=1,
            )
            assert store.save(TxEntry("commit", [])) > 0
            assert counter.count("commitTransaction") >= 2
        finally:
            _disable_failpoint(mongo_test_database.client)
    finally:
        database.dispose()
    assert mongo_test_database.database[collection_name_for(resolve_schema(TxEntry))].count_documents({}) == 2
