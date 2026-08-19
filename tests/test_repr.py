"""Repr policy: ContinuationToken round-trips; service objects render their class name."""

from types import SimpleNamespace

from httk.store.backend.mongo.store import MongoStore
from httk.store.backend.sql import Backend, SqlStore
from httk.store.entry_providers import (
    DataRecordEntryProvider,
    ReferenceEntryProvider,
    RunEntryProvider,
)
from httk.store.federated_store import FederatedStore
from httk.store.query.protocols import ContinuationToken


def test_continuation_token_repr_roundtrips() -> None:
    token = ContinuationToken("abc.def==")
    assert repr(token) == "ContinuationToken('abc.def==')"
    assert eval(repr(token)) == token


def _assert_informative(value: object, class_name: str) -> None:
    text = repr(value)
    assert text.startswith(f"{class_name}("), text
    assert " object at 0x" not in text, text


def test_sql_store_repr() -> None:
    with Backend.sqlite() as database:
        _assert_informative(SqlStore(database, entry_records={}), "SqlStore")


def test_mongo_store_repr() -> None:
    # MongoStore construction needs a live Mongo; the repr logic itself is what
    # matters here, so exercise it against a stand-in database.
    stub = SimpleNamespace(_database=SimpleNamespace())
    text = MongoStore.__repr__(stub)  # type: ignore[arg-type]
    assert text.startswith("MongoStore("), text
    assert " object at 0x" not in text, text


def test_federated_store_repr() -> None:
    store = FederatedStore({"first": object(), "second": object()})  # type: ignore[dict-item]
    text = repr(store)
    assert text.startswith("FederatedStore("), text
    assert "first" in text and "second" in text


def test_entry_provider_reprs() -> None:
    _assert_informative(ReferenceEntryProvider({}), "ReferenceEntryProvider")
    _assert_informative(RunEntryProvider({}), "RunEntryProvider")
    _assert_informative(DataRecordEntryProvider({}), "DataRecordEntryProvider")
