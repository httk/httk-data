"""Live MongoDatabase lifecycle and mode-probe coverage."""

import os

import pytest

from httk.store.mongo import MongoDatabase


@pytest.mark.usefixtures("mongo_test_database")
def test_connect_probes_replica_set_and_exposes_majority_defaults(mongo_test_database) -> None:
    """The configured test server is a transaction-capable replica set."""
    database = mongo_test_database
    assert database.supports_transactions is True
    assert database.client.write_concern.document["w"] == "majority"
    assert database.client.write_concern.document["j"] is True
    assert database.client.read_concern.level == "majority"


def test_never_forces_degraded_mode(mongo_test_database) -> None:
    """The explicit degraded pin wins even on a replica set."""
    uri = os.environ.get("HTTK_TEST_MONGODB_URI")
    if not uri:
        pytest.skip("HTTK_TEST_MONGODB_URI is not set")
    database = MongoDatabase.connect(uri, database=f"httk_test_never_{id(mongo_test_database)}", transactions="never")
    try:
        assert database.supports_transactions is False
    finally:
        database.client.drop_database(database.database.name)
        database.dispose()


def test_require_succeeds_on_the_replica_set(mongo_test_database) -> None:
    """The explicit required mode succeeds against the live server."""
    uri = os.environ.get("HTTK_TEST_MONGODB_URI")
    if not uri:
        pytest.skip("HTTK_TEST_MONGODB_URI is not set")
    database = MongoDatabase.connect(
        uri, database=f"httk_test_require_{id(mongo_test_database)}", transactions="require"
    )
    try:
        assert database.supports_transactions is True
    finally:
        database.client.drop_database(database.database.name)
        database.dispose()
