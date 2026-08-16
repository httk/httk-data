"""Federation pass-through coverage for the reserved store timestamp field."""

from dataclasses import dataclass

from httk.store import FederatedStore
from httk.store.db import Database, SqlStore


@dataclass(frozen=True)
class FederatedTimestampRecord:
    value: int


def _timestamped_store(database: Database, values: tuple[tuple[int, int], ...]) -> SqlStore:
    store = SqlStore(database, entry_records={})
    for value, clock_ns in values:
        store._clock = lambda clock_ns=clock_ns: clock_ns
        store.save(FederatedTimestampRecord(value))
    return store


def test_federated_store_timestamp_predicate_replays_to_each_source() -> None:
    with Database.sqlite() as first_database, Database.sqlite() as second_database:
        federation = FederatedStore(
            {
                "first": _timestamped_store(first_database, ((1, 1_000_000), (2, 3_000_000))),
                "second": _timestamped_store(second_database, ((3, 2_000_000), (4, 4_000_000))),
            }
        )

        for cutoff in (2_500_000, "1970-01-01T00:00:00.002500Z"):
            searcher = federation.searcher()
            variable = searcher.variable(FederatedTimestampRecord)
            searcher.add(variable.ts_start <= cutoff)
            result = searcher.results(record=variable, origin=searcher.origin)
            assert [(row.origin, row.record.value) for row in result] == [("first", 1), ("second", 3)]
