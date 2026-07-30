"""SQLite integration coverage for the generic federated-store contract."""

from dataclasses import dataclass
from fractions import Fraction

import pytest

from httk.data import FederatedStore, MultipleResultsError
from httk.data.db import Database, SqlStore


@dataclass(frozen=True)
class FederatedRecord:
    """A small exact-value record shared by two independent SQLite stores."""

    id: str
    label: str
    energy: Fraction


def _store(database: Database, records: tuple[FederatedRecord, ...]) -> SqlStore:
    store = SqlStore(database)
    with store.transaction():
        for record in records:
            store.save(record)
    return store


def test_sqlite_federation_matches_a_materialized_source_major_union() -> None:
    first_records = (
        FederatedRecord("same", "kept-first", Fraction(1, 3)),
        FederatedRecord("first-only", "discarded", Fraction(7, 9)),
    )
    second_records = (
        FederatedRecord("same", "kept-second", Fraction(2, 3)),
        FederatedRecord("second-only", "kept-third", Fraction(5, 7)),
    )
    reference_union = (
        ("first", first_records[0], "same"),
        ("second", second_records[0], "same"),
        ("second", second_records[1], "second-only"),
    )

    with Database.sqlite() as first_database, Database.sqlite() as second_database:
        federation = FederatedStore(
            {
                "first": _store(first_database, first_records),
                "second": _store(second_database, second_records),
            }
        )
        searcher = federation.searcher()
        record = searcher.variable(FederatedRecord)
        searcher.add(record.label.startswith("kept") & (record.energy >= Fraction(1, 3)))
        result = searcher.results(record=record, id=record.id, origin=searcher.origin)

        materialized = tuple((row.origin, row.record, row.id) for row in result)
        assert materialized == reference_union
        assert [row.record.id for row in result] == ["same", "same", "second-only"]
        assert all(
            row.record.energy == expected[1].energy for row, expected in zip(result, reference_union, strict=True)
        )
        assert searcher.count() == len(reference_union)
        assert len(result) == len(reference_union)
        assert (result.first().origin, result.first().record, result.first().id) == reference_union[0]
        with pytest.raises(MultipleResultsError):
            result.one()

        searcher.add_offset(1)
        searcher.set_limit(2)
        paged = searcher.results(record=record, id=record.id, origin=searcher.origin)
        assert tuple((row.origin, row.record, row.id) for row in paged) == reference_union[1:]
        assert len(paged) == len(reference_union[1:])
        assert (paged.first().origin, paged.first().record, paged.first().id) == reference_union[1]
        with pytest.raises(MultipleResultsError):
            paged.one()

        searcher.add_offset(1)
        searcher.set_limit(1)
        one = searcher.results(record=record, id=record.id, origin=searcher.origin)
        assert len(one) == 1
        assert (one.one().origin, one.one().record, one.one().id) == reference_union[2]
