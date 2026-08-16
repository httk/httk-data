"""Bounded durable entry-family federation over SQLite and DuckDB."""

import datetime
from dataclasses import dataclass
from typing import Any, ClassVar

import pytest
import sqlalchemy
from httk.core import (
    PropertyDefinition,
    load_entry_type_definition,
)
from httk.core.register import register_entry_family, register_entry_record
from httk.core.storage import StorageInfo, StoredPropertyProjection, content_id

from httk.store.db import (
    Database,
    DuplicateEntryIdError,
    SqlSearcher,
    SqlStore,
    StoredEntryFederation,
    StoredEntrySource,
)
from httk.store.query.optimade_filters import FilterTranslationError
from httk.store.store_common import EntryStore

CALCULATIONS_DEFINITION = "https://schemas.optimade.org/defs/v1.3/entrytypes/optimade/calculations"
_RESPONSES: list[str] = []


class FederatedCalculation:
    type = "calculations"
    definition_id = CALCULATIONS_DEFINITION

    @staticmethod
    def entry_type_definition():
        return load_entry_type_definition(CALCULATIONS_DEFINITION).extended(
            {
                "_httk_label": PropertyDefinition.from_simple(
                    "_httk_label", description="A label stored for federation tests."
                ),
                "_httk_store_timestamp": PropertyDefinition.from_simple(
                    "_httk_store_timestamp", description="A store timestamp test filter.", fulltype="timestamp"
                ),
            }
        )


def _label_response(record) -> str:
    label = _label_value(record)
    _RESPONSES.append(label)
    return label


def _label_value(record: Any) -> str:
    label = record.label
    assert isinstance(label, str)
    return label


def _modified_response(record: Any) -> datetime.datetime | None:
    value = record.modified
    assert value is None or isinstance(value, datetime.datetime)
    return value


def _label_query(context, operator: str, literal: object):
    return context.compare(context.field("label"), operator, context.constant(literal))


def _modified_query(context, operator: str, literal: object):
    value = context.field("modified")
    if operator == "IS_UNKNOWN":
        return context.is_null(value)
    if operator == "IS_KNOWN":
        return context.not_(context.is_null(value))
    return context.compare(value, operator, context.constant(literal))


def _store_timestamp_query(context, operator: str, literal: object):
    return context.compare(context.field("store_timestamp"), operator, context.constant(literal))


@dataclass(frozen=True)
class FederationFirst:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="stored_federation_first")

    label: str
    modified: datetime.datetime | None

    __httk_stored_properties__: ClassVar = {
        "immutable_id": StoredPropertyProjection(
            response=_label_response,
            query=_label_query,
            sort=lambda context: context.field("label"),
        ),
        "last_modified": StoredPropertyProjection(
            response=_modified_response,
            query=_modified_query,
            sort=lambda context: context.field("modified"),
        ),
        "_httk_store_timestamp": StoredPropertyProjection(
            response=lambda _record: None,
            query=_store_timestamp_query,
            sort=lambda context: context.field("store_timestamp"),
        ),
        "_httk_label": StoredPropertyProjection(
            response=_label_value,
            query=_label_query,
            sort=lambda context: context.field("label"),
        ),
    }


@dataclass(frozen=True)
class FederationSecond:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="stored_federation_second")

    label: str
    modified: datetime.datetime | None

    __httk_stored_properties__: ClassVar = FederationFirst.__httk_stored_properties__


register_entry_family(
    name="test-stored-federation-calculations",
    family=f"{__name__}:FederatedCalculation",
    definition_id=CALCULATIONS_DEFINITION,
)
register_entry_record(
    name="test-stored-federation-first",
    family="test-stored-federation-calculations",
    record=f"{__name__}:FederationFirst",
)
register_entry_record(
    name="test-stored-federation-second",
    family="test-stored-federation-calculations",
    record=f"{__name__}:FederationSecond",
)


def _record(label: str, modified: datetime.datetime | None = None, *, second: bool = False):
    cls = FederationSecond if second else FederationFirst
    return cls(label, modified)


def test_stored_entry_source_accepts_structural_entry_store() -> None:
    class MinimalEntryStore:
        entry_layout = ()

        def searcher(self):
            raise AssertionError("the acceptance test must not query the fake store")

        def fetch(self, cls: type, sid: int) -> object:
            raise AssertionError("the acceptance test must not fetch from the fake store")

        def fetch_many(self, cls: type, sids: object) -> list[object]:
            raise AssertionError("the acceptance test must not fetch from the fake store")

        def stored_property_plan(self, family: type) -> object:
            raise AssertionError("the acceptance test must not acquire a plan from the fake store")

    class MissingEntryStore:
        entry_layout = ()

        def searcher(self):
            raise AssertionError("the acceptance test must not query the fake store")

    store = MinimalEntryStore()
    source = StoredEntrySource(store, FederatedCalculation, "fake")

    assert isinstance(store, EntryStore)
    assert source.store is store
    with pytest.raises(TypeError, match="EntryStore"):
        StoredEntrySource(MissingEntryStore(), FederatedCalculation, "missing")


def test_snapshot_cutoff_ns_uses_the_coarsest_capable_source_and_skips_disabled_sources() -> None:
    resolutions = (1, 1_000, 1_000_000_000)
    databases = [Database.sqlite() for _ in resolutions]
    try:
        sources = []
        for index, (database, resolution) in enumerate(zip(databases, resolutions, strict=True)):
            store = SqlStore(
                database,
                entry_records={FederatedCalculation: (FederationFirst, FederationSecond)},
                store_timestamp_resolution=resolution,
            )
            store.save(_record(f"source-{index}"))
            sources.append(StoredEntrySource(store, FederatedCalculation, f"source-{index}", f"{index}:"))
        federation = StoredEntryFederation(tuple(sources))
        now_ns = 3_456_789_123
        assert federation.snapshot_cutoff_ns(now_ns) == 2_999_999_999
    finally:
        for database in databases:
            database.dispose()

    with Database.sqlite() as database:
        store = SqlStore(
            database,
            entry_records={FederatedCalculation: (FederationFirst, FederationSecond)},
            store_timestamps=False,
        )
        federation = StoredEntryFederation((StoredEntrySource(store, FederatedCalculation, "disabled", "disabled:"),))
        assert federation.snapshot_cutoff_ns(3_456_789_123) is None


@pytest.fixture(params=("sqlite", "duckdb"))
def databases(request):
    if request.param == "duckdb":
        pytest.importorskip("duckdb_engine")
        first = Database.duckdb()
        second = Database.duckdb()
    else:
        first = Database.sqlite()
        second = Database.sqlite()
    with first, second:
        yield first, second


def _federation(
    databases,
    first_records: tuple[FederationFirst | FederationSecond, ...],
    second_records: tuple[FederationFirst | FederationSecond, ...],
    *,
    first_prefix: str = "alpha:",
    second_prefix: str = "beta:",
    first_clock: int | None = None,
    second_clock: int | None = None,
    first_resolution: int = 1000,
    second_resolution: int = 1000,
) -> StoredEntryFederation:
    first_database, second_database = databases
    first_store = SqlStore(
        first_database,
        entry_records={FederatedCalculation: (FederationFirst, FederationSecond)},
        store_timestamp_resolution=first_resolution,
    )
    second_store = SqlStore(
        second_database,
        entry_records={FederatedCalculation: (FederationFirst, FederationSecond)},
        store_timestamp_resolution=second_resolution,
    )
    if first_clock is not None:
        first_store._clock = lambda: first_clock
    if second_clock is not None:
        second_store._clock = lambda: second_clock
    for record in first_records:
        first_store.save(record)
    for record in second_records:
        second_store.save(record)
    return StoredEntryFederation(
        (
            StoredEntrySource(first_store, FederatedCalculation, "alpha", first_prefix),
            StoredEntrySource(second_store, FederatedCalculation, "beta", second_prefix),
        )
    )


def test_stored_entry_federation_filters_store_timestamp(databases):
    federation = _federation(
        databases,
        (_record("first"),),
        (_record("second"),),
        first_clock=1_000_000,
        second_clock=3_000_000,
    )

    page = federation.query('_httk_store_timestamp <= "1970-01-01T00:00:00.002500Z"', limit=10)

    assert [row["immutable_id"] for row in page.rows] == ["first"]


def test_stored_entry_federation_sorts_store_timestamp_in_nanoseconds(databases):
    federation = _federation(
        databases,
        (_record("first"),),
        (_record("second"),),
        first_clock=2_000_000,
        second_clock=1_000_000,
        first_resolution=1,
        second_resolution=1000,
    )

    page = federation.query(sort=(("_httk_store_timestamp", False),), limit=10)

    assert [(row["immutable_id"], row["_httk_store_timestamp"]) for row in page.rows] == [
        ("second", 1_000_000),
        ("first", 2_000_000),
    ]


def test_unsorted_page_preserves_source_backing_native_order_and_hydrates_only_visible_rows(databases, monkeypatch):
    federation = _federation(
        databases,
        (_record("alpha-first"), _record("alpha-second", second=True)),
        (_record("beta-first"),),
    )
    add_sort = SqlSearcher.add_sort
    calls: list[object] = []

    def tracked(self, expression, *, descending: bool = False) -> None:
        calls.append(expression)
        add_sort(self, expression, descending=descending)

    monkeypatch.setattr(SqlSearcher, "add_sort", tracked)
    _RESPONSES.clear()
    page = federation.query(offset=1, limit=1)

    assert page.total_count == 3
    assert page.more_data_available
    assert [row["immutable_id"] for row in page.rows] == ["alpha-second"]
    assert _RESPONSES == ["alpha-second"]
    assert calls == []
    for source in federation._sources:
        for stream in source.plan.candidate_searchers():
            searcher = stream.searcher
            columns = [output.element for output in searcher._outputs]
            grouping = [variable._alias.c["sid"] for variable in searcher._variables]
            grouping.extend(
                output.element for output in searcher._outputs if output.target is None and not output.from_child
            )
            statement = searcher._base_select(columns, grouping)
            for column, descending in searcher._sorts:
                statement = statement.order_by(column._element.desc() if descending else column._element.asc())
            rendered = str(statement.compile(dialect=source.plan.store._database.engine.dialect)).upper()
            assert "ORDER BY" not in rendered


def test_unique_prefix_page_has_no_duplicate_probe_queries() -> None:
    managers = (Database.sqlite(), Database.sqlite(), Database.sqlite())
    with managers[0] as first_database, managers[1] as second_database, managers[2] as third_database:
        records = (_record("first"), _record("second"), _record("third"))
        stores = tuple(
            SqlStore(database, entry_records={FederatedCalculation: FederationFirst})
            for database in (first_database, second_database, third_database)
        )
        for store, record in zip(stores, records, strict=True):
            store.save(record)
        federation = StoredEntryFederation(
            tuple(
                StoredEntrySource(store, FederatedCalculation, name, f"{name}:")
                for store, name in zip(stores, ("alpha", "beta", "gamma"), strict=True)
            )
        )
        statements: list[str] = []

        def count_select(_conn, _cursor, statement, _parameters, _context, _executemany) -> None:
            if statement.lstrip().upper().startswith("SELECT"):
                statements.append(statement)

        for store in stores:
            sqlalchemy.event.listen(store._database.engine, "before_cursor_execute", count_select)
        try:
            page = federation.query(limit=3)
        finally:
            for store in stores:
                sqlalchemy.event.remove(store._database.engine, "before_cursor_execute", count_select)

    assert len(page.rows) == 3
    assert len(statements) == 6  # Three count queries plus three candidate queries; no duplicate probes.


def test_single_source_page_skips_probes_but_audit_detects_corrupt_cross_backing_ids() -> None:
    with Database.sqlite() as database:
        store = SqlStore(database, entry_records={FederatedCalculation: (FederationFirst, FederationSecond)})
        first = _record("first")
        second = _record("second", second=True)
        store.save(first)
        second_sid = store.save(second)
        key = content_id(first)
        with database.engine.begin() as connection:
            connection.execute(
                sqlalchemy.text("UPDATE stored_federation_second SET content_id = :content_id WHERE sid = :sid"),
                {"content_id": key, "sid": second_sid},
            )
        federation = StoredEntryFederation((StoredEntrySource(store, FederatedCalculation, "only", "same:"),))
        statements: list[str] = []

        def count_select(_conn, _cursor, statement, _parameters, _context, _executemany) -> None:
            if statement.lstrip().upper().startswith("SELECT"):
                statements.append(statement)

        sqlalchemy.event.listen(database.engine, "before_cursor_execute", count_select)
        try:
            page = federation.query(limit=1)
        finally:
            sqlalchemy.event.remove(database.engine, "before_cursor_execute", count_select)

        # Page serving intentionally returns this duplicated id without raising;
        # audit_duplicate_ids() detects same-source out-of-band corruption.
        assert len(page.rows) == 1
        assert page.total_count == 2
        assert [row["id"] for row in page.rows] == [f"same:{key}"]
        assert len(statements) == 4  # Two counts plus two page candidates; no duplicate probes.
        with pytest.raises(DuplicateEntryIdError) as caught:
            federation.audit_duplicate_ids()

    assert caught.value.public_id == f"same:{key}"
    assert {origin.backing for origin in caught.value.origins} == {
        "test-stored-federation-first",
        "test-stored-federation-second",
    }


def test_limit_zero_uses_an_id_only_sentinel_without_duplicate_probe_or_hydration(databases):
    duplicate = _record("same")
    with pytest.warns(RuntimeWarning, match="empty public_id_prefix"):
        federation = _federation(databases, (duplicate,), (duplicate,), first_prefix="", second_prefix="")
    _RESPONSES.clear()

    page = federation.query(limit=0)

    assert page.rows == ()
    assert page.total_count == 2
    assert page.more_data_available
    assert _RESPONSES == []


def test_visible_and_explicit_public_id_collision_detection_is_lazy(databases):
    duplicate = _record("aaa")
    with pytest.warns(RuntimeWarning, match="empty public_id_prefix"):
        federation = _federation(
            databases,
            (duplicate, _record("visible")),
            (duplicate,),
            first_prefix="",
            second_prefix="",
        )

    # Both duplicate candidates sort before this visible candidate.  They are
    # wholly outside the page window, so ordinary paging does not audit them.
    page = federation.query(sort=(("immutable_id", False),), offset=2, limit=1)
    assert [row["immutable_id"] for row in page.rows] == ["visible"]

    public_id = content_id(duplicate)
    with pytest.raises(DuplicateEntryIdError) as caught:
        federation.fetch(public_id)
    assert caught.value.public_id == public_id
    assert {origin.source for origin in caught.value.origins} == {"alpha", "beta"}
    assert "audit_duplicate_ids" in str(caught.value)


@pytest.mark.parametrize("descending", (False, True))
def test_sorted_heap_merge_uses_explicit_nulls_last_and_public_id_ties(databases, descending):
    early = datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC)
    late = datetime.datetime(2026, 2, 1, tzinfo=datetime.UTC)
    federation = _federation(
        databases,
        (_record("alpha-null"), _record("alpha-early", early, second=True)),
        (_record("beta-late", late), _record("beta-null")),
    )

    page = federation.query(sort=(("last_modified", descending),), limit=10)

    labels = [row["immutable_id"] for row in page.rows]
    if descending:
        assert labels == ["beta-late", "alpha-early", "alpha-null", "beta-null"]
    else:
        assert labels == ["alpha-early", "beta-late", "alpha-null", "beta-null"]
    assert page.more_data_available is False


def test_public_prefix_participates_in_id_filter_and_sort(databases):
    same = _record("same")
    federation = _federation(databases, (same,), (same,))
    all_rows = federation.query(sort=(("id", True),), limit=10).rows
    assert [row["id"][:5] for row in all_rows] == ["beta:", "alpha"]

    target = all_rows[1]["id"]
    page = federation.query(f'id = "{target}"', sort=(("id", False),), limit=10)
    assert [row["id"] for row in page.rows] == [target]


def test_sorted_pages_are_gap_free_and_equal_sort_values_use_public_id_ties(databases):
    stamped = datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC)
    federation = _federation(
        databases,
        (_record("alpha-one", stamped), _record("alpha-two", stamped, second=True)),
        (_record("beta-one", stamped), _record("beta-two", stamped, second=True)),
        first_prefix="z:",
        second_prefix="a:",
    )

    complete = federation.query(sort=(("last_modified", True),), limit=10)
    paged = tuple(
        row
        for offset in range(0, complete.total_count, 2)
        for row in federation.query(sort=(("last_modified", True),), offset=offset, limit=2).rows
    )

    assert tuple(row["id"] for row in paged) == tuple(row["id"] for row in complete.rows)
    assert [row["id"][:2] for row in complete.rows] == ["a:", "a:", "z:", "z:"]


def test_registered_definition_prefixes_reject_unknown_properties_but_filter_declared_ones(databases):
    federation = _federation(databases, (_record("declared"),), ())

    page = federation.query('_httk_label = "declared"', limit=10)
    assert [row["_httk_label"] for row in page.rows] == ["declared"]
    with pytest.raises(FilterTranslationError) as caught:
        federation.query("_httk_not_declared = 2", limit=10)
    assert caught.value.category == "unrecognized-property"


def test_constructor_rejects_mixed_entry_families_before_querying(databases):
    database, _unused = databases
    store = SqlStore(database, entry_records={FederatedCalculation: (FederationFirst,)})
    incompatible_family = type("IncompatibleFederationFamily", (), {})

    with pytest.raises(ValueError, match="one exact entry_family"):
        StoredEntryFederation(
            (
                StoredEntrySource(store, FederatedCalculation, "calculations"),
                StoredEntrySource(store, incompatible_family, "other"),
            )
        )


def test_audit_scans_bounded_id_only_batches_without_hydration(databases, monkeypatch):
    duplicate = _record("same")
    with pytest.warns(RuntimeWarning, match="empty public_id_prefix"):
        federation = _federation(
            databases,
            (duplicate, _record("a"), _record("b")),
            (duplicate, _record("c"), _record("d")),
            first_prefix="",
            second_prefix="",
        )
    _RESPONSES.clear()
    limits: list[int] = []
    offsets: list[int] = []
    original = SqlSearcher.set_limit
    add_offset = SqlSearcher.add_offset

    def tracked(self, value: int) -> None:
        limits.append(value)
        original(self, value)

    def tracked_offset(self, value: int) -> None:
        offsets.append(value)
        add_offset(self, value)

    monkeypatch.setattr(SqlSearcher, "set_limit", tracked)
    monkeypatch.setattr(SqlSearcher, "add_offset", tracked_offset)
    with pytest.raises(DuplicateEntryIdError):
        federation.audit_duplicate_ids(batch_size=1)
    assert limits and set(limits) == {1}
    assert offsets == []
    assert _RESPONSES == []


def test_page_render_batches_into_one_fetch_per_source(databases, monkeypatch):
    records = tuple(_record(f"row-{index:03d}") for index in range(24))
    federation = _federation(databases, records, ())
    fetch_many_sizes: list[int] = []
    per_row_fetches: list[int] = []
    original_many = SqlStore.fetch_many
    original_fetch = SqlStore.fetch

    def tracked_many(self, cls, sids, *, eager=False):
        materialized = list(sids)
        fetch_many_sizes.append(len(materialized))
        return original_many(self, cls, materialized, eager=eager)

    def tracked_fetch(self, cls, sid, *, eager=False):
        per_row_fetches.append(sid)
        return original_fetch(self, cls, sid, eager=eager)

    monkeypatch.setattr(SqlStore, "fetch_many", tracked_many)
    monkeypatch.setattr(SqlStore, "fetch", tracked_fetch)
    page = federation.query(limit=50)
    assert [row["immutable_id"] for row in page.rows] == [f"row-{index:03d}" for index in range(24)]
    # One batched fetch covers the whole page's single source; never one fetch per row.
    assert fetch_many_sizes == [24]
    assert per_row_fetches == []


def test_batched_page_matches_single_id_render(databases):
    federation = _federation(
        databases,
        (_record("a"), _record("c")),
        (_record("b"), _record("d")),
    )
    page = federation.query(sort=(("immutable_id", False),), limit=10)
    assert [row["immutable_id"] for row in page.rows] == ["a", "b", "c", "d"]
    assert page.total_count == 4
    assert page.more_data_available is False
    # Each batched row equals the row the single-id render path produces.
    for row in page.rows:
        assert federation.fetch(row["id"]) == dict(row)
