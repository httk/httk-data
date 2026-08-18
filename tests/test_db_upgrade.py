"""Additive schema-fingerprint upgrade coverage for SqlStore (and the pure classifier)."""

import json
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Annotated, ClassVar

import pytest
import sqlalchemy
from httk.core.storage import IdentitySkip, Indexed, StorageInfo, content_id, stored_property

from httk.store import EntryFamilyDeclaration, EntryRecordDeclaration
from httk.store.backend.sql import Backend, SqlStore, StorageLayoutUpgradeRequiredError
from httk.store.storage_layout import (
    AdditiveUpgradePlan,
    classify_schema_upgrade,
    normalize_entry_families,
    schema_fingerprint_json,
)

# An added field only classifies additive when it is excluded from content
# identity, so pre-existing rows keep their content_id across the upgrade.
_SKIP = IdentitySkip()


class UpgradeFamily:
    """Application-owned family for the scalar additive-upgrade tests."""


class UpgradeRefFamily:
    """Application-owned family whose record gains a reference on upgrade."""


@dataclass(frozen=True)
class RecOld:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="upgrade_rec", identity_name="upgrade-rec")

    value: str


@dataclass(frozen=True)
class RecNew:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="upgrade_rec", identity_name="upgrade-rec")

    value: str
    note: Annotated[str | None, _SKIP] = None


@dataclass(frozen=True)
class RecNewNonSkip:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="upgrade_rec", identity_name="upgrade-rec")

    value: str
    note: str | None = None


@dataclass(frozen=True)
class RecNewIndexed:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="upgrade_rec", identity_name="upgrade-rec")

    value: str
    code: Annotated[str | None, Indexed(), _SKIP] = None


@dataclass(frozen=True)
class RecNewRequired:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="upgrade_rec", identity_name="upgrade-rec")

    value: str
    tag: Annotated[str, _SKIP] = "x"


@dataclass(frozen=True)
class RecNewChild:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="upgrade_rec", identity_name="upgrade-rec")

    value: str
    tags: Annotated[tuple[str, ...], _SKIP] = ()


@dataclass(frozen=True)
class RecNewDerived:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="upgrade_rec", identity_name="upgrade-rec")

    value: str

    @stored_property
    def upper(self) -> str:
        return self.value.upper()


@dataclass(frozen=True)
class RecRetyped:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="upgrade_rec", identity_name="upgrade-rec")

    value: int


@dataclass(frozen=True)
class RecOtherIdentity:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="upgrade_rec", identity_name="upgrade-rec-2")

    value: str


@dataclass(frozen=True)
class UpgradeRefChild:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(
        storage_name="upgrade_ref_child", identity_name="upgrade-ref-child"
    )

    tag: str


@dataclass(frozen=True)
class RecRefOld:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(
        storage_name="upgrade_ref_parent", identity_name="upgrade-ref-parent"
    )

    value: str


@dataclass(frozen=True)
class RecRefNew:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(
        storage_name="upgrade_ref_parent", identity_name="upgrade-ref-parent"
    )

    value: str
    child: Annotated[UpgradeRefChild | None, _SKIP] = None


def _decl(family: type, name: str, record: type) -> EntryFamilyDeclaration:
    return EntryFamilyDeclaration(
        name=name,
        family=family,
        records=(EntryRecordDeclaration(name=f"{name}-record", record=record),),
    )


def _fingerprint(family: type, name: str, record: type) -> str:
    return schema_fingerprint_json(normalize_entry_families((_decl(family, name, record),)))


# --------------------------------------------------------------------------- classifier unit tests


def test_classify_added_identityskip_field_is_additive() -> None:
    old = _fingerprint(UpgradeFamily, "test-upgrade", RecOld)
    new = _fingerprint(UpgradeFamily, "test-upgrade", RecNew)
    plan = classify_schema_upgrade(old, new)
    assert isinstance(plan, AdditiveUpgradePlan)
    assert set(plan.added_columns) == {"upgrade_rec"}
    assert [column.name for column in plan.added_columns["upgrade_rec"]] == ["note"]
    assert all(column.nullable for column in plan.added_columns["upgrade_rec"])


def test_classify_no_change_is_empty_additive_plan() -> None:
    fingerprint = _fingerprint(UpgradeFamily, "test-upgrade", RecOld)
    plan = classify_schema_upgrade(fingerprint, fingerprint)
    assert isinstance(plan, AdditiveUpgradePlan)
    assert plan.added_columns == {}


def test_classify_removed_field_is_rejected() -> None:
    old = _fingerprint(UpgradeFamily, "test-upgrade", RecNew)
    new = _fingerprint(UpgradeFamily, "test-upgrade", RecOld)
    reason = classify_schema_upgrade(old, new)
    assert isinstance(reason, str)
    assert "dropped field 'note'" in reason


def test_classify_non_nullable_added_field_is_rejected() -> None:
    old = _fingerprint(UpgradeFamily, "test-upgrade", RecOld)
    new = _fingerprint(UpgradeFamily, "test-upgrade", RecNewRequired)
    reason = classify_schema_upgrade(old, new)
    assert isinstance(reason, str)
    assert "non-nullable column 'tag'" in reason


def test_classify_identity_participating_added_field_is_rejected() -> None:
    old = _fingerprint(UpgradeFamily, "test-upgrade", RecOld)
    new = _fingerprint(UpgradeFamily, "test-upgrade", RecNewNonSkip)
    reason = classify_schema_upgrade(old, new)
    assert isinstance(reason, str)
    assert "participates in content identity" in reason
    assert "IdentitySkip" in reason


def test_classify_derived_added_field_is_rejected() -> None:
    old = _fingerprint(UpgradeFamily, "test-upgrade", RecOld)
    new = _fingerprint(UpgradeFamily, "test-upgrade", RecNewDerived)
    reason = classify_schema_upgrade(old, new)
    assert isinstance(reason, str)
    assert "derived field 'upper'" in reason


def test_classify_child_added_field_is_rejected() -> None:
    old = _fingerprint(UpgradeFamily, "test-upgrade", RecOld)
    new = _fingerprint(UpgradeFamily, "test-upgrade", RecNewChild)
    reason = classify_schema_upgrade(old, new)
    assert isinstance(reason, str)
    assert "child field 'tags'" in reason


def test_classify_retyped_field_is_rejected() -> None:
    old = _fingerprint(UpgradeFamily, "test-upgrade", RecOld)
    new = _fingerprint(UpgradeFamily, "test-upgrade", RecRetyped)
    reason = classify_schema_upgrade(old, new)
    assert isinstance(reason, str)
    assert "changed field 'value'" in reason


def test_classify_changed_table_attribute_is_rejected() -> None:
    old = _fingerprint(UpgradeFamily, "test-upgrade", RecOld)
    new = _fingerprint(UpgradeFamily, "test-upgrade", RecOtherIdentity)
    reason = classify_schema_upgrade(old, new)
    assert isinstance(reason, str)
    assert "changed identity_name" in reason


def test_classify_unrecognized_table_key_is_rejected() -> None:
    old = _fingerprint(UpgradeFamily, "test-upgrade", RecOld)
    document = json.loads(old)
    document["tables"]["upgrade_rec"]["future_key"] = 1
    reason = classify_schema_upgrade(old, json.dumps(document))
    assert isinstance(reason, str)
    assert "unrecognized fingerprint key" in reason


def test_classify_corrupt_stored_is_rejected() -> None:
    reason = classify_schema_upgrade("not a fingerprint", _fingerprint(UpgradeFamily, "test-upgrade", RecOld))
    assert reason == "stored schema fingerprint is not parseable"


def test_classify_new_referenced_table_is_additive() -> None:
    old = _fingerprint(UpgradeRefFamily, "test-upgrade-ref", RecRefOld)
    new = _fingerprint(UpgradeRefFamily, "test-upgrade-ref", RecRefNew)
    plan = classify_schema_upgrade(old, new)
    assert isinstance(plan, AdditiveUpgradePlan)
    # Only the parent gains a column; the new child table is created whole.
    assert set(plan.added_columns) == {"upgrade_ref_parent"}


# --------------------------------------------------------------------------- end-to-end SqlStore tests


@pytest.fixture(params=["sqlite", "duckdb"])
def sql_database(request: pytest.FixtureRequest) -> Iterator[Backend]:
    if request.param == "duckdb":
        pytest.importorskip("duckdb_engine")
        with Backend.duckdb() as database:
            yield database
        return
    with Backend.sqlite() as database:
        yield database


def test_reopen_additive_without_upgrade_raises_with_hint(sql_database: Backend) -> None:
    SqlStore(sql_database, entry_families=(_decl(UpgradeFamily, "test-upgrade", RecOld),))
    with pytest.raises(StorageLayoutUpgradeRequiredError) as error:
        SqlStore(sql_database, entry_families=(_decl(UpgradeFamily, "test-upgrade", RecNew),))
    assert "upgrade=True" in str(error.value)
    assert error.value.hint is not None
    assert set(error.value.diff["schema"]) == {"upgrade_rec"}


def test_reopen_additive_with_upgrade_applies_and_restamps(sql_database: Backend) -> None:
    old_decl = (_decl(UpgradeFamily, "test-upgrade", RecOld),)
    new_decl = (_decl(UpgradeFamily, "test-upgrade", RecNew),)
    store = SqlStore(sql_database, entry_families=old_decl)
    sid = store.save(RecOld("kept"))

    upgraded = SqlStore(sql_database, entry_families=new_decl, upgrade=True)
    # The old row reconstructs with the new field defaulted to None.
    assert upgraded.fetch(RecNew, sid) == RecNew("kept", None)
    # A new row exercising the added column round-trips.
    new_sid = upgraded.save(RecNew("fresh", "annotated"))
    assert upgraded.fetch(RecNew, new_sid) == RecNew("fresh", "annotated")

    # A third plain reopen trusts the re-stamped fingerprint (no upgrade needed).
    reopened = SqlStore(sql_database, entry_families=new_decl)
    assert reopened.fetch(RecNew, sid) == RecNew("kept", None)


def test_upgrade_preserves_content_id_and_dedup(sql_database: Backend) -> None:
    old_decl = (_decl(UpgradeFamily, "test-upgrade", RecOld),)
    new_decl = (_decl(UpgradeFamily, "test-upgrade", RecNew),)
    store = SqlStore(sql_database, entry_families=old_decl)
    sid = store.save(RecOld("dup"))

    # The IdentitySkip'd field is excluded from identity, so a logically
    # identical record keeps the exact content_id it had before the upgrade.
    assert content_id(RecNew("dup", None)) == content_id(RecOld("dup"))

    upgraded = SqlStore(sql_database, entry_families=new_decl, upgrade=True)
    # Re-saving the old value deduplicates onto the pre-existing row.
    assert upgraded.save(RecNew("dup", None)) == sid


def test_reopen_new_referenced_table_reads_preexisting_rows() -> None:
    with Backend.sqlite() as database:
        store = SqlStore(database, entry_families=(_decl(UpgradeRefFamily, "test-upgrade-ref", RecRefOld),))
        old_sid = store.save(RecRefOld("base"))
        upgraded = SqlStore(
            database, entry_families=(_decl(UpgradeRefFamily, "test-upgrade-ref", RecRefNew),), upgrade=True
        )
        # The new referenced table is created by the upgrade, so the pre-existing
        # row no longer reads as absent; its new reference defaults to None.
        assert upgraded.fetch(RecRefNew, old_sid) == RecRefNew("base", None)
        searcher = upgraded.searcher()
        variable = searcher.variable(RecRefNew)
        searcher.output(variable, "record")
        assert searcher.count() == 1
        # A fresh row that touches the new child table still writes and reads.
        sid = upgraded.save(RecRefNew("with-child", UpgradeRefChild("t")))
        assert upgraded.fetch(RecRefNew, sid) == RecRefNew("with-child", UpgradeRefChild("t"))


def test_half_applied_upgrade_heals_on_retry() -> None:
    with Backend.sqlite() as database:
        old_decl = (_decl(UpgradeFamily, "test-upgrade", RecOld),)
        new_decl = (_decl(UpgradeFamily, "test-upgrade", RecNew),)
        store = SqlStore(database, entry_families=old_decl)
        sid = store.save(RecOld("kept"))
        # Simulate a crash after the ALTER but before the re-stamp (SQLite DDL
        # escapes the transaction): the column is physically present while the
        # stored fingerprint still says it is not.
        with database.engine.begin() as connection:
            connection.execute(sqlalchemy.text("ALTER TABLE upgrade_rec ADD COLUMN note TEXT"))
        # Retry: the upgrade skips the already-present column and completes.
        upgraded = SqlStore(database, entry_families=new_decl, upgrade=True)
        assert upgraded.fetch(RecNew, sid) == RecNew("kept", None)
        # The fingerprint is now re-stamped: a plain reopen trusts it.
        assert SqlStore(database, entry_families=new_decl).fetch(RecNew, sid) == RecNew("kept", None)


def test_reopen_non_nullable_added_field_raises_and_does_not_alter() -> None:
    with Backend.sqlite() as database:
        SqlStore(database, entry_families=(_decl(UpgradeFamily, "test-upgrade", RecOld),))
        with pytest.raises(StorageLayoutUpgradeRequiredError) as error:
            SqlStore(database, entry_families=(_decl(UpgradeFamily, "test-upgrade", RecNewRequired),), upgrade=True)
        # Non-additive: the store still raises the ordinary schema diff, unaltered.
        assert set(error.value.diff["schema"]) == {"upgrade_rec"}
        with database.engine.connect() as connection:
            columns = {row[1] for row in connection.execute(sqlalchemy.text("PRAGMA table_info(upgrade_rec)"))}
        assert "tag" not in columns


def test_reopen_identity_participating_field_has_no_hint() -> None:
    with Backend.sqlite() as database:
        SqlStore(database, entry_families=(_decl(UpgradeFamily, "test-upgrade", RecOld),))
        with pytest.raises(StorageLayoutUpgradeRequiredError) as error:
            SqlStore(database, entry_families=(_decl(UpgradeFamily, "test-upgrade", RecNewNonSkip),))
        assert error.value.hint is None
        assert "upgrade=True" not in str(error.value)


def test_reopen_removed_field_raises_even_with_upgrade() -> None:
    with Backend.sqlite() as database:
        SqlStore(database, entry_families=(_decl(UpgradeFamily, "test-upgrade", RecNew),))
        with pytest.raises(StorageLayoutUpgradeRequiredError):
            SqlStore(database, entry_families=(_decl(UpgradeFamily, "test-upgrade", RecOld),), upgrade=True)


def test_reopen_indexed_added_column_creates_the_index() -> None:
    with Backend.sqlite() as database:
        store = SqlStore(database, entry_families=(_decl(UpgradeFamily, "test-upgrade", RecOld),))
        store.save(RecOld("row"))  # materialize the table so the upgrade alters it
        SqlStore(database, entry_families=(_decl(UpgradeFamily, "test-upgrade", RecNewIndexed),), upgrade=True)
        inspector = sqlalchemy.inspect(database.engine)
        indexed_columns = {column for index in inspector.get_indexes("upgrade_rec") for column in index["column_names"]}
        assert "code" in indexed_columns


def test_upgrade_true_with_no_diff_is_a_clean_reopen() -> None:
    with Backend.sqlite() as database:
        SqlStore(database, entry_families=(_decl(UpgradeFamily, "test-upgrade", RecOld),))
        store = SqlStore(database, entry_families=(_decl(UpgradeFamily, "test-upgrade", RecOld),), upgrade=True)
        assert {family.name for family in store.entry_layout} == {"test-upgrade"}


def test_apply_is_deferred_until_after_a_failing_later_check() -> None:
    with Backend.sqlite() as database:
        store = SqlStore(database, entry_families=(_decl(UpgradeFamily, "test-upgrade", RecOld),))
        store.save(RecOld("kept"))  # materialize upgrade_rec so an ALTER would apply
        # Poison an invalid dirty: marker naming a table that does not exist; its
        # check runs after the upgrade plan is accepted but before it is applied.
        with database.engine.begin() as connection:
            connection.execute(
                sqlalchemy.text("INSERT INTO _httk_store_metadata (key, value) VALUES ('dirty:ghost', 'x')")
            )
        with pytest.raises(StorageLayoutUpgradeRequiredError):
            SqlStore(database, entry_families=(_decl(UpgradeFamily, "test-upgrade", RecNew),), upgrade=True)
        # The apply was deferred past the failing check, so no ALTER ran.
        with database.engine.connect() as connection:
            columns = {row[1] for row in connection.execute(sqlalchemy.text("PRAGMA table_info(upgrade_rec)"))}
        assert "note" not in columns


def test_index_heals_when_column_present_but_index_missing() -> None:
    with Backend.sqlite() as database:
        store = SqlStore(database, entry_families=(_decl(UpgradeFamily, "test-upgrade", RecOld),))
        store.save(RecOld("row"))
        # Simulate a crash between ADD COLUMN and CREATE INDEX: the column exists
        # while its declared index does not.
        with database.engine.begin() as connection:
            connection.execute(sqlalchemy.text("ALTER TABLE upgrade_rec ADD COLUMN code TEXT"))
        SqlStore(database, entry_families=(_decl(UpgradeFamily, "test-upgrade", RecNewIndexed),), upgrade=True)
        inspector = sqlalchemy.inspect(database.engine)
        indexed_columns = {column for index in inspector.get_indexes("upgrade_rec") for column in index["column_names"]}
        assert "code" in indexed_columns
