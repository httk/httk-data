"""Phase-one tests for frozen federated sources and target prototype binding."""

from collections.abc import Mapping

import pytest

from httk.data import FederatedSourceError, FederatedStore, UnsupportedQueryError


class ProbeSearcher:
    def __init__(self, store: "ProbeStore") -> None:
        self._store = store

    def variable(self, target: object) -> object:
        self._store.bound_targets.append(target)
        if target in self._store.unsupported_targets:
            raise UnsupportedQueryError("unsupported test target")
        return object()


class ProbeStore:
    def __init__(self, unsupported_targets: tuple[object, ...] = ()) -> None:
        self.searcher_calls = 0
        self.bound_targets: list[object] = []
        self.unsupported_targets = unsupported_targets
        self.close_calls = 0

    def searcher(self) -> ProbeSearcher:
        self.searcher_calls += 1
        return ProbeSearcher(self)

    def close(self) -> None:
        self.close_calls += 1


@pytest.mark.parametrize(
    "sources",
    ({}, {"one": ProbeStore()}, {"": ProbeStore(), "two": ProbeStore()}, {1: ProbeStore(), "two": ProbeStore()}),
)
def test_invalid_sources_fail_deterministically(sources: Mapping[object, ProbeStore]) -> None:
    with pytest.raises((TypeError, ValueError)):
        FederatedStore(sources)  # type: ignore[arg-type]


def test_sources_are_ordered_frozen_and_do_not_query_during_construction() -> None:
    first = ProbeStore()
    second = ProbeStore()
    sources = {"second": second, "first": first}
    store = FederatedStore(sources)
    sources["later"] = ProbeStore()

    assert store.source_names == ("second", "first")
    assert first.searcher_calls == second.searcher_calls == 0
    with pytest.raises(TypeError):
        store._sources["later"] = ProbeStore()  # type: ignore[index]


def test_targets_are_ordered_frozen_and_retain_exact_objects() -> None:
    store = FederatedStore({"first": ProbeStore(), "second": ProbeStore(), "third": ProbeStore()})
    second_target = object()
    first_target = object()
    targets = {"second": second_target, "first": first_target}

    target = store.target("records", targets)
    targets["third"] = object()

    assert target.name == "records"
    assert tuple(target.targets) == ("first", "second")
    assert target.targets["first"] is first_target
    assert target.targets["second"] is second_target
    with pytest.raises(TypeError):
        target.targets["first"] = object()  # type: ignore[index]
    with pytest.raises((AttributeError, TypeError)):
        target.name = "other"  # type: ignore[misc]


def test_invalid_targets_fail_deterministically() -> None:
    store = FederatedStore({"first": ProbeStore(), "second": ProbeStore()})
    for name, targets in (("", {"first": object()}), ("records", {}), ("records", {"other": object()})):
        with pytest.raises((TypeError, ValueError)):
            store.target(name, targets)


def test_direct_target_binds_the_same_target_to_every_source() -> None:
    first = ProbeStore()
    second = ProbeStore()
    store = FederatedStore({"first": first, "second": second})
    target = object()

    variable = store.searcher().variable(target)

    assert isinstance(variable, object)
    assert first.bound_targets == [target]
    assert second.bound_targets == [target]
    assert first.bound_targets[0] is second.bound_targets[0] is target


def test_explicit_target_binds_only_its_ordered_source_subset() -> None:
    first = ProbeStore()
    second = ProbeStore()
    third = ProbeStore()
    store = FederatedStore({"first": first, "second": second, "third": third})
    second_target = object()
    first_target = object()

    store.searcher().variable(store.target("records", {"second": second_target, "first": first_target}))

    assert first.bound_targets == [first_target]
    assert second.bound_targets == [second_target]
    assert third.bound_targets == []


def test_foreign_target_and_second_root_are_unsupported() -> None:
    first = FederatedStore({"first": ProbeStore(), "second": ProbeStore()})
    second = FederatedStore({"first": ProbeStore(), "second": ProbeStore()})
    foreign = first.target("records", {"first": object()})
    searcher = second.searcher()

    with pytest.raises(UnsupportedQueryError, match="another federation"):
        searcher.variable(foreign)
    searcher.variable(object())
    with pytest.raises(UnsupportedQueryError, match="second root"):
        searcher.variable(object())


def test_source_unsupported_target_binding_keeps_neutral_category_and_context() -> None:
    unsupported = object()
    first = ProbeStore()
    second = ProbeStore((unsupported,))
    store = FederatedStore({"first": first, "second": second})

    with pytest.raises(UnsupportedQueryError, match="second") as excinfo:
        store.searcher().variable(unsupported)

    assert isinstance(excinfo.value, FederatedSourceError)
    assert excinfo.value.source == "second"
    assert excinfo.value.operation == "target binding"


def test_federation_never_propagates_close_to_borrowed_sources() -> None:
    first = ProbeStore()
    second = ProbeStore()
    store = FederatedStore({"first": first, "second": second})

    assert not hasattr(store, "close")
    assert first.close_calls == second.close_calls == 0
