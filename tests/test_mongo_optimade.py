"""Live MongoDB semantic coverage for OPTIMADE-filter querying."""

from dataclasses import dataclass
from fractions import Fraction

import pytest

from httk.store.mongo import MongoStore, optimade_filter_searcher
from httk.store.query.optimade_filters import FilterTranslationError


@dataclass(frozen=True)
class Publication:
    doi: str
    year: int


@dataclass(frozen=True)
class Material:
    name: str
    x: Fraction
    symbols: list[str]
    ref: Publication | None = None


@dataclass(frozen=True)
class Part:
    label: str
    val: int


@dataclass(frozen=True)
class Assembly:
    name: str
    parts: list[Part]


PUB_A = Publication("10.1000/alpha", 1999)
PUB_B = Publication("10.2000/beta", 2005)

MAT_1 = Material("alpha oxide", Fraction(1, 2), ["O", "H"], PUB_A)
MAT_2 = Material("beta metal", Fraction(5, 2), ["Fe"], PUB_B)
MAT_3 = Material("gamma oxide", Fraction(7, 2), ["O"], None)


@pytest.fixture()
def store(mongo_test_database):
    store = MongoStore(mongo_test_database, entry_records={})
    for material in (MAT_1, MAT_2, MAT_3):
        store.save(material)
    return store


def results(searcher):
    return [item.values[0] for item in searcher]


def test_numeric_comparison_on_fraction_field(store):
    searcher = optimade_filter_searcher(store, Material, "_httk_custom_x > 1")
    assert results(searcher) == [MAT_2, MAT_3]


def test_string_operations(store):
    searcher = optimade_filter_searcher(store, Material, '_httk_custom_name STARTS WITH "beta"')
    assert results(searcher) == [MAT_2]
    searcher = optimade_filter_searcher(store, Material, '_httk_custom_name CONTAINS "oxide"')
    assert results(searcher) == [MAT_1, MAT_3]
    searcher = optimade_filter_searcher(store, Material, '_httk_custom_name = "gamma oxide"')
    assert results(searcher) == [MAT_3]


def test_has_over_list_field(store):
    searcher = optimade_filter_searcher(store, Material, '_httk_custom_symbols HAS "O"')
    assert results(searcher) == [MAT_1, MAT_3]
    searcher = optimade_filter_searcher(store, Material, '_httk_custom_symbols HAS ALL "O","H"')
    assert results(searcher) == [MAT_1]
    searcher = optimade_filter_searcher(store, Material, 'NOT _httk_custom_symbols HAS "O"')
    assert results(searcher) == [MAT_2]


def test_combined_scalar_and_related_property_semi_join(store):
    searcher = optimade_filter_searcher(
        store,
        Material,
        '_httk_custom_x > 1 AND refs._httk_custom_doi CONTAINS "10.2"',
        related_classes={"refs": Publication},
    )
    assert results(searcher) == [MAT_2]
    searcher = optimade_filter_searcher(
        store, Material, 'refs._httk_custom_doi CONTAINS "10."', related_classes={"refs": Publication}
    )
    assert results(searcher) == [MAT_1, MAT_2]


def test_related_property_comparison_and_not_complement(store):
    searcher = optimade_filter_searcher(
        store, Material, "refs._httk_custom_year >= 2000", related_classes={"refs": Publication}
    )
    assert results(searcher) == [MAT_2]
    searcher = optimade_filter_searcher(
        store, Material, "NOT refs._httk_custom_year >= 2000", related_classes={"refs": Publication}
    )
    assert results(searcher) == [MAT_1, MAT_3]


def test_related_property_no_match_is_empty_not_an_error(store):
    searcher = optimade_filter_searcher(
        store, Material, 'refs._httk_custom_doi CONTAINS "nomatch"', related_classes={"refs": Publication}
    )
    assert results(searcher) == []


def test_relationship_id_has_over_foreign_key(store):
    sid = store.sid_of(PUB_A)
    searcher = optimade_filter_searcher(
        store, Material, f'refs.id HAS "refs-{sid}"', related_classes={"refs": Publication}
    )
    assert results(searcher) == [MAT_1]


def test_relationship_id_has_only_preserves_empty_reference_vacuous_truth(store):
    sid = store.sid_of(PUB_B)
    searcher = optimade_filter_searcher(
        store, Material, f'refs.id HAS ONLY "refs-{sid}"', related_classes={"refs": Publication}
    )
    assert results(searcher) == [MAT_2, MAT_3]
    searcher = optimade_filter_searcher(
        store, Material, f'NOT refs.id HAS ONLY "refs-{sid}"', related_classes={"refs": Publication}
    )
    assert results(searcher) == [MAT_1]

    searcher = optimade_filter_searcher(
        store, Material, f'refs.id HAS ANY "refs-{sid}"', related_classes={"refs": Publication}
    )
    assert results(searcher) == [MAT_2]
    searcher = optimade_filter_searcher(
        store, Material, f'NOT refs.id HAS ANY "refs-{sid}"', related_classes={"refs": Publication}
    )
    assert results(searcher) == [MAT_1, MAT_3]


def test_relationship_id_equality_routes_through_semi_join(store):
    sid = store.sid_of(PUB_B)
    searcher = optimade_filter_searcher(
        store, Material, f'refs.id = "refs-{sid}"', related_classes={"refs": Publication}
    )
    assert results(searcher) == [MAT_2]
    searcher = optimade_filter_searcher(
        store, Material, f'refs.id != "refs-{sid}"', related_classes={"refs": Publication}
    )
    assert results(searcher) == [MAT_1]


def test_invalid_or_foreign_id_formats_match_nothing(store):
    for bad_id in ("bogus", "refs-", "refs-abc", "other-1", "refs-1-2"):
        searcher = optimade_filter_searcher(
            store, Material, f'refs.id HAS "{bad_id}"', related_classes={"refs": Publication}
        )
        assert results(searcher) == []


def test_child_of_storable_target(store):
    part_a = Part("bolt", 1)
    part_b = Part("nut", 5)
    assembly_1 = Assembly("frame", [part_a, part_b])
    assembly_2 = Assembly("hinge", [part_a])
    store.save(assembly_1)
    store.save(assembly_2)
    searcher = optimade_filter_searcher(store, Assembly, "parts._httk_custom_val > 2", related_classes={"parts": Part})
    assert results(searcher) == [assembly_1]
    part_a_sid = store.sid_of(part_a)
    searcher = optimade_filter_searcher(
        store, Assembly, f'parts.id HAS "parts-{part_a_sid}"', related_classes={"parts": Part}
    )
    assert results(searcher) == [assembly_1, assembly_2]


def test_nested_dotted_path_not_implemented(store):
    with pytest.raises(FilterTranslationError) as excinfo:
        optimade_filter_searcher(
            store, Material, "refs.other._httk_custom_x = 1", related_classes={"refs": Publication}
        )
    assert excinfo.value.category == "not-implemented"


def test_dotted_filter_without_related_classes_matches_nothing(store):
    searcher = optimade_filter_searcher(store, Material, 'refs._httk_custom_doi CONTAINS "10."')
    assert results(searcher) == []


def test_id_and_type_not_supported_without_extra_handlers(store):
    for filter_string in ('id = "materials-1"', 'type = "materials"'):
        with pytest.raises(FilterTranslationError) as excinfo:
            optimade_filter_searcher(store, Material, filter_string)
        assert excinfo.value.category == "not-implemented"


def test_unknown_prefixed_property_raises(store):
    with pytest.raises(FilterTranslationError) as excinfo:
        optimade_filter_searcher(store, Material, "_httk_bananas = 3")
    assert excinfo.value.category == "unrecognized-property"


def test_unknown_unprefixed_property_matches_nothing(store):
    searcher = optimade_filter_searcher(store, Material, "bananas = 3")
    assert results(searcher) == []


def test_unmatched_related_class_raises_value_error(store):
    with pytest.raises(ValueError):
        optimade_filter_searcher(store, Material, 'parts.id HAS "parts-1"', related_classes={"parts": Part})
