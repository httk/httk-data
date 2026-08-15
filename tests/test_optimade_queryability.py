"""Layering lock: the neutral store stays permissive for trusted callers.

Queryability (honoring ``x-optimade-requirements.query-support: "none"``) is a
PROTOCOL-BOUNDARY policy enforced at the serve engine
(``httk.serve.optimade.engine.processing._reject_hidden_filter_properties``),
applied to the client filter before any backend/adapter rewriting. The store
layer must NOT enforce it: trusted internal callers (e.g. an adapter that
rewrites public ``id`` predicates to an internal projection after validation)
must be able to filter every stored property. These tests pin that contract.
"""

from dataclasses import dataclass
from fractions import Fraction

import pytest
from httk.core import EntryTypeDefinition, PropertyDefinition

from httk.store.db import Database, SqlStore, optimade_filter_searcher


@dataclass(frozen=True)
class Material:
    name: str
    x: Fraction
    symbols: list[str]


def _query_support_none(prop: PropertyDefinition) -> PropertyDefinition:
    """Return a copy of ``prop`` declaring ``query-support: "none"``."""
    doc = dict(prop.as_optimade())
    doc["x-optimade-requirements"] = {"query-support": "none"}
    return PropertyDefinition.from_optimade(prop.name, doc)


def _definition_hiding(*names: str) -> EntryTypeDefinition:
    """A materials definition whose named properties declare query-support none."""
    properties = {
        "id": PropertyDefinition.from_simple("id", description="The id.", required_response=True),
        "type": PropertyDefinition.from_simple("type", description="The type.", required_response=True),
    }
    for name in names:
        properties[name] = _query_support_none(PropertyDefinition.from_simple(name, description="A hidden property."))
    return EntryTypeDefinition("materials", "Materials.", properties)


@pytest.fixture()
def store():
    store = SqlStore(Database.sqlite(":memory:"), entry_records={})
    store.save(Material("alpha oxide", Fraction(1, 2), ["O", "H"]))
    store.save(Material("beta metal", Fraction(5, 2), ["Fe"]))
    return store


def _results(searcher):
    return [item[0][0] for item in searcher]


def test_store_layer_filters_a_query_support_none_property(store) -> None:
    # Enforcement is at the serve engine, NOT here: the store must still filter a
    # query-support-"none" property for trusted callers.
    definition = _definition_hiding("_httk_custom_name")
    searcher = optimade_filter_searcher(store, Material, '_httk_custom_name = "alpha oxide"', definition=definition)
    assert [material.name for material in _results(searcher)] == ["alpha oxide"]


def test_store_layer_sibling_fields_unaffected(store) -> None:
    definition = _definition_hiding("_httk_custom_name")
    searcher = optimade_filter_searcher(store, Material, "_httk_custom_x > 1", definition=definition)
    assert [material.name for material in _results(searcher)] == ["beta metal"]
