"""End-to-end OPTIMADE-filter escaping through SQL translation.

httk-core's filter parser now unescapes ``\\"`` -> ``"`` and ``\\\\`` -> ``\\``
in string literals, so translated filter values can carry a literal ``"`` or
``\\`` for the first time. These tests drive such values through
``optimade_filter_searcher`` (parse + translate + execute) against real stored
rows, and pin the LIKE-escaping of ``\\``, ``%`` and ``_`` so a value character
never leaks as a SQL wildcard.
"""

from dataclasses import dataclass

import pytest

from httk.store.backend.sql import Backend, SqlStore, optimade_filter_searcher


@dataclass(frozen=True)
class Doc:
    name: str


D_QUOTE = Doc('ab"c')  # embedded double quote
D_BS = Doc("a\\b")  # embedded backslash: a\b
D_ABC = Doc("abc")  # backslash-escape control (contains "ab")
D_PCT = Doc("50%")  # embedded percent
D_505 = Doc("505")  # percent-wildcard control (contains "0")
D_US = Doc("5_5")  # embedded underscore
D_545 = Doc("545")  # underscore-wildcard control

RECORDS = (D_QUOTE, D_BS, D_ABC, D_PCT, D_505, D_US, D_545)


@pytest.fixture(params=["sqlite", "duckdb"])
def store(request):
    if request.param == "duckdb":
        pytest.importorskip("duckdb_engine")
        manager = Backend.duckdb()
    else:
        manager = Backend.sqlite(":memory:")
    with manager as database:
        value = SqlStore(database, entry_records={})
        for record in RECORDS:
            value.save(record)
        yield value


def matches(store, filter_string):
    return [item[0][0] for item in optimade_filter_searcher(store, Doc, filter_string)]


def test_equality_matches_value_with_embedded_double_quote(store):
    # Filter text: _httk_custom_name = "ab\"c" ; parser unescapes \" -> "
    assert matches(store, '_httk_custom_name = "ab\\"c"') == [D_QUOTE]


def test_equality_matches_value_with_embedded_backslash(store):
    # Filter text: _httk_custom_name = "a\\b" ; parser unescapes \\ -> \
    assert matches(store, '_httk_custom_name = "a\\\\b"') == [D_BS]


def test_contains_backslash_is_literal_not_a_like_escape(store):
    # CONTAINS "a\b": matches the literal backslash value only. If the backslash
    # were not LIKE-escaped it would be consumed as the escape char, turning the
    # pattern into %ab% and spuriously matching "abc".
    assert matches(store, '_httk_custom_name CONTAINS "a\\\\b"') == [D_BS]


def test_contains_percent_stays_literal(store):
    # CONTAINS "0%": literal, matches "50%" but not "505" (no wildcard leak).
    assert matches(store, '_httk_custom_name CONTAINS "0%"') == [D_PCT]


def test_contains_underscore_stays_literal(store):
    # CONTAINS "5_5": literal, matches "5_5" but not "545" (no single-char wildcard).
    assert matches(store, '_httk_custom_name CONTAINS "5_5"') == [D_US]
