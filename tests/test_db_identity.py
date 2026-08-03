"""Tests for content identity (httk.data.db.identity)."""

import datetime
import string
from dataclasses import dataclass
from fractions import Fraction
from typing import Annotated

from httk.core import FracVector
from httk.core.storage import Shape, Skip, stored_property

from httk.data.db import canonical_form, content_id


@dataclass(frozen=True)
class Author:
    name: str
    year: int


@dataclass(frozen=True)
class Article:
    title: str
    ratio: Fraction
    when: datetime.datetime
    cell: Annotated[FracVector, Shape(2, 2)]
    symbols: list[str]
    author: Author | None = None
    scratch: Annotated[str, Skip()] = ""

    @stored_property
    def title_length(self) -> int:
        return len(self.title)


def _article(title: str = "On Perovskites", author: Author | None = None) -> Article:
    return Article(
        title=title,
        ratio=Fraction(-13, 3),
        when=datetime.datetime(2026, 7, 24, 12, 0),
        cell=FracVector.create([[1, "1/2"], [0, "2/3"]]),
        symbols=["Ca", "Ti", "O"],
        author=author if author is not None else Author("Goldschmidt", 1926),
    )


def test_content_id_is_64_hex_characters():
    identity = content_id(_article())
    assert len(identity) == 64
    assert set(identity) <= set(string.hexdigits.lower())


def test_equal_instances_have_equal_ids():
    assert content_id(_article()) == content_id(_article())


def test_any_stored_field_change_changes_the_id():
    base = _article()
    assert content_id(_article(title="Other")) != content_id(base)
    variants = [
        Article(base.title, Fraction(-13, 4), base.when, base.cell, base.symbols, base.author),
        Article(base.title, base.ratio, datetime.datetime(2026, 7, 25), base.cell, base.symbols, base.author),
        Article(base.title, base.ratio, base.when, FracVector.create([[1, 0], [0, 1]]), base.symbols, base.author),
        Article(base.title, base.ratio, base.when, base.cell, ["Ca", "O", "Ti"], base.author),
        Article(base.title, base.ratio, base.when, base.cell, base.symbols, None),
    ]
    for variant in variants:
        assert content_id(variant) != content_id(base)


def test_distinct_but_equal_nested_instances_give_the_same_parent_id():
    one = _article(author=Author("Goldschmidt", 1926))
    other = _article(author=Author("Goldschmidt", 1926))
    assert one.author is not other.author
    assert content_id(one) == content_id(other)
    assert '"identity_name":"test_db_identity.Author"' in canonical_form(one)


def test_nested_field_change_changes_the_parent_id():
    assert content_id(_article(author=Author("Goldschmidt", 1927))) != content_id(_article())


def test_derived_and_skipped_values_do_not_affect_the_id():
    form = canonical_form(_article())
    assert "title_length" not in form
    assert "scratch" not in form
    with_scratch = Article(
        title="On Perovskites",
        ratio=Fraction(-13, 3),
        when=datetime.datetime(2026, 7, 24, 12, 0),
        cell=FracVector.create([[1, "1/2"], [0, "2/3"]]),
        symbols=["Ca", "Ti", "O"],
        author=Author("Goldschmidt", 1926),
        scratch="anything",
    )
    assert content_id(with_scratch) == content_id(_article())


def test_canonical_form_uses_exact_texts():
    form = canonical_form(_article())
    assert '"type":"rational","value":[-13,3]' in form
    assert '"denominator":6,"nominators":[[6,3],[0,4]]' in form
    assert '"2026-07-24T12:00:00.000000"' in form
