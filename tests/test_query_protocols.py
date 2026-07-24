"""Structural-conformance tests for the store/searcher query protocols (httk.data.query)."""

from typing import Any, Iterator

from httk.data import Searcher, SearchExpression, SearchField, SearchResult, SearchVariable, Store


class FakeExpression:
    def __and__(self, other: SearchExpression) -> "FakeExpression":
        return self

    def __or__(self, other: SearchExpression) -> "FakeExpression":
        return self

    def __invert__(self) -> "FakeExpression":
        return self


class FakeField:
    def has_any(self, *values: Any) -> FakeExpression:
        return FakeExpression()

    def has_only(self, *values: Any) -> FakeExpression:
        return FakeExpression()

    def contains(self, text: str) -> FakeExpression:
        return FakeExpression()

    def startswith(self, prefix: str) -> FakeExpression:
        return FakeExpression()

    def endswith(self, suffix: str) -> FakeExpression:
        return FakeExpression()


class FakeVariable:
    def always_true(self) -> FakeExpression:
        return FakeExpression()

    def always_false(self) -> FakeExpression:
        return FakeExpression()

    def __getattr__(self, name: str) -> FakeField:
        return FakeField()


class FakeSearcher:
    offset: int = 0

    def __init__(self) -> None:
        self.names: tuple[str, ...] = ()

    def variable(self, target: Any) -> Any:
        return FakeVariable()

    def output(self, variable: Any, name: str) -> None:
        self.names += (name,)

    def add(self, expression: Any) -> None:
        pass

    def count(self) -> int:
        return 0

    def set_limit(self, limit: int) -> None:
        pass

    def add_offset(self, offset: int) -> None:
        pass

    def add_sort(self, field: Any, descending: bool) -> None:
        pass

    def __iter__(self) -> Iterator[SearchResult]:
        return iter(())


class FakeStore:
    def searcher(self) -> FakeSearcher:
        return FakeSearcher()


def test_fakes_conform_to_the_protocols():
    # The annotated assignments below are the actual conformance assertions:
    # mypy/pyright verify each fake structurally satisfies its protocol.
    store: Store = FakeStore()
    searcher: Searcher = store.searcher()
    variable: SearchVariable = searcher.variable(object)
    field: SearchField = variable.anything
    expression: SearchExpression = field.has_any(1, 2)
    combined = (expression & expression) | ~expression
    searcher.add(combined)
    searcher.add(field.has_only("a"))
    searcher.add(~field.has_only("a"))
    searcher.add(field.contains("a"))
    searcher.add(field.startswith("a"))
    searcher.add(field.endswith("a"))
    searcher.add(variable.always_true())
    searcher.add(variable.always_false())
    searcher.output(variable, "out")
    searcher.add_sort(field, descending=True)
    searcher.set_limit(-1)
    searcher.add_offset(0)
    assert searcher.count() == 0
    assert list(searcher) == []


def test_search_result_is_a_two_tuple_of_values_and_names():
    # The documented shape: a 2-tuple that also names its parts, so both
    # `values, names = result` and `result[0][0]` keep working.
    result = SearchResult(("obj", 3), ("rec", "spacegroup"))
    values, names = result
    assert values == ("obj", 3)
    assert names == ("rec", "spacegroup")
    assert result[0][0] == "obj"
    assert result.values == values and result.names == names
    assert len(result) == 2


def test_comparison_operators_reachable_by_getattr_convention():
    # Handlers invoke comparisons via getattr(field, '__eq__')(value); the
    # convention must at least be callable on a conforming field object.
    field = FakeField()
    result = getattr(field, "__eq__")(42)
    assert result is NotImplemented or isinstance(result, bool)
