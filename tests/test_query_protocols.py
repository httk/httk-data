"""Structural-conformance tests for the store/searcher query protocols (httk.data.query)."""

from typing import Any, Iterator

from httk.data import SearchColumn, Searcher, SearchExpression, SearchVariable, Store


class FakeExpression:
    def __and__(self, other: SearchExpression) -> "FakeExpression":
        return self

    def __or__(self, other: SearchExpression) -> "FakeExpression":
        return self

    def __invert__(self) -> "FakeExpression":
        return self


class FakeColumn:
    def has_any(self, *values: Any) -> FakeExpression:
        return FakeExpression()

    def has_inv_any(self, *values: Any) -> FakeExpression:
        return FakeExpression()

    def has_only(self, *values: Any) -> FakeExpression:
        return FakeExpression()

    def has_inv_only(self, *values: Any) -> FakeExpression:
        return FakeExpression()

    def like(self, pattern: str) -> FakeExpression:
        return FakeExpression()


class FakeVariable:
    def __getattr__(self, name: str) -> FakeColumn:
        return FakeColumn()


class FakeSearcher:
    offset: int = 0

    def variable(self, target: Any) -> Any:
        return FakeVariable()

    def output(self, variable: Any, name: str) -> None:
        pass

    def add(self, expression: Any) -> None:
        pass

    def add_all(self, expression: Any) -> None:
        pass

    def count(self) -> int:
        return 0

    def set_limit(self, limit: int) -> None:
        pass

    def add_offset(self, offset: int) -> None:
        pass

    def add_sort(self, column: Any, descending: bool) -> None:
        pass

    def __iter__(self) -> Iterator[Any]:
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
    column: SearchColumn = variable.anything
    expression: SearchExpression = column.has_any(1, 2)
    combined = (expression & expression) | ~expression
    searcher.add(combined)
    searcher.add_all(column.has_only("a"))
    searcher.output(variable, "out")
    searcher.add_sort(column, descending=True)
    searcher.set_limit(-1)
    searcher.add_offset(0)
    assert searcher.count() == 0
    assert list(searcher) == []


def test_comparison_operators_reachable_by_getattr_convention():
    # Handlers invoke comparisons via getattr(column, '__eq__')(value); the
    # convention must at least be callable on a conforming column object.
    column = FakeColumn()
    result = getattr(column, "__eq__")(42)
    assert result is NotImplemented or isinstance(result, bool)
