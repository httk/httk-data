"""Materialized MongoDB result sets and live keyset pages."""

import copy
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any, Final, Literal

from httk.store.db.schema import resolve_schema
from httk.store.query import (
    ContinuationToken,
    MultipleResultsError,
    NoResultError,
    PageOrder,
    ResultPage,
    ResultRow,
    UnsupportedQueryError,
)
from httk.store.query.paging_tokens import (
    _decode_continuation,
    _DecodedContinuation,
    _encode_continuation,
    _plan_fingerprint,
)

from .searcher import MongoField, MongoSearcher, MongoVariable, _MongoOutput, _scalar_value, _variable_document
from .store import _DOCUMENT_LAYOUT

__all__ = ["MongoResultSet"]

_PAGE_SIZE_MAX: Final = 10_000
_PAGE_ORDER_MAX: Final = 32


@dataclass(frozen=True, slots=True)
class _PageKey:
    """One validated root scalar projection and its keyset ordering."""

    order: PageOrder
    output: _MongoOutput


@dataclass(frozen=True, slots=True)
class _Candidate:
    """One server candidate and its optional client-verification result."""

    document: dict[str, Any]
    verified: bool


class MongoResultSet:
    """A materialized MongoDB result set with named rows and scalar helpers.

    :param searcher: The search plan to freeze.
    :param outputs: Optional replacement output declarations.
    """

    def __init__(self, searcher: MongoSearcher, outputs: list[Any] | None = None) -> None:
        # Variables intentionally remain the same objects: their references
        # describe stable document paths, while the mutable lists below are
        # the actual query-plan state that must be frozen at results().
        self._plan = copy.copy(searcher)
        self._plan._variables = list(searcher._variables)
        self._plan._hidden_variables = list(searcher._hidden_variables)
        self._plan._expressions = list(searcher._expressions)
        self._plan._sorts = list(searcher._sorts)
        self._outputs = list(searcher._outputs if outputs is None else outputs)
        self._plan._outputs = list(self._outputs)
        self.names = tuple(output.name for output in self._outputs)
        self._plan._require_verifier_identity()
        self._pipeline = self._plan._pipeline(apply_window=self._plan._row_verifier is None)
        self._sorts = tuple(self._plan._sorts)
        if self._plan._row_verifier is None:
            self._rows = tuple(self._plan._execute(self._outputs))
        else:
            self._rows = tuple(
                self._document_row(document)
                for document in self._plan._verified_documents(self._pipeline, apply_window=True)
            )

    def __iter__(self) -> Iterator[ResultRow]:
        """Iterate persistent named result rows."""
        return iter(ResultRow(row, self.names) for row in self._rows)

    def __len__(self) -> int:
        """Return the exact number of materialized rows."""
        return len(self._rows)

    def first(self) -> ResultRow | None:
        """Return the first row or ``None``."""
        return next(iter(self), None)

    def one(self) -> ResultRow:
        """Return the sole row.

        :raises ~httk.store.query.NoResultError: If no row exists.
        :raises ~httk.store.query.MultipleResultsError: If multiple rows exist.
        :return: The sole result row.
        """
        if not self._rows:
            raise NoResultError("expected exactly one result, found none")
        if len(self._rows) != 1:
            raise MultipleResultsError("expected exactly one result, found more than one")
        return ResultRow(self._rows[0], self.names)

    def scalars(self, name: str | None = None) -> Iterator[Any]:
        """Iterate one named output from each row."""
        if name is None:
            if len(self.names) != 1:
                raise ValueError(f"scalars() without a name requires exactly one output; declared: {self.names}")
            name = self.names[0]
        if name not in self.names:
            raise KeyError(f"unknown output {name!r}; declared: {self.names}")
        return (row[name] for row in self)

    def column(self, name: str) -> Iterator[Any]:
        """Return an iterator over one scalar output.

        Mongo packet 4a keeps this optional capability intentionally small;
        object-output validation is performed from the frozen output plan.
        """
        if name not in self.names:
            raise KeyError(f"unknown output {name!r}; declared: {self.names}")
        index = self.names.index(name)
        if isinstance(self._outputs[index].value, MongoVariable):
            raise TypeError(f"column {name!r} is an object output")
        return self.scalars(name)

    def page(
        self,
        *,
        size: int,
        order_by: Iterable[PageOrder],
        cursor: ContinuationToken | None = None,
        include_total: bool = False,
    ) -> ResultPage:
        """Fetch one bounded live page using a strict MongoDB keyset seek.

        :param size: The maximum number of verified rows to return.
        :param order_by: Root scalar result projections used as order keys.
        :param cursor: A continuation token returned by a preceding page.
        :param include_total: Whether to calculate the exact server-side total.
        :return: The requested immutable result page.
        """
        self._validate_page_size(size)
        if not isinstance(include_total, bool):
            raise TypeError("include_total must be bool")
        keys = self._page_keys(order_by)
        fingerprint = self._page_fingerprint(keys)
        decoded = None if cursor is None else _decode_continuation(cursor, fingerprint=fingerprint, anchors=len(keys))

        documents, more_in_fetch_direction = self._page_documents(keys, decoded, size)
        if decoded is not None and decoded.direction == "backward":
            documents.reverse()
        rows = tuple(ResultRow(self._document_row(document), self.names) for document in documents)
        next_token, previous_token = self._page_tokens(documents, keys, fingerprint, decoded, more_in_fetch_direction)
        total = self._page_total() if include_total else None
        return ResultPage(rows, next_token, previous_token, total)

    @staticmethod
    def _validate_page_size(size: int) -> None:
        if isinstance(size, bool) or not isinstance(size, int):
            raise TypeError("page size must be an integer (bool is not accepted)")
        if not 1 <= size <= _PAGE_SIZE_MAX:
            raise ValueError(f"page size must be between 1 and {_PAGE_SIZE_MAX}")

    def _page_keys(self, order_by: Iterable[PageOrder]) -> tuple[_PageKey, ...]:
        """Validate the common one-root, scalar-only paging profile."""
        if len(self._plan._variables) != 1:
            raise UnsupportedQueryError("paging requires exactly one root query variable")
        if self._plan._sorts:
            raise UnsupportedQueryError("paging does not compose with add_sort(); pass PageOrder values instead")
        if self._plan.offset != 0:
            raise UnsupportedQueryError("paging does not compose with a nonzero query offset")
        if self._plan._limit is not None:
            raise UnsupportedQueryError("paging does not compose with a query limit")
        if not self._outputs:
            raise ValueError("this search has no outputs; declare outputs or pass them to results()")
        if isinstance(order_by, (str, bytes)):
            raise TypeError("order_by must be an iterable of PageOrder values")
        try:
            requested = iter(order_by)
        except TypeError as error:
            raise TypeError("order_by must be an iterable of PageOrder values") from error

        requested_orders: list[PageOrder] = []
        for order in requested:
            if len(requested_orders) >= _PAGE_ORDER_MAX:
                raise UnsupportedQueryError(f"paging supports at most {_PAGE_ORDER_MAX} order keys")
            requested_orders.append(order)

        root = self._plan._variables[0]
        keys: list[_PageKey] = []
        seen: set[str] = set()
        for order in requested_orders:
            if not isinstance(order, PageOrder):
                raise TypeError(f"order_by entries must be PageOrder values, got {type(order).__name__}")
            if order.name in seen:
                raise UnsupportedQueryError(f"duplicate paging order name {order.name!r}")
            seen.add(order.name)
            matching = [(index, output) for index, output in enumerate(self._outputs) if output.name == order.name]
            if not matching:
                raise UnsupportedQueryError(f"paging order {order.name!r} is not a declared result projection")
            if len(matching) > 1:
                raise UnsupportedQueryError(f"paging order {order.name!r} names duplicate result projections")
            _index, output = matching[0]
            if isinstance(output.value, MongoVariable):
                raise UnsupportedQueryError(f"paging order {order.name!r} is an object projection, not a scalar")
            field = output.value
            if field._child_keys or field._variable is not root:
                raise UnsupportedQueryError(
                    f"paging order {order.name!r} must be a scalar projection of the root query variable"
                )
            if field._spec.role not in {"scalar", "encoded"}:
                raise UnsupportedQueryError(f"paging order {order.name!r} has an unsupported stored representation")
            keys.append(_PageKey(order, output))
        return tuple(keys)

    def _page_documents(
        self,
        keys: tuple[_PageKey, ...],
        cursor: _DecodedContinuation | None,
        size: int,
    ) -> tuple[list[dict[str, Any]], bool]:
        """Pull verified candidates until a page and its probe have been found."""
        verified: list[dict[str, Any]] = []
        seek = cursor
        while True:
            pipeline = self._page_pipeline(keys, seek, size + 1)
            candidates = list(self._candidate_documents(pipeline))
            for candidate in candidates:
                if candidate.verified:
                    verified.append(candidate.document)
                    if len(verified) > size:
                        return verified[:size], True
            if len(candidates) < size + 1:
                return verified, False
            # A verifier may have rejected the entire batch.  Seek from the
            # final *candidate*, never a rejected row's returned anchor.
            last = candidates[-1].document
            seek = _DecodedContinuation(
                "backward" if cursor is not None and cursor.direction == "backward" else "forward",
                tuple(self._page_anchor(last, key) for key in keys),
                int(last["_id"]),
            )

    def _page_pipeline(
        self,
        keys: tuple[_PageKey, ...],
        cursor: _DecodedContinuation | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Render the page aggregation without a grouping or child-array unwind."""
        # _lookup_stages only emits 0-or-1 reference unwinds; page plans must
        # never introduce a row-multiplying child-array $unwind or a $group.
        pipeline = self._plan._lookup_stages()
        assert all("$group" not in stage for stage in pipeline)
        assert all("$unwind" not in stage or stage["$unwind"].get("preserveNullAndEmptyArrays") for stage in pipeline)
        pipeline.append({"$match": self._plan._truth_filter()})
        for index, key in enumerate(keys):
            field = self._page_field(key)
            null_rank = 0 if key.order.nulls == "first" else 1
            pipeline.append(
                {
                    "$addFields": {
                        self._page_rank_name(index): {
                            "$cond": [
                                {"$in": [{"$type": f"${field._path}"}, ["missing", "null"]]},
                                null_rank,
                                1 - null_rank,
                            ]
                        }
                    }
                }
            )
        backward = cursor is not None and cursor.direction == "backward"
        if cursor is not None:
            pipeline.append({"$match": {"$expr": self._page_seek_predicate(keys, cursor, before=backward)}})
        sort: dict[str, int] = {}
        for index, key in enumerate(keys):
            sort[self._page_rank_name(index)] = -1 if backward else 1
            sort[self._page_field(key)._path] = -1 if key.order.descending != backward else 1
        sort["_id"] = -1 if backward else 1
        pipeline.append({"$sort": sort})
        if limit is not None:
            pipeline.append({"$limit": limit})
        return pipeline

    @staticmethod
    def _page_rank_name(index: int) -> str:
        return f"_httk_page_{index}_rank"

    @staticmethod
    def _page_field(key: _PageKey) -> MongoField:
        assert isinstance(key.output.value, MongoField)
        return key.output.value

    def _page_seek_predicate(
        self,
        keys: tuple[_PageKey, ...],
        cursor: _DecodedContinuation,
        *,
        before: bool,
    ) -> dict[str, Any]:
        """Render the strict lexicographic before/after predicate in $expr form."""
        prefix: list[dict[str, Any]] = []
        choices: list[dict[str, Any]] = []
        for index, (key, anchor) in enumerate(zip(keys, cursor.anchors, strict=True)):
            rank_name = f"${self._page_rank_name(index)}"
            anchor_rank = 0 if (anchor is None) == (key.order.nulls == "first") else 1
            rank_compare = {"$lt" if before else "$gt": [rank_name, anchor_rank]}
            comparisons: list[dict[str, Any]] = [rank_compare]
            if anchor is not None:
                value_operator = "$gt" if key.order.descending == before else "$lt"
                comparisons.append(
                    {
                        "$and": [
                            {"$eq": [rank_name, anchor_rank]},
                            {value_operator: [f"${self._page_field(key)._path}", anchor]},
                        ]
                    }
                )
            choices.append({"$and": [*prefix, {"$or": comparisons}]})
            prefix.append({"$eq": [rank_name, anchor_rank]})
            if anchor is not None:
                prefix.append({"$eq": [f"${self._page_field(key)._path}", anchor]})
        choices.append({"$and": [*prefix, {"$lt" if before else "$gt": ["$_id", cursor.sid]}]})
        return {"$or": choices}

    def _page_total(self) -> int:
        if self._plan._row_verifier is not None:
            pipeline = self._plan._pipeline(apply_window=False)
            return sum(1 for _document in self._plan._verified_documents(pipeline))
        pipeline = self._plan._lookup_stages()
        pipeline.append({"$match": self._plan._truth_filter()})
        pipeline.append({"$count": "count"})
        row = next(iter(self._plan._collection().aggregate(pipeline, **self._plan._store._session_kwargs())), None)
        return 0 if row is None else int(row["count"])

    def _page_tokens(
        self,
        documents: list[dict[str, Any]],
        keys: tuple[_PageKey, ...],
        fingerprint: str,
        cursor: _DecodedContinuation | None,
        more_in_fetch_direction: bool,
    ) -> tuple[ContinuationToken | None, ContinuationToken | None]:
        if not documents:
            return None, None

        def token(document: dict[str, Any], direction: Literal["forward", "backward"]) -> ContinuationToken:
            try:
                return _encode_continuation(
                    direction=direction,
                    anchors=tuple(self._page_anchor(document, key) for key in keys),
                    sid=int(document["_id"]),
                    fingerprint=fingerprint,
                )
            except (TypeError, ValueError) as error:
                raise UnsupportedQueryError(
                    "paging order values cannot be represented in a continuation cursor"
                ) from error

        first, last = documents[0], documents[-1]
        if cursor is None:
            return (token(last, "forward") if more_in_fetch_direction else None, None)
        if cursor.direction == "forward":
            return (token(last, "forward") if more_in_fetch_direction else None, token(first, "backward"))
        return (token(last, "forward"), token(first, "backward") if more_in_fetch_direction else None)

    @staticmethod
    def _page_anchor(document: dict[str, Any], key: _PageKey) -> Any:
        field = MongoResultSet._page_field(key)
        source = _variable_document(document, field._variable)
        if source is None:
            return None
        if field._key_path == "_id":
            return source.get("_id")
        return source.get("f", {}).get(field._key_path.removeprefix("f."))

    def _page_fingerprint(self, keys: tuple[_PageKey, ...]) -> str:
        """Hash every plan property that could change a cursor's meaning."""
        return _plan_fingerprint(self._page_fingerprint_payload(keys))

    def _page_fingerprint_payload(self, keys: tuple[_PageKey, ...]) -> dict[str, Any]:
        """Return the canonical fingerprint context used by a continuation token."""
        root = self._plan._variables[0]
        return {
            "backend": "mongodb",
            "document_layout": _DOCUMENT_LAYOUT,
            "schema": self._page_schema(root._schema),
            "outputs": [
                {
                    "name": output.name,
                    "kind": "object" if isinstance(output.value, MongoVariable) else "scalar",
                    "root": output.value is root
                    if isinstance(output.value, MongoVariable)
                    else output.value._variable is root,
                    "path": None if isinstance(output.value, MongoVariable) else output.value._path,
                    "role": None if isinstance(output.value, MongoVariable) else output.value._spec.role,
                }
                for output in self._outputs
            ],
            "order": [
                {"name": key.order.name, "descending": key.order.descending, "nulls": key.order.nulls} for key in keys
            ],
            "logical_ast": "" if self._plan._row_verifier is None else self._plan._row_verifier_identity,
            "pipeline": self._page_pipeline(keys),
        }

    @staticmethod
    def _page_schema(schema: Any, seen: set[type] | None = None) -> dict[str, Any]:
        """Return the recursive structural schema form shared with SQL paging."""
        seen = set() if seen is None else seen
        cls = schema.cls
        identity = f"{cls.__module__}.{cls.__qualname__}"
        if cls in seen:
            return {"class": identity, "cycle": True}
        seen.add(cls)
        return {
            "class": identity,
            "table": schema.table_name,
            "dedup": schema.dedup,
            "indexes": schema.composite_indexes,
            "fields": [
                {
                    "field": field.field,
                    "role": field.role,
                    "columns": [
                        {
                            "name": column.name,
                            "kind": column.kind,
                            "nullable": column.nullable,
                            "indexed": column.indexed,
                            "unique": column.unique,
                        }
                        for column in field.columns
                    ],
                    "codec": field.codec_name,
                    "shape": None if field.shape is None else (field.shape.rows, field.shape.cols),
                    "target": None
                    if field.target is None
                    else MongoResultSet._page_schema(resolve_schema(field.target), seen),
                }
                for field in schema.fields
            ],
        }

    def _candidate_documents(self, pipeline: list[dict[str, Any]]) -> Iterator[_Candidate]:
        """Yield one shared candidate stream, recording optional verification."""
        for document, verified in self._plan._candidate_documents(pipeline):
            yield _Candidate(document, verified)

    def _document_row(self, document: dict[str, Any]) -> tuple[Any, ...]:
        """Decode this result set's declared outputs from one candidate document."""
        values: list[Any] = []
        for output in self._outputs:
            if isinstance(output.value, MongoVariable):
                source = _variable_document(document, output.value)
                values.append(
                    None if source is None else self._plan._store.fetch(output.value._cls, int(source["_id"]))
                )
            else:
                values.append(_scalar_value(document, output.value))
        return tuple(values)
