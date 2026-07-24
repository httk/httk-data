"""Generic translation of OPTIMADE filter syntax trees into backend search expressions.

This module turns the OPTIMADE filter language — parsed by
:func:`httk.core.parse_optimade_filter` into a :py:type:`httk.core.FilterAst`
nested-tuple syntax tree — into :class:`~httk.data.query.SearchExpression`
objects over the backend-agnostic store/searcher protocols of
:mod:`httk.data.query`. It is pure and store-independent: any
:class:`~httk.data.query.Store` whose search variables expose the queried
columns can execute the translated expressions.

The translation is driven by three inputs:

- ``property_fulltypes`` — a minimal mapping from recognized property names to
  their OPTIMADE ``fulltype`` strings (``"string"``, ``"integer"``, ``"float"``,
  ``"boolean"``, ``"timestamp"``, ``"dict"``, ``"unknown"``, or ``"list of
  ..."``), used to type-check and convert filter constants;
- ``handlers`` — a :data:`HandlerTable` mapping property names to the callables
  that build the actual search expressions (see
  :func:`simple_property_handlers` for the generic column-map-driven builder,
  and :func:`relationship_id_handler` for ``<type>.id`` relationship entries);
- ``recognized_prefixes`` — property-name prefixes the serving side claims:
  an unknown property carrying such a prefix is an error
  (``"unrecognized-property"``), while any other unknown property silently
  matches nothing, as the OPTIMADE specification prescribes.

Failures raise :class:`FilterTranslationError`, which carries a neutral
``category`` (no HTTP semantics); consumers map categories onto their
transport's error codes.

**Relationship filtering** (dotted identifiers such as ``references.doi``) is
supported for relationship types named in ``relationship_targets``:
``<type>.id HAS ...`` translates directly through the handler table, and any
other depth-1 dotted filter is resolved by a two-phase semi-join through a
supplied :data:`RelatedPropertyResolver` (see :func:`translate_filter_ast`).
Each dotted filter node is resolved *independently*: in ``references.doi
CONTAINS "x" AND references.year >= 2000``, some related reference must match
the doi condition and some (possibly different) related reference must match
the year condition.
"""

import operator
from typing import Any, Callable, Literal, Mapping

from httk.core import FilterAst, parse_optimade_filter

from httk.data.query import Searcher, SearchExpression, SearchVariable, Store

__all__ = [
    "FilterTranslationCategory",
    "FilterTranslationError",
    "HandlerTable",
    "RelatedPropertyResolver",
    "constant_types",
    "invert_op",
    "format_value",
    "translate_filter_ast",
    "simple_property_handlers",
    "relationship_id_handler",
    "filter_searcher",
    "true_handler",
    "false_handler",
    "unknown_unknown_handler",
    "known_unknown_handler",
    "unknown_comparison_handler",
    "unknown_stringmatching_handler",
    "unknown_has_handler",
    "unknown_length_handler",
    "string_handler",
    "stringmatching_handler",
    "constant_comparison_handler",
    "constant_stringmatching_handler",
    "number_handler",
    "timestamp_handler",
    "set_handler",
    "constant_set_handler",
]

type FilterTranslationCategory = Literal["unrecognized-property", "not-implemented", "type-mismatch", "internal"]
"""Why a filter could not be translated (see :class:`FilterTranslationError`)."""

type HandlerTable = Mapping[str, Mapping[str, Callable[..., Any]]]
"""Per-property translation callables, keyed by property name.

The inner mapping's keys name the filter-operation families: ``'comparison'``
(``=``, ``!=``, ``<``, ``<=``, ``>``, ``>=``), ``'stringmatching'``
(``CONTAINS``/``STARTS``/``ENDS``), ``'HAS'`` (the set operations),
``'length'`` (``LENGTH``), and ``'unknown'`` (``IS KNOWN``/``IS UNKNOWN``).
Dotted ``'<type>.id'`` entries provide relationship-id filtering (see
:func:`relationship_id_handler`).
"""

type RelatedPropertyResolver = Callable[[str, FilterAst], tuple[str, ...]]
"""Resolve a depth-1 relationship filter to the matching related-entry ids.

Called as ``resolver(related_type, sub_ast)`` where ``sub_ast`` is the filter
node with the ``<related_type>.`` prefix stripped from its identifier (and
without any surrounding ``NOT``); returns the ids of the related entries
matching the sub-filter, evaluated against the related type's own properties.
"""


class FilterTranslationError(Exception):
    """A filter cannot be translated into a search expression.

    The exception message describes the failure; :attr:`category` classifies it
    neutrally (this module knows nothing about transports, so consumers map
    each category onto their own error codes):

    - ``"unrecognized-property"`` — the filter names an unknown property
      carrying a recognized prefix (a caller error);
    - ``"type-mismatch"`` — a filter constant does not match the property's
      declared type (a caller error);
    - ``"not-implemented"`` — the filter uses a construct this translation (or
      the supplied handler table) does not support;
    - ``"internal"`` — an inconsistency in the translation itself.

    ``detail`` optionally carries extra machine-readable context.
    """

    def __init__(self, message: str, category: FilterTranslationCategory, detail: str | None = None) -> None:
        super().__init__(message)
        self.category: FilterTranslationCategory = category
        self.detail = detail


constant_types = ['String', 'Number', 'Boolean']

invert_op = {'!=': '!=', '>': '<', '<': '>', '=': '=', '<=': '>=', '>=': '<='}
_python_opmap = {
    '!=': '__ne__',
    '>': '__gt__',
    '<': '__lt__',
    '=': '__eq__',
    '<=': '__le__',
    '>=': '__ge__',
    'STARTS': 'startswith',
    'ENDS': 'endswith',
}


def format_value(fulltype: str, val: tuple[Any, ...], allow_null: bool = False) -> Any:
    """Convert a filter constant node to a Python value of the property's ``fulltype``.

    Raises:
        FilterTranslationError: With category ``"type-mismatch"`` when the
            constant does not fit the declared type, or ``"not-implemented"``
            for dictionary-typed properties.
    """
    if fulltype.startswith('list of '):
        if not isinstance(val[0], tuple):
            raise FilterTranslationError(
                "Type mismatch in filter, query had single value when list of values was expected.",
                "type-mismatch",
            )
        inner_fulltype = fulltype[len('list of ') :]
        outvals = []
        for v in val:
            outvals += [format_value(inner_fulltype, v, allow_null=allow_null)]
        return outvals
    elif allow_null and val[0] == 'Null':
        return None
    elif fulltype == 'boolean':
        if val[0] in ['Boolean']:
            return val[1] == 'TRUE'
    elif fulltype == 'integer':
        if val[0] in ['Number']:
            return int(val[1])
    elif fulltype == 'float':
        if val[0] in ['Number']:
            return float(val[1])
    elif fulltype == 'string':
        if val[0] in ['String']:
            return val[1]
    elif fulltype == 'timestamp':
        if val[0] in ['String']:
            return val[1]
    elif fulltype == 'dict':
        raise FilterTranslationError("Filtering on dictionary properties not implemented.", "not-implemented")
    elif fulltype == 'unknown':
        return val[1]
    raise FilterTranslationError(
        "Type mismatch in filter, expected:" + fulltype + ", query has:" + val[0], "type-mismatch"
    )


# ---------------------------------------------------------------------- generic handlers


def _constant_column(search_variable: SearchVariable) -> Any:
    """A column to build constant true/false expressions from (compared to itself).

    The serving-store convention is the ``hexhash`` column (any column works for
    stores whose variables serve every attribute name); variables of stricter
    backends that reject unknown attribute names (such as the SQL layer's
    :class:`~httk.data.db.searcher.SqlVariable`) fall back to their always
    present ``sid`` column.
    """
    try:
        return getattr(search_variable, 'hexhash')
    except AttributeError:
        return getattr(search_variable, 'sid')


def true_handler(search_variable: SearchVariable) -> SearchExpression:
    column = _constant_column(search_variable)
    return getattr(column, '__eq__')(column)


def false_handler(search_variable: SearchVariable) -> SearchExpression:
    column = _constant_column(search_variable)
    return getattr(column, '__ne__')(column)


def unknown_unknown_handler(entry: str, search_variable: SearchVariable, unknown_type: str) -> SearchExpression:
    if unknown_type == 'IS_UNKNOWN':
        return true_handler(search_variable)
    elif unknown_type == 'IS_KNOWN':
        return false_handler(search_variable)
    raise FilterTranslationError("Unexpected unknown operator type", "internal")


def known_unknown_handler(entry: str, search_variable: SearchVariable, unknown_type: str) -> SearchExpression:
    if unknown_type == 'IS_UNKNOWN':
        return false_handler(search_variable)
    elif unknown_type == 'IS_KNOWN':
        return true_handler(search_variable)
    raise FilterTranslationError("Unexpected unknown operator type", "internal")


def unknown_comparison_handler(entry: str, ops: Any, values: Any, search_variable: SearchVariable) -> SearchExpression:
    return false_handler(search_variable)


def unknown_stringmatching_handler(
    entry: str, values: Any, stringmatching_type: str, search_variable: SearchVariable
) -> SearchExpression:
    return false_handler(search_variable)


def unknown_has_handler(
    entry: str, op: Any, value: Any, search_variable: SearchVariable, has_type: str, inv_toggle: bool
) -> tuple[SearchExpression, bool]:
    return false_handler(search_variable), False


def unknown_length_handler(entry: str, op: str, value: Any, search_variable: SearchVariable) -> SearchExpression:
    return false_handler(search_variable)


def string_handler(entry: str, op: str, value: Any, search_variable: SearchVariable) -> SearchExpression:
    httk_op = _python_opmap[op]
    return getattr(getattr(search_variable, entry), httk_op)(value)


def stringmatching_handler(
    entry: str, value: str, stringmatching_type: str, search_variable: SearchVariable
) -> SearchExpression:
    escaped_value = value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    if stringmatching_type == 'ENDS':
        return getattr(getattr(search_variable, entry), 'like')('%' + escaped_value)
    elif stringmatching_type == 'STARTS':
        return getattr(getattr(search_variable, entry), 'like')(escaped_value + '%')
    elif stringmatching_type == 'CONTAINS':
        return getattr(getattr(search_variable, entry), 'like')('%' + escaped_value + '%')
    else:
        raise FilterTranslationError("Unexpected stringmatching operator type", "internal")


def constant_comparison_handler(val1: Any, op: str, val2: Any, search_variable: SearchVariable) -> SearchExpression:
    if getattr(operator, _python_opmap[op])(val1, val2):
        return true_handler(search_variable)
    else:
        return false_handler(search_variable)


def constant_stringmatching_handler(
    val1: Any, val2: Any, stringmatching_type: str, search_variable: SearchVariable
) -> SearchExpression:
    if getattr(val1, _python_opmap[stringmatching_type])(val2):
        return true_handler(search_variable)
    else:
        return false_handler(search_variable)


def number_handler(entry: str, op: str, value: Any, search_variable: SearchVariable) -> SearchExpression:
    httk_op = _python_opmap[op]
    return getattr(getattr(search_variable, entry), httk_op)(value)


def timestamp_handler(entry: str, op: str, value: Any, search_variable: SearchVariable) -> SearchExpression:
    raise FilterTranslationError("Timestamp comparison not yet implemented.", "not-implemented")


def set_handler(
    entry: str, ops: Any, values: Any, inv: bool, has_type: str, search_variable: SearchVariable
) -> tuple[SearchExpression, bool]:
    if has_type == 'HAS_ALL':
        if not inv:
            search = getattr(getattr(search_variable, entry), 'has_any')(values[0])
            for value in values[1:]:
                search = search & (getattr(getattr(search_variable, entry), 'has_any')(value))
            return search, False
        else:
            search = getattr(getattr(search_variable, entry), 'has_inv_any')(values[0])
            for value in values[1:]:
                search = search & (getattr(getattr(search_variable, entry), 'has_inv_any')(value))
            return search, True
    elif has_type == 'HAS_ANY':
        if not inv:
            return getattr(getattr(search_variable, entry), 'has_any')(*values), False
        else:
            return getattr(getattr(search_variable, entry), 'has_inv_any')(*values), True
    elif has_type == 'HAS_ONLY':
        if not inv:
            return getattr(getattr(search_variable, entry), 'has_only')(*values), True
        else:
            return getattr(getattr(search_variable, entry), 'has_inv_only')(*values), True
    raise FilterTranslationError("Unexpected set operator type: " + str(has_type), "internal")


def constant_set_handler(
    val1: Any, ops: Any, val2: Any, has_type: str, inv: bool, search_variable: SearchVariable
) -> tuple[SearchExpression, bool]:
    if has_type == 'HAS_ALL':
        if set(val2) <= set(val1):
            return true_handler(search_variable), False
        else:
            return false_handler(search_variable), False
    elif has_type == 'HAS_ANY':
        if set(val2).isdisjoint(val1):
            return false_handler(search_variable), False
        else:
            return true_handler(search_variable), False
    elif has_type == 'HAS_ONLY':
        if set(val1) <= set(val2):
            return true_handler(search_variable), False
        else:
            return false_handler(search_variable), False
    raise FilterTranslationError("Unexpected set operator type: " + str(has_type), "internal")


# ---------------------------------------------------------------------- the translation


def translate_filter_ast(
    node: FilterAst,
    search_variable: SearchVariable,
    entry_type: str,
    property_fulltypes: Mapping[str, str],
    handlers: HandlerTable,
    recognized_prefixes: tuple[str, ...],
    inv_toggle: bool = False,
    *,
    recursion: int = 0,
    relationship_targets: tuple[str, ...] = (),
    related_property_resolver: RelatedPropertyResolver | None = None,
) -> tuple[SearchExpression, bool]:
    """Translate one filter syntax-tree node into a search expression.

    ``node`` is a :py:type:`~httk.core.FilterAst` node (as produced by
    :func:`httk.core.parse_optimade_filter`); ``search_variable`` is the
    backend search variable the expression is built against;
    ``property_fulltypes``, ``handlers``, and ``recognized_prefixes`` drive the
    translation as described in the module docstring. ``inv_toggle`` tracks
    whether the node sits under an odd number of ``NOT``\\ s (relevant only for
    the set operations, whose inverse forms differ); ``recursion`` counts the
    nesting depth.

    Returns ``(expression, needs_post)``: the translated expression and whether
    it must *additionally* be applied in post-filter position
    (:meth:`~httk.data.query.Searcher.add_all`) because it contains set
    operations whose plain (:meth:`~httk.data.query.Searcher.add`) rendering is
    incomplete.

    **Relationship filtering:** an identifier dotted with a name in
    ``relationship_targets`` (e.g. ``('Identifier', 'references', 'doi')``)
    filters on the properties of related entries. ``<type>.id HAS ...`` is
    served directly by the ``'<type>.id'`` handler-table entry. Every other
    depth-1 dotted filter — comparisons (including ``<type>.id = ...``),
    string matching, ``IS KNOWN``/``IS UNKNOWN``, and the HAS family over
    related list properties — is resolved by a two-phase semi-join: the
    ``<type>.`` prefix is stripped from the node, the resulting sub-filter is
    passed to ``related_property_resolver`` (without any surrounding ``NOT`` —
    inversion applies to the resulting id-set membership), and the returned ids
    are substituted back as ``<type>.id HAS ANY <ids>`` (an empty id set
    translates to a constant-false expression). Each dotted node is resolved
    **independently**: ``references.doi CONTAINS "x" AND references.year >=
    2000`` matches entries where *some* related reference matches the doi
    condition and *some* — possibly different — related reference matches the
    year condition. Without a resolver, dotted filters other than ``<type>.id
    HAS ...`` are not implemented; nested (deeper than depth-1) paths and
    dotted ``LENGTH`` filters are never supported.

    Raises:
        FilterTranslationError: See :class:`FilterTranslationError` for the
            failure categories.
    """

    def recurse(sub_node: FilterAst, sub_inv_toggle: bool) -> tuple[SearchExpression, bool]:
        return translate_filter_ast(
            sub_node,
            search_variable,
            entry_type,
            property_fulltypes,
            handlers,
            recognized_prefixes,
            sub_inv_toggle,
            recursion=recursion + 1,
            relationship_targets=relationship_targets,
            related_property_resolver=related_property_resolver,
        )

    def relationship_semi_join(left: tuple[Any, ...], sub_ast: FilterAst) -> tuple[SearchExpression, bool]:
        """Resolve a dotted (relationship-property) node via the two-phase semi-join."""
        if related_property_resolver is None:
            raise FilterTranslationError(
                "Filtering on relationship " + ".".join(left[1:]) + " not implemented.", "not-implemented"
            )
        if len(left) != 3:
            raise FilterTranslationError(
                "Filtering on relationship " + ".".join(left[1:]) + " not implemented.", "not-implemented"
            )
        ids = related_property_resolver(left[1], sub_ast)
        if not ids:
            return false_handler(search_variable), False
        rewritten: FilterAst = (
            'HAS_ANY',
            ('=',) * len(ids),
            ('Identifier', left[1], 'id'),
            tuple(('String', related_id) for related_id in ids),
        )
        return recurse(rewritten, inv_toggle)

    search_expr: SearchExpression | None = None
    needs_post = False

    if node[0] in ['AND']:
        search_expr, needs_post = recurse(node[1], inv_toggle)
        rhs_search_expr, rhs_needs_post = recurse(node[2], inv_toggle)
        needs_post = needs_post or rhs_needs_post
        search_expr = search_expr & rhs_search_expr
    elif node[0] in ['OR']:
        search_expr, needs_post = recurse(node[1], inv_toggle)
        rhs_search_expr, rhs_needs_post = recurse(node[2], inv_toggle)
        needs_post = needs_post or rhs_needs_post
        search_expr = search_expr | rhs_search_expr
    elif node[0] in ['NOT']:
        search_expr, needs_post = recurse(node[1], not inv_toggle)
        search_expr = ~search_expr
    elif node[0] in ['HAS_ALL', 'HAS_ANY', 'HAS_ONLY']:
        ops = node[1]
        left = node[2]
        right = node[3]
        assert left[0] == 'Identifier'
        has_handler: Callable[..., Any] | None
        if len(left) > 2 and left[1] in relationship_targets:
            # Filtering on a relationship, e.g. `references.id HAS "ref-1"`.
            if len(left) == 3 and left[2] == 'id':
                rel_key = left[1] + '.id'
                has_handler = handlers.get(rel_key, {}).get('HAS')
                if has_handler is None:
                    raise FilterTranslationError(
                        "Filtering on relationship " + rel_key + " not implemented.", "not-implemented"
                    )
                values = format_value('list of string', right)
                if ops != tuple(['='] * len(values)):
                    raise FilterTranslationError(
                        "HAS queries with non-equal operators not implemented yet.", "not-implemented"
                    )
                search_expr, needs_post = has_handler(rel_key, ops, values, search_variable, node[0], inv_toggle)
                assert search_expr is not None
                return search_expr, needs_post
            return relationship_semi_join(left, (node[0], ops, ('Identifier',) + tuple(left[2:]), right))
        if left[1] not in property_fulltypes:
            if left[1].startswith(recognized_prefixes):
                raise FilterTranslationError(
                    "Filter invokes unrecognized property name: " + left[1], "unrecognized-property"
                )
            else:
                # TODO: this should warn
                has_handler = unknown_has_handler
                values = format_value('list of unknown', right)
        else:
            values = format_value(property_fulltypes[left[1]], right)
            has_handler = handlers.get(left[1], {}).get('HAS')
            if has_handler is None:
                raise FilterTranslationError(
                    "Filtering on property " + left[1] + " not implemented.", "not-implemented"
                )
        if ops != tuple(['='] * len(values)):
            raise FilterTranslationError("HAS queries with non-equal operators not implemented yet.", "not-implemented")
        search_expr, needs_post = has_handler(left[1], ops, values, search_variable, node[0], inv_toggle)
    elif node[0] in ['LENGTH']:
        left = node[1]
        op = node[2]
        right = node[3]
        assert left[0] == 'Identifier'
        if len(left) > 2 and left[1] in relationship_targets:
            raise FilterTranslationError(
                "Filtering on relationship " + ".".join(left[1:]) + " not implemented.", "not-implemented"
            )
        if right[0] == 'Identifier':
            raise FilterTranslationError(
                "LENGTH comparisons with non-constant right hand side not implemented.", "not-implemented"
            )
        if right[0] != 'Number':
            raise FilterTranslationError(
                "LENGTH comparison can only be done with Numbers. Unexpected right hand side type:" + right[0],
                "not-implemented",
            )
        length_handler: Callable[..., Any] | None
        if left[1] not in property_fulltypes:
            if left[1].startswith(recognized_prefixes):
                raise FilterTranslationError(
                    "Filter invokes unrecognized property name: " + left[1], "unrecognized-property"
                )
            else:
                # TODO: this should warn
                length_handler = unknown_length_handler
                value = format_value('unknown', right)
        else:
            length_handler = handlers.get(left[1], {}).get('length')
            if length_handler is None:
                raise FilterTranslationError(
                    "Filtering on property " + left[1] + " not implemented.", "not-implemented"
                )
            assert property_fulltypes[left[1]].startswith("list of ")
            value = format_value("integer", right)
        search_expr = length_handler(left[1], op, value, search_variable)
    elif node[0] in ['>', '>=', '<', '<=', '=', '!=']:
        op = node[0]
        left = node[1]
        right = node[2]
        if (left[0] == 'Boolean' or right[0] == 'Boolean') and op not in ('=', '!='):
            raise FilterTranslationError(
                "Boolean values only support the = and != comparison operators.", "not-implemented"
            )
        if left[0] in constant_types and right[0] in constant_types:
            raise FilterTranslationError("Constant vs. Constant comparisons not implemented.", "not-implemented")
        elif left[0] == 'Identifier' and right[0] == 'Identifier':
            raise FilterTranslationError("Identifier vs. Identifier comparisons not implemented.", "not-implemented")
        else:
            if right[0] == 'Identifier' and left[0] in constant_types:
                left, right = right, left
                op = invert_op[op]
            assert left[0] == 'Identifier'
            if len(left) > 2 and left[1] in relationship_targets:
                return relationship_semi_join(left, (op, ('Identifier',) + tuple(left[2:]), right))
            comparison_handler: Callable[..., Any] | None
            if left[1] not in property_fulltypes:
                if left[1].startswith(recognized_prefixes):
                    raise FilterTranslationError(
                        "Filter invokes unrecognized property name: " + left[1], "unrecognized-property"
                    )
                else:
                    # TODO: this should warn
                    comparison_handler = unknown_comparison_handler
                    value = format_value('unknown', right)
            else:
                comparison_handler = handlers.get(left[1], {}).get('comparison')
                if comparison_handler is None:
                    raise FilterTranslationError(
                        "Filtering on property " + left[1] + " not implemented.", "not-implemented"
                    )
                value = format_value(property_fulltypes[left[1]], right)
            search_expr = comparison_handler(left[1], op, value, search_variable)
    elif node[0] in ['ENDS', 'STARTS', 'CONTAINS']:
        left = node[1]
        right = node[2]
        assert left[0] == 'Identifier'
        if len(left) > 2 and left[1] in relationship_targets:
            return relationship_semi_join(left, (node[0], ('Identifier',) + tuple(left[2:]), right))
        if right[0] == 'Identifier':
            raise FilterTranslationError(
                "Identifier vs. Identifier string comparisons not implemented.", "not-implemented"
            )
        stringmatching: Callable[..., Any] | None
        if left[1] not in property_fulltypes:
            if left[1].startswith(recognized_prefixes):
                raise FilterTranslationError(
                    "Filter invokes unrecognized property name: " + left[1], "unrecognized-property"
                )
            else:
                # TODO: this should warn
                stringmatching = unknown_stringmatching_handler
                value = format_value('unknown', right)
        else:
            stringmatching = handlers.get(left[1], {}).get('stringmatching')
            if stringmatching is None:
                raise FilterTranslationError(
                    "Filtering on property " + left[1] + " not implemented.", "not-implemented"
                )
            value = format_value(property_fulltypes[left[1]], right)
        search_expr = stringmatching(left[1], value, node[0], search_variable)
    elif node[0] in ['IS_UNKNOWN', 'IS_KNOWN']:
        left = node[1]
        assert left[0] == 'Identifier'
        if len(left) > 2 and left[1] in relationship_targets:
            return relationship_semi_join(left, (node[0], ('Identifier',) + tuple(left[2:])))
        if left[1] not in property_fulltypes:
            if left[1].startswith(recognized_prefixes):
                raise FilterTranslationError(
                    "Filter invokes unrecognized property name: " + left[1], "unrecognized-property"
                )
            else:
                # TODO: this should warn
                unknown = unknown_unknown_handler
        else:
            unknown = handlers[left[1]]['unknown']
        search_expr = unknown(left[1], search_variable, node[0])
    else:
        raise FilterTranslationError("Unexpected translation error at: " + str(node[0]), "internal")
    assert search_expr is not None
    return search_expr, needs_post


# ---------------------------------------------------------------------- handler builders


def simple_property_handlers(
    entry_type: str, columns: Mapping[str, str], property_fulltypes: Mapping[str, str]
) -> dict[str, Mapping[str, Callable[..., Any]]]:
    """Build a filter handler table for an entry type from a column map.

    Always provides the standard ``id`` (matched against the ``__id`` column)
    and ``type`` (a constant equal to ``entry_type``) handlers. For every
    property named in ``columns`` (which maps property names to backend column
    names), handlers are generated from the property's fulltype in
    ``property_fulltypes`` (default ``"string"``): string properties get
    comparison and stringmatching handlers; integer and float properties get a
    numeric comparison handler; ``list of ...`` properties get a HAS (set
    membership) handler. Every generated property also gets a ``known``
    unknown handler.
    """
    handlers: dict[str, Mapping[str, Callable[..., Any]]] = {
        'id': {
            'comparison': lambda entry, op, value, sv: string_handler('__id', op, value, sv),
            'unknown': known_unknown_handler,
            'stringmatching': lambda entry, value, smtype, sv: stringmatching_handler('__id', value, smtype, sv),
        },
        'type': {
            'comparison': lambda entry, op, value, sv: constant_comparison_handler(value, op, entry_type, sv),
            'unknown': known_unknown_handler,
            'stringmatching': lambda entry, value, smtype, sv: constant_stringmatching_handler(
                value, entry_type, smtype, sv
            ),
        },
    }
    for name, column in columns.items():
        fulltype = property_fulltypes.get(name, 'string')
        table: dict[str, Callable[..., Any]] = {'unknown': known_unknown_handler}
        if fulltype.startswith('list of '):
            table['HAS'] = lambda entry, ops, values, sv, has_type, inv, col=column: set_handler(
                col, ops, values, inv, has_type, sv
            )
        elif fulltype in ('integer', 'float'):
            table['comparison'] = lambda entry, op, value, sv, col=column: number_handler(col, op, value, sv)
        elif fulltype == 'timestamp':
            # Timestamps are RFC 3339 strings; lexicographic comparison is
            # correct for same-format UTC timestamps, so string_handler applies.
            # No stringmatching handler: substring matching on timestamps is not
            # meaningful.
            table['comparison'] = lambda entry, op, value, sv, col=column: string_handler(col, op, value, sv)
        else:
            table['comparison'] = lambda entry, op, value, sv, col=column: string_handler(col, op, value, sv)
            table['stringmatching'] = lambda entry, value, smtype, sv, col=column: stringmatching_handler(
                col, value, smtype, sv
            )
        handlers[name] = table
    return handlers


def relationship_id_handler(rel_column: str) -> Mapping[str, Callable[..., Any]]:
    """A ``'<type>.id'`` handler-table entry matching related ids over a list column.

    ``rel_column`` names a list-valued backend column holding the related entry
    ids; the returned ``{'HAS': ...}`` mapping serves ``<type>.id HAS ...``
    filters (and the semi-join rewrites of other dotted filters) with the
    standard set-operation semantics of :func:`~httk.data.optimade_query.set_handler`.
    """
    return {
        'HAS': lambda entry, ops, values, sv, has_type, inv: set_handler(rel_column, ops, values, inv, has_type, sv),
    }


# ---------------------------------------------------------------------- searcher sugar


def filter_searcher(
    store: Store,
    target: Any,
    filter_string: str | FilterAst,
    *,
    entry_type: str,
    property_fulltypes: Mapping[str, str],
    columns: Mapping[str, str] | None = None,
    handlers: HandlerTable | None = None,
    recognized_prefixes: tuple[str, ...] = (),
    relationship_targets: tuple[str, ...] = (),
    related_property_resolver: RelatedPropertyResolver | None = None,
) -> Searcher:
    """Build a :class:`~httk.data.query.Searcher` over ``store`` applying an OPTIMADE filter.

    ``filter_string`` is an OPTIMADE filter string (parsed with
    :func:`httk.core.parse_optimade_filter`) or an already-parsed
    :py:type:`~httk.core.FilterAst`. The searcher binds one search variable to
    ``target`` (the store-specific query target, declared as the searcher
    output named ``entry_type``) and applies the translated filter. When
    ``handlers`` is not supplied, a default table is built with
    :func:`simple_property_handlers` from ``columns`` (or, when ``columns`` is
    also None, from an identity column map over ``property_fulltypes``). The
    remaining keyword arguments are passed through to
    :func:`translate_filter_ast`.

    Raises:
        FilterTranslationError: When the filter cannot be translated.
        httk.core.ParserSyntaxError: When a filter string does not parse.
    """
    filter_ast: FilterAst = parse_optimade_filter(filter_string) if isinstance(filter_string, str) else filter_string
    if handlers is None:
        if columns is None:
            columns = {name: name for name in property_fulltypes}
        handlers = simple_property_handlers(entry_type, columns, property_fulltypes)
    searcher = store.searcher()
    search_variable = searcher.variable(target)
    searcher.output(search_variable, entry_type)
    search_expr, needs_post = translate_filter_ast(
        filter_ast,
        search_variable,
        entry_type,
        property_fulltypes,
        handlers,
        recognized_prefixes,
        False,
        relationship_targets=relationship_targets,
        related_property_resolver=related_property_resolver,
    )
    searcher.add(search_expr)
    if needs_post:
        searcher.add_all(search_expr)
    return searcher
