"""PostgreSQL-specific SQLAlchemy dialect compilation hooks.

The rest of :mod:`httk.store.db` deals in generic SQLAlchemy objects.  This
module is the adapter boundary that rewrites the ``httk_fraction_scaled_equal``
call into inline SQL for the PostgreSQL dialect, because PostgreSQL cannot host
the per-connection Python scalar function that SQLite and DuckDB register.

Fractions are persisted as canonical ``"p/q"`` text (reduced, positive
denominator), but the caller also passes integer-form factors (for example a
constant ``1/2`` supplied as separate value and factor) that carry no ``/q``
part; those take an implicit denominator of ``1``.  Exact rational equality is
therefore evaluated by casting each argument to text, splitting into numerator
and denominator, and cross-multiplying with PostgreSQL's arbitrary-precision
``numeric`` type -- so, unlike ClickHouse's bounded Int256 inline, no digit
budget or zero-denominator guard is needed.  The only theoretical bound is
``numeric``'s ~131072-integer-digit cap, the same practical ceiling ClickHouse's
Int256 inline already accepts; no runtime guard is warranted.
"""

from typing import Any

from sqlalchemy.ext.compiler import compiles
from sqlalchemy.sql.functions import Function


def _fraction_inline(arguments: list[str]) -> str:
    """Render inline exact rational equality over four compiled fraction fragments.

    Each argument is cast to text and split on ``'/'``: the first part is the
    numerator, the second the denominator, defaulting to ``1`` when absent (an
    integer-form factor).  With arguments ``(left_value, left_factor,
    right_value, right_factor)`` the returned expression cross-multiplies to test
    ``left_value * left_factor == right_value * right_factor`` exactly, matching
    :func:`httk.store.db.engine._fraction_scaled_equal`.  A ``NULL`` argument
    propagates through ``split_part`` to a ``NULL`` comparison, matching the
    Python ``None`` result.

    :param arguments: The four compiled SQL fragments, one per fraction argument.
    :return: An inline SQL boolean expression string over PostgreSQL ``numeric``.
    :raises TypeError: If exactly four arguments are not supplied.
    """
    if len(arguments) != 4:
        raise TypeError("httk_fraction_scaled_equal expects four fraction arguments")
    numerators = [f"(split_part(({argument})::text, '/', 1)::numeric)" for argument in arguments]
    denominators = [
        f"(COALESCE(NULLIF(split_part(({argument})::text, '/', 2), ''), '1')::numeric)" for argument in arguments
    ]
    left = " * ".join((numerators[0], numerators[1], denominators[2], denominators[3]))
    right = " * ".join((numerators[2], numerators[3], denominators[0], denominators[1]))
    return f"(({left}) = ({right}))"


@compiles(Function, "postgresql")
def _compile_postgres_function(element: Function, compiler: Any, **kwargs: Any) -> str:
    """Rewrite ``httk_fraction_scaled_equal`` to inline SQL for PostgreSQL.

    :param element: The SQL function element being compiled.
    :param compiler: The active SQLAlchemy statement compiler.
    :param **kwargs: Compiler keyword arguments threaded to argument compilation.
    :return: Inline exact-equality SQL for the fraction function, else the
        default function compilation.
    """
    if element.name == "httk_fraction_scaled_equal":
        arguments = [compiler.process(argument, **kwargs) for argument in element.clause_expr.clauses]
        return _fraction_inline(arguments)
    return compiler.visit_function(element, **kwargs)
