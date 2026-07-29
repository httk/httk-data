import ast
import os
import warnings
from datetime import date
from pathlib import Path

from sphinx.deprecation import RemovedInSphinx10Warning
warnings.filterwarnings("ignore", category=RemovedInSphinx10Warning)

project = "httk-data"
author = "The httk-data AUTHORS"
copyright = f"{date.today().year}, {author}"

extensions = [
    # Core API docs
    "sphinx.ext.autodoc",        # pull docstrings
    "sphinx.ext.autosummary",    # API summary tables + stub gen
    "sphinx.ext.napoleon",       # Google/NumPy docstrings
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",        # math rendering via MathJax

    # Nice-to-haves
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",

    # Markdown + notebooks
    "myst_nb",                   # .ipynb support

    "autoapi.extension",
    "httk.core.docs.sphinx_ext",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "**/.ipynb_checkpoints"]

# Autosummary: generate stub pages automatically
autosummary_generate = True

# Autodoc defaults (tweak to taste)
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "undoc-members": False,
    "show-inheritance": True,
}
autodoc_typehints = "signature"
typehints_fully_qualified = False
typehints_document_rtype = True
typehints_defaults = "comma"
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_attr_annotations = True

# MyST / Markdown configuration (math + nice syntax)
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
    "substitution",
    "tasklist",
    "dollarmath",  # enables $...$ and $$...$$
]
myst_heading_anchors = 3

# Execute notebooks during the docs build and cache the results, so a notebook
# is verified rather than merely rendered: a cell that raises fails the build.
# Everything this needs (jupyter-cache, nbclient, ipykernel) already comes with
# myst-nb, so the "docs" extra needs nothing added. The cache lives under
# docs/_build, which `make docs-clean` removes.
nb_execution_mode = "cache"
nb_execution_raise_on_error = True

html_theme = "furo"
html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
}

# External references resolve against inventories vendored in docs/_inventories/
# so docs builds need no network access; link targets still point at the live
# sites. Refresh the committed inventories with `make docs-inventories`.
#
# httk-data builds on public httk-core objects (it serves httk-core's record
# models through the httk.core.EntryProvider contract and validates httk-core
# PropertyDefinitions), so cross-project references resolve against the published
# httk documentation site. The base URL comes from the DOCS_BASE_URL Makefile
# variable (exported as HTTK_DOCS_BASE_URL); the default below keeps bare sphinx
# invocations working.
_docs_base_url = os.environ.get("HTTK_DOCS_BASE_URL", "https://docs.httk.org")

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", "_inventories/python.inv"),
    "httk-core": (f"{_docs_base_url}/httk-core/", "_inventories/httk-core.inv"),
}

autoapi_options = [
       "members",
       "undoc-members",
       "show-inheritance",
       "show-module-summary",
       "imported-members",
]
autoapi_root = "reference/autoapi"
autoapi_ignore = []  # include everything

autoapi_type = "python"
autoapi_dirs = ["../src/httk"]
autoapi_add_toctree_entry = True
autoapi_keep_files = True
autoapi_member_order = "bysource"
autoapi_python_class_content = "module"  # docstring under class, not merged from __init__
autoapi_python_use_implicit_namespaces = True
autoapi_template_dir = "_templates/autoapi"

nitpicky = True
nitpick_ignore = [
    ("py:class", "typing.Any"),
    ("py:class", "typing.Optional"),
    ("py:class", "typing.Union"),
    ("py:class", "Ellipsis"),
    # sqlalchemy is an optional dependency ([db] extra) whose docs inventory is
    # not vendored; the SQL layer's internal-facing signatures reference it.
    ("py:class", "sqlalchemy.Engine"),
    ("py:class", "sqlalchemy.MetaData"),
    ("py:class", "sqlalchemy.Table"),
    ("py:class", "sqlalchemy.ColumnElement"),
    ("py:class", "sqlalchemy.FromClause"),
    # PEP 695 method type parameters (e.g. SqlStore.fetch[T]) are not classes.
    ("py:class", "T"),
    # AutoAPI renders the value of the RelatedPropertyResolver type alias with a
    # bare (unqualified) FilterAst class xref; the name comes from httk-core (a
    # separate distribution sharing the "httk" namespace), so it cannot resolve
    # here. Qualified httk.core.FilterAst references resolve via intersphinx.
    ("py:class", "FilterAst"),
    # RowHydrator's constructor takes the module-private hydration context; the
    # private class is deliberately undocumented, so the annotation xref cannot
    # resolve.
    ("py:class", "_Context"),
]
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

# The real cross-project references to httk-core objects (e.g. httk.core.EntryProvider
# base classes, PropertyDefinition/EntryTypeDefinition annotations) are resolved
# structurally via the httk-core intersphinx inventory above. The remaining
# "autoapi.python_import_resolution" notice is only AutoAPI's static parser being
# unable to follow the httk.core import: httk.core lives in a separate distribution
# that shares the PEP 420 "httk" namespace, so it is not among the source trees
# AutoAPI parses here. There is no source-level remedy (building on the httk-core
# contract is the intended design), so this specific subtype is suppressed while
# all reference checking stays strict.
suppress_warnings = ["myst.xref_missing", "autoapi.python_import_resolution"]

def skip_member(app, what, name, obj, skip, options):
    # Skip private members (those starting with _)
    if name.startswith('_'):
        return True
    return skip

# --- Generated example pages -------------------------------------------------
# One docs page per script in the repo's examples/ tree, written at
# builder-inited (i.e. before Sphinx reads sources). The module docstring
# becomes the page title (first line) plus prose; the code *below* the
# docstring is literal-included with an explicit ":lines: N-", so the docstring
# is never repeated inside the code block. Output mirrors the examples/
# directory layout, so nested examples cannot collide. Globbing "*.py" is what
# skips README.md and *.pyc; __init__.py and __pycache__ are skipped
# explicitly. Repo-agnostic: only paths relative to this conf.py are used.
# docs/examples/ is generated, gitignored, and removed by `make docs-clean`.
_EXAMPLES_SRC = Path(__file__).resolve().parent.parent / "examples"
_EXAMPLES_OUT = Path(__file__).resolve().parent / "examples"


def generate_example_pages(app):
    _EXAMPLES_OUT.mkdir(parents=True, exist_ok=True)
    sources = sorted(_EXAMPLES_SRC.rglob("*.py")) if _EXAMPLES_SRC.is_dir() else []
    entries = []
    for src in sources:
        if src.name == "__init__.py" or "__pycache__" in src.parts:
            continue
        text = src.read_text(encoding="utf-8")
        module = ast.parse(text)
        docstring = ast.get_docstring(module)  # cleandoc'ed: dedented, stripped
        # The docstring, when present, is always module.body[0]; code follows it.
        code_start = module.body[0].end_lineno + 1 if docstring is not None else 1
        lines = (docstring or "").splitlines()
        title = lines[0].strip() if lines else src.stem
        prose = "\n".join(lines[1:]).strip()
        has_code = any(line.strip() for line in text.splitlines()[code_start - 1 :])
        rel = src.relative_to(_EXAMPLES_SRC).with_suffix("")
        out = _EXAMPLES_OUT / (rel.as_posix() + ".md")
        out.parent.mkdir(parents=True, exist_ok=True)
        include = os.path.relpath(src, out.parent).replace(os.sep, "/")
        # An empty example (or one that is only a docstring) gets no code block:
        # literalinclude warns when a line spec pulls in nothing, and -W is fatal.
        code = f"```{{literalinclude}} {include}\n:language: python\n:lines: {code_start}-\n```" if has_code else ""
        blocks = [f"# {title}", prose, code]
        out.write_text("\n\n".join(block for block in blocks if block) + "\n", encoding="utf-8")
        entries.append(rel.as_posix())
    toctree = "```{toctree}\n:maxdepth: 1\n\n" + "\n".join(entries) + "\n```\n" if entries else ""
    intro = "Runnable scripts from the repository's `examples/` directory.\n"
    (_EXAMPLES_OUT / "index.md").write_text(f"# Examples\n\n{intro}\n{toctree}", encoding="utf-8")


def setup(sphinx):
    sphinx.connect('autoapi-skip-member', skip_member)
    sphinx.connect('builder-inited', generate_example_pages)
