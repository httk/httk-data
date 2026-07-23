import os
import warnings
from datetime import date

from sphinx.deprecation import RemovedInSphinx10Warning
warnings.filterwarnings("ignore", category=RemovedInSphinx10Warning)

project = "httk-placeholder"
author = "The httk-placeholder AUTHORS"
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

# myst-nb config: don't execute notebooks during docs build by default
nb_execution_mode = "off"

html_theme = "furo"
html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
}

# External references resolve against inventories vendored in docs/_inventories/
# so docs builds need no network access; link targets still point at the live
# sites. Refresh the committed inventories with `make docs-inventories`.
#
# When this module cross-references public objects from another httk
# distribution (e.g. httk.core), add it here against the published httk docs
# site. The base URL comes from the DOCS_BASE_URL Makefile variable (exported as
# HTTK_DOCS_BASE_URL); the default below keeps bare sphinx invocations working.
# Vendor each dependency inventory alongside python.inv, for example:
#     "httk-core": (f"{_docs_base_url}/httk-core/", "_inventories/httk-core.inv"),
_docs_base_url = os.environ.get("HTTK_DOCS_BASE_URL", "https://docs.httk.org")

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", "_inventories/python.inv"),
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
]
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

suppress_warnings = ["myst.xref_missing"]

def skip_member(app, what, name, obj, skip, options):
    # Skip private members (those starting with _)
    if name.startswith('_'):
        return True
    return skip

def setup(sphinx):
    sphinx.connect('autoapi-skip-member', skip_member)
