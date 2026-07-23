PYTHON ?= python3
DIST_DIR ?= dist

# Base URL of the published httk documentation site, used for cross-linking docs
# between httk repositories (read by docs/conf.py via HTTK_DOCS_BASE_URL).
DOCS_BASE_URL ?= https://docs.httk.org

.PHONY: docs docs-live docs-clean docs-inventories clean dist-clean dist dist-check release-check format format-check typecheck typecheck_pyright lint test test_fastfail audit

docs: docs-clean
	HTTK_DOCS_BASE_URL=$(DOCS_BASE_URL) $(PYTHON) -m sphinx -E -a -b html -W --keep-going docs docs/_build/html

docs-live:
	HTTK_DOCS_BASE_URL=$(DOCS_BASE_URL) sphinx-autobuild docs docs/_build/html

docs-clean:
	rm -rf docs/_build docs/reference/autoapi

# Refresh the committed intersphinx inventories (the one docs task that uses the
# network); docs builds themselves resolve against these vendored files offline.
docs-inventories:
	curl -fsSL https://docs.python.org/3/objects.inv -o docs/_inventories/python.inv

dist-clean:
	rm -rf build $(DIST_DIR) src/httk_placeholder.egg-info

clean: docs-clean dist-clean
	find . -name "*.pyc" -print0 | xargs -0 rm -f
	find . -name "*~" -print0 | xargs -0 rm -f
	find . -name "__pycache__" -print0 | xargs -0 rm -rf

format:
	$(PYTHON) -m ruff check src examples --select F401 --fix
	$(PYTHON) -m isort src examples
	$(PYTHON) -m black src examples

format-check: lint
	$(PYTHON) -m isort --check-only src examples
	$(PYTHON) -m black --check src examples

lint:
	$(PYTHON) -m ruff check src examples

typecheck_pyright:
	$(PYTHON) -m pyright

typecheck:
	$(PYTHON) -m mypy

test:
	$(PYTHON) -m pytest

test_fastfail:
	$(PYTHON) -m pytest -q -x

check: format-check typecheck typecheck_pyright test

ci: format-check typecheck typecheck_pyright test_fastfail

dist: dist-clean
	$(PYTHON) -m build --outdir $(DIST_DIR)

dist-check: dist
	$(PYTHON) -m twine check --strict $(DIST_DIR)/*

release-check: ci docs dist-check
