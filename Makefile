PYTHON ?= python3
DIST_DIR ?= dist

# Base URL of the published httk documentation site, used for cross-linking docs
# between httk repositories (read by docs/conf.py via HTTK_DOCS_BASE_URL).
DOCS_BASE_URL ?= https://docs.httk.org

.PHONY: docs docs-live docs-clean docs-inventories docs-lock docs-lock-check clean dist-clean dist dist-check release-check format format-check typecheck typecheck_pyright lint test test_fastfail test-extended test-extended-fastfail audit

docs: docs-clean
	HTTK_DOCS_BASE_URL=$(DOCS_BASE_URL) $(PYTHON) -m sphinx -E -a -b html -W --keep-going docs docs/_build/html

docs-live:
	HTTK_DOCS_BASE_URL=$(DOCS_BASE_URL) sphinx-autobuild docs docs/_build/html

docs-clean:
	rm -rf docs/_build docs/reference/autoapi docs/examples

# Refresh the committed intersphinx inventories (the one docs task that uses the
# network); docs builds themselves resolve against these vendored files offline.
docs-inventories:
	curl -fsSL https://docs.python.org/3/objects.inv -o docs/_inventories/python.inv
	# Requires a committed, current docs/requirements.lock; dependency release docs must be published.
	$(PYTHON) -m httk.core.docs lock-check
	$(PYTHON) -m httk.core.docs refresh-inventories --base-url $(DOCS_BASE_URL) --channel release
# Regenerate the portable documentation lock (network target).
docs-lock:
	$(PYTHON) -m httk.core.docs lock

# Verify the lock in a clean environment and run the strict documentation build
# (network target; the lock installation and build are intentionally transparent).
docs-lock-check: docs-clean
	@set -eu; \
	check_dir=$$(mktemp -d "${TMPDIR:-/tmp}/httk-data-docs-lock-check.XXXXXX"); \
	trap 'rm -rf "$$check_dir"' EXIT; \
	env -u PYTHONPATH -u PYTHONHOME $(PYTHON) -m venv "$$check_dir/venv"; \
	env -u PYTHONPATH -u PYTHONHOME "$$check_dir/venv/bin/python" -m pip install -r docs/requirements.lock; \
	env -u PYTHONPATH -u PYTHONHOME "$$check_dir/venv/bin/python" -m pip check; \
	env -u PYTHONPATH -u PYTHONHOME "$$check_dir/venv/bin/python" -m pip install . --no-deps --no-build-isolation; \
	env -u HTTK_DOCS_VERSION -u PYTHONPATH -u PYTHONHOME HTTK_DOCS_BASE_URL="$(DOCS_BASE_URL)" \
		"$$check_dir/venv/bin/python" -m sphinx -E -a -b html -W --keep-going docs "$$check_dir/html"

dist-clean:
	rm -rf build $(DIST_DIR) src/httk_data.egg-info

clean: docs-clean dist-clean
	find . -name "*.pyc" -print0 | xargs -0 rm -f
	find . -name "*~" -print0 | xargs -0 rm -f
	find . -name "__pycache__" -print0 | xargs -0 rm -rf

format:
	$(PYTHON) -m ruff check src examples --fix
	$(PYTHON) -m ruff format src examples

format-check: lint
	$(PYTHON) -m ruff format --check src examples

lint:
	$(PYTHON) -m ruff check src examples
	pydoclint --quiet src

typecheck_pyright:
	$(PYTHON) -m pyright

typecheck:
	$(PYTHON) -m mypy

test:
	$(PYTHON) -m pytest

test_fastfail:
	$(PYTHON) -m pytest -q -x

test-extended:
	HTTK_TEST_PROFILE=extended $(PYTHON) -m pytest -q -m ""

test-extended-fastfail:
	HTTK_TEST_PROFILE=extended $(PYTHON) -m pytest -q -m "" -x

check: format-check typecheck typecheck_pyright test

ci: format-check typecheck typecheck_pyright test-extended-fastfail

dist: dist-clean
	$(PYTHON) -m build --outdir $(DIST_DIR)

dist-check: dist
	$(PYTHON) -m twine check --strict $(DIST_DIR)/*

release-check: ci docs dist-check
	$(PYTHON) -m httk.core.docs lock-check
