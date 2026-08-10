PYTHON ?= python3
DIST_DIR ?= dist

# Base URL of the published httk documentation site, used for cross-linking docs
# between httk repositories (read by docs/conf.py via HTTK_DOCS_BASE_URL).
DOCS_BASE_URL ?= https://docs.httk.org

.PHONY: docs docs-live docs-clean docs-inventories docs-lock docs-lock-check clean dist-clean dist dist-check release-check format format-check typecheck typecheck_pyright lint test test_fastfail test-extended test-extended-fastfail benchmarks audit clickhouse-dev-server clickhouse-stop

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
	check_dir=$$(mktemp -d "${TMPDIR:-/tmp}/httk-store-docs-lock-check.XXXXXX"); \
	trap 'rm -rf "$$check_dir"' EXIT; \
	env -u PYTHONPATH -u PYTHONHOME $(PYTHON) -m venv "$$check_dir/venv"; \
	env -u PYTHONPATH -u PYTHONHOME "$$check_dir/venv/bin/python" -m pip install -r docs/requirements.lock; \
	env -u PYTHONPATH -u PYTHONHOME "$$check_dir/venv/bin/python" -m pip check; \
	env -u PYTHONPATH -u PYTHONHOME "$$check_dir/venv/bin/python" -m pip install . --no-deps --no-build-isolation; \
	env -u HTTK_DOCS_VERSION -u PYTHONPATH -u PYTHONHOME HTTK_DOCS_BASE_URL="$(DOCS_BASE_URL)" \
		"$$check_dir/venv/bin/python" -m sphinx -E -a -b html -W --keep-going docs "$$check_dir/html"

dist-clean:
	rm -rf build $(DIST_DIR) src/httk_store.egg-info

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

MEMGUARD = $(PYTHON) -m httk.core.memguard --max-rss-gb $(or $(HTTK_TEST_MAX_RSS_GB),$(1)) --
TEST_DUCKDB_MEMORY_BUDGET_MB = $(or $(HTTK_DUCKDB_TEST_MEMORY_BUDGET_MB),3072)
EXTENDED_DUCKDB_MEMORY_BUDGET_MB = $(or $(HTTK_DUCKDB_TEST_MEMORY_BUDGET_MB),16384)
TEST_TIMEOUT_SECONDS ?= 900
EXTENDED_TEST_TIMEOUT_SECONDS ?= 1200
BENCHMARK_TIMEOUT_SECONDS ?= 1800

# ClickHouse is an opt-in developer service, never part of test/check/ci.
CLICKHOUSE_VERSION ?= 26.8.1.1028
CLICKHOUSE_STATIC_VERSION ?= 26.7.3.19
CLICKHOUSE_DEV_ROOT ?= .clickhouse-dev
CLICKHOUSE_DEV_DIR = $(CLICKHOUSE_DEV_ROOT)/$(CLICKHOUSE_STATIC_VERSION)
CLICKHOUSE_DOWNLOAD_URL ?= https://packages.clickhouse.com/tgz/stable/clickhouse-common-static-$(CLICKHOUSE_STATIC_VERSION)-amd64.tgz
CLICKHOUSE_DEV_BINARY = $(CLICKHOUSE_DEV_DIR)/clickhouse
CLICKHOUSE_DEV_CONFIG = $(CLICKHOUSE_DEV_DIR)/config.xml
CLICKHOUSE_DEV_PID = $(CLICKHOUSE_DEV_DIR)/server.pid
CLICKHOUSE_DEV_LOG = $(CLICKHOUSE_DEV_DIR)/server.log
CLICKHOUSE_DEV_ARCHIVE = $(CLICKHOUSE_DEV_DIR)/clickhouse-common-static-$(CLICKHOUSE_STATIC_VERSION)-amd64.tgz
CLICKHOUSE_DEV_CHECKSUM = tests/clickhouse/clickhouse-common-static-$(CLICKHOUSE_STATIC_VERSION)-amd64.tgz.sha256
CLICKHOUSE_DEV_HTTP_PORT ?= 28123
CLICKHOUSE_DEV_NATIVE_PORT ?= 29000
CLICKHOUSE_DEV_DATABASE ?= httk_p5
CLICKHOUSE_SERVER_MEMGUARD_GB ?= 7
CLICKHOUSE_STANDARD_TOTAL_GB ?= 16
CLICKHOUSE_EXTENDED_TOTAL_GB ?= 24
CLICKHOUSE_STANDARD_CLIENT_MEMGUARD_GB := $(shell expr $(CLICKHOUSE_STANDARD_TOTAL_GB) - $(CLICKHOUSE_SERVER_MEMGUARD_GB))
CLICKHOUSE_EXTENDED_CLIENT_MEMGUARD_GB := $(shell expr $(CLICKHOUSE_EXTENDED_TOTAL_GB) - $(CLICKHOUSE_SERVER_MEMGUARD_GB))

define TEST_MEMGUARD
$(PYTHON) -m httk.core.memguard --max-rss-gb $(or $(HTTK_TEST_MAX_RSS_GB),$(if $(HTTK_TEST_CLICKHOUSE_URI),$(if $(filter 24,$(1)),$(CLICKHOUSE_EXTENDED_CLIENT_MEMGUARD_GB),$(CLICKHOUSE_STANDARD_CLIENT_MEMGUARD_GB)),$(1))) --
endef

clickhouse-dev-server:
	@set -eu; \
	mkdir -p "$(CLICKHOUSE_DEV_DIR)"; \
	if [ ! -f "$(CLICKHOUSE_DEV_ARCHIVE)" ]; then curl -fL --retry 3 "$(CLICKHOUSE_DOWNLOAD_URL)" -o "$(CLICKHOUSE_DEV_ARCHIVE)"; fi; \
	( cd "$(CLICKHOUSE_DEV_DIR)" && sha256sum -c "$(CURDIR)/$(CLICKHOUSE_DEV_CHECKSUM)" ); \
	if [ ! -x "$(CLICKHOUSE_DEV_BINARY)" ]; then \
		tar -xzf "$(CLICKHOUSE_DEV_ARCHIVE)" -C "$(CLICKHOUSE_DEV_DIR)"; \
		binary="$$(find "$(CLICKHOUSE_DEV_DIR)" -type f -path '*/usr/bin/clickhouse' -print -quit)"; \
		test -n "$$binary"; ln -sfn "$$(readlink -f "$$binary")" "$(CLICKHOUSE_DEV_BINARY)"; \
	fi; \
	version_output="$$("$(CLICKHOUSE_DEV_BINARY)" --version)"; \
	echo "$$version_output" | grep -Fq "$(CLICKHOUSE_STATIC_VERSION)" || { echo "ClickHouse binary version does not match $(CLICKHOUSE_STATIC_VERSION): $$version_output" >&2; exit 1; }; \
	cp tests/clickhouse/config.d/httk.xml "$(CLICKHOUSE_DEV_CONFIG)"; \
	sed -i \
		-e 's#<tcp_port replace="replace">29000</tcp_port>#<tcp_port replace="replace">$(CLICKHOUSE_DEV_NATIVE_PORT)</tcp_port>#' \
		-e 's#<http_port replace="replace">28123</http_port>#<http_port replace="replace">$(CLICKHOUSE_DEV_HTTP_PORT)</http_port>#' \
		"$(CLICKHOUSE_DEV_CONFIG)"; \
	cp tests/clickhouse/users.xml "$(CLICKHOUSE_DEV_DIR)/users.xml"; \
	mkdir -p "$(CLICKHOUSE_DEV_DIR)/data" "$(CLICKHOUSE_DEV_DIR)/tmp" "$(CLICKHOUSE_DEV_DIR)/user_files" \
		"$(CLICKHOUSE_DEV_DIR)/keeper/log" "$(CLICKHOUSE_DEV_DIR)/keeper/snapshots"; \
	python_path="$$(readlink -f "$$(command -v "$(PYTHON)")")"; binary_path="$$(readlink -f "$(CLICKHOUSE_DEV_BINARY)")"; config_path="$$(readlink -f "$(CLICKHOUSE_DEV_CONFIG)")"; \
	clickhouse_process_matches() { \
		pid="$$1"; test -r "/proc/$$pid/cmdline" || return 1; \
		cmdline="$$(tr '\\0' ' ' < "/proc/$$pid/cmdline")"; \
		case "$$cmdline" in *"$$python_path -m httk.core.memguard "*"$$binary_path"*"server"*"--config-file"*"$$config_path"*) return 0;; esac; \
		return 1; \
	}; \
	if [ -f "$(CLICKHOUSE_DEV_PID)" ]; then \
		pid="$$(cat "$(CLICKHOUSE_DEV_PID)")"; \
		case "$$pid" in ''|*[!0-9]*) echo "Refusing ClickHouse start: invalid PID file $(CLICKHOUSE_DEV_PID)" >&2; exit 1;; esac; \
		if kill -0 "$$pid" 2>/dev/null; then \
			clickhouse_process_matches "$$pid" || { echo "Refusing ClickHouse start: PID $$pid is not the expected memguard/binary/config process; inspect $(CLICKHOUSE_DEV_PID)" >&2; exit 1; }; \
			echo "ClickHouse is already running (pid $$pid)"; \
		else \
			echo "Refusing ClickHouse start: stale PID file $(CLICKHOUSE_DEV_PID) names $$pid; remove it only after verifying the process is gone" >&2; exit 1; \
		fi; \
	else \
		"$$python_path" -m httk.core.memguard --max-rss-gb "$(CLICKHOUSE_SERVER_MEMGUARD_GB)" -- \
			"$$binary_path" server --config-file "$$config_path" \
			--path="$$(pwd)/$(CLICKHOUSE_DEV_DIR)/data/" \
			--tmp_path="$$(pwd)/$(CLICKHOUSE_DEV_DIR)/tmp/" \
			--user_files_path="$$(pwd)/$(CLICKHOUSE_DEV_DIR)/user_files/" \
			--user_directories.users_xml.path="$$(pwd)/$(CLICKHOUSE_DEV_DIR)/users.xml" \
			--logger.log="$$(pwd)/$(CLICKHOUSE_DEV_LOG)" \
			--logger.errorlog="$$(pwd)/$(CLICKHOUSE_DEV_DIR)/error.log" \
			--keeper_server.log_storage_path="$$(pwd)/$(CLICKHOUSE_DEV_DIR)/keeper/log/" \
			--keeper_server.snapshot_storage_path="$$(pwd)/$(CLICKHOUSE_DEV_DIR)/keeper/snapshots/" \
			>"$(CLICKHOUSE_DEV_LOG)" 2>&1 & echo $$! > "$(CLICKHOUSE_DEV_PID)"; \
	fi; \
	client="$(CLICKHOUSE_DEV_BINARY)"; \
	bootstrap() { \
		curl -fsS "http://127.0.0.1:$(CLICKHOUSE_DEV_HTTP_PORT)/ping" >/dev/null && \
		"$$client" client --host 127.0.0.1 --port "$(CLICKHOUSE_DEV_NATIVE_PORT)" --query 'SELECT 1' >/dev/null && \
		"$$client" client --host 127.0.0.1 --port "$(CLICKHOUSE_DEV_NATIVE_PORT)" --query "SELECT count() FROM system.zookeeper WHERE path = '/'" >/dev/null && \
		"$$client" client --host 127.0.0.1 --port "$(CLICKHOUSE_DEV_NATIVE_PORT)" --query "CREATE DATABASE IF NOT EXISTS $(CLICKHOUSE_DEV_DATABASE)" && \
		for database in default "$(CLICKHOUSE_DEV_DATABASE)"; do \
			if ! "$$client" client --host 127.0.0.1 --port "$(CLICKHOUSE_DEV_NATIVE_PORT)" --database "$$database" --query "EXISTS TABLE _httk_bootstrap" | grep -q '^1$$'; then \
				"$$client" client --host 127.0.0.1 --port "$(CLICKHOUSE_DEV_NATIVE_PORT)" --database "$$database" --query "CREATE TABLE _httk_bootstrap (key String, value String) ENGINE=KeeperMap('/_httk_bootstrap') PRIMARY KEY key"; \
			fi; \
		 done; \
		return 0; \
	}; \
	for attempt in $$(seq 1 60); do if bootstrap; then exit 0; fi; sleep 1; done; \
	echo "ClickHouse native/Keeper/bootstrap readiness failed" >&2; tail -200 "$(CLICKHOUSE_DEV_LOG)"; exit 1

clickhouse-stop:
	@set -eu; \
	if [ -f "$(CLICKHOUSE_DEV_PID)" ]; then \
		pid="$$(cat "$(CLICKHOUSE_DEV_PID)")"; \
		python_path="$$(readlink -f "$$(command -v "$(PYTHON)")")"; binary_path="$$(readlink -f "$(CLICKHOUSE_DEV_BINARY)")"; config_path="$$(readlink -f "$(CLICKHOUSE_DEV_CONFIG)")"; \
		case "$$pid" in ''|*[!0-9]*) echo "Refusing ClickHouse stop: invalid PID file $(CLICKHOUSE_DEV_PID)" >&2; exit 1;; esac; \
		cmdline="$$(tr '\\0' ' ' < "/proc/$$pid/cmdline" 2>/dev/null || true)"; \
		case "$$cmdline" in *"$$python_path -m httk.core.memguard "*"$$binary_path"*"server"*"--config-file"*"$$config_path"*) ;; *) echo "Refusing ClickHouse stop: PID $$pid is stale or not the expected memguard/binary/config process; inspect $(CLICKHOUSE_DEV_PID)" >&2; exit 1;; esac; \
		kill "$$pid"; \
		for attempt in $$(seq 1 30); do kill -0 "$$pid" 2>/dev/null || break; sleep 1; done; \
		if kill -0 "$$pid" 2>/dev/null; then \
			cmdline="$$(tr '\\0' ' ' < "/proc/$$pid/cmdline" 2>/dev/null || true)"; \
			case "$$cmdline" in *"$$python_path -m httk.core.memguard "*"$$binary_path"*"server"*"--config-file"*"$$config_path"*) kill -KILL "$$pid";; *) echo "Refusing ClickHouse stop: PID $$pid changed identity; not signaling it" >&2; exit 1;; esac; \
		fi; \
		rm -f "$(CLICKHOUSE_DEV_PID)"; \
	else echo "ClickHouse is not running"; fi

test:
	HTTK_DUCKDB_TEST_MEMORY_BUDGET_MB=$(TEST_DUCKDB_MEMORY_BUDGET_MB) timeout --foreground $(TEST_TIMEOUT_SECONDS) $(call TEST_MEMGUARD,8) $(PYTHON) -m pytest

test_fastfail:
	HTTK_DUCKDB_TEST_MEMORY_BUDGET_MB=$(TEST_DUCKDB_MEMORY_BUDGET_MB) timeout --foreground $(TEST_TIMEOUT_SECONDS) $(call TEST_MEMGUARD,8) $(PYTHON) -m pytest -q -x

test-extended:
	HTTK_TEST_PROFILE=extended HTTK_DUCKDB_TEST_MEMORY_BUDGET_MB=$(EXTENDED_DUCKDB_MEMORY_BUDGET_MB) timeout --foreground $(EXTENDED_TEST_TIMEOUT_SECONDS) $(call TEST_MEMGUARD,24) $(PYTHON) -m pytest -q -m ""

test-extended-fastfail:
	HTTK_TEST_PROFILE=extended HTTK_DUCKDB_TEST_MEMORY_BUDGET_MB=$(EXTENDED_DUCKDB_MEMORY_BUDGET_MB) timeout --foreground $(EXTENDED_TEST_TIMEOUT_SECONDS) $(call TEST_MEMGUARD,24) $(PYTHON) -m pytest -q -m "" -x

benchmarks:
	timeout --foreground $(BENCHMARK_TIMEOUT_SECONDS) $(call MEMGUARD,12) $(PYTHON) benchmarks/bench50_parallel.py --workers 1 4 --replicate 5 --mode distinct --finalize parity
	timeout --foreground $(BENCHMARK_TIMEOUT_SECONDS) $(call MEMGUARD,12) $(PYTHON) benchmarks/bench50_parallel.py --workers 1 4 --replicate 5 --mode distinct --finalize deferred
	@if [ -n "$$HTTK_TEST_CLICKHOUSE_URI" ]; then \
		timeout --foreground $(BENCHMARK_TIMEOUT_SECONDS) $(call MEMGUARD,12) $(PYTHON) benchmarks/bench50_parallel.py --backend clickhouse --workers 1 4 --replicate 5 --mode distinct --finalize deferred; \
	else \
		echo "Skipping ClickHouse benchmarks: set HTTK_TEST_CLICKHOUSE_URI to enable them"; \
	fi

check: format-check typecheck typecheck_pyright test

ci: format-check typecheck typecheck_pyright test_fastfail

dist: dist-clean
	$(PYTHON) -m build --outdir $(DIST_DIR)

dist-check: dist
	$(PYTHON) -m twine check --strict $(DIST_DIR)/*

release-check: ci docs dist-check
	$(PYTHON) -m httk.core.docs lock-check
