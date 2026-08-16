# Test tiers

`pytest` and `make test` run the standard merge suite. It is deliberately
limited to quick correctness coverage: it should finish in minutes and stay
within roughly 8 GiB total RSS on a 16 GiB machine. Full-depth parameter cases
are marked `extended` and are skipped by this default profile.

Run `make test-extended` after larger implementation work. It includes every
test case and has an approximately ten-minute budget. Both targets run under
the `python -m httk.core.memguard` process-group guard supplied by
*httk-core*; `HTTK_TEST_MAX_RSS_GB` can override its limit when diagnosing a
failure.

ClickHouse is an opt-in test arm.  The exact server/client memory arithmetic,
the 7 GiB server allowance, the 4.5 GB allocator cap, setup, required
KeeperMap bootstrap DDL, and recovery procedures are in the [ClickHouse
testing guide](clickhouse-testing.md).  Start it with `make
clickhouse-dev-server` and stop it with `make clickhouse-stop`.

PostgreSQL is another opt-in test arm.  Set `HTTK_TEST_POSTGRES_URI` to a
reachable admin URI (`postgresql+psycopg://` driver) and the parameterized
backend suites include it; leave it unset and every PostgreSQL test skips.  The
[PostgreSQL testing guide](postgres-testing.md) covers the server setup and the
one known bulk-ingest limitation.  Start a local server with `make
postgres-dev-server` and stop it with `make postgres-stop`.

Performance investigations are separate from correctness testing. `make
benchmarks` runs the opt-in harnesses in `benchmarks/`; benchmark code is not
collected by pytest and is never invoked by `test`, `check`, or `ci`.
