# ClickHouse test server

The ClickHouse arm is optional.  Without `HTTK_TEST_CLICKHOUSE_URI`, its
tests skip; the shared skip message points here and to the Makefile helper:

```bash
make clickhouse-dev-server
export HTTK_TEST_CLICKHOUSE_URI='clickhousedb://default:@127.0.0.1:28123/httk_p5'
```

`clickhouse-dev-server` downloads the pinned AMD64 static binary into the
ignored, version-keyed `.clickhouse-dev/` tree, verifies its committed
SHA-256 and reported version, uses `config.d/httk.xml` as its server
configuration, and starts the server
under `python -m httk.core.memguard --max-rss-gb 7`.  The committed config
keeps the server on loopback: native port 29000, HTTP port 28123, and
embedded Keeper ports 29181/29234.  Its `max_server_memory_usage` is
4,500,000,000 bytes (4.5 GB), while memguard's 7 GiB allowance is the real
process-group safety limit.  The server is deliberately outside the pytest
process tree, but that allowance remains part of the machine-safety budget.
Use `make clickhouse-stop` when finished.

The static binary has no distro configuration tree.  The helper therefore
uses the committed XML as its server configuration and rewrites filesystem
paths into `.clickhouse-dev/` and supplies the committed passwordless,
loopback-only `users.xml`; a packaged install can instead mount or copy
`config.d/httk.xml` as `/etc/clickhouse-server/config.d/httk.xml`.  The same
fragment is mounted by the CI Docker job, which uses the image's user config.

## Required deployment bootstrap

Keeper is a hard requirement.  Before opening any httk database, provision
the deployment bootstrap table in that database.  The exact DDL is:

```sql
CREATE TABLE _httk_bootstrap (key String, value String)
ENGINE=KeeperMap('/_httk_bootstrap') PRIMARY KEY key
```

The helper provisions this table in both `default` (the deployment-presence
gate used by the tests) and `httk_p5`.  CI does the equivalent provisioning
inside the pinned container.  Fresh per-test databases are created by the
fixtures and receive the same DDL before they are opened.

## Recovery

ClickHouse writes are lease-fenced and nontransactional.  `steal_lease()` is
not available in v1.  For a dead writer, first verify that the process is
really gone, then inspect and remove only the exact observed lease value:

```sql
SELECT key, value FROM _httk_store_metadata WHERE key = 'lease';
SET keeper_map_strict_mode = 1;
DELETE FROM _httk_store_metadata
WHERE key = 'lease' AND value = '<observed lease JSON>';
```

Never clear `ingest_state` just because the lease was removed.  A marker can
mean that physical tables are partial; the default recovery is to drop the
affected database, recreate its bootstrap table, and re-ingest.  Only after a
verified cleanup/rebuild may an operator delete the exact observed marker
value.  Do not delete a live writer's value or use a key-only delete.

For marker recovery, first inspect `ingest_state`, rebuild or drop/re-ingest as
described above, then use the same strict exact-value sequence:

```sql
SELECT key, value FROM _httk_store_metadata WHERE key = 'ingest_state';
SET keeper_map_strict_mode = 1;
DELETE FROM _httk_store_metadata
WHERE key = 'ingest_state' AND value = '<observed marker JSON>';
```

The full setup, memory arithmetic, lease, marker, bootstrap-lock, and cleanup
procedures are maintained in the [canonical ClickHouse testing guide](../../docs/clickhouse-testing.md).
