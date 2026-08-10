# ClickHouse testing

The ClickHouse arm is optional.  Start the local helper with:

```bash
make clickhouse-dev-server
export HTTK_TEST_CLICKHOUSE_URI='clickhousedb://default:@127.0.0.1:28123/httk_p5'
```

The helper downloads the AMD64 static package pinned by
`CLICKHOUSE_STATIC_VERSION` (currently `26.7.3.19`) into the ignored,
version-keyed `.clickhouse-dev/` directory.  It verifies the committed SHA-256
file and runs `clickhouse --version` before launch.  The committed
`tests/clickhouse/config.d/httk.xml` supplies loopback ports 29000/28123,
embedded Keeper ports 29181/29234, the `/httk` KeeperMap prefix, and a
4,500,000,000-byte ClickHouse allocator cap.  The server runs under a separate
7 GiB memguard allowance.  Stop it with `make clickhouse-stop`.

CI uses the pinned `clickhouse/clickhouse-server:26.8.1.1028` image and the
same XML fragment.  Its validated client environment is committed in
[`constraints/clickhouse-p5.txt`](../constraints/clickhouse-p5.txt):
ClickHouse Connect 1.6.0, SQLAlchemy 2.0.48, PyArrow 25.0.0, and the recorded
transitive co-pins.  The `pyproject.toml` extra intentionally remains a
flexible developer range.

## Required deployment bootstrap

Keeper is a hard requirement.  Provision `_httk_bootstrap` in the deployment
`default` database and each database opened by httk before opening the backend.
The required DDL is:

```sql
CREATE TABLE _httk_bootstrap (key String, value String)
ENGINE=KeeperMap('/_httk_bootstrap') PRIMARY KEY key
```

The helper and CI retry native SQL, a Keeper query, database creation, and
this bootstrap provisioning sequence together before declaring the service
ready.

## Memory accounting

The standard no-server client ceiling is 8 GiB on a 16 GiB machine; the
extended no-server ceiling is 24 GiB on a 24 GiB machine.  With ClickHouse
running, its 7 GiB allowance is subtracted from the selected machine ceiling:

```text
standard: 7 GiB server + 9 GiB client memguard = 16 GiB total
extended: 7 GiB server + 17 GiB client memguard = 24 GiB total
CI:       7 GiB container server + 9 GiB client memguard = 16 GiB total
```

With `HTTK_TEST_CLICKHOUSE_URI` set, `make test` and `make test-extended`
select 9 GiB and 17 GiB respectively unless `HTTK_TEST_MAX_RSS_GB` is
explicitly supplied.  An override replaces the client number; for example,
`HTTK_TEST_MAX_RSS_GB=12` means a possible 7 + 12 = 19 GiB combined ceiling
and is suitable only on a host/container with that larger envelope.  CI uses
9 GiB explicitly so its 7 GiB server plus client guard fits the 16 GiB runner
envelope.  The server's 4.5 GB allocator cap is separate from its 7 GiB
process-group allowance.

## Manual recovery

Only recover after verifying that the former writer is dead.  Inspect the
exact value first, then use strict mode for every value-conditioned delete:

```sql
SELECT key, value FROM _httk_store_metadata WHERE key = 'lease';
SET keeper_map_strict_mode = 1;
DELETE FROM _httk_store_metadata
WHERE key = 'lease' AND value = '<observed lease JSON>';
```

Never clear `ingest_state` merely because the lease was removed.  A marker can
mean physical tables are partial; the default recovery is to drop the
affected database, recreate `_httk_bootstrap`, and re-ingest.  Only after a
verified cleanup/rebuild may an operator remove the exact marker value:

```sql
SELECT key, value FROM _httk_store_metadata WHERE key = 'ingest_state';
SET keeper_map_strict_mode = 1;
DELETE FROM _httk_store_metadata
WHERE key = 'ingest_state' AND value = '<observed marker JSON>';
```

Bootstrap-lock residue follows the same rule: inspect the exact UUID key/value,
verify the writer is dead, then execute `SET keeper_map_strict_mode = 1;`
before the exact `_httk_bootstrap` delete.  Never use a key-only delete or
delete a live writer's value.  See the [database backend details](details/db.md#clickhouse-bulk-fenced-writes)
for the lifecycle and failure semantics.
