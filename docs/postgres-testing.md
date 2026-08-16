# PostgreSQL testing

The PostgreSQL arm is optional.  Start a local server with the helper target
and export the test URI it prints:

```bash
make postgres-dev-server
export HTTK_TEST_POSTGRES_URI='postgresql+psycopg://postgres:postgres@127.0.0.1:5432/httk'
```

The helper runs the pinned `postgres:16` image as a detached container named
`httk-postgres` on the host network, with `POSTGRES_PASSWORD=postgres` and the
`httk` database, waits for `pg_isready`, and prints the URI to export.  Stop it
with `make postgres-stop`.  A raw equivalent is:

```bash
docker run --detach --name httk-postgres --network host \
  -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=httk postgres:16
```

Any reachable PostgreSQL server works; only the URI matters.

## The driver requirement

`Database.postgres()` supports the psycopg 3 driver only.  A bare
`postgresql://` URL is normalized to `postgresql+psycopg://` (SQLAlchemy 2.0
would otherwise select psycopg2), and any other explicit driver is rejected.
Install the client with the `postgresql` extra, which pins psycopg 3:

```bash
python -m pip install "httk-store[postgresql]"
```

CI installs `.[dev,postgresql,parallel]`; there is no committed constraints
file for PostgreSQL, so the `pyproject.toml` extra range applies directly.

## Skipping when unset

`HTTK_TEST_POSTGRES_URI` must name a reachable admin URI (used to create and
drop a fresh isolated database per test).  When it is unset every PostgreSQL
test skips with a pointer to this setup, so the default suite stays green
without a server.  The parameterized backend suites add PostgreSQL under the
`postgres` xdist group so its per-test databases do not collide.

## Known limitation

Under PostgreSQL **bulk** ingest, a `NaN` inside a stored **list-of-floats
(child) field** is not preserved — it reads back as `NULL`.  Bulk ingest stages
rows through SQLite shards, and SQLite has no `NaN`, so the value is lost in the
list column.  A **scalar** float `NaN` IS preserved under bulk ingest, and the
serial `save()` path preserves `NaN` in both scalar and list fields on all
backends.  See the [database backend details](details/db.md#bulk-ingestion).
