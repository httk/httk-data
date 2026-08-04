# Releasing `httk-data`

Releases are built and published by GitHub Actions. PyPI authentication uses
Trusted Publishing, so the repository does not need a stored PyPI API token.

## One-time setup

1. Create accounts on [PyPI](https://pypi.org) and
   [TestPyPI](https://test.pypi.org), and enable two-factor authentication.
2. In the GitHub repository settings, create environments named `pypi` and
   `testpypi`. Configure a required reviewer for `pypi` (and optionally for
   `testpypi`); restricting the `pypi` environment to tags matching `v*` is
   also recommended.
3. On PyPI, add a pending GitHub Trusted Publisher with these values:

   - PyPI project name: `httk-data`
   - Owner: `httk`
   - Repository: `httk-data`
   - Workflow: `release.yml`
   - Environment: `pypi`

4. Add the corresponding pending publisher on TestPyPI, using the environment
   `testpypi` instead.

A pending publisher creates the project during its first upload. It does not
reserve the project name before then.

## Prepare and check a release

Update `project.version` in `pyproject.toml`. After making dependency changes,
regenerate and commit the documentation lock:

```console
make docs-lock
```

`make docs-lock` requires every internal `httk-*` dependency to be published
and resolvable on PyPI at a version satisfying this project's dependency floors.
`make release-check` only verifies an existing lock offline, so it remains
hard-gated until that lock has been generated and committed. Until the
dependencies are published, the development docs workflow uses its explicit
bootstrap-fallback mode: it clones internal dependencies first, emits a warning,
installs those checkouts, and then performs fresh external docs dependency
resolution. Releases remain impossible by design until the lock can be
generated and committed.

Before tagging, refresh and commit the dependency inventories from the exact
versions pinned by that lock:

```console
make docs-inventories
```

The dependency release documentation must already be published at those
versions. `make release-check` validates lock freshness; the workflow's
`check-release` validates the inventory headers against the lock pins. From a
Python 3.12 environment,
install the development tools and run the complete local check:

```console
python -m pip install -e ".[dev,docs,release]"
make release-check
```

`make release-check` includes the offline documentation lock-freshness check,
in addition to formatting, static analysis, tests, strict documentation, an
isolated sdist/wheel build, and strict package-metadata checks. Before tagging,
run `make docs-lock-check` for the required full clean-environment locked
installation and strict docs build; this is a network check. The resulting
package files are written to `dist/`.

Versions on package indexes are immutable. Use a new development or release
candidate version when repeating an upload, for example `0.1.0rc1` followed by
`0.1.0`.

## TestPyPI

Run the **Publish package** workflow manually in GitHub Actions. A manual run
publishes to TestPyPI only. To retry a TestPyPI upload without committing a version bump, pass the
optional `version_suffix` workflow input (e.g. `.post1` or `rc2`); it is
appended to `project.version` for that build only.
When the workflow run has completed (approving the
`testpypi` environment first, if it has a required reviewer), test the artifact
in a fresh environment:

```console
python -m venv /tmp/httk-data-test
/tmp/httk-data-test/bin/python -m pip install \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ httk-data==0.1.0
/tmp/httk-data-test/bin/python -c "import httk.data"
```

Replace `0.1.0` with the version being tested. Unlike `httk-core`, `httk-data`
has a runtime dependency (`httk-core`), so `--no-deps` is not appropriate here:
`import httk.data` pulls in `httk.core` at import time. The
`--extra-index-url` lets pip resolve that dependency (once it is published to the
real PyPI) while the package under test comes from TestPyPI.

## PyPI

1. Confirm that `make release-check` succeeds on the exact commit to release.
2. Push the commit and create a GitHub release whose tag is `v` followed by the
   package version, for example `v0.1.0`.
3. Publish the GitHub release and approve the protected `pypi` environment.
4. Verify the release from a fresh environment with `pip install httk-data`.

The workflow rejects a Git tag that does not match `project.version`, rebuilds
the distributions from the tagged source, checks them, and publishes them via
PyPI Trusted Publishing.
