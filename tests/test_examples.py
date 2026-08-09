"""Smoke test: every script under ``examples/`` runs standalone to a clean exit.

The examples are also the source of the generated docs pages (see the
``builder-inited`` hook in ``docs/conf.py``), so a broken example is a broken
documentation page. This test is deliberately repo-agnostic: it discovers the
scripts rather than listing them, and the two things an example may need to say
about itself are declared *in the example*, as module-level constants:

``HTTK_EXAMPLE_NO_AUTORUN = True``
    "Do not run me unattended." For examples that bind a port, wait for input,
    or otherwise never return (e.g. a server that calls ``serve_forever()``).

``HTTK_EXAMPLE_REQUIRES = ["numpy", ...]``
    "Only run me when these optional imports are available." Missing ones make
    the example skip rather than fail.

Both are read straight out of the file's *source* with :mod:`ast` — the example
is never imported into the test process, so declaring ``NO_AUTORUN`` is enough
to guarantee the example cannot hang the test run even at import time. A
declaration in the example (rather than a hard-coded path list here) also keeps
this file byte-identical across repos and puts the fact next to the code it
describes, where the author of a new server example will see it.
"""

import ast
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"

#: Wall-clock budget for one example. A hung example must fail the run, not block it.
EXAMPLE_TIMEOUT_SECONDS = 120

NO_AUTORUN_SENTINEL = "HTTK_EXAMPLE_NO_AUTORUN"
REQUIRES_SENTINEL = "HTTK_EXAMPLE_REQUIRES"


def discover_examples() -> list[Path]:
    """Return every example script, recursing into subdirectories."""
    if not EXAMPLES_DIR.is_dir():
        return []
    return sorted(
        path for path in EXAMPLES_DIR.rglob("*.py") if path.name != "__init__.py" and "__pycache__" not in path.parts
    )


def _module_constants(source: str) -> dict[str, object]:
    """Evaluate module-level assignments of literal values, without importing."""
    constants: dict[str, object] = {}
    for node in ast.parse(source).body:
        if not isinstance(node, ast.Assign):
            continue
        try:
            value = ast.literal_eval(node.value)
        except (ValueError, TypeError, SyntaxError):
            continue  # not a literal (a call, a name, ...) -- not a declaration
        for target in node.targets:
            if isinstance(target, ast.Name):
                constants[target.id] = value
    return constants


def _example_id(path: Path) -> str:
    return path.relative_to(EXAMPLES_DIR).as_posix()


def test_examples_are_discovered() -> None:
    """Guard against a silently empty parametrization (drop in a repo with no examples)."""
    assert discover_examples(), f"no example scripts found under {EXAMPLES_DIR}"


@pytest.mark.parametrize("example", discover_examples(), ids=_example_id)
def test_example_runs_cleanly(example: Path, tmp_path: Path) -> None:
    constants = _module_constants(example.read_text(encoding="utf-8"))

    if constants.get(NO_AUTORUN_SENTINEL):
        pytest.skip(f"{_example_id(example)} declares {NO_AUTORUN_SENTINEL} (run it by hand)")

    requirements = constants.get(REQUIRES_SENTINEL) or ()
    if isinstance(requirements, (list, tuple)):
        for requirement in requirements:
            if importlib.util.find_spec(str(requirement)) is None:
                pytest.skip(f"{_example_id(example)} requires the optional dependency {requirement!r}")

    # Subprocess, not import: an example is a *script*, and a separate process is
    # the only way a timeout can actually stop a runaway one. cwd is a tmp_path so
    # examples that write files cannot pollute the working tree.
    try:
        result = subprocess.run(
            [sys.executable, str(example)],
            capture_output=True,
            text=True,
            timeout=EXAMPLE_TIMEOUT_SECONDS,
            cwd=tmp_path,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(
            f"{_example_id(example)} did not finish within {EXAMPLE_TIMEOUT_SECONDS}s; "
            f"if it is meant to run indefinitely, declare {NO_AUTORUN_SENTINEL} = True in it"
        )

    assert result.returncode == 0, f"{_example_id(example)} exited {result.returncode}\n{result.stderr}"
    # Every example is meant to show something; silence means it demonstrated nothing.
    assert result.stdout.strip(), f"{_example_id(example)} printed nothing"
