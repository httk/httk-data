"""Session-wide setup shared by this repository's tests.

``tests/test_examples.py`` runs each example script in a *subprocess* whose
working directory is a fresh temporary one, so an example that writes files
cannot pollute the checkout. That interacts badly with one thing: Python
resolves a **relative** ``PYTHONPATH`` entry against each process's own working
directory, not against the directory pytest was started in.

It matters whenever a repository is tested straight from a source checkout
rather than from an install — the sibling httk repositories are developed that
way, with invocations such as ``PYTHONPATH=src:../httk-data/src pytest``. Left
alone, those relative entries would resolve against the temporary directory in
the child process and point at nothing, so every example would fail to import
its own package: a false failure that says nothing about the example.

Absolutizing the inherited entries once, up front, makes them mean what the
caller meant — in this process and in every subprocess it spawns. It is a no-op
when ``PYTHONPATH`` is unset (the installed case, including CI) or when its
entries are already absolute.
"""

import os

_PYTHONPATH = os.environ.get("PYTHONPATH")
if _PYTHONPATH:
    os.environ["PYTHONPATH"] = os.pathsep.join(
        os.path.abspath(entry) if entry else entry for entry in _PYTHONPATH.split(os.pathsep)
    )
