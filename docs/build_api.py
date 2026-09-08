"""Regenerate the API reference under ``docs/api/`` with pdoc.

Run from the repository root::

    python docs/build_api.py

The output (``docs/api/*.html`` + ``search.js``) is git-ignored; Quarto picks
up ``docs/api/`` as a static resource and ``docs/api/api.qmd`` embeds it in an
iframe.

Every module is listed explicitly rather than letting pdoc walk the package,
because ``trimes.__all__`` (and ``trimes.synchrophasor.__all__``) would
otherwise hide most submodules from discovery. pdoc prints a harmless
"module ... has already been added" warning for each submodule that a parent's
``__all__`` re-exports; the pages are still generated correctly.
"""

from __future__ import annotations

import pathlib
import shutil
import subprocess
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
PACKAGE_ROOT = REPO_ROOT / "src" / "trimes"
OUTPUT_DIR = REPO_ROOT / "docs" / "api"

# Implementation details / backward-compatible shims -- no dedicated page.
EXCLUDE = {
    "trimes._optional",
    "trimes.synchrophasor.signal",  # alias of trimes.analytic
    "trimes.synchrophasor.util",  # alias of trimes.analytic
}


def discover_modules() -> list[str]:
    modules: list[str] = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        parts = list(path.relative_to(PACKAGE_ROOT.parent).with_suffix("").parts)
        if "__pycache__" in parts:
            continue
        if parts[-1] == "__init__":
            parts.pop()
        name = ".".join(parts)
        if name and name not in EXCLUDE:
            modules.append(name)
    return modules


def main() -> int:
    modules = discover_modules()

    # Clean previous output so removed modules don't leave stale pages behind.
    # `api.qmd` is the only hand-written file in the directory; keep it.
    for item in OUTPUT_DIR.iterdir():
        if item.name == "api.qmd":
            continue
        shutil.rmtree(item) if item.is_dir() else item.unlink()

    cmd = [
        sys.executable,
        "-m",
        "pdoc",
        *modules,
        "--docformat",
        "numpy",
        "--math",
        "--output-directory",
        str(OUTPUT_DIR),
    ]
    print("$", " ".join(cmd))
    return subprocess.run(cmd, cwd=REPO_ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
