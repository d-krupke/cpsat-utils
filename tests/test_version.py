"""
Ensure __version__ stays in sync with pyproject.toml.
"""

from pathlib import Path

import cpsat_utils


def test_version_matches_pyproject():
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    for line in pyproject.read_text().splitlines():
        if line.startswith("version"):
            pyproject_version = line.split('"')[1]
            break
    else:
        raise RuntimeError("version not found in pyproject.toml")
    assert cpsat_utils.__version__ == pyproject_version, (
        f"__version__ ({cpsat_utils.__version__}) != "
        f"pyproject.toml ({pyproject_version})"
    )
