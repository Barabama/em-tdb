"""Root conftest — skip tests requiring local data dirs not tracked in git."""

import ast
from pathlib import Path

import pytest

_ROOT = Path(__file__).parent.parent

_DATA_DIRS: dict[str, Path] = {
    "ref_tdb": _ROOT / "ref_tdb",
    "fits-dat": _ROOT / "tests" / "fits-dat",
    "fits-json": _ROOT / "tests" / "fits-json",
}


def _find_dep(item) -> str | None:
    """Return the first missing data directory this test depends on, or None."""
    item_path = str(item.fspath)
    for name, dirpath in _DATA_DIRS.items():
        if not dirpath.exists() and name in item_path:
            return name

    # Parse the test file with AST — collect all identifiers and string literals
    try:
        source = Path(item.fspath).read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(item.fspath))
    except (OSError, SyntaxError):
        return None

    tokens: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            tokens.add(node.id)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            # Add both the full string and each path component
            tokens.add(node.value)
            for part in Path(node.value).parts:
                tokens.add(part)

    for name, dirpath in _DATA_DIRS.items():
        if not dirpath.exists() and name in tokens:
            return name
    return None


def pytest_collection_modifyitems(items):
    for item in items:
        dep = _find_dep(item)
        if dep is not None:
            item.add_marker(
                pytest.mark.skipif(True, reason=f"{dep} not available")
            )
