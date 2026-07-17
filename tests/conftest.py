"""Root conftest — skip tests requiring ref_tdb when it's not tracked."""

import inspect
from pathlib import Path

import pytest

REF_TDB_DIR = Path(__file__).parent.parent / "ref_tdb"


def _uses_ref_tdb(item) -> bool:
    if "ref_tdb" in str(item.fspath):
        return True
    if "REF_TDB" in getattr(item, "fixturenames", ()):
        return True
    # Check the test source AND any class it belongs to
    if hasattr(item, "obj"):
        try:
            src = inspect.getsource(item.obj)
            if "REF_TDB_DIR" in src or "ref_tdb" in src:
                return True
        except OSError:
            pass
        # Also check class-level source for helper methods
        cls = getattr(item, "cls", None)
        if cls is not None:
            try:
                cls_src = inspect.getsource(cls)
                if "REF_TDB_DIR" in cls_src:
                    return True
            except OSError:
                pass
    return False


def pytest_collection_modifyitems(items):
    for item in items:
        if _uses_ref_tdb(item):
            item.add_marker(
                pytest.mark.skipif(not REF_TDB_DIR.exists(), reason="ref_tdb not available")
            )
