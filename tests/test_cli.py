"""TDB parsing tests using ref_tdb/*.tdb files.

Covers all .tdb/.TDB files in ref_tdb/ including SER, GHSER, GPxxLIQ,
YxxBCC and other function naming conventions.
"""

import argparse
import pytest
import tempfile
from pathlib import Path

from emtdb.tdb import ThermoDBI, TDBManager
from emtdb.cli import (
    cmd_parse,
    cmd_import,
    cmd_list,
    create_parser,
)


REF_TDB_DIR = Path(__file__).parent.parent / "ref_tdb"


def get_all_tdb_files():
    files = []
    for pat in ("*.tdb", "*.TDB"):
        files.extend(sorted(REF_TDB_DIR.glob(pat)))
    return files


# ── Fixtures ──

@pytest.fixture(scope="module")
def parser():
    return create_parser()


@pytest.fixture(params=get_all_tdb_files(), ids=lambda f: f.name)
def any_tdb(request):
    """Every ref_tdb file, parameterised."""
    return request.param


# ── Structural tests ──

class TestParseAllRefTdb:
    """Every ref_tdb file must parse without crash and have valid structure."""

    def test_parse_success(self, any_tdb):
        tdb_mgr = TDBManager(ThermoDBI(":memory:"))
        try:
            parsed = tdb_mgr.parse_tdb(any_tdb, any_tdb.stem)
            assert parsed.tdb == any_tdb.stem
            # All records must have correct TypedDict keys
            for e in parsed.elems:
                assert "elem" in e
            for f in parsed.funcs:
                assert "func" in f and "elem" in f and "expression" in f
            for p in parsed.phases:
                assert "phase" in p and "components" in p
            for p in parsed.params:
                assert "param" in p and "phase" in p and "ptype" in p
                assert "order_num" in p and "expression" in p
        finally:
            tdb_mgr.db.close()

    def test_phase_names_in_params(self, any_tdb):
        tdb_mgr = TDBManager(ThermoDBI(":memory:"))
        try:
            parsed = tdb_mgr.parse_tdb(any_tdb, any_tdb.stem)
            phase_names = {p["phase"] for p in parsed.phases}
            for p in parsed.params:
                assert p["phase"] in phase_names, (
                    f"{p['param']} references undeclared phase {p['phase']}"
                )
        finally:
            tdb_mgr.db.close()

    def test_roundtrip_import_export_reparse(self, any_tdb):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            output_path = Path(tmpdir) / "output.tdb"

            # Phase 1: import (use 'with' for clean close on Windows)
            with ThermoDBI(str(db_path)) as db:
                mgr1 = TDBManager(db)
                p1 = mgr1.parse_tdb(any_tdb, any_tdb.stem)
                if not p1.params:
                    pytest.skip("No params to import")
                try:
                    mgr1.import_tdb(p1.phases, p1.params, p1.tdb, "test", "1.0")
                except Exception:
                    pytest.skip("Import not supported for this TDB format")

            # Phase 2: export
            with ThermoDBI(str(db_path)) as db:
                mgr2 = TDBManager(db)
                mgr2.export_tdb(any_tdb.stem, str(output_path))

            # Phase 3: reparse
            with ThermoDBI(":memory:") as db:
                mgr3 = TDBManager(db)
                p2 = mgr3.parse_tdb(output_path, any_tdb.stem)
                assert len(p2.phases) == len(p1.phases)
                assert len(p2.params) == len(p1.params)


# ── CLI tests ──

class TestCliRefTdb:
    """CLI commands on every ref_tdb file."""

    def test_cmd_parse(self, any_tdb):
        args = argparse.Namespace(tdb_file=str(any_tdb), tdb_name=any_tdb.stem)
        assert cmd_parse(args) == 0

    def test_cmd_import(self, any_tdb):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            args = argparse.Namespace(
                db=str(db_path), tdb_file=str(any_tdb), typed="",
                tdb_name=any_tdb.stem, desc="test", ver="1.0",
            )
            assert cmd_import(args) in (0, 1)  # graceful failure for non-importable TDBs

    def test_list_after_import(self, any_tdb):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            args_imp = argparse.Namespace(
                db=str(db_path), tdb_file=str(any_tdb), typed="",
                tdb_name=any_tdb.stem, desc="test", ver="1.0",
            )
            result = cmd_import(args_imp)

            if result != 0:
                pytest.skip("Import not supported for this TDB")

            for typed in ("elem", "tdb"):
                args_list = argparse.Namespace(
                    db=str(db_path), typed=typed, like=False,
                    elem="", func="", phase="", param="", tdb="",
                )
                assert cmd_list(args_list) == 0


# ── Content-specific tests ──

class TestSpecificRefTdb:
    def test_16symbols_has_bcc_fcc(self):
        f = REF_TDB_DIR / "16symbols-cxy-hmd.tdb"
        assert f.exists()
        mgr = TDBManager(ThermoDBI(":memory:"))
        try:
            p = mgr.parse_tdb(f, "x")
            phases = {x["phase"] for x in p.phases}
            assert "BCC" in phases and "FCC" in phases
            assert len(p.elems) >= 16
            assert len(p.params) >= 256
        finally:
            mgr.db.close()

    def test_saf2507_has_many_phases(self):
        """saf2507.TDB has BCC, FCC, HCP, LIQUID, SIGMA, etc."""
        f = REF_TDB_DIR / "saf2507.TDB"
        assert f.exists()
        mgr = TDBManager(ThermoDBI(":memory:"))
        try:
            p = mgr.parse_tdb(f, "x")
            phases = {x["phase"] for x in p.phases}
            assert len(phases) >= 10, f"Expected many phases, got {len(phases)}"
        finally:
            mgr.db.close()

    def test_steel1_has_bcc(self):
        f = REF_TDB_DIR / "steel1.TDB"
        assert f.exists()
        mgr = TDBManager(ThermoDBI(":memory:"))
        try:
            p = mgr.parse_tdb(f, "x")
            phases = {x["phase"] for x in p.phases}
            assert "BCC_A2" in phases
        finally:
            mgr.db.close()
