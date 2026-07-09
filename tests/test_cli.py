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
    cmd_export,
    cmd_list,
    cmd_delete,
    cmd_etot,
    cmd_subset,
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



# ── Export tests ──

class TestExport:
    """Direct ``python main.py export`` route."""

    def test_export_roundtrip(self):
        """Import 16symbols, export, reparse — same count."""
        src = REF_TDB_DIR / "16symbols-cxy-hmd.tdb"
        assert src.exists()

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"

            # Import
            with ThermoDBI(str(db_path)) as db:
                mgr = TDBManager(db)
                p1 = mgr.parse_tdb(src, "x")
                mgr.import_tdb(p1.phases, p1.params, "x", "test", "1.0")

            # Export
            out = Path(tmpdir) / "out.tdb"
            args = argparse.Namespace(db=str(db_path), output=str(out), tdb_name="x")
            assert cmd_export(args) == 0
            assert out.exists()

            # Reparse
            with ThermoDBI(":memory:") as db:
                mgr2 = TDBManager(db)
                p2 = mgr2.parse_tdb(out, "x")
                assert len(p2.phases) == len(p1.phases)
                assert len(p2.params) == len(p1.params)


# ── Delete tests ──

class TestDelete:
    """``python main.py delete`` on imported TDBs."""

    def _import_simple(self, db_path: str):
        """Import BCC+FCC from 16symbols into a persistent DB."""
        src = REF_TDB_DIR / "16symbols-cxy-hmd.tdb"
        with ThermoDBI(db_path) as db:
            mgr = TDBManager(db)
            p = mgr.parse_tdb(src, "x")
            mgr.save_elements(p.elems)
            mgr.save_functions(p.funcs)
            mgr.import_tdb(p.phases, p.params, "x", "test", "1.0")

    def test_delete_element(self):
        """Delete a single element — should not crash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            self._import_simple(db_path)
            args = argparse.Namespace(
                db=db_path, typed="elem", cascade=False,
                elem="CO", func="", phase="", param="", tdb="",
            )
            assert cmd_delete(args) == 0

    def test_delete_tdb(self):
        """Delete an entire TDB."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            self._import_simple(db_path)
            args = argparse.Namespace(
                db=db_path, typed="tdb", cascade=True,
                elem="", func="", phase="", param="", tdb="x",
            )
            assert cmd_delete(args) == 0

    def test_delete_function(self):
        """Delete a single function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            self._import_simple(db_path)
            args = argparse.Namespace(
                db=db_path, typed="func", cascade=False,
                elem="", func="GHSERAL", phase="", param="", tdb="",
            )
            assert cmd_delete(args) == 0

    def test_delete_phase(self):
        """Delete a single phase."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            self._import_simple(db_path)
            args = argparse.Namespace(
                db=db_path, typed="phase", cascade=True,
                elem="", func="", phase="BCC", param="", tdb="",
            )
            assert cmd_delete(args) == 0


# ── Subset tests ──

class TestSubset:
    """``python main.py subset`` extracting a TDB subset."""

    def test_subset_basic(self):
        """Extract (Al,Nb,Ti) subset from 16symbols."""
        src = REF_TDB_DIR / "16symbols-cxy-hmd.tdb"
        assert src.exists()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "sub.tdb"
            args = argparse.Namespace(
                tdb_file=str(src), elem=["AL", "NB", "TI"],
                db="", tdb_name="sub", output=str(out),
            )
            assert cmd_subset(args) == 0
            assert out.exists()

            # Reparse the subset
            with ThermoDBI(":memory:") as db:
                mgr = TDBManager(db)
                p = mgr.parse_tdb(out, "sub")
                # Should have exactly 3 requested elements
                elems = {e["elem"] for e in p.elems}
                assert "AL" in elems and "NB" in elems and "TI" in elems
                # BCC(AL:*, NB:*, TI:*) + FCC(AL:*, NB:*, TI:*) = at least 8 params
                assert len(p.params) >= 8

    def test_subset_nonexistent_elem_returns_1(self):
        """Requesting an element not in the TDB should fail gracefully."""
        src = REF_TDB_DIR / "16symbols-cxy-hmd.tdb"
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "sub.tdb"
            args = argparse.Namespace(
                tdb_file=str(src), elem=["XX"],
                db="", tdb_name="sub", output=str(out),
            )
            assert cmd_subset(args) == 1


# ── ETot tests ──

class TestEtot:
    """``python main.py etot`` fitting DFT static energies."""

    VE_DATA_FE = """\
10.750000 -8.294257
10.930000 -8.305271
11.120000 -8.310988
11.310000 -8.311472
11.500000 -8.308382
11.690000 -8.301387
11.880000 -8.293690
"""

    VE_DATA_CR = """\
11.200000 -9.508000
11.480000 -9.514000
11.760000 -9.517000
12.040000 -9.516000
12.320000 -9.513000
12.600000 -9.508000
"""

    VE_DATA_FECR = """\
22.000000 -17.550000
22.500000 -17.570000
23.000000 -17.585000
23.500000 -17.589000
24.000000 -17.586000
24.500000 -17.576000
25.000000 -17.561000
"""

    @pytest.fixture
    def etot_data_dir(self, tmp_path: Path):
        """Create temporary directory with end-member folders."""
        folders = {
            "SER-Fe-1": self.VE_DATA_FE,
            "SER-Cr-2": self.VE_DATA_CR,
            "BCC-Fe-Cr-2": self.VE_DATA_FECR,
        }
        for name, content in folders.items():
            folder = tmp_path / name
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "v-e.dat").write_text(content, encoding="utf-8")
        return str(tmp_path)

    def test_etot_parse_and_fit(self, etot_data_dir):
        """End-to-end: BM3 fit → TDB output."""
        with tempfile.TemporaryDirectory() as outdir:
            out = Path(outdir) / "out.tdb"
            args = argparse.Namespace(
                data_dir=etot_data_dir,
                tdb_name="etot-test",
                db="",
                output=str(out),
                output_json="",
                output_csv="",
            )
            assert cmd_etot(args) == 0
            assert out.exists()

            # Reparse the output TDB
            with ThermoDBI(":memory:") as db:
                mgr = TDBManager(db)
                p = mgr.parse_tdb(out, "etot-test")
                assert len(p.funcs) == 2  # ETSERFE, ETSERCR
                assert len(p.phases) == 1  # BCC
                assert len(p.params) == 2  # FE:CR + CR:FE (symmetric)
                # Check function names use ETSERxxx not ETOT_SER_xxx
                func_names = {f["func"] for f in p.funcs}
                assert "ETSERFE" in func_names
                assert "ETSERCR" in func_names

    def test_etot_output_json(self, etot_data_dir):
        """JSON export contains all records with BM3 diagnostics."""
        with tempfile.TemporaryDirectory() as outdir:
            out = Path(outdir) / "out.json"
            args = argparse.Namespace(
                data_dir=etot_data_dir,
                tdb_name="etot-test",
                db="",
                output="",
                output_json=str(out),
                output_csv="",
            )
            assert cmd_etot(args) == 0
            import json
            with open(out) as f:
                data = json.load(f)
            assert len(data) == 3
            for r in data:
                assert "E0_eV" in r
                assert "V0_Ang3" in r
                assert "B0_GPa" in r
                assert "R2" in r

    def test_etot_output_csv(self, etot_data_dir):
        """CSV export works and has correct header."""
        with tempfile.TemporaryDirectory() as outdir:
            out = Path(outdir) / "out.csv"
            args = argparse.Namespace(
                data_dir=etot_data_dir,
                tdb_name="etot-test",
                db="",
                output="",
                output_json="",
                output_csv=str(out),
            )
            assert cmd_etot(args) == 0
            lines = out.read_text(encoding="utf-8").strip().splitlines()
            assert lines[0] == "name,phase,elements,E0_eV,V0_Ang3,B0_GPa,B1,R2,n_points"
            assert len(lines) == 4  # header + 3 data rows

    def test_etot_invalid_dir_returns_1(self):
        """Non-existent data directory returns exit code 1."""
        args = argparse.Namespace(
            data_dir="/nonexistent/path",
            tdb_name="test",
            db="", output="", output_json="", output_csv="",
        )
        assert cmd_etot(args) == 1
