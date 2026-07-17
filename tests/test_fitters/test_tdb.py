"""Tests for ``emtdb.fitters.tdb``."""

from __future__ import annotations

import numpy as np
import pytest

from emtdb.fitters.base import FitResult
from emtdb.fitters.tdb import (
    _pad,
    format_tdb_etser,
    format_tdb_func,
    format_tdb_param_with_etser,
    format_tdb_parameter,
    write_tdb_file,
)


# ---------------------------------------------------------------------------
# format_tdb_parameter
# ---------------------------------------------------------------------------


class TestFormatTdbParameter:
    def test_bcc_binary(self):
        """BCC binary with equal metrics."""
        line = format_tdb_parameter(
            "BCC", ["FE", "CR"], [0.5, 0.5],
            "+1.234E+05-5.678E+01*T",
        )
        assert line.startswith("PARAMETER G(BCC,FE:CR;0)")
        assert "-0.5*GHSERFE#" in line
        assert "-0.5*GHSERCR#" in line
        assert line.endswith("N !")

    def test_fcc_binary(self):
        """FCC with unequal metrics."""
        line = format_tdb_parameter(
            "FCC", ["AL", "NI"], [0.25, 0.75],
            "+1.23E+05+2.34E+01*T-3.45E+00*T*LN(T)",
        )
        assert line.startswith("PARAMETER G(FCC,AL:NI;0)")
        assert "-0.25*GHSERAL#" in line
        assert "-0.75*GHSERNI#" in line

    def test_ser_single_element(self):
        """SER with single element → still a PARAMETER for consistency."""
        line = format_tdb_parameter(
            "SER", ["NB"], [1.0],
            "+1.23E+05-4.56E+01*T+...",
        )
        assert line.startswith("PARAMETER G(SER,NB;0)")
        assert "-1*GHSERNB#" in line

    def test_ternary(self):
        """Three elements."""
        line = format_tdb_parameter(
            "OTH", ["AL", "TI", "NB"], [0.25, 0.25, 0.5],
            "+1.23E+05",
        )
        assert line.startswith("PARAMETER G(OTH,AL:TI:NB;0)")
        assert "-0.25*GHSERAL#" in line
        assert "-0.5*GHSERNB#" in line

    def test_case_normalised(self):
        """Phase and elements are uppercased."""
        line = format_tdb_parameter("bcc", ["fe", "cr"], [0.5, 0.5], "")
        assert "G(BCC,FE:CR;0)" in line


# ---------------------------------------------------------------------------
# format_tdb_func
# ---------------------------------------------------------------------------


class TestFormatTdbFunc:
    def test_two_letter_elem(self):
        """Two-letter element (NB) → GHSERNB."""
        line = format_tdb_func("NB", "+1.234E+05-5.678E+01*T")
        assert line.startswith("FUNCTION GHSERNB")
        assert "+1.234E+05-5.678E+01*T" in line
        assert line.endswith("N !")

    def test_single_letter_elem(self):
        """Single-letter element (V) → padded to VV."""
        line = format_tdb_func("V", "+3.000E+00")
        assert "GHSERVV" in line

    def test_w_element(self):
        """W → GHSERWW."""
        line = format_tdb_func("W", "")
        assert "GHSERWW" in line

    def test_already_uppercase(self):
        """Case-normalised."""
        line = format_tdb_func("nb", "")
        assert "GHSERNB" in line


# ---------------------------------------------------------------------------
# _pad (internal helper)
# ---------------------------------------------------------------------------


class TestPad:
    @pytest.mark.parametrize(
        "inp, exp",
        [
            ("NB", "NB"),
            ("CR", "CR"),
            ("V", "VV"),
            ("W", "WW"),
            ("AL", "AL"),
            ("Fe", "FE"),
            ("va", "VA"),
        ],
    )
    def test_pad(self, inp, exp):
        assert _pad(inp) == exp


# ---------------------------------------------------------------------------
# format_tdb_etser
# ---------------------------------------------------------------------------


class TestFormatTdbEtser:
    def test_basic(self):
        line = format_tdb_etser("NB", -5.461234e5)
        assert line.startswith("FUNCTION ETSERNB")
        assert "-5.461234E+05" in line
        assert line.endswith("N !")

    def test_single_letter(self):
        line = format_tdb_etser("V", 0.0)
        assert "ETSERVV" in line

    def test_zero(self):
        line = format_tdb_etser("FE", 0.0)
        assert "+0.000000E+00" in line


# ---------------------------------------------------------------------------
# format_tdb_param_with_etser
# ---------------------------------------------------------------------------


class TestFormatTdbParamWithEtser:
    def test_binary(self):
        line = format_tdb_param_with_etser(
            "BCC", ["FE", "CR"], [0.5, 0.5], -5.461234e5,
        )
        assert line.startswith("PARAMETER G(BCC,FE:CR;0)")
        assert "-0.5*ETSERFE#" in line
        assert "-0.5*ETSERCR#" in line
        assert "-5.461234E+05" in line

    def test_single_letter(self):
        line = format_tdb_param_with_etser(
            "BCC", ["V", "W"], [0.5, 0.5], -1.0e5,
        )
        assert "ETSERVV" in line
        assert "ETSERWW" in line


# ---------------------------------------------------------------------------
# write_tdb_file
# ---------------------------------------------------------------------------


class TestWriteTdbFile:
    def test_ser_only(self, tmp_path):
        """SER results → TDB file with FUNCTION lines."""
        results = [
            FitResult(
                name="SER-FE", phase="SER", elements=["FE"],
                metrics=[1.0], atom_num=1, params=[], r2=0.99,
                expression="", tdb_line="FUNCTION ETSERFE  1.00 -8.0E+05; 6000.00 N !",
                x_data=np.array([]), y_data=np.array([]), y_fit=np.array([]),
            ),
        ]
        path = tmp_path / "out.tdb"
        write_tdb_file(str(path), results)
        text = path.read_text()
        assert "ETSERFE" in text
        assert "TYPE_DEFINITION" in text
        assert text.strip().endswith("N !")

    def test_param_only(self, tmp_path):
        """Non-SER results → PARAMETER lines."""
        results = [
            FitResult(
                name="BCC-FE-CR", phase="BCC", elements=["FE", "CR"],
                metrics=[0.5, 0.5], atom_num=2, params=[], r2=0.99,
                expression="",
                tdb_line="PARAMETER G(BCC,FE:CR;0)   1.00 -1.7E+06-0.5*ETSERFE#-0.5*ETSERCR#; 6000.00 N !",
                x_data=np.array([]), y_data=np.array([]), y_fit=np.array([]),
            ),
        ]
        path = tmp_path / "out.tdb"
        write_tdb_file(str(path), results)
        text = path.read_text()
        assert "G(BCC,FE:CR;0)" in text

    def test_mixed(self, tmp_path):
        """SER + param results → both sections present."""
        results = [
            FitResult(
                name="SER-FE", phase="SER", elements=["FE"],
                metrics=[1.0], atom_num=1, params=[], r2=0.99,
                expression="", tdb_line="FUNCTION ETSERFE  1.00 -8.0E+05; 6000.00 N !",
                x_data=np.array([]), y_data=np.array([]), y_fit=np.array([]),
            ),
            FitResult(
                name="BCC-FE-CR", phase="BCC", elements=["FE", "CR"],
                metrics=[0.5, 0.5], atom_num=2, params=[], r2=0.99,
                expression="",
                tdb_line="PARAMETER G(BCC,FE:CR;0)   1.00 -1.7E+06-0.5*ETSERFE#-0.5*ETSERCR#; 6000.00 N !",
                x_data=np.array([]), y_data=np.array([]), y_fit=np.array([]),
            ),
        ]
        path = tmp_path / "out.tdb"
        write_tdb_file(str(path), results, description="test")
        text = path.read_text()
        assert "ETSERFE" in text
        assert "G(BCC,FE:CR;0)" in text
        assert "test" in text
