"""Tests for ``emtdb.fitters.gibbs`` — full pipeline with real data."""

from __future__ import annotations

import numpy as np
import pytest

from emtdb.fitters.base import expand_results
from emtdb.fitters.gibbs import GibbsFitter


# ---------------------------------------------------------------------------
# Fitting — DAT files
# ---------------------------------------------------------------------------


class TestFitGibbsDat:
    """End-to-end fits using ``tests/fits-dat/`` gibbs-temperature.dat files."""

    @pytest.fixture
    def fitter(self) -> GibbsFitter:
        return GibbsFitter(max_trials=100)

    @pytest.mark.parametrize(
        "folder, phase, elems, metrics, atom_num, min_r2, dat_subpath",
        [
            ("BCC-Mn-W",   "BCC", ["MN", "W"],   [0.5, 0.5],       2, 0.95, "QHA-MnW/gibbs-temperature.dat"),
            ("BCC-TiNb",   "BCC", ["TI", "NB"],  [0.5, 0.5],       2, 0.95, "gibbs-temperature.dat"),
            ("BCC-VV-2",   "BCC", ["V", "V"],    [0.5, 0.5],       2, 0.95, "QHA-VV/gibbs-temperature.dat"),
            ("FCC-CrAl3",  "FCC", ["CR", "AL"],  [0.25, 0.75],     4, 0.95, "gibbs-temperature.dat"),
            ("HCP-W2Ni6",  "HCP", ["W", "NI"],   [0.25, 0.75],     8, 0.95, "QHA-WNi/gibbs-temperature.dat"),
            ("OTH-Al-Ti-Nb-4", "OTH", ["AL", "TI", "NB"], [0.25, 0.25, 0.5], 4, 0.95, "gibbs-temperature.dat"),
            ("OTH-Nb2TiAl", "OTH", ["NB", "TI", "AL"], [0.33, 0.33, 0.34], 4, 0.95, "gibbs-temperature.dat"),
            ("SER-Re-8a",  "SER", ["RE"],        [1.0],             8, 0.95, "QHA-ReRe/gibbs-temperature.dat"),
        ],
    )
    def test_fit(self, fitter, folder, phase, elems, metrics, atom_num, min_r2, dat_subpath):
        path = f"tests/fits-dat/{folder}/{dat_subpath}"

        result = fitter.fit_one(path, folder, phase, elems, metrics, atom_num)

        assert result.r2 >= min_r2, (
            f"R² {result.r2:.6f} < {min_r2} for {folder}"
        )
        assert len(result.params) == 6
        assert result.expression.startswith("+") or result.expression.startswith("-")
        assert isinstance(result.tdb_line, str)
        assert len(result.tdb_line) > 10
        assert len(result.x_data) == len(result.y_data) == len(result.y_fit) > 10

    def test_ser_outputs_function(self, fitter):
        """SER phase → tdb_line is a FUNCTION, not PARAMETER."""
        result = fitter.fit_one(
            "tests/fits-dat/SER-Re-8a/QHA-ReRe/gibbs-temperature.dat",
            "SER-Re-8a", "SER", ["RE"], [1.0], 8,
        )
        assert result.tdb_line.startswith("FUNCTION GHSER")

    def test_bcc_outputs_parameter(self, fitter):
        """BCC phase → tdb_line is a PARAMETER."""
        result = fitter.fit_one(
            "tests/fits-dat/BCC-TiNb/gibbs-temperature.dat",
            "BCC-TiNb", "BCC", ["TI", "NB"], [0.5, 0.5], 2,
        )
        assert result.tdb_line.startswith("PARAMETER G(")

    def test_expand_bcc(self, fitter):
        """BCC binary with different elements → two results after expand."""
        result = fitter.fit_one(
            "tests/fits-dat/BCC-Mn-W/QHA-MnW/gibbs-temperature.dat",
            "BCC-Mn-W", "BCC", ["MN", "W"], [0.5, 0.5], 2,
        )
        expanded = expand_results(result)
        assert len(expanded) == 2
        assert expanded[1].name == "BCC-Mn-W-ex"
        assert expanded[1].elements == ["W", "MN"]

    def test_expand_same_elem(self, fitter):
        """BCC with identical elements → no expansion."""
        result = fitter.fit_one(
            "tests/fits-dat/BCC-VV-2/QHA-VV/gibbs-temperature.dat",
            "BCC-Al-Al-2", "BCC", ["AL", "AL"], [0.5, 0.5], 2,
        )
        assert len(expand_results(result)) == 1

    def test_invalid_path(self, fitter):
        with pytest.raises((FileNotFoundError, OSError)):
            fitter.fit_one("/nonexistent/file.dat", "err", "BCC",
                           ["FE"], [1.0], 1)


# ---------------------------------------------------------------------------
# Fitting — JSON files (Format B / QhaData)
# ---------------------------------------------------------------------------


class TestFitGibbsJson:
    """End-to-end fits using ``tests/fits-json/`` QhaData files."""

    @pytest.fixture
    def fitter(self) -> GibbsFitter:
        return GibbsFitter(max_trials=100)

    @pytest.mark.parametrize(
        "folder, phase, elems, metrics, atom_num, min_r2",
        [
            ("BCC-Al-V", "BCC", ["AL", "V"], [0.5, 0.5], 2, 0.90),
            ("BCC-Nb-V", "BCC", ["NB", "V"], [0.5, 0.5], 2, 0.90),
            ("BCC-Ti-Ti", "BCC", ["TI", "TI"], [0.5, 0.5], 2, 0.90),
            ("SER-Ti", "SER", ["TI"], [1.0], 1, 0.90),
        ],
    )
    def test_fit(self, fitter, folder, phase, elems, metrics, atom_num, min_r2):
        path = f"tests/fits-json/{folder}/{folder}-qha.json"

        result = fitter.fit_one(path, folder, phase, elems, metrics, atom_num)

        assert result.r2 >= min_r2, (
            f"R² {result.r2:.6f} < {min_r2} for {folder}"
        )
        assert len(result.params) == 6
        assert len(result.x_data) == len(result.y_data) == len(result.y_fit) > 10

    def test_ser_ti_outputs_function(self, fitter):
        """SER phase from JSON → FUNCTION."""
        result = fitter.fit_one(
            "tests/fits-json/SER-Ti/SER-Ti-qha.json",
            "SER-Ti", "SER", ["TI"], [1.0], 1,
        )
        assert result.tdb_line.startswith("FUNCTION GHSER")
        assert "GHSERTI" in result.tdb_line or "GHSERTI" in result.tdb_line

    def test_bcc_alv_expands(self, fitter):
        """BCC-Al-V with different elements → expands to two."""
        result = fitter.fit_one(
            "tests/fits-json/BCC-Al-V/BCC-Al-V-qha.json",
            "BCC-Al-V", "BCC", ["AL", "V"], [0.5, 0.5], 2,
        )
        expanded = expand_results(result)
        assert len(expanded) == 2
        assert expanded[1].elements == ["V", "AL"]

    def test_bcc_titi_no_expand(self, fitter):
        """BCC-Ti-Ti same element → no expansion."""
        result = fitter.fit_one(
            "tests/fits-json/BCC-Ti-Ti/BCC-Ti-Ti-qha.json",
            "BCC-Ti-Ti", "BCC", ["TI", "TI"], [0.5, 0.5], 2,
        )
        expanded = expand_results(result)
        assert len(expanded) == 1
