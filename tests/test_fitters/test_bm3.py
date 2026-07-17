"""Tests for ``emtdb.fitters.bm3`` — using synthetic v-e.dat fixtures."""

from __future__ import annotations

import numpy as np
import pytest

from emtdb.fitters.base import expand_results
from emtdb.fitters.bm3 import Bm3Fitter
from emtdb.config import B0_EV_ANG3_TO_GPA


class TestFitBm3:
    """End-to-end BM3 fits with synthesised data (known ground truth)."""

    SERSynthetic = "bm3_synthetic_ser"
    BCCSynthetic = "bm3_synthetic_bcc"

    def test_ser_fit(self, bm3_synthetic_ser):
        """SER BM3 fit: recovers parameters and produces FUNCTION TDB line."""
        result = Bm3Fitter(max_trials=30).fit_one(
            bm3_synthetic_ser, "test-SER-V", "SER", ["V"], [1.0], 1,
        )

        assert result.r2 > 0.99
        assert len(result.params) == 4

        # Parameter recovery (should be close to injected truth).
        e0, v0, b0, b1 = result.params
        assert abs(e0 - (-10.0)) < 0.1    # E0 < 0.1 eV error
        assert 14.5 < v0 < 15.5             # V0 within 0.5 Å³
        assert 100 < b0 < 200               # B0 reasonable range (GPa)
        assert 3.0 < b1 < 5.0                # B1 reasonable

        assert result.tdb_line.startswith("FUNCTION ETSER")
        assert len(result.tdb_line) > 10

    def test_ser_expression_format(self, bm3_synthetic_ser):
        """Expression string contains key BM3 parameter labels."""
        result = Bm3Fitter().fit_one(
            bm3_synthetic_ser, "test", "SER", ["V"], [1.0], 1,
        )
        assert "E0=" in result.expression
        assert "V0=" in result.expression
        assert "B0=" in result.expression
        assert "B1=" in result.expression

    def test_bcc_fit(self, bm3_synthetic_bcc):
        """BCC BM3 fit: recovers parameters with different seed."""
        result = Bm3Fitter(max_trials=30).fit_one(
            bm3_synthetic_bcc, "test-BCC-FE-CR", "BCC",
            ["FE", "CR"], [0.5, 0.5], 2,
        )

        assert result.r2 > 0.99

        e0, v0, b0, b1 = result.params
        assert abs(e0 - (-8.0)) < 0.1
        assert abs(v0 - 13.5) < 0.5
        assert 80 < b0 < 160
        assert 2.5 < b1 < 4.5

        assert result.tdb_line.startswith("PARAMETER G(BCC,FE:CR;0)")
        assert "-0.5*ETSERFE#" in result.tdb_line
        assert "-0.5*ETSERCR#" in result.tdb_line

    def test_bcc_expand(self, bm3_synthetic_bcc):
        """BCC with different elements → expands to two results."""
        result = Bm3Fitter().fit_one(
            bm3_synthetic_bcc, "test-BCC-FE-CR", "BCC",
            ["FE", "CR"], [0.5, 0.5], 2,
        )
        expanded = expand_results(result)
        assert len(expanded) == 2
        assert expanded[1].name == "test-BCC-FE-CR-ex"
        assert expanded[1].elements == ["CR", "FE"]

    def test_same_element_no_expand(self, tmp_path):
        """BCC with identical elements → single result."""
        # Minimal synthetic data.
        rng = np.random.default_rng(0)
        v = np.linspace(12, 20, 12)
        e = -10.0 + (v - 15.0) ** 2 * 0.05 + rng.normal(0, 0.01, 12)

        path = tmp_path / "v-e.dat"
        np.savetxt(path, np.column_stack([v, e]), fmt="%.10f")
        path_str = str(path)

        result = Bm3Fitter(max_trials=10).fit_one(
            path_str, "test", "BCC", ["AL", "AL"], [0.5, 0.5], 2,
        )
        assert len(expand_results(result)) == 1

    def test_too_few_points_raises(self, tmp_path):
        """Fewer than 4 data points → ValueError."""
        path = tmp_path / "short.dat"
        path.write_text("10.0  -5.0\n11.0  -5.3\n12.0  -5.1\n")
        with pytest.raises(ValueError, match="At least 4"):
            Bm3Fitter().fit_one(
                str(path), "err", "BCC", ["FE"], [1.0], 1,
            )

    def test_fallback_on_failed_fit(self, tmp_path, monkeypatch):
        """When curve_fit never succeeds, fallback uses data minimum."""
        rng = np.random.default_rng(0)
        v = np.linspace(12, 20, 12)
        ideal = -10.0 + (v - 15.0) ** 2 * 0.05
        e = ideal + rng.normal(0, 0.01, 12)
        path = tmp_path / "v-e.dat"
        np.savetxt(path, np.column_stack([v, e]), fmt="%.10f")

        # Force curve_fit to fail for all trials.
        def _broken(*args, **kwargs):
            raise RuntimeError("deliberate failure")

        monkeypatch.setattr(
            "emtdb.fitters.bm3.curve_fit", _broken
        )

        result = Bm3Fitter(max_trials=5).fit_one(
            str(path), "fallback", "SER", ["V"], [1.0], 1,
        )
        # Fallback still returns reasonable values.
        assert result.r2 == -1.0
        assert abs(result.params[0]) > 0  # E0 from data min
