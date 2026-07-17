"""Tests for ``emtdb.fitters.base``."""

from __future__ import annotations

import numpy as np
import pytest

from emtdb.fitters.base import FitResult, expand_results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make(name="test", phase="BCC", elements=None, metrics=None,
          atom_num=2) -> FitResult:
    return FitResult(
        name=name,
        phase=phase,
        elements=elements or ["FE", "CR"],
        metrics=metrics or [0.5, 0.5],
        atom_num=atom_num,
        params=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        r2=0.9999,
        expression="+1.0+2.0*T+3.0*T*LN(T)+4.0*T**2+5.0*T**3+6.0/T",
        tdb_line="PARAMETER G(BCC,FE:CR;0) ...",
        x_data=np.array([300.0, 400.0]),
        y_data=np.array([-10.0, -8.0]),
        y_fit=np.array([-9.99, -8.01]),
    )


# ---------------------------------------------------------------------------
# FitResult
# ---------------------------------------------------------------------------

class TestFitResult:
    def test_fields_accessible(self):
        """All expected fields exist and are readable."""
        r = _make()
        assert r.name == "test"
        assert r.phase == "BCC"
        assert r.elements == ["FE", "CR"]
        assert r.metrics == [0.5, 0.5]
        assert r.atom_num == 2
        assert len(r.params) == 6
        assert r.r2 == 0.9999
        assert isinstance(r.expression, str)
        assert isinstance(r.tdb_line, str)

    def test_array_fields_default_to_empty(self):
        """x_data / y_data / y_fit default to empty arrays."""
        r = FitResult(
            name="n", phase="SER", elements=["NB"],
            metrics=[1.0], atom_num=1,
            params=[], r2=0.0, expression="",
        )
        assert r.x_data.shape == (0,)
        assert r.y_data.shape == (0,)
        assert r.y_fit.shape == (0,)

    def test_tdb_line_default_empty(self):
        """tdb_line defaults to empty string."""
        r = FitResult(
            name="n", phase="SER", elements=["NB"],
            metrics=[1.0], atom_num=1,
            params=[], r2=0.0, expression="",
        )
        assert r.tdb_line == ""


# ---------------------------------------------------------------------------
# expand_results
# ---------------------------------------------------------------------------

class TestExpandResults:
    def test_binary_equal_metrics(self):
        """Two different elements with equal metrics → 2 results, second -ex."""
        r = _make(elements=["FE", "CR"], metrics=[0.5, 0.5])
        results = expand_results(r)
        assert len(results) == 2
        assert results[0] is r
        assert results[1].name == "test-ex"
        assert results[1].elements == ["CR", "FE"]
        assert results[1].metrics == [0.5, 0.5]
        # Shared fields
        assert results[1].params == r.params
        assert results[1].r2 == r.r2

    def test_same_element_no_expand(self):
        """Identical elements → no expansion."""
        r = _make(elements=["AL", "AL"], metrics=[0.5, 0.5])
        assert len(expand_results(r)) == 1

    def test_unequal_metrics_no_expand(self):
        """Different metrics → no expansion."""
        r = _make(elements=["FE", "CR"], metrics=[0.25, 0.75])
        assert len(expand_results(r)) == 1

    def test_single_element_no_expand(self):
        """Single element (SER) → no expansion."""
        r = _make(elements=["NB"], metrics=[1.0], atom_num=2)
        assert len(expand_results(r)) == 1

    def test_three_elements_no_expand(self):
        """Three elements → no expansion."""
        r = _make(elements=["AL", "TI", "NB"], metrics=[0.25, 0.25, 0.5])
        assert len(expand_results(r)) == 1

    def test_expand_preserves_arrays(self):
        """-ex result gets copies of the original arrays."""
        r = _make()
        results = expand_results(r)
        assert len(results[1].x_data) == 2
        assert np.allclose(results[1].x_data, [300.0, 400.0])
        assert np.allclose(results[1].y_fit, [-9.99, -8.01])

    def test_fp_tolerance_respected(self):
        """Metrics within 1e-12 are treated as equal."""
        r = _make(metrics=[0.5, 0.5 + 1e-13])
        assert len(expand_results(r)) == 2

    def test_fp_tolerance_boundary(self):
        """Difference just above 1e-12 → no expansion."""
        r = _make(metrics=[0.5, 0.5 + 2e-12])
        assert len(expand_results(r)) == 1
