"""pytest fixtures for ``tests/test_fitters/``."""

from __future__ import annotations

import numpy as np
import pytest

from emtdb.config import B0_EV_ANG3_TO_GPA


def _bm3_eos(v, e0, v0, b0, b1):
    """3rd-order Birch-Murnaghan EOS (eV units)."""
    x = (v0 / v) ** (2.0 / 3.0)
    return e0 + (9.0 / 16.0) * v0 * b0 * (
        (x - 1.0) ** 3 * b1 + (x - 1.0) ** 2 * (6.0 - 4.0 * x)
    )


@pytest.fixture
def bm3_synthetic_ser(tmp_path) -> str:
    """Synthetic v-e.dat for a single-element (SER) BM3 fit.

    Generated from known BM3 parameters with 1% Gaussian noise.
    """
    rng = np.random.default_rng(42)
    volumes = np.linspace(12.0, 20.0, 20)

    # Physical parameters (eV units for B0).
    e0_true = -10.0       # eV
    v0_true = 15.0        # Å³
    b0_ev = 150.0 / B0_EV_ANG3_TO_GPA  # 150 GPa → eV/Å³
    b1_true = 4.0

    energies = _bm3_eos(volumes, e0_true, v0_true, b0_ev, b1_true)
    energies += rng.normal(0, 0.005, 20)  # 1 % noise

    path = tmp_path / "v-e.dat"
    np.savetxt(path, np.column_stack([volumes, energies]), fmt="%.10f")
    return str(path)


@pytest.fixture
def bm3_synthetic_bcc(tmp_path) -> str:
    """Synthetic v-e.dat for a binary (BCC) BM3 fit.

    Different parameters from the SER case.
    """
    rng = np.random.default_rng(123)
    volumes = np.linspace(10.0, 18.0, 16)

    e0_true = -8.0
    v0_true = 13.5
    b0_ev = 120.0 / B0_EV_ANG3_TO_GPA
    b1_true = 3.5

    energies = _bm3_eos(volumes, e0_true, v0_true, b0_ev, b1_true)
    energies += rng.normal(0, 0.003, 16)

    path = tmp_path / "v-e.dat"
    np.savetxt(path, np.column_stack([volumes, energies]), fmt="%.10f")
    return str(path)
