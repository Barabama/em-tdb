"""
3rd-order Birch-Murnaghan EOS fitter for DFT static energy vs volume.

Fits the function ``E(V) = E0 + (9/16)·V0·B0·[((V0/V)^{2/3}−1)³·B1
                                   + ((V0/V)^{2/3}−1)²·(6−4·(V0/V)^{2/3})]``

Logic matches ``demo/etotfit.py`` verbatim.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit

from emtdb.config import B0_EV_ANG3_TO_GPA, F_CONST
from emtdb.fitters.base import FitResult
from emtdb.fitters.readers import read_ve_dat
from emtdb.fitters.tdb import (
    format_tdb_etser,
    format_tdb_param_with_etser,
)


# ---------------------------------------------------------------------------
# Model function
# ---------------------------------------------------------------------------


def bm3_eos(
    v: np.ndarray,
    e0: float,
    v0: float,
    b0: float,
    b1: float,
) -> np.ndarray:
    """3rd-order Birch-Murnaghan equation of state ``E(V)``."""
    x = (v0 / v) ** (2.0 / 3.0)
    return e0 + (9.0 / 16.0) * v0 * b0 * (
        (x - 1.0) ** 3 * b1 + (x - 1.0) ** 2 * (6.0 - 4.0 * x)
    )


# ---------------------------------------------------------------------------
# Fitter
# ---------------------------------------------------------------------------


class Bm3Fitter:
    """Fit the 3rd-order BM3 EOS to DFT energy–volume data.

    Uses ``scipy.optimize.curve_fit`` with up to *max_trials* random-restart
    attempts.  Falls back to the minimum-energy data point when all attempts
    fail.
    """

    def __init__(self, max_trials: int = 30) -> None:
        self.max_trials = max_trials

    def fit_one(
        self,
        path: str,
        name: str,
        phase: str,
        elements: list[str],
        metrics: list[float],
        atom_num: int,
    ) -> FitResult:
        """Read volume–energy data and fit the BM3 equation of state.

        Parameters
        ----------
        path:
            Path to a ``v-e.dat`` file.
        name:
            Dataset identifier.
        phase:
            Phase name (``"SER"``, ``"BCC"``, …).
        elements:
            Element symbols in upper case.
        metrics:
            Normalised stoichiometry ratios.
        atom_num:
            Atoms per formula unit.

        Returns
        -------
        FitResult
            *params* = ``[E0_eV, V0_Ang3, B0_GPa, B1]``
            *x_data* = volumes, *y_data* = energies.
        """
        volumes, energies = read_ve_dat(path)

        if len(volumes) < 4:
            raise ValueError(
                f"At least 4 data points required for BM3 fit, "
                f"got {len(volumes)}"
            )

        # --- initial guess ---
        idx_mid = len(volumes) // 2
        e0_0 = float(energies[idx_mid])
        v0_0 = float(volumes[idx_mid])
        b0_0 = 1.2       # eV/Å³  (will be converted to GPa after fit)
        b1_0 = 3.5

        # --- multi-trial fit ---
        best_params: tuple[float, float, float, float] | None = None
        best_r2 = -1e9

        rng = np.random.default_rng()

        for trial in range(self.max_trials):
            if trial == 0:
                p0 = (e0_0, v0_0, b0_0, b1_0)
            else:
                p0 = (
                    e0_0 * (1.0 + rng.uniform(-1e-3, 1e-3)),
                    v0_0 * (1.0 + rng.uniform(-1e-2, 1e-2)),
                    max(0.5, b0_0 * (1.0 + rng.uniform(-0.3, 0.3))),
                    max(2.0, b1_0 * (1.0 + rng.uniform(-0.2, 0.2))),
                )
            try:
                params, _ = curve_fit(
                    bm3_eos, volumes, energies, p0=p0, maxfev=2000,
                )
            except Exception:
                continue

            residuals = energies - bm3_eos(volumes, *params)
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((energies - np.mean(energies)) ** 2)
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else -1.0

            if r2 > best_r2:
                best_r2 = r2
                best_params = tuple(params)
                if r2 > 0.9999:
                    break

        # --- fallback when all trials fail ---
        if best_params is None:
            idx_min = int(np.argmin(energies))
            e0_eV = float(energies[idx_min])
            v0_ang3 = float(volumes[idx_min])
            best_params = (e0_eV, v0_ang3, 0.0, 3.5)
            best_r2 = -1.0

        e0_eV, v0_ang3, b0_ev, b1 = best_params
        b0_gpa = b0_ev * B0_EV_ANG3_TO_GPA

        # --- TDB output ---
        e0_j = e0_eV * F_CONST / atom_num
        expression = (
            f"E0={e0_eV:.6f} eV  V0={v0_ang3:.2f} Å³  "
            f"B0={b0_gpa:.1f} GPa  B1={b1:.2f}"
        )

        if phase.upper() == "SER":
            tdb_line = format_tdb_etser(elements[0], e0_j)
        else:
            tdb_line = format_tdb_param_with_etser(
                phase, elements, metrics, e0_j,
            )

        # --- fitted curve ---
        y_fit = bm3_eos(volumes, e0_eV, v0_ang3, b0_ev, b1)

        return FitResult(
            name=name,
            phase=phase.upper(),
            elements=[e.upper() for e in elements],
            metrics=metrics,
            atom_num=atom_num,
            params=[e0_eV, v0_ang3, b0_gpa, b1],
            r2=best_r2,
            expression=expression,
            tdb_line=tdb_line,
            x_data=volumes,
            y_data=energies,
            y_fit=y_fit,
        )
