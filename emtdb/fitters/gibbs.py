"""
SGTE polynomial fitter for Gibbs free energy vs temperature.

Fits the function ``G(T) = A + B*T + C*T*ln(T) + D*T² + E*T³ + F/T``
using ``scipy.optimize.curve_fit`` with multiple random-restart trials.

Logic matches ``demo/gibbsfit.py`` verbatim.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

from emtdb.fitters.base import FitResult
from emtdb.fitters.readers import read_gibbs_dat, read_gibbs_json
from emtdb.fitters.tdb import format_tdb_func, format_tdb_parameter

# SGTE polynomial template used for expression formatting.
_FORMULA = "+A+B*T+C*T*LN(T)+D*T**2+E*T**3+F*T**(-1)"


# ---------------------------------------------------------------------------
# Model function
# ---------------------------------------------------------------------------


def sgte_poly(t: np.ndarray, a: float, b: float, c: float,
              d: float, e: float, f: float) -> np.ndarray:
    """SGTE polynomial evaluated at temperature(s) *t*."""
    return a + b * t + c * t * np.log(t) + d * t ** 2 + e * t ** 3 + f / t


# ---------------------------------------------------------------------------
# Fitter
# ---------------------------------------------------------------------------


class GibbsFitter:
    """Fit the SGTE polynomial ``G(T)`` to Gibbs free energy data.

    Uses repeated ``scipy.optimize.curve_fit`` trials and retains the
    result with the highest :math:`R^2`.
    """

    def __init__(self, max_trials: int = 100) -> None:
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
        """Read temperature–Gibbs data and fit the SGTE polynomial.

        Parameters
        ----------
        path:
            Path to a ``gibbs-temperature.dat`` or QHA JSON file.
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
            The best fit from ``max_trials`` attempts.
        """
        # --- read data ---
        path_obj = Path(path)
        if path_obj.suffix == ".json":
            t_data, g_data = read_gibbs_json(str(path_obj), atom_num)
        else:
            t_data, g_data = read_gibbs_dat(str(path_obj), atom_num)

        # --- fitting ---
        best_params: list[float] | None = None
        best_r2 = 0.0

        a_guess = float(g_data[0])
        if t_data[-1] != t_data[0]:
            b_guess = float((g_data[-1] - g_data[0]) / (t_data[-1] - t_data[0]))
        else:
            b_guess = 0.0
        p0_base = np.array([a_guess, b_guess, 0.0, 0.0, 0.0, 0.0])
        rng = np.random.default_rng()

        for trial in range(self.max_trials):
            if trial == 0:
                p0 = p0_base
            else:
                p0 = p0_base * (1.0 + rng.uniform(-0.01, 0.01, size=6))
            try:
                raw_params, _ = curve_fit(
                    sgte_poly, t_data, g_data, p0=p0, maxfev=5000,
                )
            except Exception:
                continue

            residuals = g_data - sgte_poly(t_data, *raw_params)
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((g_data - np.mean(g_data)) ** 2)
            r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

            if r2 > best_r2:
                best_params = raw_params.tolist()
                best_r2 = r2
                if r2 > 0.9999:
                    break

        if best_params is None:
            raise RuntimeError(
                f"curve_fit failed to converge after {self.max_trials} trials"
            )

        expression = _format_expression(best_params)
        g_fit = sgte_poly(t_data, *best_params)

        # --- TDB output: FUNCTION for SER, PARAMETER otherwise ---
        if phase.upper() == "SER":
            tdb_line = format_tdb_func(elements[0], expression)
        else:
            tdb_line = format_tdb_parameter(
                phase, elements, metrics, expression,
            )

        return FitResult(
            name=name,
            phase=phase.upper(),
            elements=[e.upper() for e in elements],
            metrics=metrics,
            atom_num=atom_num,
            params=best_params,
            r2=best_r2,
            expression=expression,
            tdb_line=tdb_line,
            x_data=t_data,
            y_data=g_data,
            y_fit=g_fit,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_expression(params: list[float]) -> str:
    """Format ``[A, B, C, D, E, F]`` into an SGTE expression string.

    Matches ``demo/gibbsfit.py`` ``_formula_str``.
    """
    return (
        _FORMULA.replace("+A", f"{params[0]:+E}")
        .replace("+B", f"{params[1]:+E}")
        .replace("+C", f"{params[2]:+E}")
        .replace("+D", f"{params[3]:+E}")
        .replace("+E", f"{params[4]:+E}")
        .replace("+F", f"{params[5]:+E}")
    )
