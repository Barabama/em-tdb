"""
Core data model for thermodynamic fitting results.

Defines the unified ``FitResult`` dataclass used by all fitters,
and the ``expand_results()`` helper for BCC symmetric exchange.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class FitResult:
    """Container for a single thermodynamic fitting result.

    Attributes
    ----------
    name:
        Dataset identifier (e.g. ``"BCC-TiNb-2"``).
    phase:
        Phase name in upper case (``"SER"``, ``"BCC"``, ``"FCC"``, …).
    elements:
        Element symbols in upper case (e.g. ``["TI", "NB"]``).
    metrics:
        Normalised sublattice stoichiometry (sums to 1).
    atom_num:
        Number of atoms per formula unit.
    params:
        Fitted model parameters.
    r2:
        Coefficient of determination.
    expression:
        Human-readable expression string of the fitted model.
    tdb_line:
        TDB output line — a ``FUNCTION`` for SER, or a ``PARAMETER`` for
        other phases.
    x_data:
        Independent variable (temperature for Gibbs, volume for BM3).
    y_data:
        Observed dependent variable.
    y_fit:
        Fitted curve evaluated at *x_data*.
    """

    name: str
    phase: str
    elements: list[str]
    metrics: list[float]
    atom_num: int
    params: list[float]
    r2: float
    expression: str
    tdb_line: str = ""
    x_data: np.ndarray = field(default_factory=lambda: np.array([]))
    y_data: np.ndarray = field(default_factory=lambda: np.array([]))
    y_fit: np.ndarray = field(default_factory=lambda: np.array([]))


def expand_results(result: FitResult) -> list[FitResult]:
    """Expand a result with a BCC symmetric-exchange entry if applicable.

    When the result has exactly two *different* elements with equal
    stoichiometric metrics, a second ``FitResult`` is produced with the
    element order swapped and ``name`` suffixed by ``"-ex"``.

    Parameters
    ----------
    result:
        The primary fitting result.

    Returns
    -------
    list[FitResult]
        Always contains *result* as the first element.  A second swapped
        entry is appended only when the BCC symmetric condition holds.
    """
    elems = result.elements
    metrics = result.metrics

    if (
        len(elems) == 2
        and elems[0] != elems[1]
        and len(metrics) == 2
        and abs(metrics[0] - metrics[1]) < 1e-12
    ):
        swapped = FitResult(
            name=f"{result.name}-ex",
            phase=result.phase,
            elements=[elems[1], elems[0]],
            metrics=[metrics[1], metrics[0]],
            atom_num=result.atom_num,
            params=result.params[:],
            r2=result.r2,
            expression=result.expression,
            tdb_line=result.tdb_line,
            x_data=result.x_data.copy(),
            y_data=result.y_data.copy(),
            y_fit=result.y_fit.copy(),
        )
        return [result, swapped]

    return [result]
