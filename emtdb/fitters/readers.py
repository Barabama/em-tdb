"""
Data-file readers for thermodynamic fitting inputs.

Three public functions, each returning ``(x_array, y_array)``:

* ``read_gibbs_dat`` — ``gibbs-temperature.dat`` (eV → J/mol)
* ``read_gibbs_json`` — QHA JSON with ``gibbs_temperature`` (kJ/mol → J/mol)
* ``read_ve_dat`` — ``v-e.dat`` (raw V, E)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from emtdb.config import F_CONST, T_MIN, T_MAX


def read_gibbs_dat(path: str | Path, atom_num: int) -> tuple[np.ndarray, np.ndarray]:
    """Read a ``gibbs-temperature.dat`` file.

    The file contains two white-space-separated columns: temperature (K)
    and energy (eV, total for the supercell).  Energy is converted to
    J/mol/atom via ``G = G_eV * F_CONST / atom_num``.

    Temperatures outside ``[T_MIN, T_MAX]`` (100–2900 K) are discarded.
    """
    data = np.loadtxt(str(path))
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"Expected 2-column data, got shape {data.shape}")

    t = data[:, 0]
    g = data[:, 1] * F_CONST / atom_num

    mask = (t >= T_MIN) & (t <= T_MAX)
    t, g = t[mask], g[mask]

    if len(t) == 0:
        raise ValueError("No data points remain after temperature filtering")

    return t, g


def read_gibbs_json(path: str | Path, atom_num: int) -> tuple[np.ndarray, np.ndarray]:
    """Read a QHA JSON file (Format B / ``QhaData``).

    Extracts ``temperatures`` and ``gibbs_temperature`` arrays.
    ``gibbs_temperature`` is stored in kJ/mol and converted to J/mol.

    The Gibbs array is one element shorter than temperatures (no data at
    0 K), so the leading temperature point is dropped for alignment.
    """
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)

    t = np.array(data["temperatures"])
    g = np.array(data["gibbs_temperature"])

    # gibbs_temperature has no entry at T = 0 K → align by dropping first T.
    if len(t) == len(g) + 1:
        t = t[1:]
    elif len(t) != len(g):
        raise ValueError(
            f"Temperature / gibbs_temperature length mismatch: "
            f"{len(t)} vs {len(g)}"
        )

    g = g * 1000.0  # kJ/mol → J/mol

    mask = (t >= T_MIN) & (t <= T_MAX)
    t, g = t[mask], g[mask]

    if len(t) == 0:
        raise ValueError("No data points remain after temperature filtering")

    return t, g


def read_ve_dat(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a ``v-e.dat`` file containing volume (Å³) and energy (eV) columns.

    Returns ``(volumes, energies)`` sorted by ascending volume.
    """
    data = np.loadtxt(str(path))
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"Expected 2-column data, got shape {data.shape}")

    v = data[:, 0]
    e = data[:, 1]

    idx = np.argsort(v)
    return v[idx], e[idx]
