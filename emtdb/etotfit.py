"""
EM-TDB - DFT static energy (E0) fitter.

Fits 3rd-order Birch-Murnaghan EOS to VASP v-e.dat data,
extracting E0 (minimum energy), V0 (equilibrium volume),
B0 (bulk modulus), and B1 (dB/dP) for each end-member.

Outputs TDB FUNCTION (ETSERxx) / PARAMETER (G(phase,e1:e2;0))
entries that can be imported into the database.

Usage:
    from emtdb.etotfit import ETotFitter

    fitter = ETotFitter(PHASE_METRICS)
    parsed = fitter.process_folders("path/to/data")
    # parsed.funcs    → SER reference functions (ETSERxx)
    # parsed.phases   → phase definitions
    # parsed.params   → G parameters with SER ref subtraction
"""

import json
import logging
import random
import re
from dataclasses import dataclass, asdict
from itertools import groupby
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

from emtdb.tdb.tdbi import Func, Phase, Param
from emtdb.tdb.tdbmgr import ParsedData

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Data containers
# ──────────────────────────────────────────────────────────────────────

@dataclass
class E0FitResult:
    """Result of BM3 fitting for one end-member folder."""
    name: str
    phase: str
    elements: list[str]
    metrics: list[float]
    atom_num: int
    E0_eV: float          # equilibrium energy (eV/atom)
    V0_Ang3: float        # equilibrium volume (Å³)
    B0_GPa: float         # bulk modulus (GPa)
    B1: float             # dB/dP
    R2: float             # goodness of fit
    n_points: int         # number of V-E data points


# ──────────────────────────────────────────────────────────────────────
# BM3 equation of state
# ──────────────────────────────────────────────────────────────────────

def _bm3_eos(v: np.ndarray, E0: float, V0: float, B0: float, B1: float) -> np.ndarray:
    """3rd-order Birch-Murnaghan E(V).

    E(V) = E0 + (9/16)*V0*B0*[((V0/V)^{2/3}-1)^3*B1
                                + ((V0/V)^{2/3}-1)^2*(6-4*(V0/V)^{2/3})]
    """
    x = (V0 / v) ** (2.0 / 3.0)
    return E0 + (9.0 / 16.0) * V0 * B0 * (
        (x - 1.0) ** 3 * B1 + (x - 1.0) ** 2 * (6.0 - 4.0 * x)
    )


def _auto_initial_guess(volumes: np.ndarray, energies: np.ndarray
                        ) -> tuple[float, float, float, float]:
    """Auto-guess BM3 parameters — no user input needed.

    P1(E0), P2(V0) — midpoint of sorted data.
    P3(B0)  — 1.2 eV/Å³ (~192 GPa), typical metal.
    P4(B1)  — 3.5 (standard for metals).
    """
    idx = len(volumes) // 2
    return (float(energies[idx]), float(volumes[idx]), 1.2, 3.5)


def _fit_birch_murnaghan(volumes: np.ndarray, energies: np.ndarray,
                         max_trials: int = 30) -> tuple[float, float, float, float, float]:
    """Fit BM3 EOS with random restarts, return (E0, V0, B0, B1, R²).

    Uses multiple ``curve_fit`` attempts (up to *max_trials*) with small
    random perturbations to the auto-guess, keeping the highest R².
    """
    E0_0, V0_0, B0_0, B1_0 = _auto_initial_guess(volumes, energies)

    best_params: tuple[float, float, float, float] | None = None
    best_r2 = -1e9

    for trial in range(max_trials):
        if trial == 0:
            p0 = (E0_0, V0_0, B0_0, B1_0)
        else:
            p0 = (
                E0_0 * (1.0 + random.uniform(-1e-3, 1e-3)),
                V0_0 * (1.0 + random.uniform(-1e-2, 1e-2)),
                max(0.5, B0_0 * (1.0 + random.uniform(-0.3, 0.3))),
                max(2.0, B1_0 * (1.0 + random.uniform(-0.2, 0.2))),
            )
        try:
            params, _ = curve_fit(_bm3_eos, volumes, energies, p0=p0, maxfev=2000)
        except (RuntimeError, ValueError):
            continue

        residuals = energies - _bm3_eos(volumes, *params)
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((energies - np.mean(energies)) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else -1.0

        if r2 > best_r2:
            best_r2 = r2
            best_params = tuple(params)
            if r2 > 0.9999:
                break

    if best_params is None:
        idx_min = int(np.argmin(energies))
        log.warning("BM3 fit failed — falling back to data minimum")
        return (float(energies[idx_min]), float(volumes[idx_min]), 0.0, 3.5, -1.0)

    return (*best_params, best_r2)


# ──────────────────────────────────────────────────────────────────────
# Folder / file helpers
# ──────────────────────────────────────────────────────────────────────

def _parse_folder_name(name: str, phase_metrics: dict
                       ) -> dict:
    """Parse folder name → {phase, elems, metrics, atom_num}.

    Supports::

        SER-Fe-1          → SER, [Fe],  [1.0],   atom_num=1
        BCC-Fe-Mn-2       → BCC, [Fe,Mn], [0.5,0.5], atom_num=2
        FCC-Al-Co-4       → FCC, [Al,Co], [0.25,0.75], atom_num=4
        HCP-Co-Ni-8       → HCP, [Co,Ni], [0.25,0.75], atom_num=8
    """
    parts = name.split("-")
    if len(parts) < 2:
        raise ValueError(f"Invalid folder name: {name}")
    phase = parts[0].upper()
    if phase not in phase_metrics:
        raise ValueError(f"Unknown phase '{phase}' in '{name}'")

    metrics = list(phase_metrics[phase])
    m_sum = sum(metrics)
    metrics = [m / m_sum for m in metrics]
    elems = parts[1: 1 + len(metrics)]

    m = re.search(r"-(\d+)(?:atoms?)?$", name)
    atom_num = int(m.group(1)) if m else 1

    return {"phase": phase, "elems": elems, "metrics": metrics, "atom_num": atom_num}


def _pad(elem: str) -> str:
    """Pad element symbol to 2 chars by repeating the last character.

    V → VV,  Ni → Ni,  Al → Al
    """
    return elem.strip().upper().ljust(2, elem.strip()[-1].upper())


def _find_v_e_dat(folder: Path) -> Path | None:
    """Find the deepest v-e.dat file under *folder*."""
    files = [p for p in folder.rglob("*v-e.dat") if p.is_file()]
    return files[-1] if files else None


def _read_v_e_dat(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read v-e.dat (two columns: volume(Å³)  energy(eV)) → (volumes, energies)."""
    volumes: list[float] = []
    energies: list[float] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                volumes.append(float(parts[0]))
                energies.append(float(parts[1]))
            except ValueError:
                log.warning("Skipping unparseable line in %s: %s", path, line)

    if len(volumes) < 4:
        raise ValueError(
            f"Need at least 4 data points for BM3 fit, got {len(volumes)} in {path}"
        )
    return np.array(volumes), np.array(energies)


# ──────────────────────────────────────────────────────────────────────
# ETotFitter class (the public API)
# ──────────────────────────────────────────────────────────────────────

class ETotFitter:
    """Fit DFT static energies (E0) from v-e.dat files.

    Args:
        phase_metrics: Dict mapping phase name → tuple of sublattice
            occupancies, e.g. ``{"SER": (1,), "BCC": (1, 1), ...}``.

    Example::

        from emtdb.config import PHASE_METRICS
        fitter = ETotFitter(PHASE_METRICS)
        parsed = fitter.process_folders("/path/to/data")
        # parsed.funcs  → SER functions (ETSERxx)
        # parsed.params → G parameters
    """

    def __init__(self, phase_metrics: dict):
        self.phase_metrics = phase_metrics

    # ── Public methods ────────────────────────────────────────────────

    def process_folders(self, directory: Path | str) -> list[E0FitResult]:
        """Walk *directory*, fit E0 for each end-member subfolder.

        Args:
            directory: Root directory whose immediate subdirectories are
                end-member folders (e.g. ``SER-Fe-1/``, ``BCC-Fe-Mn-2/``).

        Returns:
            List of :class:`E0FitResult` objects.
        """
        directory = Path(directory) if isinstance(directory, str) else directory
        results: list[E0FitResult] = []

        for item in sorted(directory.iterdir()):
            if not item.is_dir():
                continue

            try:
                info = _parse_folder_name(item.name, self.phase_metrics)
            except (ValueError, KeyError) as exc:
                log.warning("Skipping %s: %s", item.name, exc)
                continue

            ve_path = _find_v_e_dat(item)
            if not ve_path:
                log.warning("Skipping %s: no v-e.dat found", item.name)
                continue

            try:
                volumes, energies = _read_v_e_dat(ve_path)
            except ValueError as exc:
                log.warning("Skipping %s: %s", item.name, exc)
                continue

            E0, V0, B0, B1, r2 = _fit_birch_murnaghan(volumes, energies)

            # Store a result
            result = E0FitResult(
                name=item.name,
                phase=info["phase"],
                elements=[e.upper() for e in info["elems"]],
                metrics=info["metrics"],
                atom_num=info["atom_num"],
                E0_eV=E0,
                V0_Ang3=V0,
                B0_GPa=B0 * 160.2189,
                B1=B1,
                R2=r2,
                n_points=len(volumes),
            )
            results.append(result)

            emsg = ""
            if r2 < 0.5:
                emsg = " (poor fit)"
            log.info("  %s: E0=%.6f eV  V0=%.2f  B0=%.1f GPa  R²=%.6f%s",
                     item.name, E0, V0, result.B0_GPa, r2, emsg)

        return results

    def results_to_parsed(self, results: list[E0FitResult],
                          tdb_name: str = "") -> ParsedData:
        """Convert fitted results into :class:`ParsedData` for DB / TDB export.

        SER results become ``ETSERxx`` functions.
        Non-SER results become ``G(phase,e1:e2;0)`` parameters with
        reference-function subtraction.

        Args:
            results: List from :meth:`process_folders`.
            tdb_name: Logical TDB name (default: empty string).

        Returns:
            :class:`ParsedData` with ``funcs``, ``phases``, ``params``.
        """

        funcs: list[Func] = []
        phases: list[Phase] = []
        params: list[Param] = []

        ser_results = [r for r in results if r.phase == "SER"]
        non_ser = [r for r in results if r.phase != "SER"]

        # ── SER functions ──
        seen_funcs: set[str] = set()
        for r in ser_results:
            elem = r.elements[0]
            func_name = f"ETSER{_pad(elem)}"
            if func_name not in seen_funcs:
                seen_funcs.add(func_name)
                e0_j = r.E0_eV * 96485 / r.atom_num
                funcs.append(Func(
                    func=func_name,
                    elem=elem,
                    temp_start=1.0,
                    temp_end=6000.0,
                    expression=f"{e0_j:+E}",
                    is_continued="N",
                ))

        # ── Phase definitions & parameters ──
        # Group results by phase
        phase_groups = {
            ph: list(grp)
            for ph, grp in groupby(
                sorted(non_ser, key=lambda r: r.phase),
                key=lambda r: r.phase,
            )
        }

        for phase, fits in phase_groups.items():
            # Phase definition from the first result's metrics
            metrics = tuple(fits[0].metrics)
            # Normalised metrics → raw occupancies
            raw_metrics = self.phase_metrics.get(phase)
            if raw_metrics is None:
                continue
            # Collect components per sublattice
            n_subl = len(metrics)
            comps_by_subl: list[list[str]] = [[] for _ in range(n_subl)]
            for fit in fits:
                for i, elem in enumerate(fit.elements):
                    if elem not in comps_by_subl[i]:
                        comps_by_subl[i].append(elem)
            components = ":".join(
                ",".join(sorted(set(c))) for c in comps_by_subl
            )

            phases.append(Phase(
                phase=phase,
                sub_lattices=len(raw_metrics),
                stoichiometry=" ".join(str(m) for m in raw_metrics),
                components=components,
                tdb=tdb_name,
            ))

            # Parameters for each fit
            for fit in fits:
                elems = fit.elements
                comp_str = ":".join(elems)
                e0_j = fit.E0_eV * 96485 / fit.atom_num
                ser_ref = "".join(
                    f"-{m}*ETSER{_pad(e)}#" for e, m in zip(elems, fit.metrics)
                )
                params.append(Param(
                    param="",
                    ptype="G",
                    phase=phase,
                    components=comp_str,
                    order_num=0,
                    temp_start=1.0,
                    temp_end=6000.0,
                    tdb=tdb_name,
                    expression=f"{e0_j:+E}{ser_ref}",
                    is_continued="N",
                ))

                # Symmetric exchange for equal-sublattice binary phases
                if (len(elems) == 2 and elems[0] != elems[1]
                        and fit.metrics[0] == fit.metrics[1]):
                    ex_elems = list(reversed(elems))
                    ex_comp = ":".join(ex_elems)
                    ex_ref = "".join(
                        f"-{m}*ETSER{_pad(e)}#" for e, m in zip(ex_elems, fit.metrics)
                    )
                    params.append(Param(
                        param="",
                        ptype="G",
                        phase=phase,
                        components=ex_comp,
                        order_num=0,
                        temp_start=1.0,
                        temp_end=6000.0,
                        tdb=tdb_name,
                        expression=f"{e0_j:+E}{ex_ref}",
                        is_continued="N",
                    ))

        return ParsedData(elems=[], funcs=funcs, phases=phases, params=params, tdb=tdb_name)

    def export_json(self, results: list[E0FitResult], output: Path | str) -> None:
        """Export fitting results to JSON (includes E0, V0, B0, B1, R²)."""
        output = Path(output) if isinstance(output, str) else output
        rows = []
        for r in results:
            rows.append(asdict(r))
        with open(output, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)
        log.info("Exported %d fit results to %s", len(results), output)

    def export_csv(self, results: list[E0FitResult], output: Path | str) -> None:
        """Export fitting results to CSV (name, phase, elements, E0, V0, B0, B1, R²)."""
        import csv

        output = Path(output) if isinstance(output, str) else output
        fieldnames = ["name", "phase", "elements", "E0_eV", "V0_Ang3",
                      "B0_GPa", "B1", "R2", "n_points"]
        with open(output, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                row = asdict(r)
                row["elements"] = " ".join(r.elements)
                # Only keep fields listed in fieldnames
                writer.writerow({k: row[k] for k in fieldnames})
        log.info("Exported %d fit results to %s", len(results), output)
