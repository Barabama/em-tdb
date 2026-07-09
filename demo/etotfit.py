#!/usr/bin/env python
"""E0 / Birch-Murnaghan 3rd-order EOS fitting — single-file demo.

Two modes:
  --single  One folder/file, user supplies phase/elements/atom-num
  --batch   Root folder, subfolder names are parsed automatically

Core: scipy.curve_fit → BM3 EOS:
  E(V) = E0 + (9/16)*V0*B0*[((V0/V)^{2/3}-1)^3*B1
                              + ((V0/V)^{2/3}-1)^2*(6-4*(V0/V)^{2/3})]
"""

import argparse
import re
import sys
import traceback
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ── constants ───────────────────────────────────────────────────────────

VERSION = "1.0.0"
F_CONST = 96485  # Faraday constant (C/mol)
B0_EV_ANG3_TO_GPA = 160.2189  # eV/Å³ → GPa

PHASE_METRICS = {
    "SER": (1,),
    "BCC": (1, 1),
    "FCC": (1, 3),
    "HCP": (2, 6),
}


# ── BM3 equation of state ───────────────────────────────────────────────


def _bm3_eos(v: np.ndarray, E0: float, V0: float, B0: float, B1: float) -> np.ndarray:
    """3rd-order Birch-Murnaghan E(V)."""
    x = (V0 / v) ** (2.0 / 3.0)
    return E0 + (9.0 / 16.0) * V0 * B0 * (
        (x - 1.0) ** 3 * B1 + (x - 1.0) ** 2 * (6.0 - 4.0 * x)
    )


def _auto_initial_guess(volumes: np.ndarray, energies: np.ndarray
                        ) -> tuple[float, float, float, float]:
    """Auto-guess BM3 parameters from data midpoint.

    Returns (E0, V0, B0_guess, B1_guess).
    """
    idx = len(volumes) // 2
    return (float(energies[idx]), float(volumes[idx]), 1.2, 3.5)


def _fit_birch_murnaghan(volumes: np.ndarray, energies: np.ndarray,
                         max_trials: int = 30
                         ) -> tuple[float, float, float, float, float]:
    """Fit BM3 EOS with random restarts, return (E0, V0, B0, B1, R²).

    Uses *max_trials* curve_fit attempts with small perturbations to the
    initial guess, keeping the highest R².
    """
    E0_0, V0_0, B0_0, B1_0 = _auto_initial_guess(volumes, energies)

    best_params: tuple[float, float, float, float] | None = None
    best_r2 = -1e9

    for trial in range(max_trials):
        if trial == 0:
            p0 = (E0_0, V0_0, B0_0, B1_0)
        else:
            p0 = (
                E0_0 * (1.0 + np.random.uniform(-1e-3, 1e-3)),
                V0_0 * (1.0 + np.random.uniform(-1e-2, 1e-2)),
                max(0.5, B0_0 * (1.0 + np.random.uniform(-0.3, 0.3))),
                max(2.0, B1_0 * (1.0 + np.random.uniform(-0.2, 0.2))),
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
        print(f"  [WARN] BM3 fit failed — falling back to data minimum")
        return (float(energies[idx_min]), float(volumes[idx_min]), 0.0, 3.5, -1.0)

    return (*best_params, best_r2)


# ── file / folder helpers ───────────────────────────────────────────────


def _find_v_e_dat(folder: Path) -> Path | None:
    """Find the deepest v-e.dat file under *folder*."""
    files = sorted(folder.rglob("*v-e.dat"))
    return files[-1] if files else None


def _read_v_e_dat(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read v-e.dat (two cols: volume(Å³)  energy(eV)) → (volumes, energies)."""
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
                print(f"  [WARN] Skipping unparseable line in {path}: {line}")

    if len(volumes) < 4:
        raise ValueError(
            f"Need at least 4 data points for BM3 fit, got {len(volumes)} in {path}"
        )
    return np.array(volumes), np.array(energies)


def _pad(elem: str) -> str:
    """Pad element symbol to 2 chars (V → VV,  Ni → Ni,  Al → Al)."""
    return elem.strip().upper().ljust(2, elem.strip()[-1].upper())


def _parse_folder_name(name: str, phase_metrics: dict | None = None
                       ) -> dict | None:
    """Parse subfolder name → {phase, elems, metrics, atom_num}.

    Supports::
        SER-Fe-1          → SER, [Fe],   [1.0],          atom_num=1
        BCC-Fe-Mn-2       → BCC, [Fe,Mn], [0.5,0.5],      atom_num=2
        FCC-Al-Co-4       → FCC, [Al,Co], [0.25,0.75],    atom_num=4
        HCP-Co-Ni-8       → HCP, [Co,Ni], [0.25,0.75],    atom_num=8

    Returns None on parse failure.
    """
    if phase_metrics is None:
        phase_metrics = PHASE_METRICS

    parts = name.split("-")
    if len(parts) < 2:
        return None
    phase = parts[0].upper()
    if phase not in phase_metrics:
        return None

    metrics = list(phase_metrics[phase])
    m_sum = sum(metrics)
    metrics = [m / m_sum for m in metrics]
    elems = parts[1: 1 + len(metrics)]
    if not elems:
        return None

    m = re.search(r"-(\d+)(?:atoms?)?$", name)
    atom_num = int(m.group(1)) if m else 1

    return {"phase": phase, "elems": elems, "metrics": metrics, "atom_num": atom_num}


# ── fit logic ───────────────────────────────────────────────────────────


def fit_one(ve_path: Path, name: str, phase: str,
            elems: list[str], metrics: list[float], atom_num: int
            ) -> dict:
    """Fit BM3 to one v-e.dat file and return a result dict.

    Returns::

        {
            "name": str,
            "phase": str,
            "elems": list[str],
            "metrics": list[float],   # normalised stoichiometry
            "atom_num": int,
            "E0_eV": float,           # equilibrium energy (eV/atom)
            "V0_Ang3": float,         # equilibrium volume (Å³)
            "B0_GPa": float,          # bulk modulus (GPa)
            "B1": float,              # dB/dP
            "R2": float,              # goodness of fit
            "n_points": int,
            "volumes": np.ndarray,
            "energies": np.ndarray,
        }
    """
    volumes, energies = _read_v_e_dat(ve_path)
    E0, V0, B0_eV, B1, r2 = _fit_birch_murnaghan(volumes, energies)
    B0_GPa = B0_eV * B0_EV_ANG3_TO_GPA

    return {
        "name": name,
        "phase": phase,
        "elems": elems,
        "metrics": metrics,
        "atom_num": atom_num,
        "E0_eV": E0,
        "V0_Ang3": V0,
        "B0_GPa": B0_GPa,
        "B1": B1,
        "R2": r2,
        "n_points": len(volumes),
        "volumes": volumes,
        "energies": energies,
    }


# ── TDB formatting ──────────────────────────────────────────────────────


def _format_tdb_func(elem: str, e0_j: float) -> str:
    """Format a SER reference function line."""
    elem_pad = _pad(elem)
    return (
        f" FUNCTION ETSER{elem_pad}   1.00 {e0_j:+E}; 6000.00 N !\n"
    )


def _format_tdb_param(phase: str, elems: list[str],
                      metrics: list[float], e0_j: float) -> str:
    """Format one G parameter line (no symmetric exchange — handled upstream)."""
    comp_str = ":".join(e.upper() for e in elems)
    ser_ref = "".join(
        f"-{m:.4g}*ETSER{_pad(e)}#" for e, m in zip(elems, metrics)
    )
    return (
        f" PARAMETER G({phase.upper()},{comp_str};0)   1.00"
        f" {e0_j:+E}{ser_ref}; 6000.00 N !\n"
    )


def _result_to_tdb(result: dict) -> list[str]:
    """Convert a fit result to TDB lines (funcs or params)."""
    lines: list[str] = []
    e0_j = result["E0_eV"] * F_CONST / result["atom_num"]

    if result["phase"].upper() == "SER":
        lines.append(_format_tdb_func(result["elems"][0], e0_j))
    else:
        lines.append(_format_tdb_param(
            result["phase"], result["elems"], result["metrics"], e0_j,
        ))
    return lines


# ── plotting ────────────────────────────────────────────────────────────


def _plot_single(name: str, result: dict, output: str | Path | None = None
                 ) -> None:
    """Plot one BM3 fit (E-V data + fitted curve) and save to PNG."""
    vol = result["volumes"]
    ene = result["energies"]

    fig, ax = plt.subplots(figsize=(7, 5))

    # Data points
    ax.plot(vol, ene, "o", ms=5, label="VASP data", zorder=3)

    # Fitted BM3 curve (smooth)
    v_min, v_max = vol.min(), vol.max()
    v_smooth = np.linspace(v_min * 0.95, v_max * 1.05, 200)
    e_fit = _bm3_eos(
        v_smooth,
        result["E0_eV"], result["V0_Ang3"],
        result["B0_GPa"] / B0_EV_ANG3_TO_GPA, result["B1"],
    )
    ax.plot(v_smooth, e_fit, "-", lw=2, label="BM3 fit", zorder=2)

    # Annotate key parameters
    txt = (
        f"E₀ = {result['E0_eV']:.6f} eV\n"
        f"V₀ = {result['V0_Ang3']:.2f} Å³\n"
        f"B₀ = {result['B0_GPa']:.1f} GPa\n"
        f"B₁ = {result['B1']:.2f}\n"
        f"R² = {result['R2']:.6f}"
    )
    ax.text(0.97, 0.95, txt, transform=ax.transAxes,
            va="top", ha="right", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.8))

    ax.set_xlabel("Volume (Å³)")
    ax.set_ylabel("Energy (eV)")
    ax.set_title(f"{result['phase']}  {','.join(e.upper() for e in result['elems'])}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = Path(output) if output else Path(f"{name}_bm3_fit.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  [plot] Saved to {out_path}")


def _plot_batch(results: list[dict], out_dir: Path,
                max_subplots_per_image: int = 64) -> None:
    """Grid-plot all batch results, splitting into multiple images if needed."""
    n = len(results)
    num_batches = (n + max_subplots_per_image - 1) // max_subplots_per_image

    for batch_idx in range(num_batches):
        start = batch_idx * max_subplots_per_image
        end = min(start + max_subplots_per_image, n)
        batch = results[start:end]
        n_batch = len(batch)

        grid = int(np.ceil(np.sqrt(n_batch)))
        fig, axes = plt.subplots(
            grid, grid, figsize=(grid * 5, grid * 4), squeeze=False
        )

        for ax, r in zip(axes.flatten(), batch):
            vol = r["volumes"]
            ene = r["energies"]
            ax.plot(vol, ene, "o", ms=2, label="Data", zorder=3)

            v_s = np.linspace(vol.min() * 0.95, vol.max() * 1.05, 200)
            e_fit = _bm3_eos(
                v_s, r["E0_eV"], r["V0_Ang3"],
                r["B0_GPa"] / B0_EV_ANG3_TO_GPA, r["B1"],
            )
            ax.plot(v_s, e_fit, "-", lw=1.5, label=f"R²={r['R2']:.4f}", zorder=2)

            ax.set_title(r["name"], fontsize=7)
            ax.tick_params(labelsize=6)
            ax.legend(fontsize=5)

        for ax in axes.flatten()[n_batch:]:
            ax.axis("off")

        fig.tight_layout()
        if num_batches > 1:
            out = out_dir / f"bm3_fits_batch{batch_idx + 1}.png"
        else:
            out = out_dir / "bm3_fits.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  [plot] Saved to {out}")


# ── single mode ─────────────────────────────────────────────────────────


def _expand_results(result: dict) -> list[dict]:
    """Return results list; add exchanged entry for equal-sublattice binary."""
    out = [result]
    elems = result["elems"]
    metrics = result["metrics"]
    if (len(elems) == 2 and elems[0] != elems[1]
            and len(metrics) > 1 and metrics[0] == metrics[1]):
        out.append({
            **result,
            "name": f"{result['name']}-ex",
            "elems": [elems[1], elems[0]],
            "metrics": [metrics[1], metrics[0]],
        })
    return out


def run_single(args: argparse.Namespace) -> int:
    """Single-fit mode: process one v-e.dat file."""
    filepath = Path(args.filepath)
    if filepath.is_dir():
        ve_path = _find_v_e_dat(filepath)
        if ve_path is None:
            print(f"[ERROR] No v-e.dat found under {filepath}")
            return 1
        name = filepath.name
    else:
        ve_path = filepath
        name = filepath.stem

    phase = args.phase.upper()
    elems = [e.upper() for e in args.elem]
    atom_num = int(args.atom_num)

    if phase not in PHASE_METRICS:
        print(f"[ERROR] Unknown phase {phase!r}. Choose from: {', '.join(PHASE_METRICS)}")
        return 1

    metrics_raw = list(PHASE_METRICS[phase])
    if args.metrics:
        if len(args.metrics) != len(metrics_raw):
            print(f"[ERROR] --metrics length ({len(args.metrics)}) "
                  f"doesn't match {phase} ({len(metrics_raw)})")
            return 1
        metrics_raw = [float(m) for m in args.metrics]
    m_sum = sum(metrics_raw)
    metrics = [m / m_sum for m in metrics_raw]

    if len(elems) != len(metrics):
        print(f"[ERROR] len(elems)={len(elems)} != len(metrics)={len(metrics)}")
        return 1

    print(f"v-e.dat:   {ve_path}")
    print(f"Phase:     {phase}")
    print(f"Elements:  {','.join(elems)}")
    print(f"Metrics:   {', '.join(f'{m:.4f}' for m in metrics)}")
    print(f"Atom num:  {atom_num}")
    print()

    try:
        result = fit_one(ve_path, name, phase, elems, metrics, atom_num)
    except ValueError as e:
        print(f"[ERROR] {e}")
        return 1

    print(f"  E₀  = {result['E0_eV']:.6f} eV")
    print(f"  V₀  = {result['V0_Ang3']:.2f} Å³")
    print(f"  B₀  = {result['B0_GPa']:.1f} GPa")
    print(f"  B₁  = {result['B1']:.2f}")
    print(f"  R²  = {result['R2']:.6f}")
    print(f"  N   = {result['n_points']} points")
    print()

    # Print TDB output for each expanded result
    print("── TDB output ──────────────────────────────────")
    for r in _expand_results(result):
        for line in _result_to_tdb(r):
            print(line, end="")

    # Plot
    output_plot = args.output_plot
    if output_plot:
        _plot_single(name, result, output_plot)
    else:
        _plot_single(name, result)

    # Full TDB file
    if args.output:
        _write_tdb(args.output, _expand_results(result))
        print(f"  [tdb] Saved to {args.output}")

    return 0


# ── batch mode ──────────────────────────────────────────────────────────


def _write_tdb(path: str | Path, results: list[dict]) -> None:
    """Write all results to a TDB file."""
    with open(path, "w", encoding="utf-8") as f:
        f.write("$ Thermodynamic database generated by demo/etotfit.py\n")
        f.write(f"$ {len(results)} end-member(s)\n")
        f.write("$ \n")
        f.write(" TYPE_DEFINITION % SEQ *;\n")
        f.write(" DEFAULT_COMMAND DEFSYMBOL $ N  !\n\n")

        has_funcs = any(r["phase"].upper() == "SER" for r in results)
        has_params = any(r["phase"].upper() != "SER" for r in results)

        if has_funcs:
            f.write("$ ── SER reference functions ──\n")
            for r in results:
                if r["phase"].upper() == "SER":
                    for line in _result_to_tdb(r):
                        f.write(line)

        if has_params:
            f.write("$ ── Phase parameters ──\n")
            for r in results:
                if r["phase"].upper() != "SER":
                    for line in _result_to_tdb(r):
                        f.write(line)


def run_batch(args: argparse.Namespace) -> int:
    """Batch-fit mode: scan subfolders, parse names, fit each."""
    root = Path(args.root)
    if not root.is_dir():
        print(f"[ERROR] {root} is not a directory")
        return 1

    phase_filter = args.phase.upper() if args.phase else None

    results: list[dict] = []
    skipped: list[tuple[str, str]] = []

    for subdir in sorted(root.iterdir()):
        if not subdir.is_dir():
            continue

        parsed = _parse_folder_name(subdir.name)
        if parsed is None:
            skipped.append((subdir.name, "cannot parse folder name"))
            continue
        if phase_filter and parsed["phase"] != phase_filter:
            continue

        ve_path = _find_v_e_dat(subdir)
        if ve_path is None:
            skipped.append((subdir.name, "no v-e.dat found"))
            continue

        try:
            result = fit_one(
                ve_path, subdir.name,
                parsed["phase"], parsed["elems"],
                parsed["metrics"], parsed["atom_num"],
            )
        except ValueError as e:
            skipped.append((subdir.name, str(e)))
            continue

        # Expand exchanged entries
        for r in _expand_results(result):
            results.append(r)

    print(f"\nFitted: {len(results)} result(s)")
    for r in results:
        print(f"\n  {'─' * 50}")
        print(f"  Name:      {r['name']}")
        print(f"  Phase:     {r['phase']}")
        print(f"  Elements:  {','.join(e.upper() for e in r['elems'])}")
        print(f"  E₀ = {r['E0_eV']:.6f} eV  "
              f"V₀ = {r['V0_Ang3']:.2f}  "
              f"B₀ = {r['B0_GPa']:.1f} GPa  "
              f"R² = {r['R2']:.6f}")

    if skipped:
        print(f"\n[WARNING] {len(skipped)} folder(s) skipped:")
        for name, reason in skipped:
            print(f"  - {name}: {reason}")

    if not results:
        print("[ERROR] No results fitted")
        return 1

    # Plot
    output_plot_dir = args.output_plot
    if output_plot_dir:
        plot_dir = Path(output_plot_dir)
        plot_dir.mkdir(parents=True, exist_ok=True)
    else:
        plot_dir = root
    _plot_batch(results, plot_dir)

    # TDB file output
    if args.output:
        _write_tdb(args.output, results)
        print(f"\n  [tdb] Saved to {args.output}")

    return 0


# ── CLI ─────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description=f"BM3 E0 (E-V) Fitter v{VERSION} — single-file demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    sub = parser.add_subparsers(dest="mode", required=True)

    # ── single ──
    p_s = sub.add_parser("single", help="Fit one v-e.dat file")
    p_s.add_argument("--filepath", required=True,
                     help="v-e.dat file or folder containing one")
    p_s.add_argument("--phase", required=True,
                     help="Phase name (BCC, FCC, HCP, SER, …)")
    p_s.add_argument("--elem", required=True, nargs="+",
                     help="Element symbols (e.g. Fe Cr)")
    p_s.add_argument("--atom-num", required=True, type=int,
                     help="Number of atoms per formula unit")
    p_s.add_argument("--metrics", nargs="+", type=float, default=None,
                     help="Stoichiometry ratios (default: from PHASE_METRICS)")
    p_s.add_argument("--output", "-o", type=str, default="",
                     help="Output TDB file path")
    p_s.add_argument("--output-plot", type=str, default="",
                     help="Output plot image path (default: {name}_bm3_fit.png)")

    # ── batch ──
    p_b = sub.add_parser("batch", help="Fit all subfolders under a root")
    p_b.add_argument("--root", required=True,
                     help="Root directory containing end-member folders")
    p_b.add_argument("--phase", default="",
                     help="Phase filter (omit to process all phases)")
    p_b.add_argument("--output", "-o", type=str, default="",
                     help="Output TDB file path")
    p_b.add_argument("--output-plot", type=str, default="",
                     help="Output plot directory (default: root dir)")

    args = parser.parse_args()

    try:
        if args.mode == "single":
            return run_single(args)
        else:
            return run_batch(args)
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
