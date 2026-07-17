#!/usr/bin/env python
"""E0 / Birch-Murnaghan 3rd-order EOS fitting — single-file demo.

Two modes:
  --single  One folder/file, user supplies phase/elements/atom-num
  --batch   Root folder, subfolder names are parsed automatically

Core fitting delegated to ``emtdb.fitters.bm3.Bm3Fitter``.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from emtdb.config import F_CONST, B0_EV_ANG3_TO_GPA, PHASE_METRICS
from emtdb.fitters import (
    Bm3Fitter,
    FitResult,
    expand_results,
    parse_folder_name,
)
from emtdb.fitters.bm3 import bm3_eos
from emtdb.fitters.readers import read_ve_dat
from emtdb.fitters.tdb import (
    format_tdb_etser,
    format_tdb_param_with_etser,
    write_tdb_file,
)

VERSION = "1.0.0"


# ── file / folder helpers ───────────────────────────────────────────────

def _find_v_e_dat(folder: Path) -> Path | None:
    """Find the deepest v-e.dat file under *folder*."""
    files = sorted(folder.rglob("*v-e.dat"))
    return files[-1] if files else None


# ── plotting ────────────────────────────────────────────────────────────

def _plot_single(name: str, result: FitResult, output: str | Path | None = None) -> None:
    """Plot one BM3 fit (E-V data + fitted curve) and save to PNG."""
    vol = result.x_data
    ene = result.y_data
    if len(vol) == 0:
        return

    e0, v0, b0_gpa, b1 = result.params
    b0_ev = b0_gpa / B0_EV_ANG3_TO_GPA

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(vol, ene, "o", ms=5, label="VASP data", zorder=3)

    v_min, v_max = vol.min(), vol.max()
    v_smooth = np.linspace(v_min * 0.95, v_max * 1.05, 200)
    e_fit = bm3_eos(v_smooth, e0, v0, b0_ev, b1)
    ax.plot(v_smooth, e_fit, "-", lw=2, label="BM3 fit", zorder=2)

    txt = (
        f"E₀ = {e0:.6f} eV\n"
        f"V₀ = {v0:.2f} Å³\n"
        f"B₀ = {b0_gpa:.1f} GPa\n"
        f"B₁ = {b1:.2f}\n"
        f"R² = {result.r2:.6f}"
    )
    ax.text(0.97, 0.95, txt, transform=ax.transAxes,
            va="top", ha="right", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.8))
    ax.set_xlabel("Volume (Å³)")
    ax.set_ylabel("Energy (eV)")
    ax.set_title(f"{result.phase}  {','.join(e.upper() for e in result.elements)}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = Path(output) if output else Path(f"{name}_bm3_fit.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  [plot] Saved to {out_path}")


def _plot_batch(results: list[FitResult], out_dir: Path,
                max_subplots_per_image: int = 64) -> None:
    """Grid-plot all batch results, splitting into multiple images."""
    n = len(results)
    num_batches = (n + max_subplots_per_image - 1) // max_subplots_per_image
    for batch_idx in range(num_batches):
        start = batch_idx * max_subplots_per_image
        end = min(start + max_subplots_per_image, n)
        batch = results[start:end]
        n_batch = len(batch)
        grid = int(np.ceil(np.sqrt(n_batch)))
        fig, axes = plt.subplots(grid, grid, figsize=(grid * 5, grid * 4), squeeze=False)
        for ax, r in zip(axes.flatten(), batch):
            vol = r.x_data
            ene = r.y_data
            if len(vol) == 0:
                ax.axis("off")
                continue
            e0, v0, b0_gpa, b1 = r.params
            b0_ev = b0_gpa / B0_EV_ANG3_TO_GPA
            ax.plot(vol, ene, "o", ms=2, label="Data", zorder=3)
            v_s = np.linspace(vol.min() * 0.95, vol.max() * 1.05, 200)
            e_fit = bm3_eos(v_s, e0, v0, b0_ev, b1)
            ax.plot(v_s, e_fit, "-", lw=1.5, label=f"R²={r.r2:.4f}", zorder=2)
            ax.set_title(r.name, fontsize=7)
            ax.tick_params(labelsize=6)
            ax.legend(fontsize=5)
        for ax in axes.flatten()[n_batch:]:
            ax.axis("off")
        fig.tight_layout()
        suffix = f"_batch{batch_idx + 1}" if num_batches > 1 else ""
        out = out_dir / f"bm3_fits{suffix}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  [plot] Saved to {out}")


# ── single mode ─────────────────────────────────────────────────────────

def _print_result(result: FitResult) -> None:
    """Print a single BM3 FitResult."""
    e0, v0, b0, b1 = result.params
    print(f"  Name:      {result.name}")
    print(f"  Phase:     {result.phase}")
    print(f"  Elements:  {','.join(e.upper() for e in result.elements)}")
    print(f"  Metrics:   {', '.join(f'{m:.4f}' for m in result.metrics)}")
    print(f"  R² =       {result.r2:.6f}")
    print(f"  Expression: {result.expression}")
    print(f"  TDB =       {result.tdb_line}")


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
    metrics_raw = list(PHASE_METRICS.get(phase, (1,)))
    if args.metrics:
        if len(args.metrics) != len(metrics_raw):
            print(f"[ERROR] --metrics length ({len(args.metrics)}) "
                  f"doesn't match {phase} ({len(metrics_raw)})")
            return 1
        metrics_raw = [float(m) for m in args.metrics]
    total = sum(metrics_raw)
    metrics = [m / total for m in metrics_raw]

    if len(elems) != len(metrics):
        print(f"[ERROR] len(elems)={len(elems)} != len(metrics)={len(metrics)}")
        return 1

    print(f"v-e.dat:   {ve_path}")
    print(f"Phase:     {phase}")
    print(f"Elements:  {','.join(elems)}")
    print(f"Metrics:   {', '.join(f'{m:.4f}' for m in metrics)}")
    print(f"Atom num:  {atom_num}")
    print()

    volumes, energies = read_ve_dat(str(ve_path))

    fitter = Bm3Fitter(max_trials=30)
    try:
        result = fitter.fit_one(
            str(ve_path), name, phase, elems, metrics, atom_num,
        )
    except ValueError as e:
        print(f"[ERROR] {e}")
        return 1

    e0, v0, b0, b1 = result.params
    print(f"  E₀  = {e0:.6f} eV")
    print(f"  V₀  = {v0:.2f} Å³")
    print(f"  B₀  = {b0:.1f} GPa")
    print(f"  B₁  = {b1:.2f}")
    print(f"  R²  = {result.r2:.6f}")
    print()

    expanded = expand_results(result)
    for r in expanded:
        e0_j = r.params[0] * F_CONST / r.atom_num
        if r.phase.upper() == "SER":
            r.tdb_line = format_tdb_etser(r.elements[0], e0_j)
        else:
            r.tdb_line = format_tdb_param_with_etser(r.phase, r.elements, r.metrics, e0_j)
    print("── TDB output ──────────────────────────────────")
    for r in expanded:
        print(r.tdb_line)

    output_plot = args.output_plot
    if output_plot:
        _plot_single(name, result, output_plot)
    else:
        _plot_single(name, result)

    if args.output:
        write_tdb_file(args.output, expanded)
        print(f"  [tdb] Saved to {args.output}")

    return 0


# ── batch mode ──────────────────────────────────────────────────────────

def run_batch(args: argparse.Namespace) -> int:
    """Batch-fit mode: scan subfolders, parse names, fit each."""
    root = Path(args.root)
    if not root.is_dir():
        print(f"[ERROR] {root} is not a directory")
        return 1

    phase_filter = args.phase.upper() if args.phase else None
    results: list[FitResult] = []
    skipped: list[tuple[str, str]] = []

    for subdir in sorted(root.iterdir()):
        if not subdir.is_dir():
            continue

        parsed = parse_folder_name(subdir.name)
        if parsed is None:
            skipped.append((subdir.name, "cannot parse folder name"))
            continue
        p, elems, atom_num = parsed
        if phase_filter and p != phase_filter:
            continue

        ve_path = _find_v_e_dat(subdir)
        if ve_path is None:
            skipped.append((subdir.name, "no v-e.dat found"))
            continue

        metrics_raw = list(PHASE_METRICS.get(p, (1,)))
        total = sum(metrics_raw)
        metrics = [m / total for m in metrics_raw]

        fitter = Bm3Fitter(max_trials=30)
        try:
            result = fitter.fit_one(
                str(ve_path), subdir.name, p, elems, metrics, atom_num,
            )
        except ValueError as e:
            skipped.append((subdir.name, str(e)))
            continue

        for r in expand_results(result):
            e0_j = r.params[0] * F_CONST / r.atom_num
            if r.phase.upper() == "SER":
                r.tdb_line = format_tdb_etser(r.elements[0], e0_j)
            else:
                r.tdb_line = format_tdb_param_with_etser(r.phase, r.elements, r.metrics, e0_j)
            results.append(r)

    print(f"\nFitted: {len(results)} result(s)")
    for r in results:
        e0, v0, b0, b1 = r.params
        print(f"\n  {'─' * 50}")
        print(f"  Name:      {r.name}")
        print(f"  Phase:     {r.phase}")
        print(f"  Elements:  {','.join(e.upper() for e in r.elements)}")
        print(f"  E₀ = {e0:.6f} eV  "
              f"V₀ = {v0:.2f}  "
              f"B₀ = {b0:.1f} GPa  "
              f"R² = {r.r2:.6f}")

    if skipped:
        print(f"\n[WARNING] {len(skipped)} folder(s) skipped:")
        for name, reason in skipped:
            print(f"  - {name}: {reason}")

    if not results:
        print("[ERROR] No results fitted")
        return 1

    plot_dir = Path(args.output_plot) if args.output_plot else root
    if args.output_plot:
        plot_dir.mkdir(parents=True, exist_ok=True)
    _plot_batch(results, plot_dir)

    if args.output:
        write_tdb_file(args.output, results)
        print(f"\n  [tdb] Saved to {args.output}")

    return 0


# ── CLI ─────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description=f"BM3 E0 (E-V) Fitter v{VERSION} — single-file demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    p_s = sub.add_parser("single", help="Fit one v-e.dat file")
    p_s.add_argument("--filepath", required=True,
                     help="v-e.dat file or folder containing one")
    p_s.add_argument("--phase", required=True,
                     help="Phase name (BCC, FCC, HCP, SER, ...)")
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
        return run_batch(args)
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
