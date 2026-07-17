#!/usr/bin/env python
"""Gibbs-Temperature Fitting — single-file demo.

Two modes:
  --single  One folder/file, user supplies all parameters
  --batch   Root folder, subfolder names are parsed automatically

Core fitting delegated to ``emtdb.fitters.gibbs.GibbsFitter``.
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

from emtdb.config import F_CONST, PHASE_METRICS
from emtdb.fitters import (
    GibbsFitter,
    expand_results,
    normalize_metrics,
    parse_folder_name,
)
from emtdb.fitters.tdb import format_tdb_func, format_tdb_parameter

VERSION = "2.0.0"
FORMULA = "+A+B*T+C*T*LN(T)+D*T**2+E*T**3+F*T**(-1)"


# ── helpers ─────────────────────────────────────────────────────────────

def _find_gibbs_dat(filepath: Path) -> Path:
    """Locate gibbs-temperature.dat from a file path or folder path."""
    if filepath.is_file():
        return filepath
    if filepath.is_dir():
        hits = sorted(filepath.rglob("gibbs[_-]temperature.dat"))
        if hits:
            return hits[0]
    raise FileNotFoundError(
        f"No gibbs-temperature.dat / gibbs_temperature.dat found in or under {filepath}"
    )


# ── plotting ────────────────────────────────────────────────────────────

def _plot_single(result, filepath):
    """Plot one fit and save to png."""
    if len(result.x_data) == 0:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(result.x_data, result.y_data, "o", ms=3, label="Data")
    ax.plot(
        result.x_data,
        result.y_fit,
        "-",
        lw=2,
        label=f"Fit  R²={result.r2:.6f}",
    )
    ax.set_xlabel("T (K)")
    ax.set_ylabel("G (J/mol/atom)")
    ax.set_title(f"{result.phase}  {','.join(result.elements)}")
    ax.legend()
    parent = filepath.parent if filepath.is_file() else filepath
    out = parent / f"{result.name}_fit.png"
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  [plot] Saved to {out}")


def _plot_batch(results, root, max_subplots_per_image=64):
    """Grid-plot all batch results, splitting into multiple images if needed."""
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
            if len(r.x_data) > 0:
                ax.plot(r.x_data, r.y_data, "o", ms=2, label="Data")
                ax.plot(r.x_data, r.y_fit, "-", lw=1.5, label=f"R²={r.r2:.4f}")
            ax.set_title(r.name, fontsize=7)
            ax.tick_params(labelsize=6)
            ax.legend(fontsize=5)
        for ax in axes.flatten()[n_batch:]:
            ax.axis("off")
        fig.tight_layout()
        suffix = f"_batch{batch_idx + 1}" if num_batches > 1 else ""
        out = root / f"fit_results{suffix}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  [plot] Saved to {out}")


# ── single mode ─────────────────────────────────────────────────────────

def _print_result(result):
    """Print a single FitResult."""
    print(f"  Name:      {result.name}")
    print(f"  Phase:     {result.phase}")
    print(f"  Elements:  {','.join(result.elements)}")
    print(f"  Metrics:   {', '.join(f'{m:.4f}' for m in result.metrics)}")
    print(f"  R² =       {result.r2:.6f}")
    print(f"  G(T) =     {result.expression}")
    print(f"  TDB =      {result.tdb_line}")


def run_single(args):
    """Single-fit mode."""
    filepath = Path(args.filepath)
    dat_path = _find_gibbs_dat(filepath)

    name = filepath.stem if filepath.is_file() else filepath.name
    phase = args.phase.upper()
    metrics_raw = [float(x) for x in args.metrics]
    metrics = normalize_metrics(metrics_raw)
    elements = [e.upper() for e in args.elem]
    atom_num = int(args.atom_num)

    if len(elements) != len(metrics):
        print(f"[ERROR] len(elements)={len(elements)} != len(metrics)={len(metrics)}")
        return 1

    print(f"Folder:    {filepath}")
    print(f"Dat file:  {dat_path}")
    print(f"Phase:     {phase}")
    print(f"Elements:  {','.join(elements)}")
    print(f"Metrics:   {', '.join(f'{m:.4f}' for m in metrics)}")
    print(f"Atom num:  {atom_num}")

    fitter = GibbsFitter(max_trials=100)
    try:
        result = fitter.fit_one(str(dat_path), name, phase, elements, metrics, atom_num)
    except Exception as e:
        print(f"[ERROR] {e}")
        return 1

    for r in expand_results(result):
        if r.phase == "SER":
            r.tdb_line = format_tdb_func(r.elements[0], r.expression)
        else:
            r.tdb_line = format_tdb_parameter(r.phase, r.elements, r.metrics, r.expression)
        print()
        _print_result(r)

    _plot_single(result, filepath)
    return 0


# ── batch mode ──────────────────────────────────────────────────────────

def run_batch(args):
    """Batch-fit mode. Scans subfolders, parses names, fits each."""
    root = Path(args.filepath)
    if not root.is_dir():
        print(f"[ERROR] {root} is not a directory")
        return 1

    phase = args.phase.upper()
    if not args.metrics:
        pm = PHASE_METRICS.get(phase)
        if pm:
            metrics_raw = list(pm)
        else:
            print(f"[ERROR] Unknown phase {phase!r} — please provide --metrics")
            return 1
    else:
        metrics_raw = [float(x) for x in args.metrics]

    metrics = normalize_metrics(metrics_raw)

    print(f"Phase:     {phase}")
    print(f"Metrics:   {', '.join(f'{m:.4f}' for m in metrics)}")
    print(f"Root:      {root}\n")

    results = []
    skipped: list[tuple[str, str]] = []

    for subdir in sorted(root.iterdir()):
        if not subdir.is_dir():
            continue

        parsed = parse_folder_name(subdir.name)
        if parsed is None:
            skipped.append((subdir.name, "cannot parse folder name"))
            continue
        p, elems, atom_num = parsed
        if p.upper() != phase:
            continue
        if len(elems) != len(metrics):
            skipped.append((subdir.name, f"elems ({len(elems)}) != metrics ({len(metrics)})"))
            continue

        try:
            dat_path = _find_gibbs_dat(subdir)
        except FileNotFoundError:
            skipped.append((subdir.name, "no gibbs-temperature.dat"))
            continue

        fitter = GibbsFitter(max_trials=100)
        try:
            result = fitter.fit_one(str(dat_path), subdir.name, phase, elems, metrics, atom_num)
        except Exception as e:
            skipped.append((subdir.name, str(e)))
            continue

        for r in expand_results(result):
            if r.phase == "SER":
                r.tdb_line = format_tdb_func(r.elements[0], r.expression)
            else:
                r.tdb_line = format_tdb_parameter(r.phase, r.elements, r.metrics, r.expression)
            results.append(r)

    print(f"\nFitted: {len(results)} result(s)")
    for r in results:
        print(f"\n  {'─' * 50}")
        _print_result(r)

    if skipped:
        print(f"\n[WARNING] {len(skipped)} folder(s) skipped:")
        print(f"  [HINT] Expected format: {phase}-<elem1>-<elem2>-<atoms>")
        for name, reason in skipped:
            print(f"  - {name}: {reason}")

    if results:
        _plot_batch(results, root)
    return 0 if results else 1


# ── CLI ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=f"Gibbs-Temperature Fitter v{VERSION}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    p_s = sub.add_parser("single", help="Fit one dataset")
    p_s.add_argument("--filepath", required=True, help=".dat file or folder")
    p_s.add_argument("--phase", required=True, help="Phase name (BCC, FCC, SER, ...)")
    p_s.add_argument("--metrics", required=True, nargs="+", type=float,
                     help="Stoichiometry ratios (e.g. 1 3 for FCC)")
    p_s.add_argument("--elem", required=True, nargs="+",
                     help="Element symbols (e.g. Fe Cr)")
    p_s.add_argument("--atom_num", required=True, type=int,
                     help="Number of atoms per formula unit")

    p_b = sub.add_parser("batch", help="Fit all subfolders matching a phase")
    p_b.add_argument("--filepath", required=True, help="Root folder")
    p_b.add_argument("--phase", required=True, help="Phase name")
    p_b.add_argument("--metrics", nargs="+", type=float, default=None,
                     help="Stoichiometry ratios (omit for built-in defaults)")

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
