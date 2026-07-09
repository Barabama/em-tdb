#!/usr/bin/env python
"""Gibbs-Temperature Fitting — single-file demo.

Two modes:
  --single  One folder/file, user supplies all parameters
  --batch   Root folder, subfolder names are parsed automatically

Core: scipy curve_fit → SGTE polynomial A+B*T+C*T*LN(T)+D*T**2+E*T**3+F*T**(-1)
"""

import argparse
import re
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from tqdm import tqdm

# ── constants ───────────────────────────────────────────────────────────

VERSION = "2.0.0"
F_CONST = 96485  # Faraday constant (C/mol)

FORMULA = "+A+B*T+C*T*LN(T)+D*T**2+E*T**3+F*T**(-1)"

T_MIN, T_MAX = 100, 2900

PHASE_METRICS = {
    "SER": (1,),
    "BCC": (1, 1),
    "FCC": (1, 3),
    "HCP": (2, 6),
}

ELEMENTS = frozenset({
    "AC", "AG", "AL", "AM", "AR", "AS", "AT", "AU",
    "B", "BA", "BE", "BH", "BI", "BK", "BR",
    "C", "CA", "CD", "CE", "CF", "CL", "CM", "CN", "CO", "CR", "CS", "CU",
    "DB", "DS", "DY",
    "ER", "ES", "EU",
    "F", "FE", "FL", "FM", "FR",
    "GA", "GD", "GE",
    "H", "HE", "HF", "HG", "HO", "HS",
    "I", "IN", "IR",
    "K", "KR",
    "LA", "LI", "LR", "LU", "LV",
    "MC", "MD", "MG", "MN", "MO", "MT",
    "N", "NA", "NB", "ND", "NE", "NH", "NI", "NO", "NP",
    "O", "OG", "OS",
    "P", "PA", "PB", "PD", "PM", "PO", "PR", "PT", "PU",
    "RA", "RB", "RE", "RF", "RG", "RH", "RN", "RU",
    "S", "SB", "SC", "SE", "SG", "SI", "SM", "SN", "SR",
    "TA", "TB", "TC", "TE", "TH", "TI", "TL", "TM", "TS",
    "U",
    "V", "VA",
    "W",
    "XE",
    "Y", "YB",
    "ZN", "ZR",
})

# ── helpers ─────────────────────────────────────────────────────────────

def _find_gibbs_dat(filepath: Path) -> Path:
    """Locate gibbs-temperature.dat from a file path or folder path."""
    if filepath.is_file():
        return filepath
    if filepath.is_dir():
        hits = sorted(filepath.rglob("gibbs[_-]temperature.dat"))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"No gibbs-temperature.dat / gibbs_temperature.dat found in or under {filepath}")


def _read_dat(filepath: Path, atom_num: int) -> pd.DataFrame:
    """Read gibbs-temperature.dat, return T, G (J/mol/atom) as DataFrame."""
    data = pd.read_csv(filepath, sep=r"\s+", skiprows=1, header=None, names=["T", "G"])
    data = data[(data["T"] >= T_MIN) & (data["T"] <= T_MAX)]
    data["G"] = data["G"] * F_CONST / atom_num
    return data.reset_index(drop=True)


_ELEM_TOKEN_RE = re.compile(r"([A-Z][a-z]?)(\d*)")


def _parse_folder_name(name: str) -> dict | None:
    """Parse subfolder name → {phase, elements, atom_num}.

    Supports:
        BCC-Al-Al-2      → phase=BCC,  elems=[Al, Al], atom_num=2
        BCC-TiNb-2       → phase=BCC,  elems=[Ti, Nb], atom_num=2
        BCC-Ta2-Al-8     → phase=BCC,  elems=[Ta, Al], atom_num=8
        SER-Nb-2atoms    → phase=SER,  elems=[Nb],    atom_num=2
        OTH-Al-Ti-Nb-4   → phase=OTH,  elems=[Al,Ti,Nb], atom_num=4
        FCC-Fe-Mn        → phase=FCC,  elems=[Fe, Mn], atom_num=1

    Returns None on parse failure.
    """
    parts = name.split("-")
    if len(parts) < 2:
        return None
    phase = parts[0].upper()

    # Walk tokens from right to extract atom_num first
    atom_num = None
    elem_tokens = []
    for part in reversed(parts[1:]):
        if atom_num is None:
            m = re.match(r"^(\d+)(?:atoms?)?$", part)
            if m:
                atom_num = int(m.group(1))
                continue
        elem_tokens.insert(0, part)

    if atom_num is None:
        atom_num = 1

    # Parse element tokens
    elements = []
    for tok in elem_tokens:
        if tok.upper() in ELEMENTS:
            elements.append(tok.upper())
        else:
            pos = 0
            local = []
            while pos < len(tok):
                m2 = _ELEM_TOKEN_RE.match(tok, pos)
                if m2 is None:
                    return None
                sym = m2.group(1).upper()
                if sym not in ELEMENTS:
                    return None
                local.append(sym)
                pos = m2.end()
            if not local:
                return None
            elements.extend(local)

    if not elements:
        return None
    return {"phase": phase, "elements": elements, "atom_num": atom_num}


# ── fit logic ───────────────────────────────────────────────────────────

def _fit_func(x, A, B, C, D, E, F):
    return A + B * x + C * x * np.log(x) + D * x**2 + E * x**3 + F / x


def _formula_str(params):
    """Format [A,B,C,D,E,F] → SGTE expression string."""
    return (
        FORMULA.replace("+A", f"{params[0]:+E}")
        .replace("+B", f"{params[1]:+E}")
        .replace("+C", f"{params[2]:+E}")
        .replace("+D", f"{params[3]:+E}")
        .replace("+E", f"{params[4]:+E}")
        .replace("+F", f"{params[5]:+E}")
    )


def _format_tdb_parameter(phase, elements, norm_metrics, expression):
    """Build a full TDB PARAMETER line.

    Example:
        PARAMETER G(FCC,ZR:ZR;0) 1.00 -8.138924E+05+... -0.25*GHSERZR#-0.75*GHSERZR#; 6000.00 N !
    """
    elems_str = ":".join(e.upper() for e in elements)
    ser_terms = "".join(
        f"-{m:.4g}*GHSER{e.upper()}#"
        for e, m in zip(elements, norm_metrics)
    )
    return (
        f"PARAMETER G({phase.upper()},{elems_str};0) 1.00"
        f" {expression}{ser_terms}; 6000.00 N !"
    )


def fit_one(dat_path, name, phase, elements, metrics, atom_num):
    """Fit one gibbs-temperature.dat and return a result dict."""
    data = _read_dat(dat_path, atom_num)
    x = data["T"].values
    y = data["G"].values

    best_params = []
    best_r2 = 0.0
    for _ in tqdm(range(100), desc=f"Fitting {name}", ncols=80, leave=False):
        try:
            params, _ = curve_fit(_fit_func, x, y, maxfev=5000)
        except Exception:
            continue
        residuals = y - _fit_func(x, *params)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        if r2 > best_r2:
            best_params = params.tolist()
            best_r2 = r2
            if r2 > 0.9999:
                break

    return {
        "name": name,
        "phase": phase,
        "elements": elements,
        "metrics": metrics,
        "params": best_params,
        "r2": best_r2,
        "expression": _formula_str(best_params) if best_params else "",
        "data": data,
    }


# ── single mode ─────────────────────────────────────────────────────────

def _expand_results(result, name, phase, elements, metrics, atom_num):
    """Return results list; add BCC exchanged entry if applicable."""
    out = [result]
    if (
        len(elements) == 2
        and elements[0] != elements[1]
        and len(metrics) > 1
        and metrics[0] == metrics[1]
    ):
        out.append({
            **result,
            "name": f"{name}-ex",
            "elements": [elements[1], elements[0]],
            "metrics": [metrics[1], metrics[0]],
        })
    return out


def run_single(args):
    """Single-fit mode."""
    filepath = Path(args.filepath)
    dat_path = _find_gibbs_dat(filepath)

    name = filepath.stem if filepath.is_file() else filepath.name
    phase = args.phase.upper()
    metrics_raw = [float(x) for x in args.metrics]
    m_sum = sum(metrics_raw)
    metrics = [x / m_sum for x in metrics_raw]
    elements = [e.upper() for e in args.elem]
    atom_num = int(args.atom_num)

    if len(elements) != len(metrics):
        print(
            f"[ERROR] len(elements)={len(elements)} != len(metrics)={len(metrics)}"
        )
        return 1

    print(f"Folder:    {filepath}")
    print(f"Dat file:  {dat_path}")
    print(f"Phase:     {phase}")
    print(f"Elements:  {','.join(elements)}")
    print(f"Metrics:   {', '.join(f'{m:.4f}' for m in metrics)}")
    print(f"Atom num:  {atom_num}")

    result = fit_one(dat_path, name, phase, elements, metrics, atom_num)
    if not result["params"]:
        print("[ERROR] Fitting failed — no parameters obtained")
        return 1

    for r in _expand_results(result, name, phase, elements, metrics, atom_num):
        print()
        print(f"  Name:      {r['name']}")
        print(f"  Phase:     {r['phase']}")
        print(f"  Elements:  {','.join(r['elements'])}")
        print(f"  Metrics:   {','.join(f'{m:.4f}' for m in r['metrics'])}")
        print(f"  R² =       {r['r2']:.6f}")
        print(f"  G(T) =     {r['expression']}")
        print(f"  TDB =      {_format_tdb_parameter(r['phase'], r['elements'], r['metrics'], r['expression'])}")

    _plot_single(name, result, filepath)
    return 0


def _plot_single(name, result, filepath):
    """Plot one fit and save to png."""
    data = result["data"]
    if data is None:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(data["T"], data["G"], "o", ms=3, label="Data")
    if result["params"]:
        x = data["T"].values
        ax.plot(
            x,
            _fit_func(x, *result["params"]),
            "-",
            lw=2,
            label=f"Fit  R²={result['r2']:.6f}",
        )
    ax.set_xlabel("T (K)")
    ax.set_ylabel("G (J/mol/atom)")
    ax.set_title(f"{result['phase']}  {','.join(result['elements'])}")
    ax.legend()
    parent = filepath.parent if filepath.is_file() else filepath
    out = parent / f"{name}_fit.png"
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  [plot] Saved to {out}")


# ── batch mode ──────────────────────────────────────────────────────────

def run_batch(args):
    """Batch-fit mode. Scans subfolders, parses names, fits each."""
    root = Path(args.filepath)
    if not root.is_dir():
        print(f"[ERROR] {root} is not a directory")
        return 1

    phase = args.phase.upper()
    if not args.metrics:
        if phase in PHASE_METRICS:
            metrics_raw = list(PHASE_METRICS[phase])
        else:
            print(f"[ERROR] Unknown phase {phase!r} — please provide --metrics")
            return 1
    else:
        metrics_raw = [float(x) for x in args.metrics]

    m_sum = sum(metrics_raw)
    metrics = [x / m_sum for x in metrics_raw]

    print(f"Phase:     {phase}")
    print(f"Metrics:   {', '.join(f'{m:.4f}' for m in metrics)}")
    print(f"Root:      {root}\n")

    results = []
    skipped = []

    for subdir in sorted(root.iterdir()):
        if not subdir.is_dir():
            continue
        parsed = _parse_folder_name(subdir.name)
        if parsed is None:
            skipped.append((subdir.name, "cannot parse folder name"))
            continue
        if parsed["phase"].upper() != phase:
            continue
        elems = parsed["elements"]
        atom_num = parsed["atom_num"]
        if len(elems) != len(metrics):
            skipped.append(
                (subdir.name, f"elems ({len(elems)}) != metrics ({len(metrics)})")
            )
            continue

        try:
            dat_path = _find_gibbs_dat(subdir)
        except FileNotFoundError:
            skipped.append((subdir.name, "no gibbs-temperature.dat"))
            continue

        try:
            result = fit_one(dat_path, subdir.name, phase, elems, metrics, atom_num)
            if result["params"]:
                for r in _expand_results(
                    result, subdir.name, phase, elems, metrics, atom_num
                ):
                    results.append(r)
        except Exception as e:
            skipped.append((subdir.name, str(e)))
            continue

    print(f"\nFitted: {len(results)} result(s)")
    for r in results:
        print(f"\n  {'─' * 50}")
        print(f"  Name:      {r['name']}")
        print(f"  Phase:     {r['phase']}")
        print(f"  Elements:  {','.join(r['elements'])}")
        print(f"  Metrics:   {','.join(f'{m:.4f}' for m in r['metrics'])}")
        print(f"  R² =       {r['r2']:.6f}")
        print(f"  G(T) =     {r['expression']}")
        print(f"  TDB =      {_format_tdb_parameter(r['phase'], r['elements'], r['metrics'], r['expression'])}")

    if skipped:
        print(f"\n[WARNING] {len(skipped)} folder(s) skipped:")
        print(f"  [HINT] Expected format: {phase}-<elem1>-<elem2>-<atoms>")
        print(f"  [HINT] e.g. {phase}-Fe-Cr-2  or  {phase}-FeCr-2")
        for name, reason in skipped:
            print(f"  - {name}: {reason}")

    if results:
        _plot_batch(results, root)
    return 0 if results else 1


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
        fig, axes = plt.subplots(
            grid, grid, figsize=(grid * 5, grid * 4), squeeze=False
        )
        for ax, r in zip(axes.flatten(), batch):
            d = r["data"]
            if d is not None and r["params"]:
                ax.plot(d["T"], d["G"], "o", ms=2, label="Data")
                ax.plot(
                    d["T"],
                    _fit_func(d["T"], *r["params"]),
                    "-",
                    lw=1.5,
                    label=f"R²={r['r2']:.4f}",
                )
            ax.set_title(r["name"], fontsize=7)
            ax.tick_params(labelsize=6)
            ax.legend(fontsize=5)
        for ax in axes.flatten()[n_batch:]:
            ax.axis("off")
        fig.tight_layout()
        if num_batches > 1:
            out = root / f"fit_results_batch{batch_idx + 1}.png"
        else:
            out = root / "fit_results.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  [plot] Saved to {out}")


# ── CLI ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=f"Gibbs-Temperature Fitter v{VERSION}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    sub = parser.add_subparsers(dest="mode", required=True)

    # -- single --
    p_s = sub.add_parser("single", help="Fit one dataset")
    p_s.add_argument("--filepath", required=True, help=".dat file or folder")
    p_s.add_argument("--phase", required=True, help="Phase name (BCC, FCC, SER, …)")
    p_s.add_argument(
        "--metrics", required=True, nargs="+", type=float,
        help="Stoichiometry ratios (e.g. 1 3 for FCC)",
    )
    p_s.add_argument(
        "--elem", required=True, nargs="+", help="Element symbols (e.g. Fe Cr)"
    )
    p_s.add_argument(
        "--atom_num", required=True, type=int,
        help="Number of atoms per formula unit",
    )

    # -- batch --
    p_b = sub.add_parser("batch", help="Fit all subfolders matching a phase")
    p_b.add_argument("--filepath", required=True, help="Root folder")
    p_b.add_argument("--phase", required=True, help="Phase name")
    p_b.add_argument(
        "--metrics", nargs="+", type=float, default=None,
        help="Stoichiometry ratios (omit to use built-in PHASE_METRICS)",
    )

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
