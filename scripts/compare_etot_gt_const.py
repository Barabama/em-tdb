"""
compare_etot_gt_const.py — Compare two TDB files (etot vs gt) at constant level.

Strategy (as discussed):
 1. Compare FUNCTION constants: GHSERXX.A vs ETOT_SER_XX (they should match — same DFT energy)
 2. Compare PARAMETER formation energies: for each end-member, compute
    net = coeff_A - Σ(weight_i × reference_func_constant_i)
    and compare between the two TDBs.

Usage:
    python compare_etot_gt_const.py \\
        --etot  ref_tdb/20260201-tdb-etot-bcc+fcc-by-gml.tdb \\
        --gt    ref_tdb/20260627-16s-bcc+fcc-cxy-hmd.TDB \\
        [--tol 0.1]         # eV tolerance (default 0.1)
        [--output report]   # output CSV prefix (default None = print only)
        [-v]                # verbose: show all params even when OK
"""

import re
import sys
import math
from pathlib import Path
from typing import Optional


# ── Helper: extract the constant term A from an expression ──────────────────

def extract_const_term(expr: str) -> Optional[float]:
    """Extract the leading constant term A from a TDB expression.

    Handles:
        -3.584235E+05                         → -358423.5
        -3.614407E+05+1.808228E+02*T-...      → -361440.7
        1.0 -3.512708E+05 -0.5*ETOT_SER_AL#   → N/A (1.0 is temp_start, not expr)
    """
    # Strip leading whitespace and commas
    expr = expr.strip()

    # Look for the first number at the start of the expression
    # The constant term A is the first digit-bearing token before any *T, *LN, etc.
    m = re.match(
        r"\s*([+-]?\d+(?:\.\d*)?(?:[Ee][+-]?\d+)?)\s*(?:\+|\-|$)",
        expr,
    )
    if m:
        return float(m.group(1))

    # Maybe it starts with just a number, no sign (shouldn't happen in practice)
    m2 = re.match(r"\s*(\d+(?:\.\d*)?(?:[Ee][+-]?\d+)?)\s*", expr)
    if m2:
        return float(m2.group(1))

    return None


def extract_poly_a(expr: str) -> Optional[float]:
    """Extract the constant 'A' coefficient from a Gibbs polynomial
    of the form:  A + B*T + C*T*LN(T) + D*T**2 + E*T**3 + F*T**(-1)

    The constant A is the first numeric value in the expression.
    """
    expr = expr.strip().replace(" ", "")
    if not expr:
        return None

    # Try to parse the leading number (with optional sign)
    m = re.match(r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)", expr)
    if m:
        return float(m.group(1))
    return None


# ── Pure-text TDB parser (no dependency on emtdb) ───────────────────────────

def parse_tdb_functions(filepath: str) -> dict[str, dict]:
    """Parse FUNCTION blocks from a TDB file.

    Returns dict like::
        {"GHSERAL": {"expr_raw": " -3.614407E+05+1.808228E+02*T-...",
                     "temp_start": 1.0, "temp_end": 6000.0,
                     "const_A": -361440.7},
         "ETOT_SER_AL": {"expr_raw": " -3.584235E+05",
                         "const_A": -358423.5}}
    """
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Strip comments and concatenate (preserving multi-line FUNCTION/PARAMETER)
    text_stripped = "".join(
        s.split("$", 1)[0].strip() for s in lines
    )
    sentences = [s.strip() for s in text_stripped.split("!")]

    funcs = {}
    for clause in sentences:
        if not clause:
            continue
        clause = clause + " !"
        # MULTILINE fix: use DOTALL so . matches \n, but also the expression
        # may have embedded newlines — collapse whitespace first.
        clause_norm = re.sub(r"\s+", " ", clause)

        m = re.match(
            r"FUNCTION\s+(\S+)\s+(\S+)\s+(.+?)\s*;\s*(\S+)\s*([YN]?)\s*!",
            clause_norm,
            re.DOTALL,
        )
        if m:
            func_name = m.group(1)
            temp_start = float(m.group(2))
            expr_raw = m.group(3).strip()
            temp_end = float(m.group(4))
            is_cont = m.group(5) or "N"
            const_A = extract_poly_a(expr_raw)
            funcs[func_name] = {
                "expr_raw": expr_raw,
                "temp_start": temp_start,
                "temp_end": temp_end,
                "const_A": const_A,
                "is_continued": is_cont,
            }
    return funcs


def parse_tdb_parameters(filepath: str) -> dict[str, dict]:
    """Parse PARAMETER blocks from a TDB file."""
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    text_stripped = "".join(
        s.split("$", 1)[0].strip() for s in lines
    )
    sentences = [s.strip() for s in text_stripped.split("!")]

    params = {}
    for clause in sentences:
        if not clause:
            continue
        clause = clause + " !"
        clause_norm = re.sub(r"\s+", " ", clause)

        m = re.match(
            r"PARAMETER\s+([GL])\((\S+),(\S+);(\d)\)\s+(\S+)\s+(.+?)\s*;\s*(\S+)\s*([YN]?)\s*!",
            clause_norm,
            re.DOTALL,
        )
        if m:
            ptype = m.group(1)
            phase = m.group(2)
            components = m.group(3)
            order = int(m.group(4))
            temp_start = float(m.group(5))
            expr_raw = m.group(6).strip()
            temp_end = float(m.group(7))
            is_cont = m.group(8) or "N"

            param_key = f"{ptype}({phase},{components};{order})"
            const_A = extract_poly_a(expr_raw)

            # Extract function reference terms: e.g. -0.5*GHSERAL#
            ref_terms = []
            for rm in re.finditer(
                r"([+-]?\s*\d+(?:\.\d*)?(?:[Ee][+-]?\d+)?)\s*\*\s*([A-Za-z_0-9]+)#",
                expr_raw,
            ):
                coeff_str = rm.group(1).replace(" ", "")
                func_name = rm.group(2)
                try:
                    coeff_val = float(coeff_str)
                except ValueError:
                    coeff_val = 0.0
                ref_terms.append((coeff_val, func_name))

            params[param_key] = {
                "ptype": ptype,
                "phase": phase,
                "components": components,
                "order": order,
                "temp_start": temp_start,
                "temp_end": temp_end,
                "expr_raw": expr_raw,
                "const_A": const_A,
                "ref_terms": ref_terms,
                "is_continued": is_cont,
            }
    return params


# ── Element extraction from function names ──────────────────────────────────

_KNOWN_ELEMENTS = {
    "AL", "CO", "CR", "CU", "FE", "HF", "MN", "MO", "NB", "NI",
    "RE", "TA", "TI", "V", "W", "ZR",
}


def extract_elem_from_func(func_name: str) -> str:
    """Extract element symbol from the suffix of a function name.
    Tries 3-char then 2-char suffixes.
    """
    upper = func_name.upper()
    for l in (3, 2, 1):
        suffix = upper[-l:]
        if suffix in _KNOWN_ELEMENTS:
            return suffix
    return ""


# ── Phase sublattice stoichiometry ──────────────────────────────────────────

PHASE_STOICHIOMETRY = {
    "BCC": (0.5, 0.5),   # 2 sublattices, equal
    "FCC": (0.25, 0.75), # 2 sublattices, 1:3
}


# ── Main comparison logic ────────────────────────────────────────────────────

def compare_functions(funcs_etot: dict, funcs_gt: dict, tol: float = 1.0):
    """Compare FUNCTION constants between etot and gt.

    Returns dict of mismatches.
    """
    results = []
    for fname_gt, fdata_gt in sorted(funcs_gt.items()):
        # Try to find matching etot function
        elem = extract_elem_from_func(fname_gt)
        if not elem:
            continue
        fname_etot_candidates = [k for k in funcs_etot if k.upper().endswith(elem.upper())]
        if not fname_etot_candidates:
            continue
        fname_etot = fname_etot_candidates[0]
        fdata_etot = funcs_etot[fname_etot]

        a_gt = fdata_gt["const_A"]
        a_etot = fdata_etot["const_A"]

        if a_gt is None or a_etot is None:
            results.append({
                "element": elem,
                "gt_func": fname_gt,
                "etot_func": fname_etot,
                "gt_const_A": a_gt,
                "etot_const": a_etot,
                "diff": None,
                "status": "SKIP (incomplete)",
            })
            continue

        diff = a_gt - a_etot
        diff_ev = diff / 96485.33212  # J → eV (TC uses J/mol, 1 eV = 96485 J)
        status = "OK" if abs(diff_ev) < tol else "MISMATCH"

        results.append({
            "element": elem,
            "gt_func": fname_gt,
            "etot_func": fname_etot,
            "gt_const_A": a_gt,
            "etot_const": a_etot,
            "diff_J": diff,
            "diff_eV": diff_ev,
            "status": status,
        })

    return results


def compute_net_formation(param_data: dict, func_map: dict[str, float],
                          src_label: str) -> Optional[float]:
    """Compute net formation energy: const_A - Σ(weight × func_const)."""
    if param_data["const_A"] is None:
        return None

    net = param_data["const_A"]
    missing = []
    for coeff, func_name in param_data["ref_terms"]:
        func_const = func_map.get(func_name)
        if func_const is None:
            missing.append(func_name)
        else:
            net -= coeff * func_const

    if missing:
        return None
    return net


def get_stoichiometry(phase: str, components: str) -> list[float]:
    """Get sublattice weights for each component in the parameter.
    BCC  : (0.5, 0.5) → two components, each gets one weight
    FCC  : (0.25, 0.75) → first sublattice, second sublattice
    """
    weights = PHASE_STOICHIOMETRY.get(phase)
    if not weights:
        return None
    return list(weights)


def compare_parameters(params_etot: dict, params_gt: dict,
                       func_map_etot: dict[str, float],
                       func_map_gt: dict[str, float],
                       tol: float = 1.0):
    """Compare PARAMETER net formation energies."""
    common_keys = sorted(set(params_etot.keys()) & set(params_gt.keys()))

    results = []
    for key in common_keys:
        p_etot = params_etot[key]
        p_gt = params_gt[key]

        net_etot = compute_net_formation(p_etot, func_map_etot, "etot")
        net_gt = compute_net_formation(p_gt, func_map_gt, "gt")

        if net_etot is None or net_gt is None:
            results.append({
                "param": key,
                "phase": p_etot["phase"],
                "components": p_etot["components"],
                "net_etot": net_etot,
                "net_gt": net_gt,
                "diff_J": None,
                "diff_eV": None,
                "status": "SKIP (missing func ref)",
            })
            continue

        diff = net_gt - net_etot
        diff_ev = diff / 96485.33212
        status = "OK" if abs(diff_ev) < tol else "MISMATCH"

        results.append({
            "param": key,
            "phase": p_etot["phase"],
            "components": p_etot["components"],
            "net_etot": net_etot,
            "net_gt": net_gt,
            "diff_J": diff,
            "diff_eV": diff_ev,
            "status": status,
        })

    return results


def print_func_results(results: list[dict], tol: float = 0.05):
    """Pretty-print function comparison results."""
    mismatches = [r for r in results if r["status"] == "MISMATCH"]
    skipped = [r for r in results if r["status"].startswith("SKIP")]
    ok_count = len(results) - len(mismatches) - len(skipped)

    print(f"\n{'='*90}")
    print(f"FUNCTION COMPARISON: GT polynomial constant A vs ETOT constant value")
    print(f"{'='*90}")
    print(f"{'Elem':6s} {'GT Func':16s} {'ETOT Func':16s} {'GT A(J/mol)':20s} "
          f"{'Etot(J/mol)':20s} {'Diff(eV)':12s} {'Status'}")
    print(f"{'-'*90}")

    for r in results:
        if r["diff_J"] is not None:
            diff_str = f"{r['diff_eV']:.6f}"
            a_gt_str = f"{r['gt_const_A']:.2f}" if r['gt_const_A'] is not None else "N/A"
            a_etot_str = f"{r['etot_const']:.2f}" if r['etot_const'] is not None else "N/A"
        else:
            diff_str = "N/A"
            a_gt_str = "N/A"
            a_etot_str = "N/A"

        status_tag = r["status"]
        if r["status"] == "MISMATCH":
            status_tag = "⚠️  MISMATCH"
        elif r["status"] == "OK":
            status_tag = "✓ OK"

        print(f"{r['element']:6s} {r['gt_func']:16s} {r['etot_func']:16s} "
              f"{a_gt_str:20s} {a_etot_str:20s} {diff_str:12s} {status_tag}")

    print(f"{'-'*90}")
    print(f"Total: {len(results)}  |  OK: {ok_count}  |  "
          f"⚠️  MISMATCH (>={tol}eV): {len(mismatches)}  |  SKIP: {len(skipped)}")


def print_param_results(results: list[dict], tol: float, verbose: bool = False):
    """Pretty-print parameter comparison results."""
    mismatches = [r for r in results if r["status"] == "MISMATCH"]
    skipped = [r for r in results if r["status"].startswith("SKIP")]
    ok_results = [r for r in results if r["status"] == "OK"]

    print(f"\n{'='*110}")
    print(f"PARAMETER COMPARISON: net formation energy (subtracting function refs)")
    print(f"{'='*110}")
    print(f"{'Parameter':50s} {'Phase':6s} {'Net Etot(J)':18s} {'Net GT(J)':18s} "
          f"{'Diff(eV)':12s} {'Status'}")
    print(f"{'-'*110}")

    # Print mismatches first
    for r in mismatches:
        d = f"{r['diff_eV']:.6f}" if r['diff_eV'] is not None else "N/A"
        ne = f"{r['net_etot']:.2f}" if r['net_etot'] is not None else "N/A"
        ng = f"{r['net_gt']:.2f}" if r['net_gt'] is not None else "N/A"
        print(f"{r['param']:50s} {r['phase']:6s} {ne:18s} {ng:18s} {d:12s} ⚠️  MISMATCH")

    if verbose:
        for r in ok_results:
            d = f"{r['diff_eV']:.6f}" if r['diff_eV'] is not None else "N/A"
            ne = f"{r['net_etot']:.2f}" if r['net_etot'] is not None else "N/A"
            ng = f"{r['net_gt']:.2f}" if r['net_gt'] is not None else "N/A"
            print(f"{r['param']:50s} {r['phase']:6s} {ne:18s} {ng:18s} {d:12s} ✓ OK")

    for r in skipped:
        print(f"{r['param']:50s} {r['phase']:6s} {'N/A':18s} {'N/A':18s} {'N/A':12s} "
              f"? {r['status']}")

    print(f"{'-'*110}")
    print(f"Total: {len(results)}  |  OK: {len(ok_results)}  |  "
          f"⚠️  MISMATCH (>{tol:.2f}eV): {len(mismatches)}  |  SKIP: {len(skipped)}")

    if mismatches:
        print(f"\n{'='*110}")
        print(f"TOP MISMATCHES BY |diff|:")
        mm_sorted = sorted(mismatches, key=lambda r: abs(r["diff_eV"]) if r["diff_eV"] else 0, reverse=True)
        for r in mm_sorted[:10]:
            print(f"  {r['param']:50s}  diff = {r['diff_eV']:.6f} eV")


# ── Entry point ─────────────────────────────────────────────────────────────

def main():
    import argparse

    ap = argparse.ArgumentParser(description="Compare etot vs gt TDB at constant level")
    ap.add_argument("--etot", required=True, help="Path to etot TDB file")
    ap.add_argument("--gt", required=True, help="Path to gt (Gibbs-temperature) TDB file")
    ap.add_argument("--tol", type=float, default=0.05,
                    help="Tolerance in eV (default: 0.05)")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="Show all params even when OK")
    ap.add_argument("--output", help="CSV output prefix (optional)")
    args = ap.parse_args()

    for fp in [args.etot, args.gt]:
        if not Path(fp).exists():
            print(f"ERROR: file not found: {fp}")
            sys.exit(1)

    print(f"Parsing etot TDB: {args.etot}")
    funcs_etot = parse_tdb_functions(args.etot)
    params_etot = parse_tdb_parameters(args.etot)

    print(f"Parsing gt TDB:   {args.gt}")
    funcs_gt = parse_tdb_functions(args.gt)
    params_gt = parse_tdb_parameters(args.gt)

    print(f"  ├─ ETOT: {len(funcs_etot)} functions, {len(params_etot)} parameters")
    print(f"  └─ GT:   {len(funcs_gt)} functions, {len(params_gt)} parameters")

    # ── Build function constant maps ──
    func_map_etot = {name: data["const_A"] for name, data in funcs_etot.items()
                     if data["const_A"] is not None}
    func_map_gt = {name: data["const_A"] for name, data in funcs_gt.items()
                   if data["const_A"] is not None}

    # ── Compare functions ──
    func_results = compare_functions(funcs_etot, funcs_gt, tol=args.tol)
    print_func_results(func_results, tol=args.tol)

    # ── Compare parameters ──
    param_results = compare_parameters(
        params_etot, params_gt,
        func_map_etot, func_map_gt,
        tol=args.tol,
    )
    print_param_results(param_results, tol=args.tol, verbose=args.verbose)

    # ── Summary ──
    func_mm = [r for r in func_results if r.get("status") == "MISMATCH"]
    param_mm = [r for r in param_results if r.get("status") == "MISMATCH"]

    print(f"\n{'='*50}")
    print(f"SUMMARY")
    print(f"{'='*50}")
    print(f"  Function mismatches (>{args.tol}eV): {len(func_mm)} / {len(func_results)}")
    print(f"  Parameter mismatches (>{args.tol}eV): {len(param_mm)} / {len(param_results)}")

    if func_mm:
        print(f"\n  ⚠️  Function mismatches indicate DFT static energy differences "
              f"between the two datasets.")
        print(f"     This propagates to ALL parameters referencing those elements.")

    if param_mm and not func_mm:
        print(f"\n  ⚠️  Parameter mismatches without function mismatches indicate")
        print(f"     differences in the fitting (polynomial) of the formation energy.")

    # ── Export CSV ──
    if args.output:
        from csv import DictWriter
        prefix = args.output

        with open(f"{prefix}_funcs.csv", "w", newline="") as f:
            w = DictWriter(f, fieldnames=list(func_results[0].keys()))
            w.writeheader()
            w.writerows(func_results)
        print(f"\n  → Wrote {prefix}_funcs.csv")

        with open(f"{prefix}_params.csv", "w", newline="") as f:
            w = DictWriter(f, fieldnames=list(param_results[0].keys()))
            w.writeheader()
            w.writerows(param_results)
        print(f"  → Wrote {prefix}_params.csv")


if __name__ == "__main__":
    main()
