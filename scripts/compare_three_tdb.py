"""
compare_three_tdb.py — three-way comparison of TDB constant terms.

Compares FUNCTION constants and PARAMETER net formation energies
across three TDB files (etot / hamid / cxy-hmd), using only
phases common to all three (BCC and FCC).

For each end-member the "net" formation energy is:
    net = param_const_A - Σ(weight_i × ref_func_const_i)

This strips out the different reference-function naming conventions
(ETOT_SER_xxx vs GHSERxxx) and compares only the physical quantity.

Usage:
    python scripts/compare_three_tdb.py [--output report.csv]

Output:
    Terminal tables for functions and parameters
    Optional CSV export
"""

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Optional


# ── Known element symbols ──

_KNOWN_ELEMENTS = frozenset({
    "AL", "CO", "CR", "CU", "FE", "HF", "MN", "MO", "NB", "NI",
    "RE", "TA", "TI", "V", "W", "ZR",
})

TDB_PATHS = {
    "etot":     "ref_tdb/20260201-tdb-etot-bcc+fcc-by-gml.tdb",
    "hamid":    "ref_tdb/20250516-bcc+fcc+hcp_by-hamid-cxy-wubo.TDB",
    "cxy-hmd":  "ref_tdb/20260627-16s-bcc+fcc-cxy-hmd.TDB",
}
PHASES = ["BCC", "FCC"]

FC = 96485.33212  # Faraday constant J/mol → eV


# ═══════════════════════════════════════════════════════════════════
#  0.  Low-level TDB parsing helpers
# ═══════════════════════════════════════════════════════════════════

def _read_clean(filepath: str) -> str:
    """Read a TDB file, strip $ comments, collapse whitespace."""
    text = Path(filepath).read_text(encoding="utf-8")
    text = re.sub(r"\$[^!]*", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _extract_elem(func_name: str) -> Optional[str]:
    """Extract element symbol from the suffix of a function name."""
    upper = func_name.upper()
    for l in (3, 2, 1):
        suffix = upper[-l:]
        if suffix in _KNOWN_ELEMENTS:
            return suffix
    return None


def _map_etot_to_ghser(name: str) -> str:
    """ETOT_SER_AL → GHSERAL,  ETOT_SER_FE → GHSERFE."""
    if name.startswith("ETOT_SER_"):
        elem = name.replace("ETOT_SER_", "")
        return f"GHSER{elem}"
    return name  # already GHSERxxx


# ═══════════════════════════════════════════════════════════════════
#  1.  Parse FUNCTION blocks: extract constant A term
# ═══════════════════════════════════════════════════════════════════

def parse_functions(filepath: str) -> dict[str, dict]:
    """Return ``{func_name: {const_A: float, elem: str, expr: str}}``.

    ``const_A`` is the first numeric value in the expression — the
    SGTE polynomial constant A.  For ``ETOT_SER_xxx`` this **is** the
    whole expression.  For ``GHSERxxx`` it is the A coefficient.
    """
    text = _read_clean(filepath)
    funcs: dict[str, dict] = {}

    for clause in text.split("!"):
        m = re.match(
            r"FUNCTION\s+(\S+)\s+(\S+)\s+(.+?)\s*;\s*(\S+)\s*([YN]?)",
            clause.strip(),
        )
        if not m:
            continue
        name = m.group(1)
        expr_raw = m.group(3).strip()
        elem = _extract_elem(name)
        if not elem:
            continue

        # Extract the constant A term (first number)
        const_A: Optional[float] = None
        m_a = re.match(
            r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)",
            expr_raw,
        )
        if m_a:
            const_A = float(m_a.group(1))

        funcs[name] = {"const_A": const_A, "elem": elem, "expr": expr_raw}
    return funcs


# ═══════════════════════════════════════════════════════════════════
#  2.  Parse PARAMETER blocks
# ═══════════════════════════════════════════════════════════════════

def parse_parameters(filepath: str, phases: list[str]) -> dict[str, dict]:
    """Return ``{param_key: {phase, comps, const_A, ref_terms, expr}}``.

    ``ref_terms`` is a list of ``(weight, func_name)`` tuples from
    ``XXX#`` references in the expression.

    Only parameters for phases in *phases* are retained.
    """
    text = _read_clean(filepath)
    params: dict[str, dict] = {}

    for clause in text.split("!"):
        m = re.match(
            r"PARAMETER\s+([GL])\((\S+),(\S+);(\d)\)\s+(\S+)\s+(.+?)\s*;\s*(\S+)\s*([YN]?)",
            clause.strip(),
        )
        if not m:
            continue
        ptype, phase, comps, order, tstart, expr, tend, cont = m.groups()
        if phase not in phases:
            continue

        key = f"{ptype}({phase},{comps};{order})"
        order_num = int(order)
        t_start = float(tstart)
        t_end = float(tend)
        is_cont = cont or "N"

        # Constant A term
        const_A: Optional[float] = None
        m_a = re.match(
            r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)",
            expr.strip(),
        )
        if m_a:
            const_A = float(m_a.group(1))

        # Function reference terms (e.g. -0.5*GHSERAL#)
        ref_terms: list[tuple[float, str]] = []
        for rm in re.finditer(
            r"([+-]?\s*\d+(?:\.\d*)?(?:[Ee][+-]?\d+)?)\s*\*\s*([A-Za-z_0-9]+)#",
            expr,
        ):
            coeff = float(rm.group(1).replace(" ", ""))
            fname = rm.group(2)
            ref_terms.append((coeff, fname))

        params[key] = {
            "ptype": ptype,
            "phase": phase,
            "components": comps,
            "order": order_num,
            "temp_start": t_start,
            "temp_end": t_end,
            "const_A": const_A,
            "ref_terms": ref_terms,
            "expr": expr.strip(),
            "is_continued": is_cont,
        }
    return params


# ═══════════════════════════════════════════════════════════════════
#  3.  Build reference-function constant map (GHSER-name space)
# ═══════════════════════════════════════════════════════════════════

def build_ref_const_map(funcs: dict[str, dict]) -> dict[str, float]:
    """Build ``{GHSERAL: A, GHSERFE: A, ...}`` from a function dict.

    Renames ETOT_SER_xxx → GHSERxxx so callers can look up by the
    same key regardless of which TDB the data came from.
    """
    ref_map: dict[str, float] = {}
    for name, data in funcs.items():
        g_name = _map_etot_to_ghser(name)
        if data["const_A"] is not None:
            ref_map[g_name] = data["const_A"]
    return ref_map


# ═══════════════════════════════════════════════════════════════════
#  4.  Compute net formation energy
# ═══════════════════════════════════════════════════════════════════

def compute_net(param: dict, ref_map: dict[str, float]) -> Optional[float]:
    """net = const_A - Σ(weight × ref_func_const).

    Each reference function name is mapped through
    :func:`_map_etot_to_ghser` first so ETOT_SER_* and GHSER* resolve
    to the same physical energy.
    """
    if param["const_A"] is None:
        return None
    net = param["const_A"]
    for coeff, fname in param["ref_terms"]:
        g_name = _map_etot_to_ghser(fname)
        rv = ref_map.get(g_name)
        if rv is None:
            return None  # missing reference → cannot compute
        net -= coeff * rv
    return net


# ═══════════════════════════════════════════════════════════════════
#  5.  Comparison runners
# ═══════════════════════════════════════════════════════════════════

def compare_funcs(all_funcs: dict[str, dict], labels: list[str]) -> list[dict]:
    """Compare FUNCTION constant A across all three TDBs.

    Match functions by the **element** they belong to.
    Returns list of dicts for printing.
    """
    # Build element→label→const_A map
    elem_data: dict[str, dict[str, Optional[float]]] = {}
    for label, funcs in zip(labels, all_funcs):
        for name, data in funcs.items():
            elem = data["elem"]
            if elem not in elem_data:
                elem_data[elem] = {l: None for l in labels}
            # Last one wins if duplicates (shouldn't happen per TDB)
            elem_data[elem][label] = data["const_A"]

    results = []
    for elem in sorted(elem_data):
        vals = elem_data[elem]
        row = {"element": elem}
        diffs = []
        for label in labels:
            v = vals[label]
            row[label] = v
        # Pairwise differences in eV
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                vi = vals[labels[i]]
                vj = vals[labels[j]]
                if vi is not None and vj is not None:
                    d = (vj - vi) / FC
                else:
                    d = None
                diffs.append(d)
                row[f"delta{labels[i]}-{labels[j]}_eV"] = d
        results.append(row)
    return results


def compare_params(all_params: list[dict], all_ref_maps: list[dict[str, float]],
                   labels: list[str], phases: list[str]) -> list[dict]:
    """Three-way comparison of net formation energies.

    Only parameters that exist in **all three** TDBs are included.
    """
    # Find common key set
    key_sets = [set(p.keys()) for p in all_params]
    common_keys = sorted(set.intersection(*key_sets))

    print(f"  Common end-members across all 3 TDBs: {len(common_keys)}")

    results = []
    for key in common_keys:
        nets = []
        for pi, ref_map in zip(all_params, all_ref_maps):
            if key in pi:
                nets.append(compute_net(pi[key], ref_map))
            else:
                nets.append(None)

        row = {"param": key, "phase": key.split("(")[1].split(",")[0] if "(" in key else ""}
        for label, net in zip(labels, nets):
            row[f"net_{label}"] = net

        # Pairwise differences in eV
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                ni, nj = nets[i], nets[j]
                if ni is not None and nj is not None:
                    d = (nj - ni) / FC
                else:
                    d = None
                row[f"delta{labels[i]}-{labels[j]}_eV"] = d
        results.append(row)
    return results


# ═══════════════════════════════════════════════════════════════════
#  6.  Printing
# ═══════════════════════════════════════════════════════════════════

def _fmt(v: Optional[float], width: int = 14, decimals: int = 1) -> str:
    if v is None:
        return "N/A".rjust(width)
    return f"{v:>{width}.{decimals}f}"


def _fmt_ev(v: Optional[float], width: int = 12) -> str:
    if v is None:
        return "N/A".rjust(width)
    return f"{v:>{width}.6f}"


def print_func_results(results: list[dict], labels: list[str], tol: float = 0.05):
    """Pretty-print function constant comparison."""
    mismatches = [r for r in results
                  if any(r.get(f"delta{l1}-{l2}_eV") is not None
                         and abs(r[f"delta{l1}-{l2}_eV"]) > tol
                         for i, l1 in enumerate(labels)
                         for l2 in labels[i + 1:])]

    print(f"\n{'=' * 100}")
    print(f"FUNCTION constant A comparison")
    print(f"{'=' * 100}")
    hdr = f"{'Elem':6s}"
    for label in labels:
        hdr += f"  {label + '_A':>16s}"
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            hdr += f"  delta{labels[i]}-{labels[j]}(eV)"
    print(hdr)
    print("-" * len(hdr))

    for r in results:
        line = f"{r['element']:6s}"
        for label in labels:
            line += _fmt(r[label], 16, 1)
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                line += _fmt_ev(r.get(f"delta{labels[i]}-{labels[j]}_eV"), 14)
        print(line)

    print(f"\n  Total: {len(results)}  |  "
          f"Mismatch (>|{tol}| eV): {len(mismatches)}")
    if mismatches:
        print(f"  ⚠️  Elements with diff >{tol} eV:")
        for r in mismatches:
            print(f"      {r['element']}")


def print_param_results(results: list[dict], labels: list[str], tol: float = 0.05,
                        top_n: int = 15):
    """Pretty-print parameter net formation energy comparison."""
    mismatches = [r for r in results
                  if any(r.get(f"delta{l1}-{l2}_eV") is not None
                         and abs(r[f"delta{l1}-{l2}_eV"]) > tol
                         for i, l1 in enumerate(labels)
                         for l2 in labels[i + 1:])]

    print(f"\n{'=' * 130}")
    print(f"PARAMETER net formation energy comparison")
    print(f"{'=' * 130}")

    hdr = f"{'Parameter':50s}  {'Phase':6s}"
    for label in labels:
        hdr += f"  {'net_' + label:>16s}"
    for i, l1 in enumerate(labels):
        for l2 in labels[i + 1:]:
            hdr += f"  delta{l1}-{l2}_eV"

    # Print mismatches
    print(f"\n--- Mismatches (>{tol} eV) ---")
    for r in sorted(mismatches, key=lambda x: max(
        abs(x.get(f"delta{l1}-{l2}_eV") or 0) for i, l1 in enumerate(labels) for l2 in labels[i + 1:]
    ), reverse=True):
        line = f"{r['param']:50s}  {r['phase']:6s}"
        for label in labels:
            line += _fmt(r.get(f"net_{label}"), 16, 1)
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                line += _fmt_ev(r.get(f"delta{labels[i]}-{labels[j]}_eV"), 14)
        print(line)

    # Summary stats
    all_diffs = []
    for r in results:
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                d = r.get(f"delta{labels[i]}-{labels[j]}_eV")
                if d is not None:
                    all_diffs.append(abs(d))

    if all_diffs:
        print(f"\n  Total common: {len(results)}")
        print(f"  Mismatches (>{tol} eV): {len(mismatches)}")
        print(f"  Mean |delta|: {sum(all_diffs) / len(all_diffs):.6f} eV")
        print(f"  Max  |delta|: {max(all_diffs):.6f} eV")
        print(f"  Median |delta|: {sorted(all_diffs)[len(all_diffs) // 2]:.6f} eV")

    # Top N worst
    if mismatches:
        print(f"\n--- TOP {min(top_n, len(mismatches))} worst mismatches ---")
        worst = sorted(mismatches, key=lambda r: max(
            abs(r.get(f"delta{l1}-{l2}_eV") or 0) for i, l1 in enumerate(labels) for l2 in labels[i + 1:]
        ), reverse=True)[:top_n]
        for r in worst:
            max_d = max(abs(r.get(f"delta{l1}-{l2}_eV") or 0) for i, l1 in enumerate(labels) for l2 in labels[i + 1:])
            print(f"  {r['param']:50s}  max|delta|={max_d:.6f} eV")


# ═══════════════════════════════════════════════════════════════════
#  7.  Main
# ═══════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Three-way TDB constant-term comparison",
    )
    ap.add_argument("--tdb1", default=TDB_PATHS["etot"],
                    help="First TDB file (default: etot)")
    ap.add_argument("--tdb2", default=TDB_PATHS["hamid"],
                    help="Second TDB file (default: hamid)")
    ap.add_argument("--tdb3", default=TDB_PATHS["cxy-hmd"],
                    help="Third TDB file (default: cxy-hmd)")
    ap.add_argument("--labels", nargs=3, default=["0201", "0516", "0627"],
                    help="Three labels for the TDBs")
    ap.add_argument("--phases", nargs="+", default=PHASES,
                    help="Phases to compare (default: BCC FCC)")
    ap.add_argument("--tol", type=float, default=0.05,
                    help="Tolerance in eV (default: 0.05)")
    ap.add_argument("--output", type=str, default="",
                    help="CSV output prefix (optional)")
    ap.add_argument("--top", type=int, default=15,
                    help="Number of worst mismatches to show")
    args = ap.parse_args()

    tdb_paths = [args.tdb1, args.tdb2, args.tdb3]
    labels = args.labels

    # Validate files
    for fp in tdb_paths:
        if not Path(fp).exists():
            print(f"ERROR: file not found: {fp}")
            sys.exit(1)

    # ── Parse ──
    print("Parsing TDB files …")
    all_funcs = []
    all_params = []
    all_ref_maps = []

    for fp, label in zip(tdb_paths, labels):
        funcs = parse_functions(fp)
        params = parse_parameters(fp, args.phases)
        ref_map = build_ref_const_map(funcs)
        all_funcs.append(funcs)
        all_params.append(params)
        all_ref_maps.append(ref_map)
        print(f"  {label:12s}: {len(funcs)} funcs, {len(params)} params, "
              f"{len(ref_map)} ref functions")

    # ── FUNCTION comparison ──
    func_results = compare_funcs(all_funcs, labels)
    print_func_results(func_results, labels, tol=args.tol)

    # ── PARAMETER comparison ──
    param_results = compare_params(all_params, all_ref_maps, labels, args.phases)
    print_param_results(param_results, labels, tol=args.tol, top_n=args.top)

    # ── CSV export ──
    if args.output:
        prefix = args.output
        if func_results:
            fkeys = ["element"] + labels
            for i in range(len(labels)):
                for j in range(i + 1, len(labels)):
                    fkeys.append(f"delta{labels[i]}-{labels[j]}_eV")
            with open(f"{prefix}_funcs.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fkeys)
                w.writeheader()
                w.writerows(func_results)
            print(f"\n  → {prefix}_funcs.csv")

        if param_results:
            pkeys = ["param", "phase"] + [f"net_{l}" for l in labels]
            for i in range(len(labels)):
                for j in range(i + 1, len(labels)):
                    pkeys.append(f"delta{labels[i]}-{labels[j]}_eV")
            with open(f"{prefix}_params.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=pkeys)
                w.writeheader()
                # Filter out keys that aren't in pkeys
                clean = [{k: r[k] for k in pkeys if k in r} for r in param_results]
                w.writerows(clean)
            print(f"  → {prefix}_params.csv")


if __name__ == "__main__":
    main()
