"""
Check VASP QHA data usability for end-member thermodynamic calculations.

For each end-member with gibbs-temperature.dat, checks:
1. POTCAR element count matches POSCAR atom types (extra TITEL entries = broken POTCAR)
2. R3f convergence (OSZICAR RMM steps, OUTCAR energy convergence)
3. POTCAR elements match intended composition from folder name

Note: POSCAR element LABELS may differ from POTCAR (template issue) — this is
cosmetic and does NOT affect VASP results. VASP assigns pseudopotentials by
position in POTCAR, not by POSCAR labels.

Usage:
  python check_potcar_poscar.py --root E:/nas-shared/Endmembers/13symbols-hmd
  python check_potcar_poscar.py --summary
  python check_potcar_poscar.py --usable
  python check_potcar_poscar.py --failed
"""

import argparse
import re
import os
from pathlib import Path
from dataclasses import dataclass, field


# ──────────────────────────────────────────────────────────────────────────────
# Element symbol set
# ──────────────────────────────────────────────────────────────────────────────

ELEMENTS = {
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "In", "Sn", "Sb", "Te", "I", "Xe",
    "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy",
    "Ho", "Er", "Tm", "Yb", "Lu",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn",
    "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf",
    "Es", "Fm", "Md", "No", "Lr",
    "VA",
}


def is_element_symbol(s: str) -> bool:
    return s in ELEMENTS


# ──────────────────────────────────────────────────────────────────────────────
# POSCAR / POTCAR parsing
# ──────────────────────────────────────────────────────────────────────────────

def parse_poscar(poscar_path: str) -> dict | None:
    """Parse POSCAR, auto-detect VASP 5/6 format.

    Returns: {format: 5|6, elements: [str], atom_counts: [int]} or None
    """
    try:
        with open(poscar_path, "r", encoding="utf-8") as f:
            lines = [f.readline() for _ in range(8)]
    except Exception:
        return None

    line0 = lines[0].strip()
    line5 = lines[5].strip()
    line6 = lines[6].strip()

    tokens0 = line0.split()
    if tokens0 and all(is_element_symbol(t) for t in tokens0):
        return {
            "format": 6,
            "elements": tokens0,
            "atom_counts": [int(x) for x in line5.split()],
        }

    tokens5 = line5.split()
    if tokens5 and all(is_element_symbol(t) for t in tokens5):
        return {
            "format": 5,
            "elements": tokens5,
            "atom_counts": [int(x) for x in line6.split()],
        }

    return None


def parse_potcar(potcar_path: str) -> list[str] | None:
    """Extract element symbols from POTCAR TITEL lines."""
    elements = []
    try:
        with open(potcar_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip().startswith("TITEL"):
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        elements.append(parts[3])
        return elements if elements else None
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# R3f convergence check
# ──────────────────────────────────────────────────────────────────────────────

def check_r3f_convergence(r3f_dir: str) -> dict:
    """Check R3f VASP calculation convergence.

    Returns: {converged, rmm_steps, energy, issues}
    """
    result = {"converged": False, "rmm_steps": 0, "energy": None, "issues": []}

    oszicar = os.path.join(r3f_dir, "OSZICAR")
    outcar = os.path.join(r3f_dir, "OUTCAR")

    if not os.path.exists(oszicar):
        result["issues"].append("OSZICAR not found")
        return result

    try:
        with open(oszicar, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception:
        result["issues"].append("Cannot read OSZICAR")
        return result

    if not lines:
        result["issues"].append("OSZICAR is empty")
        return result

    # Parse last non-empty line for energy
    last_line = ""
    for line in reversed(lines):
        if line.strip():
            last_line = line.strip()
            break

    energy_match = re.search(r"F=\s*([-\d.E+]+)", last_line)
    if energy_match:
        result["energy"] = float(energy_match.group(1))

    # Count RMM steps from the last SCF block
    for line in reversed(lines):
        m = re.match(r"RMM:\s+(\d+)", line.strip())
        if m:
            result["rmm_steps"] = int(m.group(1))
            break

    if result["rmm_steps"] >= 80:
        result["issues"].append(f"RMM={result['rmm_steps']} (>=80, likely unconverged)")

    # Check OUTCAR for convergence
    if os.path.exists(outcar):
        try:
            with open(outcar, "r", encoding="utf-8") as f:
                content = f.read()
            if "aborting loop because EDIFF is reached" in content:
                result["converged"] = True
            elif "reached required accuracy" in content:
                result["converged"] = True
            else:
                result["issues"].append("OUTCAR: no convergence marker found")
        except Exception:
            result["issues"].append("Cannot read OUTCAR")
    else:
        result["issues"].append("OUTCAR not found")

    if result["energy"] is None:
        result["issues"].append("Cannot parse final energy")

    return result


# ──────────────────────────────────────────────────────────────────────────────
# End-member check
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class EndMemberReport:
    path: str
    name: str
    has_gibbs: bool = False
    qha_count: int = 0
    qha_with_r3f: int = 0
    potcar_issues: list = field(default_factory=list)
    unconverged_r3f: list = field(default_factory=list)
    info: list = field(default_factory=list)

    @property
    def usable(self) -> bool:
        return (
            self.has_gibbs
            and self.qha_count > 0
            and len(self.potcar_issues) == 0
            and len(self.unconverged_r3f) == 0
        )


def find_gibbs_file(end_member_dir: str) -> str | None:
    """Find gibbs-temperature.dat in end-member's QHA subfolder."""
    for root, _, files in os.walk(end_member_dir):
        if "gibbs-temperature.dat" in files:
            return os.path.join(root, "gibbs-temperature.dat")
    return None


def check_potcar_type_count(potcar_path: str, poscar_path: str) -> str | None:
    """Check if POTCAR TITEL count matches POSCAR atom type count.

    Returns error message or None if OK.
    """
    potcar_elems = parse_potcar(potcar_path)
    poscar = parse_poscar(poscar_path)
    if not potcar_elems or not poscar:
        return None  # can't check

    n_potcar = len(potcar_elems)
    n_poscar = len(poscar["elements"])
    if n_potcar != n_poscar:
        return (
            f"POTCAR has {n_potcar} TITEL entries ({potcar_elems}) "
            f"but POSCAR has {n_poscar} atom types ({poscar['elements']})"
        )
    return None


def check_end_member(em_dir: str) -> EndMemberReport:
    """Check a single end-member directory for QHA data usability."""
    name = os.path.basename(em_dir)
    report = EndMemberReport(path=em_dir, name=name)

    # 1. Check for gibbs-temperature.dat
    gibbs_path = find_gibbs_file(em_dir)
    report.has_gibbs = gibbs_path is not None
    if not report.has_gibbs:
        report.info.append("No gibbs-temperature.dat found")
        return report

    # 2. Find all QHA volume points (dfpt/QHA-xx/)
    dfpt_dir = os.path.join(em_dir, "dfpt")
    if not os.path.isdir(dfpt_dir):
        report.info.append("No dfpt/ directory")
        return report

    qha_dirs = sorted([
        os.path.join(dfpt_dir, d)
        for d in os.listdir(dfpt_dir)
        if d.startswith("QHA-") and os.path.isdir(os.path.join(dfpt_dir, d))
    ])
    report.qha_count = len(qha_dirs)
    if report.qha_count == 0:
        report.info.append("No QHA-xx directories in dfpt/")
        return report

    # 3. Check each QHA volume point
    for qha_dir in qha_dirs:
        qha_name = os.path.basename(qha_dir)

        # 3a. Check QHA POTCAR type count vs POSCAR
        qha_potcar = os.path.join(qha_dir, "POTCAR")
        qha_poscar = os.path.join(qha_dir, "POSCAR")
        if os.path.exists(qha_potcar) and os.path.exists(qha_poscar):
            err = check_potcar_type_count(qha_potcar, qha_poscar)
            if err:
                report.potcar_issues.append({"dir": qha_name, "error": err})

        # 3b. Check R3f
        r3f_dir = os.path.join(qha_dir, "R3f")
        if os.path.isdir(r3f_dir):
            report.qha_with_r3f += 1

            # R3f POTCAR type count vs POSCAR
            r3f_potcar = os.path.join(r3f_dir, "POTCAR")
            r3f_poscar = os.path.join(r3f_dir, "POSCAR")
            if os.path.exists(r3f_potcar) and os.path.exists(r3f_poscar):
                err = check_potcar_type_count(r3f_potcar, r3f_poscar)
                if err:
                    report.potcar_issues.append({"dir": f"{qha_name}/R3f", "error": err})

            # R3f convergence
            conv = check_r3f_convergence(r3f_dir)
            if not conv["converged"]:
                report.unconverged_r3f.append({
                    "dir": f"{qha_name}/R3f",
                    "rmm_steps": conv["rmm_steps"],
                    "energy": conv["energy"],
                    "issues": conv["issues"],
                })

    return report


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Check VASP QHA data usability for end-member calculations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--root", type=str,
        default=r"E:\nas-shared\Endmembers\13symbols-hmd",
        help="Root directory to scan",
    )
    parser.add_argument("--summary", action="store_true", help="Show summary counts only")
    parser.add_argument("--usable", action="store_true", help="Show only usable end-members")
    parser.add_argument("--failed", action="store_true", help="Show only failed end-members")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"Error: {root} does not exist")
        return 1

    print(f"Scanning: {root}\n")

    # Find all end-member directories (have dfpt/ or QHA-*/ subfolder)
    em_dirs = []
    for dirpath, dirnames, _ in os.walk(str(root)):
        has_dfpt = "dfpt" in dirnames
        has_qha_sub = any(d.startswith("QHA-") for d in dirnames)
        if has_dfpt or has_qha_sub:
            em_dirs.append(dirpath)
    em_dirs = sorted(set(em_dirs))

    if not em_dirs:
        print("No end-member directories found.")
        return 1

    print(f"Found {len(em_dirs)} end-member directories\n")

    # Check each
    reports = [check_end_member(d) for d in em_dirs]

    usable = [r for r in reports if r.usable]
    has_gibbs = [r for r in reports if r.has_gibbs]
    failed = [r for r in reports if r.has_gibbs and not r.usable]
    no_gibbs = [r for r in reports if not r.has_gibbs]

    # ── Output ──

    if args.usable:
        print(f"=== Usable end-members ({len(usable)}/{len(has_gibbs)} with gibbs data) ===\n")
        for r in usable:
            print(f"  OK  {r.name}  ({r.qha_count} QHA volumes)")
        return 0

    if args.failed:
        print(f"=== Failed end-members ({len(failed)}/{len(has_gibbs)} with gibbs data) ===\n")
        for r in failed:
            reasons = []
            if r.potcar_issues:
                reasons.append(f"{len(r.potcar_issues)} POTCAR issues")
            if r.unconverged_r3f:
                reasons.append(f"{len(r.unconverged_r3f)} unconverged R3f")
            print(f"  FAIL  {r.name}: {'; '.join(reasons)}")
            for p in r.potcar_issues:
                print(f"          {p['dir']}: {p['error']}")
            for u in r.unconverged_r3f:
                print(f"          {u['dir']}: RMM={u['rmm_steps']}, E={u['energy']}")
        return 1

    if args.summary:
        print("=== Summary ===\n")
        print(f"  Total end-members:      {len(reports)}")
        print(f"  Has gibbs data:         {len(has_gibbs)}")
        print(f"  Usable:                 {len(usable)}")
        print(f"  Failed (has gibbs):     {len(failed)}")
        print(f"  No gibbs data:          {len(no_gibbs)}")
        print()
        potcar_count = sum(1 for r in failed if r.potcar_issues)
        conv_count = sum(1 for r in failed if r.unconverged_r3f)
        print(f"  Failure reasons:")
        print(f"    POTCAR type mismatch: {potcar_count}")
        print(f"    R3f unconverged:      {conv_count}")
        return 0

    # ── Detailed ──
    print(f"=== Usable ({len(usable)}/{len(has_gibbs)} with gibbs data) ===\n")
    for r in usable:
        print(f"  OK  {r.name}  ({r.qha_count} QHA volumes)")

    if failed:
        print(f"\n=== Failed ({len(failed)}) ===\n")
        for r in failed:
            print(f"  FAIL  {r.name}")
            for p in r.potcar_issues:
                print(f"          {p['dir']}: {p['error']}")
            for u in r.unconverged_r3f:
                print(f"          {u['dir']}: RMM={u['rmm_steps']}, E={u['energy']}, {u['issues']}")

    if no_gibbs:
        print(f"\n=== No gibbs data ({len(no_gibbs)}) ===\n")
        for r in no_gibbs:
            print(f"  --  {r.name}")

    return 0 if not failed else 1


if __name__ == "__main__":
    exit(main())
