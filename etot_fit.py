"""
Fit DFT static energies (E0 from VASP v-e.dat) into TDB format.

Extracts the lowest DFT total energy for each end-member structure
and outputs as TDB FUNCTION/PARAMETER entries.

Folder naming: {PHASE}-{ELEM1}-{ELEM2}[-{atom_num}]
  e.g. SER-Fe-1, BCC-Fe-Mn-2, FCC-Al-Co-4
"""

import argparse
import re
import logging
from pathlib import Path

from emtdb.config import PHASE_METRICS
from emtdb.tdb.tdbi import Func, Phase, Param
from emtdb.tdb.tdbmgr import ParsedData

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _pad(elem: str) -> str:
    """Pad element symbol to 2 chars by repeating the last character.

    V → VV,  Nj → Nj (already 2 chars),  Al → Al (already 2 chars)
    Used for function names (e.g. SERVV, ETSERNi) and SER# references.
    """
    return elem.strip().upper().ljust(2, elem.strip()[-1].upper())


def _find_v_e_dat(folder: Path) -> Path | None:
    """Find the deepest v-e.dat file under folder."""
    candidates = sorted(
        folder.rglob("*v-e.dat"),
        key=lambda p: len(p.parts),
    )
    # rglob may also match *v-e.dat parent directories as files;
    # filter for actual files
    files = [p for p in candidates if p.is_file()]
    return files[-1] if files else None


def _parse_v_e_dat(path: Path, atom_num: int = 1) -> float:
    """Extract minimum total energy from a v-e.dat file.

    v-e.dat format: volume(V)  energy(eV)
    Returns energy in J/mol/atom.
    """
    energies = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                energies.append(float(parts[1]))
            except ValueError:
                log.warning("Skipping unparseable line in %s: %s", path, line)
    if not energies:
        raise ValueError(f"No energy data found in {path}")
    return min(energies) * 96485 / atom_num


def _parse_folder_name(name: str) -> dict:
    """Parse folder name into phase, elements, metrics, atom_num.

    Returns dict with keys: phase, elems (raw), metrics (normalised), atom_num
    """
    parts = name.split("-")
    if len(parts) < 2:
        raise ValueError(f"Invalid folder name: {name}")
    phase = parts[0].upper()
    if phase not in PHASE_METRICS:
        raise ValueError(f"Unknown phase '{phase}' in '{name}'")

    metrics = list(PHASE_METRICS[phase])
    m_sum = sum(metrics)
    metrics = [m / m_sum for m in metrics]
    elems = parts[1 : 1 + len(metrics)]

    m = re.search(r"-(\d+)(?:atoms?)?$", name)
    atom_num = int(m.group(1)) if m else 1

    return {"phase": phase, "elems": elems, "metrics": metrics, "atom_num": atom_num}


# ──────────────────────────────────────────────────────────────────────
# E0 → TDB entry
# ──────────────────────────────────────────────────────────────────────

def e0_to_func(elem: str, energy: float) -> Func:
    """Create a SER reference Func from a single-element E0."""
    name = f"ETSER{_pad(elem)}"
    return Func(
        func=name,
        elem=elem,
        temp_start=1.0,
        temp_end=6000.0,
        expression=f"{energy:+E}",
        is_continued="N",
    )


def e0_to_param(phase: str, elems: list[str], metrics: list[float],
                energy: float) -> Param:
    """Create a G parameter for a binary end-member."""
    components = ":".join(elems)  # raw, no padding
    ser_ref = "".join(
        f"-{m}*ETSER{_pad(e)}#" for e, m in zip(elems, metrics)
    )
    return Param(
        param="",
        ptype="G",
        phase=phase,
        components=components,
        order_num=0,
        temp_start=1.0,
        temp_end=6000.0,
        tdb="",
        expression=f"{energy:+E}{ser_ref}",
        is_continued="N",
    )


def process_folder(folder: Path) -> ParsedData:
    """Parse a single end-member folder and return ParsedData.

    For SER phase → returns a Func.
    For binary phases (BCC/FCC/HCP) → returns a Phase + Param(s).
    Symmetric phases with different elements also generate an exchanged Param.
    """
    info = _parse_folder_name(folder.name)
    phase = info["phase"]
    elems = info["elems"]
    metrics = info["metrics"]
    atom_num = info["atom_num"]

    ve_path = _find_v_e_dat(folder)
    if not ve_path:
        raise FileNotFoundError(f"No v-e.dat found under {folder}")
    energy = _parse_v_e_dat(ve_path, atom_num)

    funcs: list[Func] = []
    phases: list[Phase] = []
    params: list[Param] = []

    if phase == "SER":
        funcs.append(e0_to_func(elems[0], energy))
    else:
        phases.append(Phase(
            phase=phase,
            sub_lattices=len(metrics),
            stoichiometry=" ".join(str(m) for m in metrics),
            components=":".join(elems),
            tdb="",
        ))
        params.append(e0_to_param(phase, elems, metrics, energy))

        # Symmetric exchange: BCC(1,1) with distinct elements → reversed entry
        if len(metrics) > 1 and metrics[0] == metrics[1] and elems[0] != elems[1]:
            ex_elems = list(reversed(elems))
            params.append(e0_to_param(phase, ex_elems, metrics, energy))

    return ParsedData(elems=[], funcs=funcs, phases=phases, params=params, tdb="")


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fit DFT E0 from v-e.dat to TDB format"
    )
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Root dir containing end-member folders")
    parser.add_argument("--tdb-name", type=str, default="",
                        help="Name for the TDB (default: data-dir basename)")
    parser.add_argument("--output", type=str, default="",
                        help="Output TDB file path (default: stdout)")
    args = parser.parse_args()

    root = Path(args.data_dir)
    if not root.is_dir():
        log.error("Directory not found: %s", root)
        return 1

    tdb_name = args.tdb_name or root.stem

    # Collect all end-member folders
    folders = sorted(root.iterdir())

    all_funcs: list[Func] = []
    all_phases: list[Phase] = []
    all_params: list[Param] = []
    errors = []

    for f in folders:
        if not f.is_dir():
            continue
        try:
            pd = process_folder(f)
            all_funcs.extend(pd.funcs)
            all_phases.extend(pd.phases)
            all_params.extend(pd.params)
        except Exception as e:
            errors.append((f.name, str(e)))
            log.warning("Skipping %s: %s", f.name, e)

    # Deduplicate SER functions
    seen = set()
    deduped_funcs = []
    for fun in all_funcs:
        if fun["func"] not in seen:
            seen.add(fun["func"])
            deduped_funcs.append(fun)

    # Assign tdb name
    for p in all_phases:
        p["tdb"] = tdb_name
    for p in all_params:
        p["tdb"] = tdb_name

    # Build output lines
    lines: list[str] = []

    if deduped_funcs:
        lines.append("$ FUNCTIONS")
        for fun in deduped_funcs:
            lines.append(
                f"FUNCTION {fun['func']} {fun['temp_start']:.2f} "
                f"{fun['expression']}; {fun['temp_end']:.2f} {fun['is_continued']} !"
            )

    if all_phases:
        lines.append("\n$ PHASE AND PARAMETER DATA END")

    for ph in all_phases:
        lines.append(
            f"\nPHASE {ph['phase']} % {ph['sub_lattices']} {ph['stoichiometry']} !"
        )
        # CONSTITUENT line from components
        constituents = ":".join(
            c.strip() for c in ph["components"].split(":")
        )
        lines.append(f"CONSTITUENT {ph['phase']} : {constituents} : !")

    for p in all_params:
        ser_expr = p["expression"]
        lines.append(
            f"PARAMETER G({p['phase']},{p['components']};{p['order_num']}) "
            f"{p['temp_start']:.2f} {ser_expr}; {p['temp_end']:.2f} {p['is_continued']} !"
        )

    output = "\n".join(lines)

    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
        log.info("Written %d lines to %s", len(lines), args.output)
    else:
        print(output)

    if errors:
        log.warning("\nSkipped %d folders:", len(errors))
        for name, reason in errors:
            log.warning("  %s: %s", name, reason)

    return 0 if not errors else 1


if __name__ == "__main__":
    exit(main())
