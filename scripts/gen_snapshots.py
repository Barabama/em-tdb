#!/usr/bin/env python3
"""Generate cross-validation snapshots from the Python demo.

Runs ``backup_old/gibbsfit.py`` on each test dataset and saves
the R² and full TDB PARAMETER string as JSON files under
``tests/snapshots/``.  Rust integration tests read these
snapshots and assert that their outputs match.

Usage:
    python scripts/gen_snapshots.py

Output (one .json per dataset):
    tests/snapshots/SER-Nb-2atoms.json
    tests/snapshots/BCC-TiNb-2.json
    tests/snapshots/BCC-Al-Al-2.json
"""

import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEMO = REPO_ROOT / "backup_old" / "gibbsfit.py"
SNAPSHOT_DIR = REPO_ROOT / "tests" / "snapshots"
TEST_DATA = REPO_ROOT / "tests" / "fits-dat"

DATASETS = [
    {
        "name": "SER-Nb-2atoms",
        "filepath": str(TEST_DATA / "SER-Nb-2atoms" / "gibbs-temperature.dat"),
        "phase": "SER",
        "metrics": ["1"],
        "elem": ["Nb"],
        "atom_num": "2",
    },
    {
        "name": "BCC-TiNb-2",
        "filepath": str(TEST_DATA / "BCC-TiNb-2" / "gibbs-temperature.dat"),
        "phase": "BCC",
        "metrics": ["1", "1"],
        "elem": ["Ti", "Nb"],
        "atom_num": "2",
    },
    {
        "name": "BCC-Al-Al-2",
        "filepath": str(TEST_DATA / "BCC-Al-Al-2" / "QHA-AlAl" / "gibbs-temperature.dat"),
        "phase": "BCC",
        "metrics": ["1", "1"],
        "elem": ["Al", "Al"],
        "atom_num": "2",
    },
]

# Regex to extract output fields
RX_R2 = re.compile(r"R² =\s+([\d.]+)")
RX_TDB = re.compile(r"TDB =\s+(.+)")

# Regex to find "ex" (exchanged) blocks
RX_EX = re.compile(r"Name:\s+(\S+-ex)")


def main():
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

    for ds in DATASETS:
        name = ds["name"]
        p = Path(ds["filepath"])
        if not p.exists():
            print(f"[SKIP] {name} — file not found: {p}")
            continue

        cmd = [
            sys.executable, str(DEMO), "single",
            "--filepath", ds["filepath"],
            "--phase", ds["phase"],
            "--metrics", *ds["metrics"],
            "--elem", *ds["elem"],
            "--atom_num", ds["atom_num"],
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[SKIP] {name} — demo returned {result.returncode}: {result.stderr.strip()}")
            continue

        output = result.stdout + result.stderr  # tqdm writes to stderr

        # Parse R² — take the first one (main entry, not -ex)
        r2_match = RX_R2.search(output)
        if r2_match is None:
            print(f"[SKIP] {name} — could not parse R²")
            continue
        r2 = float(r2_match.group(1))

        # Parse TDB lines — collect all, skip -ex ones
        tdb_lines = RX_TDB.findall(output)
        main_tdbs = [t for t in tdb_lines if "-ex" not in t]

        if not main_tdbs:
            print(f"[SKIP] {name} — no TDB line found")
            continue

        snapshot = {
            "name": name,
            "r2": r2,
            "tdb_parameter": main_tdbs[0],
        }

        snap_path = SNAPSHOT_DIR / f"{name}.json"
        with open(snap_path, "w") as f:
            json.dump(snapshot, f, indent=2)
        print(f"[OK]   {name} → {snap_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
