# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

**mpea-tdb-fit** — CLI tool for managing MPEA (Multi-Principle Element Alloys) End-Member Thermodynamic Database files and fitting Gibbs-temperature data from DFT/Phonopy calculations.

## Commands

```bash
# Install dev
pip install -e ".[dev]"

# Run all tests
pytest

# Run a single test class/test
pytest tests/test_cli.py::TestCmdParse
pytest tests/test_cli.py::TestCmdParse::test_parse_with_tdb_name

# Run the CLI
python main.py parse --tdb-file path/to/file.tdb
python main.py import --tdb-file path/to/file.tdb --db my.db
python main.py fit --data-dir path/to/data --data_type json
```

## Architecture

### Data flow

```
TDB file ──[TDBParser]──▶ ParsedData (in-memory) ──[TDBManager]──▶ SQLite DB (ThermoDBI)
                                                          │
Phonopy/QHA data ──[GTFitter]──▶ FitResult[] ──[fit2db]──┘
```

### Layers

1. **`src/tdb/tdbi.py`** — `ThermoDBI`: Low-level SQLite CRUD interface. Five tables: `elements`, `functions` (SER functions), `tdbs` (metadata), `phases` (phase definitions with sublattice stoichiometry), `parameters` (Gibbs/Lambda parameters referencing phases). Connection uses `sqlite3.Row` row factory; all writes use transactions via `with self.db.conn:`.

2. **`src/tdb/tdbmgr.py`** — `TDBParser` (regex-based TDB file parsing, expression reformatting, export formatting) + `TDBManager` (high-level import/export logic: parse → deduplicate against DB → create or update). `ParsedData` is a `@dataclass` holding `elems`, `funcs`, `phases`, `params`, `tdb`.

3. **`src/gibbsfit.py`** — `GTFitter`: Fits `A + B*T + C*T*LN(T) + D*T**2 + E*T**3 + F*T**(-1)` to Gibbs-temperature data. Runs scipy `curve_fit` up to 100 times, keeping best R². Handles two data formats via `handle_dat` (Phonopy `gibbs-temperature.dat`, CSV) and `handle_json` (atomate QHA JSON). Key detail: BCC symmetric end-members (equal metrics, different elements) generate an additional exchanged entry.

4. **`src/cli.py`** — Argparse CLI with subcommands: `parse` (TDB→JSON), `import` (TDB→SQLite), `export` (SQLite→TDB), `fit` (data→TDB), `list`, `delete`. Each `cmd_*` function returns exit code 0/1.

5. **`src/config.py`** — Constants: `PHASE_METRICS` defines sublattice stoichiometries per phase (`SER: (1,)`, `BCC: (1,1)`, `FCC: (1,3)`, `HCP: (2,6)`).

### Standalone scripts (not part of the core package)

- **`compare_tdb.py`** — Compares two TDB files by evaluating functions/parameters numerically over temperature range and plotting differences.
- **`etot_fit.py`** — Parses VASP `v-e.dat` files to extract DFT static energies and format as TDB FUNCTION/PARAMETER entries.
- **`sftp_fit.py`** — Extends `GTFitter` with SFTP download capability (paramiko). Downloads remote QHA data, processes locally, fits, exports TDB. Hardcoded server configs — not for general use.
- **`scripts/extract_magnetic_moment.py`** — Extracts magnetic moments from VASP r3f files and compares against reference values.

### Database schema (`src/tdb/schema.sql`)

Foreign keys: `functions.elem → elements.elem`, `phases.tdb → tdbs.tdb`, `parameters(phase, tdb) → phases(phase, tdb)`. Parameter type CHECK: `G` or `L`. Order number CHECK: `≥ 0`. Continued flag CHECK: `Y` or `N`.

### Naming conventions for fit data folders

Folder names follow pattern: `{PHASE}-{elem1}-{elem2}[-{atom_count}]` (e.g., `BCC-Fe-Mn`, `HCP-Co-Ni-4`). The phase prefix must match a key in `PHASE_METRICS`.
