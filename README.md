# mpea-tdb-fit

MPEA (Multi-Principle Element Alloys) End-Member Thermodynamic Database management tool — parse, store, fit, and export CALPHAD-style TDB files with Gibbs-temperature data fitting from DFT/Phonopy calculations.

## Installation

```bash
pip install -e ".[dev]"
```

Requires Python >= 3.10. Core dependencies: numpy, pandas, matplotlib, scipy, sympy, openpyxl, tqdm.

## Quick start

```bash
# Parse a TDB file to JSON
python main.py parse -f my_alloy.tdb

# Import into a persistent SQLite database
python main.py import -f my_alloy.tdb --db alloy.db

# List elements in the database
python main.py list --db alloy.db -t elem

# Fit Gibbs-temperature data and produce a TDB
python main.py fit -d path/to/fit-data/ --data_type json

# Export a TDB from the database
python main.py export --db alloy.db -n my_alloy -o output.tdb
```

## CLI reference

| Command    | Description                                      |
|------------|--------------------------------------------------|
| `parse`    | Parse TDB file → JSON (no database)              |
| `import`   | Parse TDB file → SQLite database                 |
| `export`   | SQLite database → TDB file                       |
| `fit`      | Fit Gibbs-temperature data, store in DB, export  |
| `list`     | Query database contents (elements, functions, phases, parameters, TDB metadata) |
| `delete`   | Remove entries from database (supports cascade)  |

Use `python main.py <COMMAND> --help` for detailed options.

## Data format: fit directories

The `fit` command expects a directory tree where each subfolder is named `{PHASE}-{elem1}-{elem2}[-{atom_count}]`, e.g.:

```
fit-data/
  BCC-Fe-Mn/
    gibbs-temperature.dat       # Phonopy format
  FCC-Co-Ni-4/
    qha_output.json             # atomate QHA Flow format
  SER-Fe/
  SER-Mn/
```

Supported data types: `dat` (Phonopy `gibbs-temperature.dat`) and `json` (atomate QHA Flow JSON). The phase prefix must match a key in `PHASE_METRICS` in [src/config.py](src/config.py).

## Fitting formula

The fitting engine fits the standard CALPHAD Gibbs expression:

```
G(T) = A + B·T + C·T·ln(T) + D·T² + E·T³ + F/T
```

Fitting runs `scipy.optimize.curve_fit` up to 100 times per dataset, retaining the best R² result. Data is restricted to the 100–2900 K range.

## Architecture

```
main.py ──▶ src/cli.py (argparse subcommands)
                │
                ├──▶ src/tdb/tdbmgr.py (TDBManager: parse/import/export)
                │        └──▶ src/tdb/tdbi.py (ThermoDBI: SQLite CRUD)
                │
                └──▶ src/gibbsfit.py (GTFitter: curve fitting + plotting)
```

- **`src/tdb/tdbi.py`** — SQLite interface with 5 tables: `elements`, `functions`, `tdbs`, `phases`, `parameters`. See [src/tdb/schema.sql](src/tdb/schema.sql).
- **`src/tdb/tdbmgr.py`** — Regex-based TDB parser/formatter. Converts between TDB text format and structured `ParsedData`.
- **`src/gibbsfit.py`** — Gibbs-temperature curve fitter. Reads `.dat` or `.json` data, fits 6-parameter expression, generates plots, converts results to database-ready objects.
- **`src/config.py`** — Phase stoichiometry definitions (`SER`, `BCC`, `FCC`, `HCP`) and type constants.

## Standalone scripts

These scripts are self-contained and not part of the core package:

| Script                                            | Purpose                                          |
|---------------------------------------------------|--------------------------------------------------|
| [compare_tdb.py](compare_tdb.py)                  | Numerically compare two TDB files and plot differences |
| [etot_fit.py](etot_fit.py)                        | Extract DFT static energies from VASP `v-e.dat` files |
| [sftp_fit.py](sftp_fit.py)                        | Remote fitting pipeline via SFTP (paramiko)      |
| [scripts/extract_magnetic_moment.py](scripts/extract_magnetic_moment.py) | Extract magnetic moments from VASP r3f files |

## Development

```bash
# Run all tests
pytest

# Run a single test
pytest tests/test_cli.py::TestCmdParse
pytest tests/test_cli.py::TestCmdParse::test_parse_with_tdb_name
```

Tests use temporary SQLite databases — no external setup required. The file `tests/test1.tdb` serves as the test fixture TDB.

## License

MIT — see [LICENSE](LICENSE).
