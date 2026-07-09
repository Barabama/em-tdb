"""
EM-TDB - Command-line interface for parsing and managing thermodynamic database files.
"""

import sys
import json
import logging
import argparse
import traceback
from pathlib import Path

from emtdb.tdb import ThermoDBI
from emtdb.tdb import TDBManager
from emtdb.gibbsfit import GTFitter
from emtdb.etotfit import ETotFitter
    
from emtdb.config import VERSION, DB_CHOICES, PHASE_METRICS, DATA_TYPES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s[%(levelname)s]%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def cmd_parse(args: argparse.Namespace) -> int:
    tdb_file = args.tdb_file
    tdb_file = Path(tdb_file) if isinstance(tdb_file, str) else tdb_file
    tdb_name = args.tdb_name or str(tdb_file.stem)
    try:
        tdb_mgr = TDBManager(ThermoDBI(":memory:"))
        parsed = tdb_mgr.parse_tdb(tdb_file, tdb_name)
        output = f"{tdb_name}.json"
        with open(output, "w", encoding="utf-8") as f:
            json.dump(parsed.to_dict(), f, indent=2)
        log.info(f"TDB {tdb_name} parsed to {output}")
        return 0
    except Exception as e:
        log.error(traceback.format_exc())
        log.error(f"Error parsing {tdb_file}: {e}")
        return 1


def cmd_import(args: argparse.Namespace) -> int:
    db_path = args.db or ":memory:"
    tdb_file = args.tdb_file
    tdb_file = Path(tdb_file) if isinstance(tdb_file, str) else tdb_file
    tdb_name = args.tdb_name or str(tdb_file.stem)
    typed = args.typed
    desc = args.desc or ""
    ver = args.ver or "1.0"
    try:
        tdb_mgr = TDBManager(ThermoDBI(db_path))
        parsed = tdb_mgr.parse_tdb(tdb_file, tdb_name)
        if typed == "elem":
            tdb_mgr.save_elements(parsed.elems)
            log.info(f"Imported {len(parsed.elems)} elements")
        elif typed == "func":
            tdb_mgr.save_functions(parsed.funcs)
            log.info(f"Imported {len(parsed.funcs)} functions")
        elif typed == "tdb":
            tdb_mgr.import_tdb(parsed.phases, parsed.params, parsed.tdb, desc, ver)
            log.info(f"Imported TDB '{tdb_name}' with phases and parameters")
        elif not typed:
            tdb_mgr.save_elements(parsed.elems)
            tdb_mgr.save_functions(parsed.funcs)
            tdb_mgr.import_tdb(parsed.phases, parsed.params, parsed.tdb, desc, ver)
            log.info(f"Imported TDB '{tdb_name}' all data")
        else:
            log.error(f"Unknown type {typed}")
            return 1
        return 0
    except Exception as e:
        log.error(traceback.format_exc())
        log.error(f"Error importing {tdb_file} into db {db_path}: {e}")
        return 1
    finally:
        tdb_mgr.db.close()


def cmd_export(args: argparse.Namespace) -> int:
    db_path = args.db
    output = args.output
    tdb_name = args.tdb_name
    try:
        tdb_mgr = TDBManager(ThermoDBI(db_path))
        tdb_mgr.export_tdb(tdb_name, output)
        log.info(f"Exported TDB '{tdb_name}' to {output}")
        return 0
    except Exception as e:
        log.error(traceback.format_exc())
        log.error(f"Error exporting TDB '{tdb_name}' from {db_path}: {e}")
        return 1
    finally:
        tdb_mgr.db.close()


def cmd_fit(args: argparse.Namespace) -> int:
    db_path = args.db or ":memory:"
    data_dir = args.data_dir
    data_dir = Path(data_dir) if isinstance(data_dir, str) else data_dir
    data_type = args.data_type
    tdb_name = args.tdb_name or str(data_dir.stem)
    try:
        tdb_mgr = TDBManager(ThermoDBI(db_path))
        fitter = GTFitter(PHASE_METRICS)
        results = fitter.process_folders(data_dir, data_type)
        log.info(f"Processed {len(results)} fits")

        img_path = data_dir.joinpath("fit_results.png")
        log.info(f"Plotting fit results to {img_path}...")
        fitter.plot_fits(results, img_path)

        if args.output_json:
            fitter.export_json(results, args.output_json)

        if args.output_csv:
            fitter.export_csv(results, args.output_csv)

        parsed = fitter.fit2db(results, tdb_name)

        log.info(f"Saving {len(parsed.funcs)} functions...")
        tdb_mgr.save_functions(parsed.funcs)

        log.info(f"Importing {len(parsed.params)} parameters...")
        tdb_mgr.import_tdb(
            parsed.phases,
            parsed.params,
            parsed.tdb,
            desc=f"fitted from {data_dir}",
            ver=VERSION,
        )
        log.info(f"Fitted {tdb_name} from {data_dir} and imported")

        # Export TDB if in memory
        if db_path == ":memory:":
            output = f"{tdb_name}.tdb"
            tdb_mgr.export_tdb(tdb_name, output)
            log.info(f"Exported {tdb_name} to {output}")

        return 0
    except Exception as e:
        log.error(traceback.format_exc())
        log.error(f"Failed to fit {tdb_name} from {data_dir}: {e}")
        return 1
    finally:
        tdb_mgr.db.close()


def cmd_etot(args: argparse.Namespace) -> int:
    """Fit DFT static energies (E0) from v-e.dat → TDB."""
    data_dir = Path(args.data_dir)
    tdb_name = args.tdb_name or data_dir.stem
    try:
        fitter = ETotFitter(PHASE_METRICS)
        results = fitter.process_folders(data_dir)
        log.info(f"Processed {len(results)} end-members")

        if args.output_json:
            fitter.export_json(results, args.output_json)

        if args.output_csv:
            fitter.export_csv(results, args.output_csv)

        parsed = fitter.results_to_parsed(results, tdb_name)
        log.info(f"Converted: {len(parsed.funcs)} functions, "
                 f"{len(parsed.phases)} phases, {len(parsed.params)} parameters")

        # Import into DB (in-memory unless --db given)
        db_path = args.db or ":memory:"
        tdb_mgr = TDBManager(ThermoDBI(db_path))

        if parsed.funcs:
            tdb_mgr.save_functions(parsed.funcs)
        if parsed.phases or parsed.params:
            tdb_mgr.import_tdb(
                parsed.phases,
                parsed.params,
                parsed.tdb,
                desc=f"fitted from {data_dir}",
                ver=VERSION,
            )

        # Export TDB file
        if args.output:
            tdb_mgr.export_tdb(tdb_name, args.output)
            log.info("Exported TDB to %s", args.output)
        elif db_path == ":memory:":
            output = f"{tdb_name}.tdb"
            tdb_mgr.export_tdb(tdb_name, output)
            log.info("Exported TDB to %s", output)

        return 0
    except Exception as e:
        log.error(traceback.format_exc())
        log.error(f"Failed to process {data_dir}: {e}")
        return 1


def cmd_subset(args: argparse.Namespace) -> int:
    """Extract a TDB subset containing only specified elements."""
    tdb_file = Path(args.tdb_file)
    tdb_name = args.tdb_name or tdb_file.stem
    output = args.output or f"{tdb_name}-subset.tdb"
    db_path = args.db or ":memory:"
    try:
        tdb_mgr = TDBManager(ThermoDBI(":memory:"))
        parsed = tdb_mgr.parse_tdb(tdb_file, tdb_name)
        log.info(
            f"Parsed {tdb_file}: "
            f"{len(parsed.elems)} elements, {len(parsed.funcs)} functions, "
            f"{len(parsed.phases)} phases, {len(parsed.params)} parameters"
        )

        from emtdb.tdb.tdbmgr import filter_parsed_data

        elements = set(args.elem)
        filtered = filter_parsed_data(parsed, elements)
        log.info(
            f"Filtered to {len(filtered.elems)} elements, {len(filtered.funcs)} functions, "
            f"{len(filtered.phases)} phases, {len(filtered.params)} parameters"
        )

        if not filtered.params:
            log.error("No parameters match the specified elements")
            return 1

        # Import to DB then export
        if db_path != ":memory:":
            tdb_mgr.db.close()
            import os
            if os.path.exists(db_path):
                os.remove(db_path)
            tdb_mgr = TDBManager(ThermoDBI(db_path))

        tdb_mgr.save_elements(filtered.elems)
        tdb_mgr.save_functions(filtered.funcs)
        tdb_mgr.import_tdb(
            filtered.phases, filtered.params, tdb_name,
            desc=f"subset from {tdb_file.name}",
            ver="1.0",
        )
        tdb_mgr.export_tdb(tdb_name, output)
        log.info(f"Exported subset to {output}")
        return 0
    except Exception as e:
        log.error(traceback.format_exc())
        log.error(f"Error creating subset: {e}")
        return 1
    finally:
        tdb_mgr.db.close()


def cmd_list(args: argparse.Namespace) -> int:
    db_path = args.db or ":memory:"
    typed = args.typed
    like = args.like
    filters = {
        "elem": args.elem,
        "func": args.func,
        "phase": args.phase,
        "param": args.param,
        "tdb": args.tdb,
    }
    try:
        db = ThermoDBI(db_path)
        func_map = {
            "elem": db.read_element,
            "func": db.read_function,
            "phase": db.read_phase,
            "param": db.read_parameter,
            "tdb": db.read_tdb,
        }
        result = func_map[typed](**filters, use_like=like)
        log.info(json.dumps(result, indent=2))
        return 0
    except Exception as e:
        log.error(traceback.format_exc())
        log.error(f"Error listing {typed} from {db_path}: {e}")
        return 1
    finally:
        db.close()


def cmd_delete(args: argparse.Namespace) -> int:
    db_path = args.db or ":memory:"
    typed = args.typed
    cascade = args.cascade
    filters = {
        "elem": args.elem,
        "func": args.func,
        "phase": args.phase,
        "param": args.param,
        "tdb": args.tdb,
    }
    try:
        db = ThermoDBI(db_path)
        func_map = {
            "elem": db.delete_element,
            "func": db.delete_function,
            "phase": db.delete_phase,
            "param": db.delete_parameter,
            "tdb": db.delete_tdb,
        }
        func_map[typed](**filters, cascade=cascade)
        log.info(f"Deleted {typed} from {db_path}")
        return 0
    except Exception as e:
        log.error(traceback.format_exc())
        log.error(f"Failed to delete {typed} from {db_path}: {e}")
        return 1
    finally:
        db.close()


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CLI for TDB management",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"{VERSION}")
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    # Parse ------------------------------------------------
    p_parse = subparsers.add_parser(
        "parse",
        help="Parse TDB file to structured format",
    )
    p_parse.add_argument(
        "--tdb-file",
        "-f",
        type=str,
        required=True,
        help="TDB file path",
    )
    p_parse.add_argument(
        "--tdb-name",
        "-n",
        type=str,
        default="",
        help="Optional logical tdb name in DB (default: filename)",
    )

    # Import ------------------------------------------------
    p_import = subparsers.add_parser(
        "import",
        help="Import TDB file into database",
    )
    p_import.add_argument(
        "--db",
        default="",
        help="Optional database path (default: in-memory)",
    )
    p_import.add_argument(
        "--tdb-file",
        "-f",
        type=str,
        required=True,
        help="TDB file path",
    )
    p_import.add_argument(
        "--typed",
        "-t",
        choices=DB_CHOICES,
        default="",
        help="Type of entry to import (default: all)",
    )
    p_import.add_argument(
        "--tdb-name",
        "-n",
        default="",
        help="Optional logical tdb name in DB (default: filename)",
    )
    p_import.add_argument(
        "--desc",
        "-d",
        default="",
        help="Optional description of the tdb",
    )
    p_import.add_argument(
        "--ver",
        "-v",
        default="",
        help="Optional version of the tdb",
    )

    # Export ------------------------------------------------
    p_export = subparsers.add_parser(
        "export",
        help="Export TDB from database to file",
    )
    p_export.add_argument(
        "--db",
        type=str,
        required=True,
        help="Database path",
    )
    p_export.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Output TDB file path",
    )
    p_export.add_argument(
        "--tdb-name",
        "-n",
        type=str,
        required=True,
        help="Logical tdb name in DB",
    )

    # Fit Gibbs ------------------------------------------------
    p_fit = subparsers.add_parser(
        "fit",
        help="Fit Gibbs-Temperature data from Phonopy output",
    )
    p_fit.add_argument(
        "--db",
        default="",
        help="Optional database path (default: in-memory)",
    )
    p_fit.add_argument(
        "--data-dir",
        "-d",
        required=True,
        help="Directory containing data",
    )
    p_fit.add_argument(
        "--data_type",
        "-t",
        choices=DATA_TYPES,
        default="dat",
        help="Data file type",
    )
    p_fit.add_argument(
        "--tdb-name",
        "-n",
        default="",
        help="Optional logical tdb name in DB (default: data_dir name)",
    )
    p_fit.add_argument(
        "--output-json",
        type=str,
        default="",
        help="Output JSON file path for fit results (includes R², params A-F, raw data)",
    )
    p_fit.add_argument(
        "--output-csv",
        type=str,
        default="",
        help="Output CSV file path for fit results (flat columns: name/phase/elements/r2/A/B/C/D/E/F)",
    )

    # Fit E0 (etot) ------------------------------------------------
    p_etot = subparsers.add_parser(
        "etot",
        help="Fit DFT static energies (E0) from v-e.dat files",
    )
    p_etot.add_argument(
        "--db",
        default="",
        help="Optional database path (default: in-memory)",
    )
    p_etot.add_argument(
        "--data-dir",
        "-d",
        required=True,
        help="Directory containing end-member folders",
    )
    p_etot.add_argument(
        "--tdb-name",
        "-n",
        default="",
        help="Logical TDB name in DB (default: data_dir name)",
    )
    p_etot.add_argument(
        "--output", "-o",
        type=str,
        default="",
        help="Output TDB file path (default: {tdb-name}.tdb)",
    )
    p_etot.add_argument(
        "--output-json",
        type=str,
        default="",
        help="Output JSON file path for fitted E0/V0/B0 results",
    )
    p_etot.add_argument(
        "--output-csv",
        type=str,
        default="",
        help="Output CSV file path for fitted E0/V0/B0 results",
    )

    # Subset ------------------------------------------------
    p_subset = subparsers.add_parser(
        "subset",
        help="Extract TDB subset containing only specified elements",
    )
    p_subset.add_argument(
        "--tdb-file", "-f",
        type=str,
        required=True,
        help="Input TDB file path",
    )
    p_subset.add_argument(
        "--elem", "-e",
        type=str,
        nargs="+",
        required=True,
        metavar="ELEM",
        help="Element symbols to keep (e.g. FE CR NI)",
    )
    p_subset.add_argument(
        "--db",
        default="",
        help="Optional database path (default: in-memory)",
    )
    p_subset.add_argument(
        "--tdb-name", "-n",
        default="",
        help="Optional logical tdb name (default: input filename)",
    )
    p_subset.add_argument(
        "--output", "-o",
        type=str,
        default="",
        help="Output TDB file path (default: {tdb-name}-subset.tdb)",
    )

    # List ------------------------------------------------
    p_list = subparsers.add_parser(
        "list",
        help="List entries in database",
    )
    p_list.add_argument(
        "--db",
        default="",
        help="Optional database path (default: in-memory)",
    )
    p_list.add_argument(
        "--typed",
        "-t",
        required=True,
        choices=DB_CHOICES,
        help="Entry type to list",
    )
    p_list.add_argument(
        "--like",
        "-l",
        action="store_true",
        default=False,
        help="Use LIKE in query",
    )
    p_list.add_argument(
        "--elem",
        default="",
        help="Filter by element for elements and functions",
    )
    p_list.add_argument(
        "--func",
        default="",
        help="Filter by function for functions",
    )
    p_list.add_argument(
        "--phase",
        default="",
        help="Filter by phase for phases and parameters",
    )
    p_list.add_argument(
        "--param",
        default="",
        help="Filter by parameter for parameters",
    )
    p_list.add_argument(
        "--tdb",
        default="",
        help="Filter by tdb for tdbs, phases and parameters",
    )

    # Delete ------------------------------------------------
    p_delete = subparsers.add_parser(
        "delete",
        help="Delete entry from database",
    )
    p_delete.add_argument(
        "--db",
        default="",
        help="Optional database path (default: in-memory)",
    )
    p_delete.add_argument(
        "--typed",
        "-t",
        required=True,
        choices=DB_CHOICES,
        help="Entry type to delete",
    )
    p_delete.add_argument(
        "--cascade",
        "-c",
        action="store_true",
        default=False,
        help="Delete dependent entries",
    )
    p_delete.add_argument(
        "--elem",
        default="",
        help="Filter by element for elements and functions",
    )
    p_delete.add_argument(
        "--func",
        default="",
        help="Filter by function for functions",
    )
    p_delete.add_argument(
        "--phase",
        default="",
        help="Filter by phase for phases and parameters",
    )
    p_delete.add_argument(
        "--param",
        default="",
        help="Filter by parameter for parameters",
    )
    p_delete.add_argument(
        "--tdb",
        default="",
        help="Filter by tdb for tdbs, phases and parameters",
    )

    # Register handlers
    p_parse.set_defaults(func=cmd_parse)
    p_import.set_defaults(func=cmd_import)
    p_export.set_defaults(func=cmd_export)
    p_list.set_defaults(func=cmd_list)
    p_delete.set_defaults(func=cmd_delete)
    p_fit.set_defaults(func=cmd_fit)
    p_etot.set_defaults(func=cmd_etot)
    p_subset.set_defaults(func=cmd_subset)

    return parser


def main() -> int:
    parser = create_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
