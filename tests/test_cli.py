import argparse
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile

from src.cli import (
    cmd_parse,
    cmd_import,
    cmd_export,
    cmd_list,
    cmd_delete,
    cmd_fit,
    create_parser,
)
from src.tdb.tdbi import ThermoDBI
from src.tdb.tdbmgr import TDBManager


@pytest.fixture
def test_tdb_file():
    return Path(__file__).parent / "test1.tdb"


@pytest.fixture(scope="module")
def parser():
    return create_parser()


class TestCmdParse:
    def test_parse_with_json_output(self, test_tdb_file):
        args = argparse.Namespace(tdb_file=str(test_tdb_file), output="json", tdb_name="test")
        result = cmd_parse(args)
        assert result == 0

    def test_parse_with_repr_output(self, test_tdb_file):
        args = argparse.Namespace(tdb_file=str(test_tdb_file), output="repr", tdb_name="")
        result = cmd_parse(args)
        assert result == 0

    def test_parse_nonexistent_file(self):
        args = argparse.Namespace(tdb_file="nonexistent.tdb", output="json", tdb_name="")
        result = cmd_parse(args)
        assert result == 1

    def test_parse_with_tdb_name(self, test_tdb_file):
        args = argparse.Namespace(
            tdb_file=str(test_tdb_file), output="json", tdb_name="custom_tdb_name"
        )
        result = cmd_parse(args)
        assert result == 0


class TestParser:
    def test_parse_command_required_args(self, parser, test_tdb_file):
        args = parser.parse_args(["parse", "--tdb-file", str(test_tdb_file)])
        assert args.command == "parse"
        assert args.tdb_file == str(test_tdb_file)
        assert args.output == "json"
        assert args.tdb_name == ""

    def test_parse_command_with_all_options(self, parser, test_tdb_file):
        args = parser.parse_args(
            [
                "parse",
                "--tdb-file",
                str(test_tdb_file),
                "--output",
                "repr",
                "--tdb-name",
                "test_tdb",
            ]
        )
        assert args.command == "parse"
        assert args.tdb_file == str(test_tdb_file)
        assert args.output == "repr"
        assert args.tdb_name == "test_tdb"

    def test_parse_command_short_options(self, parser, test_tdb_file):
        args = parser.parse_args(
            ["parse", "-f", str(test_tdb_file), "-o", "json", "-n", "test"]
        )
        assert args.command == "parse"
        assert args.tdb_file == str(test_tdb_file)
        assert args.output == "json"
        assert args.tdb_name == "test"

    def test_parse_command_missing_tdb_file(self, parser):
        with pytest.raises(SystemExit):
            parser.parse_args(["parse"])

    def test_parse_command_invalid_output(self, parser, test_tdb_file):
        with pytest.raises(SystemExit):
            parser.parse_args(
                ["parse", "--tdb-file", str(test_tdb_file), "--output", "invalid"]
            )

    def test_parser_has_parse_subcommand(self, parser):
        with pytest.raises(SystemExit):
            parser.parse_args(["parse", "--help"])


class TestCmdImport:
    def test_import_elements(self, test_tdb_file):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            args = argparse.Namespace(
                db=str(db_path),
                tdb_file=str(test_tdb_file),
                typed="elem",
                tdb_name="test",
                desc="test description",
                ver="1.0",
            )
            result = cmd_import(args)
            assert result == 0

    def test_import_functions(self, test_tdb_file):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            args = argparse.Namespace(
                db=str(db_path),
                tdb_file=str(test_tdb_file),
                typed="elem",
                tdb_name="test",
                desc="test description",
                ver="1.0",
            )
            result = cmd_import(args)
            assert result == 0

            args = argparse.Namespace(
                db=str(db_path),
                tdb_file=str(test_tdb_file),
                typed="func",
                tdb_name="test",
                desc="test description",
                ver="1.0",
            )
            result = cmd_import(args)
            assert result == 0

    def test_import_phases_and_params(self, test_tdb_file):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            args = argparse.Namespace(
                db=str(db_path),
                tdb_file=str(test_tdb_file),
                typed="phase",
                tdb_name="test",
                desc="test description",
                ver="1.0",
            )
            result = cmd_import(args)
            assert result == 0

    def test_import_nonexistent_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            args = argparse.Namespace(
                db=str(db_path),
                tdb_file="nonexistent.tdb",
                typed="elem",
                tdb_name="test",
                desc="test description",
                ver="1.0",
            )
            result = cmd_import(args)
            assert result == 1

    def test_import_with_custom_tdb_name(self, test_tdb_file):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            args = argparse.Namespace(
                db=str(db_path),
                tdb_file=str(test_tdb_file),
                typed="elem",
                tdb_name="custom_tdb_name",
                desc="custom description",
                ver="2.0",
            )
            result = cmd_import(args)
            assert result == 0


class TestParserImport:
    def test_import_command_required_args(self, parser, test_tdb_file):
        args = parser.parse_args(
            ["import", "--typed", "elem", "--tdb-file", str(test_tdb_file)]
        )
        assert args.command == "import"
        assert args.tdb_file == str(test_tdb_file)
        assert args.typed == "elem"
        assert args.db == ":memory:"
        assert args.tdb_name == ""
        assert args.desc == ""
        assert args.ver == ""

    def test_import_command_with_all_options(self, parser, test_tdb_file):
        args = parser.parse_args(
            [
                "import",
                "--db",
                "test.db",
                "--typed",
                "func",
                "--tdb-file",
                str(test_tdb_file),
                "--tdb-name",
                "test_tdb",
                "--desc",
                "test description",
                "--ver",
                "1.0",
            ]
        )
        assert args.command == "import"
        assert args.db == "test.db"
        assert args.typed == "func"
        assert args.tdb_file == str(test_tdb_file)
        assert args.tdb_name == "test_tdb"
        assert args.desc == "test description"
        assert args.ver == "1.0"

    def test_import_command_short_options(self, parser, test_tdb_file):
        args = parser.parse_args(
            [
                "import",
                "-t",
                "phase",
                "-f",
                str(test_tdb_file),
                "-n",
                "test",
                "-d",
                "desc",
                "-v",
                "2.0",
            ]
        )
        assert args.command == "import"
        assert args.typed == "phase"
        assert args.tdb_file == str(test_tdb_file)
        assert args.tdb_name == "test"
        assert args.desc == "desc"
        assert args.ver == "2.0"

    def test_import_command_missing_typed(self, parser, test_tdb_file):
        with pytest.raises(SystemExit):
            parser.parse_args(["import", "--tdb-file", str(test_tdb_file)])

    def test_import_command_missing_tdb_file(self, parser):
        with pytest.raises(SystemExit):
            parser.parse_args(["import", "--typed", "elem"])

    def test_import_command_invalid_typed(self, parser, test_tdb_file):
        with pytest.raises(SystemExit):
            parser.parse_args(
                ["import", "--typed", "invalid", "--tdb-file", str(test_tdb_file)]
            )

    def test_parser_has_import_subcommand(self, parser):
        with pytest.raises(SystemExit):
            parser.parse_args(["import", "--help"])


class TestCmdExport:
    def test_export_to_file(self, test_tdb_file):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            output_path = Path(tmpdir) / "output.tdb"

            tdb_mgr = TDBManager(ThermoDBI(str(db_path)))
            parsed = tdb_mgr.parse_tdb(test_tdb_file, "test")
            tdb_mgr.import_tdb(
                parsed.phases, parsed.params, parsed.tdb, "test description", "1.0"
            )

            args = argparse.Namespace(db=str(db_path), output=str(output_path), tdb_name="test")
            result = cmd_export(args)
            assert result == 0
            assert output_path.exists()

    def test_export_nonexistent_tdb(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            output_path = Path(tmpdir) / "output.tdb"

            ThermoDBI(str(db_path))

            args = argparse.Namespace(
                db=str(db_path), output=str(output_path), tdb_name="nonexistent"
            )
            result = cmd_export(args)
            assert result == 1

    def test_export_with_custom_tdb_name(self, test_tdb_file):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            output_path = Path(tmpdir) / "output.tdb"

            tdb_mgr = TDBManager(ThermoDBI(str(db_path)))
            parsed = tdb_mgr.parse_tdb(test_tdb_file, "custom_tdb")
            tdb_mgr.import_tdb(
                parsed.phases, parsed.params, parsed.tdb, "custom description", "2.0"
            )

            args = argparse.Namespace(
                db=str(db_path), output=str(output_path), tdb_name="custom_tdb"
            )
            result = cmd_export(args)
            assert result == 0
            assert output_path.exists()


class TestParserExport:
    def test_export_command_required_args(self, parser):
        args = parser.parse_args(["export", "--output", "output.tdb"])
        assert args.command == "export"
        assert args.output == "output.tdb"
        assert args.db == ":memory:"
        assert args.tdb_name == ""

    def test_export_command_with_all_options(self, parser):
        args = parser.parse_args(
            ["export", "--db", "test.db", "--output", "output.tdb", "--tdb-name", "test"]
        )
        assert args.command == "export"
        assert args.db == "test.db"
        assert args.output == "output.tdb"
        assert args.tdb_name == "test"

    def test_export_command_short_options(self, parser):
        args = parser.parse_args(["export", "-o", "output.tdb", "-n", "test"])
        assert args.command == "export"
        assert args.output == "output.tdb"
        assert args.tdb_name == "test"

    def test_export_command_missing_output(self, parser):
        with pytest.raises(SystemExit):
            parser.parse_args(["export"])

    def test_parser_has_export_subcommand(self, parser):
        with pytest.raises(SystemExit):
            parser.parse_args(["export", "--help"])


class TestCmdList:
    def test_list_elements(self, test_tdb_file):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"

            tdb_mgr = TDBManager(ThermoDBI(str(db_path)))
            parsed = tdb_mgr.parse_tdb(test_tdb_file, "test")
            tdb_mgr.save_elements(parsed.elems)

            args = argparse.Namespace(
                db=str(db_path),
                typed="elem",
                like=False,
                elem="",
                func="",
                phase="",
                param="",
                tdb="",
            )
            result = cmd_list(args)
            assert result == 0

    def test_list_phases(self, test_tdb_file):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"

            tdb_mgr = TDBManager(ThermoDBI(str(db_path)))
            parsed = tdb_mgr.parse_tdb(test_tdb_file, "test")
            tdb_mgr.import_tdb(
                parsed.phases, parsed.params, parsed.tdb, "test description", "1.0"
            )

            args = argparse.Namespace(
                db=str(db_path),
                typed="phase",
                like=False,
                elem="",
                func="",
                phase="",
                param="",
                tdb="test",
            )
            result = cmd_list(args)
            assert result == 0

    def test_list_with_like(self, test_tdb_file):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"

            tdb_mgr = TDBManager(ThermoDBI(str(db_path)))
            parsed = tdb_mgr.parse_tdb(test_tdb_file, "test")
            tdb_mgr.save_elements(parsed.elems)

            args = argparse.Namespace(
                db=str(db_path),
                typed="elem",
                like=True,
                elem="A",
                func="",
                phase="",
                param="",
                tdb="",
            )
            result = cmd_list(args)
            assert result == 0

    def test_list_tdbs(self, test_tdb_file):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"

            tdb_mgr = TDBManager(ThermoDBI(str(db_path)))
            parsed = tdb_mgr.parse_tdb(test_tdb_file, "test")
            tdb_mgr.import_tdb(
                parsed.phases, parsed.params, parsed.tdb, "test description", "1.0"
            )

            args = argparse.Namespace(
                db=str(db_path),
                typed="tdb",
                like=False,
                elem="",
                func="",
                phase="",
                param="",
                tdb="",
            )
            result = cmd_list(args)
            assert result == 0


class TestParserList:
    def test_list_command_required_args(self, parser):
        args = parser.parse_args(["list", "--typed", "elem"])
        assert args.command == "list"
        assert args.typed == "elem"
        assert args.db == ":memory:"
        assert args.like == False

    def test_list_command_with_all_options(self, parser):
        args = parser.parse_args(
            [
                "list",
                "--db",
                "test.db",
                "--typed",
                "phase",
                "--like",
                "--phase",
                "TEST_PHASE",
                "--tdb",
                "test",
            ]
        )
        assert args.command == "list"
        assert args.db == "test.db"
        assert args.typed == "phase"
        assert args.like == True
        assert args.phase == "TEST_PHASE"
        assert args.tdb == "test"

    def test_list_command_short_options(self, parser):
        args = parser.parse_args(["list", "-t", "elem", "-l"])
        assert args.command == "list"
        assert args.typed == "elem"
        assert args.like == True

    def test_list_command_missing_typed(self, parser):
        with pytest.raises(SystemExit):
            parser.parse_args(["list"])

    def test_parser_has_list_subcommand(self, parser):
        with pytest.raises(SystemExit):
            parser.parse_args(["list", "--help"])


class TestCmdDelete:
    def test_delete_element(self, test_tdb_file):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"

            tdb_mgr = TDBManager(ThermoDBI(str(db_path)))
            parsed = tdb_mgr.parse_tdb(test_tdb_file, "test")
            tdb_mgr.save_elements(parsed.elems)

            args = argparse.Namespace(
                db=str(db_path),
                typed="elem",
                cascade=False,
                elem=parsed.elems[0]["elem"],
                func="",
                phase="",
                param="",
                tdb="",
            )
            result = cmd_delete(args)
            assert result == 0

    def test_delete_tdb_with_cascade(self, test_tdb_file):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"

            tdb_mgr = TDBManager(ThermoDBI(str(db_path)))
            parsed = tdb_mgr.parse_tdb(test_tdb_file, "test")
            tdb_mgr.import_tdb(
                parsed.phases, parsed.params, parsed.tdb, "test description", "1.0"
            )

            args = argparse.Namespace(
                db=str(db_path),
                typed="tdb",
                cascade=True,
                elem="",
                func="",
                phase="",
                param="",
                tdb="test",
            )
            result = cmd_delete(args)
            assert result == 0

    def test_delete_phase(self, test_tdb_file):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"

            tdb_mgr = TDBManager(ThermoDBI(str(db_path)))
            parsed = tdb_mgr.parse_tdb(test_tdb_file, "test")
            tdb_mgr.import_tdb(
                parsed.phases, parsed.params, parsed.tdb, "test description", "1.0"
            )

            args = argparse.Namespace(
                db=str(db_path),
                typed="phase",
                cascade=True,
                elem="",
                func="",
                phase=parsed.phases[0]["phase"],
                param="",
                tdb="test",
            )
            result = cmd_delete(args)
            assert result == 0


class TestParserDelete:
    def test_delete_command_required_args(self, parser):
        args = parser.parse_args(["delete", "--typed", "elem", "--elem", "A"])
        assert args.command == "delete"
        assert args.typed == "elem"
        assert args.elem == "A"
        assert args.db == ":memory:"
        assert args.cascade == False

    def test_delete_command_with_all_options(self, parser):
        args = parser.parse_args(
            [
                "delete",
                "--db",
                "test.db",
                "--typed",
                "phase",
                "--cascade",
                "--phase",
                "TEST_PHASE",
                "--tdb",
                "test",
            ]
        )
        assert args.command == "delete"
        assert args.db == "test.db"
        assert args.typed == "phase"
        assert args.cascade == True
        assert args.phase == "TEST_PHASE"
        assert args.tdb == "test"

    def test_delete_command_short_options(self, parser):
        args = parser.parse_args(["delete", "-t", "elem", "-c"])
        assert args.command == "delete"
        assert args.typed == "elem"
        assert args.cascade == True

    def test_delete_command_missing_typed(self, parser):
        with pytest.raises(SystemExit):
            parser.parse_args(["delete"])

    def test_parser_has_delete_subcommand(self, parser):
        with pytest.raises(SystemExit):
            parser.parse_args(["delete", "--help"])


class TestCmdFit:
    @patch("src.cli.GTFitter")
    def test_fit_with_json_data_type(self, mock_fitter):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            data_dir = Path(__file__).parent / "fits-json"

            mock_instance = MagicMock()
            mock_instance.process_folders.return_value = []
            mock_instance.plot_fits.return_value = None
            mock_instance.fit2db.return_value = MagicMock(phases=[], params=[], tdb="test_fit")
            mock_fitter.return_value = mock_instance

            args = argparse.Namespace(
                db=str(db_path), data_dir=str(data_dir), data_type="json", tdb_name="test_fit"
            )
            result = cmd_fit(args)
            assert result == 0
            assert db_path.exists()

    @patch("src.cli.GTFitter")
    def test_fit_with_dat_data_type(self, mock_fitter):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            data_dir = Path(__file__).parent / "fits-json"

            mock_instance = MagicMock()
            mock_instance.process_folders.return_value = []
            mock_instance.plot_fits.return_value = None
            mock_instance.fit2db.return_value = MagicMock(phases=[], params=[], tdb="test_fit")
            mock_fitter.return_value = mock_instance

            args = argparse.Namespace(
                db=str(db_path), data_dir=str(data_dir), data_type="dat", tdb_name="test_fit"
            )
            result = cmd_fit(args)
            assert result == 0
            assert db_path.exists()

    @patch("src.cli.GTFitter")
    def test_fit_with_custom_tdb_name(self, mock_fitter):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            data_dir = Path(__file__).parent / "fits-json"

            mock_instance = MagicMock()
            mock_instance.process_folders.return_value = []
            mock_instance.plot_fits.return_value = None
            mock_instance.fit2db.return_value = MagicMock(
                phases=[], params=[], tdb="custom_tdb_name"
            )
            mock_fitter.return_value = mock_instance

            args = argparse.Namespace(
                db=str(db_path),
                data_dir=str(data_dir),
                data_type="json",
                tdb_name="custom_tdb_name",
            )
            result = cmd_fit(args)
            assert result == 0


class TestParserFit:
    def test_fit_command_required_args(self, parser):
        args = parser.parse_args(["fit", "--data-dir", "/path/to/data", "--tdb-name", "test"])
        assert args.command == "fit"
        assert args.data_dir == "/path/to/data"
        assert args.tdb_name == "test"
        assert args.db == ":memory:"
        assert args.data_type == "dat"

    def test_fit_command_with_all_options(self, parser):
        args = parser.parse_args(
            [
                "fit",
                "--db",
                "test.db",
                "--data-dir",
                "/path/to/data",
                "--data_type",
                "json",
                "--tdb-name",
                "test",
            ]
        )
        assert args.command == "fit"
        assert args.db == "test.db"
        assert args.data_dir == "/path/to/data"
        assert args.data_type == "json"
        assert args.tdb_name == "test"

    def test_fit_command_short_options(self, parser):
        args = parser.parse_args(["fit", "-d", "/path/to/data", "-t", "json", "-n", "test"])
        assert args.command == "fit"
        assert args.data_dir == "/path/to/data"
        assert args.data_type == "json"
        assert args.tdb_name == "test"

    def test_fit_command_missing_data_dir(self, parser):
        with pytest.raises(SystemExit):
            parser.parse_args(["fit", "--tdb-name", "test"])

    def test_fit_command_missing_tdb_name(self, parser):
        with pytest.raises(SystemExit):
            parser.parse_args(["fit", "--data-dir", "/path/to/data"])

    def test_parser_has_fit_subcommand(self, parser):
        with pytest.raises(SystemExit):
            parser.parse_args(["fit", "--help"])
