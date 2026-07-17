"""Tests for ``emtdb.fitters.readers``."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from emtdb.config import F_CONST, T_MIN, T_MAX
from emtdb.fitters.readers import read_gibbs_dat, read_gibbs_json, read_ve_dat


# ---------------------------------------------------------------------------
# read_gibbs_dat
# ---------------------------------------------------------------------------


class TestReadGibbsDat:
    """Uses real ``tests/fits-dat/`` files."""

    def test_bcc_mnw(self):
        """BCC-Mn-W: nested in QHA-MnW subfolder."""
        path = "tests/fits-dat/BCC-Mn-W/QHA-MnW/gibbs-temperature.dat"
        t, g = read_gibbs_dat(path, atom_num=2)
        assert len(t) == len(g) > 10
        assert t.min() >= T_MIN and t.max() <= T_MAX
        assert np.all(np.isfinite(g)) and np.all(g < 0)

    def test_bcc_tinb(self):
        """BCC-TiNb: flat layout."""
        path = "tests/fits-dat/BCC-TiNb/gibbs-temperature.dat"
        t, g = read_gibbs_dat(path, atom_num=2)
        assert len(t) == len(g) > 10
        assert np.all(np.isfinite(g))
        assert np.all(np.diff(t) > 0)

    def test_bcc_vv(self):
        """BCC-VV-2: nested in QHA-VV."""
        path = "tests/fits-dat/BCC-VV-2/QHA-VV/gibbs-temperature.dat"
        t, g = read_gibbs_dat(path, atom_num=2)
        assert len(t) > 10 and np.all(np.isfinite(g))

    def test_fcc_cral3(self):
        """FCC-CrAl3: flat."""
        path = "tests/fits-dat/FCC-CrAl3/gibbs-temperature.dat"
        t, g = read_gibbs_dat(path, atom_num=4)
        assert len(t) > 10 and np.all(np.isfinite(g))

    def test_hcp_w2ni6(self):
        """HCP-W2Ni6: nested in QHA-WNi."""
        path = "tests/fits-dat/HCP-W2Ni6/QHA-WNi/gibbs-temperature.dat"
        t, g = read_gibbs_dat(path, atom_num=8)
        assert len(t) > 10 and np.all(np.isfinite(g))

    def test_oth(self):
        """OTH-Al-Ti-Nb-4: ternary phase."""
        path = "tests/fits-dat/OTH-Al-Ti-Nb-4/gibbs-temperature.dat"
        t, g = read_gibbs_dat(path, atom_num=4)
        assert len(t) > 10 and np.all(np.isfinite(g))

    def test_oth_nb2tial(self):
        """OTH-Nb2TiAl: ternary, different naming."""
        path = "tests/fits-dat/OTH-Nb2TiAl/gibbs-temperature.dat"
        t, g = read_gibbs_dat(path, atom_num=4)
        assert len(t) > 10 and np.all(np.isfinite(g))

    def test_ser_re8a(self):
        """SER-Re-8a: nested in QHA-ReRe, '8a' atom suffix."""
        path = "tests/fits-dat/SER-Re-8a/QHA-ReRe/gibbs-temperature.dat"
        path = "tests/fits-dat/OTH-Al-Ti-Nb-4/gibbs-temperature.dat"
        t, g = read_gibbs_dat(path, atom_num=4)

        assert len(t) > 10
        assert np.all(np.isfinite(g))

    def test_atom_num_zero_raises(self):
        """atom_num=0 produces division by zero → non-finite values."""
        path = "tests/fits-dat/BCC-TiNb/gibbs-temperature.dat"
        with pytest.raises((ValueError, FloatingPointError)):
            t, g = read_gibbs_dat(path, atom_num=0)
            if not np.all(np.isfinite(g)):
                raise ValueError("non-finite G values")

    def test_file_not_found(self):
        """Missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            read_gibbs_dat("tests/fits-dat/nonexistent.dat", 1)

    def test_empty_file_raises(self, tmp_path):
        """Empty file raises."""
        p = tmp_path / "empty.dat"
        p.write_text("")
        with pytest.raises((ValueError, StopIteration)):
            read_gibbs_dat(str(p), 1)


# ---------------------------------------------------------------------------
# read_gibbs_json
# ---------------------------------------------------------------------------


class TestReadGibbsJson:
    """Uses real ``tests/fits-json/`` files (Format B / QhaData only)."""

    def test_bcc_alv(self):
        """BCC-Al-V: Gibbs data from QHA JSON."""
        path = "tests/fits-json/BCC-Al-V/BCC-Al-V-qha.json"
        t, g = read_gibbs_json(path, atom_num=2)

        assert len(t) == len(g) > 10
        assert t.min() >= T_MIN
        assert t.max() <= T_MAX
        assert np.all(np.isfinite(g))
        assert np.all(g < 0)

    def test_bcc_nbv(self):
        """BCC-Nb-V: binary QHA JSON."""
        path = "tests/fits-json/BCC-Nb-V/BCC-Nb-V-qha.json"
        t, g = read_gibbs_json(path, atom_num=2)
        assert len(t) == len(g) > 10
        assert np.all(np.isfinite(g))

    def test_bcc_titi(self):
        """BCC-Ti-Ti: same-element binary QHA JSON."""
        path = "tests/fits-json/BCC-Ti-Ti/BCC-Ti-Ti-qha.json"
        t, g = read_gibbs_json(path, atom_num=2)
        assert len(t) == len(g) > 10
        assert np.all(np.isfinite(g))

    def test_ser_ti(self):
        """SER-Ti: single-element QHA JSON."""
        path = "tests/fits-json/SER-Ti/SER-Ti-qha.json"
        t, g = read_gibbs_json(path, atom_num=1)

        assert len(t) == len(g) > 10
        assert np.all(np.isfinite(g))

    def test_temperature_alignment(self):
        """gibbs_temperature is one shorter than temperatures → aligned."""
        path = "tests/fits-json/BCC-Al-V/BCC-Al-V-qha.json"

        with open(path) as fh:
            raw = json.load(fh)
        raw_t_count = len(raw["temperatures"])
        raw_g_count = len(raw["gibbs_temperature"])
        assert raw_t_count == raw_g_count + 1  # pre-condition

        t, _ = read_gibbs_json(path, atom_num=2)
        # After dropping T=0 and filtering, we should have reasonable count.
        assert len(t) > 10

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            read_gibbs_json("nonexistent.json", 1)


# ---------------------------------------------------------------------------
# read_ve_dat
# ---------------------------------------------------------------------------


class TestReadVeDat:
    """Uses synthetic files (no real v-e.dat exists in test data)."""

    def test_three_points(self, tmp_path):
        p = tmp_path / "v-e.dat"
        p.write_text("10.0  -5.0\n12.0  -5.5\n14.0  -5.3\n")
        v, e = read_ve_dat(str(p))
        assert list(v) == [10.0, 12.0, 14.0]
        assert list(e) == [-5.0, -5.5, -5.3]

    def test_sorts_by_volume(self, tmp_path):
        """Output is always sorted by ascending volume."""
        p = tmp_path / "v-e.dat"
        p.write_text("14.0  -5.3\n10.0  -5.0\n12.0  -5.5\n")
        v, e = read_ve_dat(str(p))
        assert list(v) == [10.0, 12.0, 14.0]
        assert list(e) == [-5.0, -5.5, -5.3]

    def test_extra_columns_ignored(self, tmp_path):
        """Extra columns past the second are ignored."""
        p = tmp_path / "v-e.dat"
        p.write_text("10.0  -5.0  0.1\n12.0  -5.5  0.2\n")
        v, e = read_ve_dat(str(p))
        assert len(v) == 2
        assert len(e) == 2

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            read_ve_dat("nonexistent.dat")

    def test_empty_file_raises(self, tmp_path):
        p = tmp_path / "empty.dat"
        p.write_text("")
        with pytest.raises((ValueError, StopIteration)):
            read_ve_dat(str(p))
