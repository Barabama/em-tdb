"""Tests for ``emtdb.fitters.parsers``."""

from __future__ import annotations

import pytest

from emtdb.fitters.parsers import (
    gen_exchange,
    normalize_metrics,
    parse_folder_name,
)


# ---------------------------------------------------------------------------
# parse_folder_name
# ---------------------------------------------------------------------------


class TestParseFolderName:
    """Folder names taken from the real ``tests/fits-dat/`` and ``tests/fits-json/`` directories."""

    @pytest.mark.parametrize(
        "folder, exp_phase, exp_elems, exp_atom",
        [
            # fits-dat — defaults from PHASE_METRICS sum
            ("BCC-Mn-W", "BCC", ["MN", "W"], 2),
            ("BCC-TiNb", "BCC", ["TI", "NB"], 2),
            ("BCC-VV-2", "BCC", ["V", "V"], 2),
            ("FCC-CrAl3", "FCC", ["CR", "AL"], 4),
            ("HCP-W2Ni6", "HCP", ["W", "NI"], 8),
            ("OTH-Al-Ti-Nb-4", "OTH", ["AL", "TI", "NB"], 4),
            ("OTH-Nb2TiAl", "OTH", ["NB", "TI", "AL"], 1),
            # fits-json
            ("BCC-Al-V", "BCC", ["AL", "V"], 2),
            ("BCC-Nb-V", "BCC", ["NB", "V"], 2),
            ("BCC-Ti-Ti", "BCC", ["TI", "TI"], 2),
            ("SER-Ti", "SER", ["TI"], 1),
            # Edge: single-letter, concatenated, "8a" suffix
            ("BCC-VV-2", "BCC", ["V", "V"], 2),
            ("SER-V-1", "SER", ["V"], 1),
            ("BCC-FeCr-2", "BCC", ["FE", "CR"], 2),
            ("SER-Re-8a", "SER", ["RE"], 8),
        ],
    )
    def test_valid(self, folder, exp_phase, exp_elems, exp_atom):
        p, e, a = parse_folder_name(folder)
        assert p == exp_phase
        assert e == exp_elems
        assert a == exp_atom

    @pytest.mark.parametrize(
        "folder",
        [
            "",
            "BCC",
            "BCC-XX-YY-2",
            "OTH--2",
        ],
    )
    def test_invalid(self, folder):
        assert parse_folder_name(folder) is None

    def test_case_insensitive_phase(self):
        """Phase token is case-insensitive."""
        p, e, a = parse_folder_name("bcc-ti-nb-4")
        assert p == "BCC"
        assert e == ["TI", "NB"]
        assert a == 4

    def test_mixed_case_elements(self):
        """Element tokens are normalised to upper case."""
        p, e, a = parse_folder_name("BCC-FeCr-2")
        assert e == ["FE", "CR"]

    def test_no_atom_suffix_uses_phase_default(self):
        """Folder without atom_count defaults to PHASE_METRICS sum."""
        p, e, a = parse_folder_name("FCC-Fe-Mn")
        assert a == 4  # FCC metrics (1,3) sum = 4
        p, e, a = parse_folder_name("BCC-Fe-Mn")
        assert a == 2  # BCC metrics (1,1) sum = 2
        p, e, a = parse_folder_name("SER-Fe")
        assert a == 1  # SER metrics (1,) sum = 1


# ---------------------------------------------------------------------------
# normalize_metrics
# ---------------------------------------------------------------------------


class TestNormalizeMetrics:
    @pytest.mark.parametrize(
        "raw, exp",
        [
            ([1.0, 1.0], [0.5, 0.5]),
            ([1.0, 3.0], [0.25, 0.75]),
            ([2.0, 6.0], [0.25, 0.75]),
            ([1.0], [1.0]),
            ([10.0, 10.0], [0.5, 0.5]),
        ],
    )
    def test_normalize(self, raw, exp):
        result = normalize_metrics(raw)
        assert len(result) == len(exp)
        for a, b in zip(result, exp):
            assert abs(a - b) < 1e-12

    def test_zero_sum_returns_copy(self):
        """Zero-sum input returns the same list (unchanged)."""
        raw = [0.0, 0.0]
        result = normalize_metrics(raw)
        assert result == raw
        assert result is not raw  # different object


# ---------------------------------------------------------------------------
# gen_exchange
# ---------------------------------------------------------------------------


class TestGenExchange:
    def test_binary_equal(self):
        """Equal metrics → swapped pair."""
        e, m = gen_exchange(["FE", "CR"], [0.5, 0.5])
        assert e == ["CR", "FE"]
        assert m == [0.5, 0.5]

    def test_same_element(self):
        """Same element → None."""
        assert gen_exchange(["AL", "AL"], [0.5, 0.5]) is None

    def test_unequal_metrics(self):
        """Different metrics → None."""
        assert gen_exchange(["FE", "CR"], [0.25, 0.75]) is None

    def test_single_element(self):
        """Single element → None."""
        assert gen_exchange(["FE"], [1.0]) is None

    def test_three_elements(self):
        """Three elements → None."""
        assert gen_exchange(["A", "B", "C"], [0.33, 0.33, 0.34]) is None
