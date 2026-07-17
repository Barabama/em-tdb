"""
Folder-name parsers and metric helper functions.

Logic matches ``demo/gibbsfit.py`` and ``demo/etotfit.py`` verbatim.
"""

from __future__ import annotations

import re

from emtdb.config import PHASE_METRICS

# Regex for splitting concatenated element symbols (e.g. "TiNb" → ["Ti","Nb"]).
_ELEM_TOKEN_RE = re.compile(r"([A-Z][a-z]?)(\d*)")


def parse_folder_name(name: str) -> tuple[str, list[str], int] | None:
    """Parse a subfolder name into ``(phase, elements, atom_num)``.

    ``atom_num`` defaults to the sum of ``PHASE_METRICS[phase]``
    when no explicit suffix is present (e.g. ``BCC-Mn-W`` → 2).

    Supported naming patterns:
        ``BCC-Mn-W``          → ``("BCC", ["MN","W"], 2)``
        ``BCC-TiNb``          → ``("BCC", ["TI","NB"], 2)``
        ``BCC-Al-Al-2``       → ``("BCC", ["AL","AL"], 2)``
        ``BCC-TiNb-2``        → ``("BCC", ["TI","NB"], 2)``
        ``BCC-Ta2-Al-8``      → ``("BCC", ["TA","AL"], 8)``
        ``SER-Nb-2atoms``     → ``("SER", ["NB"], 2)``
        ``SER-Re-8a``         → ``("SER", ["RE"], 8)``
        ``OTH-Al-Ti-Nb-4``    → ``("OTH", ["AL","TI","NB"], 4)``
        ``FCC-Fe-Mn``         → ``("FCC", ["FE","MN"], 4)``
        ``HCP-W2Ni6``         → ``("HCP", ["W","NI"], 8)``

    Returns ``None`` on parse failure (unknown element, empty, …).
    """
    parts = name.split("-")
    if len(parts) < 2:
        return None

    phase = parts[0].upper()

    # Walk from the right to find an optional atom_num suffix.
    atom_num: int | None = None
    elem_end = len(parts)
    if len(parts) > 1:
        last = parts[-1]
        parsed_atom = _parse_atom_num(last)
        if parsed_atom is not None:
            atom_num = parsed_atom
            elem_end = len(parts) - 1

    # Fallback: use sum of phase metrics as default atom count.
    if atom_num is None:
        raw = PHASE_METRICS.get(phase, (1,))
        atom_num = int(sum(raw))

    # Collect element tokens between phase and optional atom_num.
    raw_tokens = parts[1:elem_end]
    elements: list[str] = []
    for tok in raw_tokens:
        upper = tok.upper()
        if _is_element(upper):
            elements.append(upper)
        else:
            # Try to split concatenated token (e.g. "TiNb" → ["Ti", "Nb"]).
            split = _split_elements(tok)
            if split is None:
                return None
            elements.extend(split)

    if not elements:
        return None

    return (phase, elements, atom_num)


def _parse_atom_num(s: str) -> int | None:
    """Try to parse an atom-count suffix such as ``"2"``, ``"2atoms"``, ``"8a"``.

    Supports ``atoms``, ``atom``, ``ATOMS``, ``ATOM`` suffixes, single trailing
    letter (e.g. ``"a"`` in ``"8a"``), or bare number.
    """
    stripped = s
    for suffix in ("atoms", "atom", "ATOMS", "ATOM"):
        if stripped.endswith(suffix):
            stripped = stripped[: -len(suffix)]
            break
    else:
        # Single trailing lowercase abbreviation, e.g. "8a" → "8".
        if len(stripped) > 1 and stripped[-1].isalpha():
            stripped = stripped[:-1]
    try:
        return int(stripped)
    except ValueError:
        return None


def _split_elements(s: str) -> list[str] | None:
    """Split a concatenated element string (e.g. ``"TiNb"``, ``"AlNb3"``).

    Each element is one uppercase letter + optional one lowercase letter.
    Digits immediately after an element token are skipped (they encode
    stoichiometric ratios, not used here).

    Returns ``None`` if any unrecognised element is encountered.
    """
    chars = list(s)
    result: list[str] = []
    i = 0
    while i < len(chars):
        if not chars[i].isascii() or not chars[i].isupper():
            return None

        elem = chars[i]
        i += 1

        # Optional one lowercase letter (e.g. "Ti", "Nb").
        if i < len(chars) and chars[i].isascii() and chars[i].islower():
            elem += chars[i]
            i += 1

        # Skip trailing digits (ratio markers).
        while i < len(chars) and chars[i].isascii() and chars[i].isdigit():
            i += 1

        upper = elem.upper()
        if not _is_element(upper):
            return None
        result.append(upper)

    return result


# ── Known element symbols (uppercase, from the periodic table). ──────────

_ELEMENTS: frozenset[str] = frozenset({
    "AC", "AG", "AL", "AM", "AR", "AS", "AT", "AU",
    "B", "BA", "BE", "BH", "BI", "BK", "BR",
    "C", "CA", "CD", "CE", "CF", "CL", "CM", "CN", "CO", "CR", "CS", "CU",
    "DB", "DS", "DY",
    "ER", "ES", "EU",
    "F", "FE", "FL", "FM", "FR",
    "GA", "GD", "GE",
    "H", "HE", "HF", "HG", "HO", "HS",
    "I", "IN", "IR",
    "K", "KR",
    "LA", "LI", "LR", "LU", "LV",
    "MC", "MD", "MG", "MN", "MO", "MT",
    "N", "NA", "NB", "ND", "NE", "NH", "NI", "NO", "NP",
    "O", "OG", "OS",
    "P", "PA", "PB", "PD", "PM", "PO", "PR", "PT", "PU",
    "RA", "RB", "RE", "RF", "RG", "RH", "RN", "RU",
    "S", "SB", "SC", "SE", "SG", "SI", "SM", "SN", "SR",
    "TA", "TB", "TC", "TE", "TH", "TI", "TL", "TM", "TS",
    "U",
    "V", "VA",
    "W",
    "XE",
    "Y", "YB",
    "ZN", "ZR",
})


def _is_element(s: str) -> bool:
    """Check whether *s* is a known element symbol (case-sensitive, upper)."""
    return s in _ELEMENTS


# ── Metric helpers ──────────────────────────────────────────────────────


def normalize_metrics(metrics: list[float]) -> list[float]:
    """Normalise stoichiometric ratios so that they sum to 1.0.

    Examples
    --------
    ``[1, 1]`` → ``[0.5, 0.5]``
    ``[1, 3]`` → ``[0.25, 0.75]``
    ``[1]``    → ``[1.0]``
    """
    total = sum(metrics)
    if total == 0.0:
        return metrics[:]
    return [m / total for m in metrics]


def gen_exchange(
    elements: list[str], metrics: list[float]
) -> tuple[list[str], list[float]] | None:
    """Generate a BCC symmetric-exchange pair ``(swapped_elements, swapped_metrics)``.

    Returns ``(swapped_elements, swapped_metrics)`` when:
    - exactly two *different* elements are present, **and**
    - both metrics are equal (within ``1e-12`` tolerance).

    Returns ``None`` otherwise.
    """
    if (
        len(elements) != 2
        or elements[0] == elements[1]
        or len(metrics) != 2
        or abs(metrics[0] - metrics[1]) > 1e-12
    ):
        return None
    return ([elements[1], elements[0]], [metrics[1], metrics[0]])
