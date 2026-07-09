"""
Merge specific end-member PARAMETER lines from 0516 TDB into 16symbols TDB.

Usage: .conda/python.exe scripts/merge_tdb.py
"""

import re
from pathlib import Path

SRC = Path("ref_tdb/20250516-bcc+fcc+hcp_by-hamid-cxy-wubo.TDB")
DST = Path("ref_tdb/16symbols-cxy-hmd.tdb")
OUT = Path("ref_tdb/16symbols-cxy-hmd-merged.TDB")

REPLACE = [
    ("BCC", "CO", "CO"), ("BCC", "CO", "TI"), ("BCC", "TI", "CO"),
    ("BCC", "HF", "NI"), ("BCC", "NI", "HF"),
    ("BCC", "TI", "V"), ("BCC", "V", "TI"),
    ("FCC", "AL", "MO"), ("FCC", "AL", "V"),
    ("FCC", "CR", "CU"), ("FCC", "CR", "HF"), ("FCC", "CU", "AL"),
    ("FCC", "MN", "NB"),
    ("FCC", "NI", "AL"), ("FCC", "NI", "FE"),
    ("FCC", "TI", "CU"), ("FCC", "V", "MN"),
]


def read_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.readlines()


def find_param_block(lines, phase, e1, e2):
    """Find the index range of a PARAMETER G(phase,e1:e2;0) block.

    Returns (start_idx, end_idx) where end_idx is exclusive (after ! line).
    The block includes the PARAMETER line + possibly indented continuation lines.
    """
    tag = f"PARAMETER G({phase},{e1}:{e2};0)"
    for i, line in enumerate(lines):
        if tag.upper() in line.upper():
            start = i
            # Find the ! that ends this block
            end = start
            while end < len(lines):
                if "!" in lines[end]:
                    end += 1
                    break
                end += 1
            return (start, end)
    return None


def main():
    src_lines = read_lines(SRC)
    dst_lines = read_lines(DST)

    src_text = "".join(src_lines)
    dst_text = "".join(dst_lines)

    # For each target, extract the multi-line PARAMETER block from source
    # and replace it in dest
    for phase, e1, e2 in REPLACE:
        tag = f"PARAMETER G({phase},{e1}:{e2};0)"

        # Extract from source
        m_src = re.search(
            rf"PARAMETER\s+G\({phase},{e1}:{e2};0\).*?(?:\n\s+.*?)*?!(?:\n|$)",
            src_text,
            re.IGNORECASE | re.DOTALL,
        )
        if not m_src:
            print(f"⚠  NOT FOUND in 0516: {tag}")
            continue

        new_block = m_src.group(0)

        # Find in dest and replace
        m_dst = re.search(
            rf"PARAMETER\s+G\({phase},{e1}:{e2};0\).*?(?:\n\s+.*?)*?!(?:\n|$)",
            dst_text,
            re.IGNORECASE | re.DOTALL,
        )
        if not m_dst:
            print(f"⚠  NOT FOUND in 16symbols: {tag}")
            continue

        dst_text = dst_text[: m_dst.start()] + new_block + dst_text[m_dst.end() :]
        print(f"✓  Replaced: {tag}")

    OUT.write_text(dst_text, encoding="utf-8")
    print(f"\n✅  Written to {OUT}")

    # Validate: count params in output
    for phase, e1, e2 in REPLACE:
        tag = f"PARAMETER G({phase},{e1}:{e2};0)"
        if tag.upper() in dst_text.upper():
            print(f"  ✓  Verified: {tag}")
        else:
            print(f"  ✗  MISSING: {tag}")


if __name__ == "__main__":
    main()
