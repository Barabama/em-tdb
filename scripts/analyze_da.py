"""分析 compare_tdb_endmembers.py 的输出，识别 A 参数偏离很大的端元。"""
import re
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict

text = Path(sys.argv[1] if len(sys.argv) > 1 else "temp.txt").read_text(encoding="utf-8")

# 解析各 phase 的表格行
sections = re.split(r"Phase: (\w+)", text)
phase_data = {}

for i in range(1, len(sections) - 1, 2):
    phase = sections[i]
    body = sections[i + 1]
    rows = []
    for m in re.finditer(
        r"^\s{2}(\S+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s+([-+]?[\d.]+)\s+([-+]?[\d.]+)\s+([-+]?[\d.]+)",
        body,
        re.MULTILINE,
    ):
        label, maxd, rmsd, tatmax, a1, a2, da = m.groups()
        rows.append(
            {
                "label": label,
                "maxd": float(maxd),
                "rmsd": float(rmsd),
                "tatmax": int(tatmax),
                "a1": float(a1),
                "a2": float(a2),
                "da": float(da),
            }
        )
    if rows:
        phase_data[phase] = rows

# ── 分析 ──
for phase, rows in phase_data.items():
    sorted_by_da = sorted(rows, key=lambda r: abs(r["da"]), reverse=True)

    print(f"\n{'=' * 100}")
    print(f"  {phase}  — 按 |ΔA| 排序 (前 40)")
    print(f"{'=' * 100}")
    print(f"  {'end-member':<18s}  {'|ΔA|':>12s}  {'A₁':>12s}  {'A₂':>12s}  "
          f"{'max|ΔG|':>10s}  {'RMSD':>10s}  {'ratio A₂/A₁':>10s}")
    print(f"  {'─' * 18}  {'─' * 12}  {'─' * 12}  {'─' * 12}  "
          f"{'─' * 10}  {'─' * 10}  {'─' * 10}")

    for r in sorted_by_da[:40]:
        ratio = abs(r["a2"] / r["a1"]) if r["a1"] != 0 else float("inf")
        print(f"  {r['label']:<18s}  {abs(r['da']):>12.1f}  {r['a1']:>12.1f}  {r['a2']:>12.1f}  "
              f"{r['maxd']:>10.1f}  {r['rmsd']:>10.1f}  {ratio:>10.2f}")

    # ── 统计按元素的聚集偏差 ──
    print(f"\n  ── 汇总: 每组相同第一元素的平均 |ΔA|, max|ΔA|, count ──")
    by_first = defaultdict(list)
    for r in rows:
        first = r["label"].split(":")[0]
        by_first[first].append(r["da"])

    print(f"  {'elem':<6s}  {'count':>6s}  {'mean|ΔA|':>10s}  {'max|ΔA|':>10s}  {'median|ΔA|':>10s}")
    print(f"  {'─' * 6}  {'─' * 6}  {'─' * 10}  {'─' * 10}  {'─' * 10}")
    for elem in sorted(by_first.keys()):
        das = by_first[elem]
        das_abs = [abs(d) for d in das]
        print(f"  {elem:<6s}  {len(das):>6d}  {np.mean(das_abs):>10.0f}  {np.max(das_abs):>10.0f}  {np.median(das_abs):>10.0f}")

    # ── ΔA 直方图 ──
    das = np.array([r["da"] for r in rows])
    print(f"\n  ΔA 分布统计:")
    p = np.percentile(abs(das), [0, 25, 50, 75, 90, 95, 99, 100])
    for i, v in enumerate(p):
        print(f"    P{[0,25,50,75,90,95,99,100][i]:3d} = {v:.0f} J/mol")

    # ── 判断异常阈值 ──
    q75, q90, q95, q99 = np.percentile(abs(das), [75, 90, 95, 99])
    print(f"\n  异常判定:")
    print(f"    Q75 = {q75:.0f} J/mol (以上标记为偏高)")
    print(f"    Q90 = {q90:.0f} J/mol (以上标记为高)")
    print(f"    Q99 = {q99:.0f} J/mol (以上标记为极高)")

    # ── 打印偏离很大的 ──
    threshold = q90
    outliers = [r for r in rows if abs(r["da"]) > threshold]
    print(f"\n  |ΔA| > Q90 ({threshold:.0f} J/mol) 的端元 ({len(outliers)} 个):")
    for r in sorted(outliers, key=lambda x: abs(x["da"]), reverse=True):
        print(f"    {r['label']:<18s}  ΔA={r['da']:>+10.1f}  A₁={r['a1']:>12.1f}  A₂={r['a2']:>12.1f}")
