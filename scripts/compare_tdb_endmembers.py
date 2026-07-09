"""Compare end-member excess Gibbs values between two TDBs.

Extracts the raw PARAMETER polynomial for each common end-member,
strips all XXX# function references, and performs full-curve
comparison (max|Δ|, RMSD, Wald statistic) across the T range.

Usage: .conda/python.exe scripts/compare_tdb_endmembers.py
"""
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2

TDB1 = Path("ref_tdb/20250516-bcc+fcc+hcp_by-hamid-cxy-wubo.TDB")
TDB2 = Path("ref_tdb/16symbols-cxy-hmd.tdb")

T_RANGE = np.linspace(100, 2900, 1000)
T_EVAL = np.linspace(100, 2900, 200)  # for per-end-member plots
N_COLS = 4


# ── Design matrix & Wald test ──

def basis(T):
    """SGTE polynomial basis: A + B*T + C*T*LN(T) + D*T**2 + E*T**3 + F/T."""
    return np.column_stack([
        np.ones_like(T), T, T * np.log(T), T**2, T**3, 1.0 / T,
    ])


def compare_fits(theta_A, theta_B, cov_A=None, cov_B=None):
    """Compare two coefficient vectors with full-curve statistics.

    Parameters
    ----------
    theta_A, theta_B : (6,) arrays — [A, B, C, D, E, F]
    cov_A, cov_B : (6,6) arrays or None — parameter covariances

    Returns dict with keys:
        max_dev, rmsd, t_at_max, wald_statistic, p_value
    """
    X = basis(T_RANGE)
    diff_curve = X @ (theta_A - theta_B)
    max_dev = np.max(np.abs(diff_curve))
    t_at_max = T_RANGE[np.argmax(np.abs(diff_curve))]
    rmsd = np.sqrt(np.mean(diff_curve ** 2))

    result = {
        "max_dev": max_dev,
        "rmsd": rmsd,
        "t_at_max": t_at_max,
    }

    # Wald test (only if covariances available)
    if cov_A is not None and cov_B is not None:
        delta = theta_A - theta_B
        cov_sum = cov_A + cov_B
        try:
            W = delta @ np.linalg.solve(cov_sum, delta)
        except np.linalg.LinAlgError:
            W = delta @ np.linalg.pinv(cov_sum) @ delta
        p_value = 1.0 - chi2.cdf(W, df=6)
        result["wald_statistic"] = W
        result["p_value"] = p_value

    return result


# ── TDB parsing helpers ──

def _norm_expr(s: str) -> str:
    """Normalise TDB expression for sympy."""
    s = re.sub(r'(?<!\w)LN(?!\w)', 'ln', s, flags=re.IGNORECASE)
    s = re.sub(r'(?<!\w)EXP(?!\w)', 'exp', s, flags=re.IGNORECASE)
    s = re.sub(r'(?<!\w)LOG(?!\w)', 'log', s, flags=re.IGNORECASE)
    return s


def strip_function_refs(expr: str) -> str:
    """Remove all XXX# function references."""
    s = re.sub(r'[+-]?\s*[\d.]*\s*\*?\s*[A-Za-z0-9_]+#', '', expr)
    s = re.sub(r'(?<=[\d*])\s+(?=-)', '', s)
    return s.strip().strip('+')


def extract_param_raw(tdb_path: Path, phase: str, components: str) -> str | None:
    """Read TDB; return PARAMETER expression for G(phase,components;0)."""
    pattern = re.compile(
        rf"PARAMETER\s+G\({re.escape(phase)},{re.escape(components)};\d+\)"
        r"\s+\S+\s+(.+?);\s*\S+",
        re.IGNORECASE,
    )
    text = tdb_path.read_text(encoding="utf-8", errors="ignore")
    text = re.sub(r'\$.*', '', text)
    text = re.sub(r'\s+', ' ', text)
    m = pattern.search(text)
    return m.group(1).strip() if m else None


def expr_to_coeffs(expr_str: str) -> np.ndarray | None:
    """Extract SGTE coefficients [A,B,C,D,E,F] from a polynomial expression.

    Fits the polynomial at 6 temperatures by solving the linear system.
    """
    import sympy
    t = sympy.Symbol('T')
    s = _norm_expr(expr_str)
    # Clean whitespace
    s = re.sub(r'(?<=[\d*Ee+])\s+(?=-)', '', s)
    s = re.sub(r'(?<=[\dEe])\s+(?=[+\-])', '', s)
    s = re.sub(r'(?<=\d)\s+(?=\d)', '', s)
    try:
        expr = sympy.sympify(s)
        f = sympy.lambdify(t, expr, modules='numpy')
    except Exception:
        return None

    # Solve X @ theta = y at 6 well-spaced temperatures
    T_fit = np.linspace(200, 2800, 6)
    X = basis(T_fit)
    y = f(T_fit)
    try:
        theta, *_ = np.linalg.lstsq(X, y, rcond=None)
        return theta
    except Exception:
        return None


def coeffs_to_key(theta: np.ndarray) -> str:
    """Format coefficient vector as human-readable key."""
    return (f"A={theta[0]:+.1f} B={theta[1]:+.4f} C={theta[2]:+.4f} "
            f"D={theta[3]:+.2e} E={theta[4]:+.2e} F={theta[5]:+.1f}")


# ── Main ──

if __name__ == "__main__":
    all_results = {}  # phase → list of (label, stats_dict)

    for phase in ["BCC", "FCC"]:
        print(f"\n{'=' * 120}")
        print(f"  Phase: {phase}")
        print(f"{'=' * 120}")

        # Discover common keys
        def find_keys(path, ph):
            keys = set()
            text = path.read_text(encoding="utf-8", errors="ignore")
            for m in re.finditer(
                rf"PARAMETER\s+G\({ph},([A-Za-z0-9:,]+);\d+\)",
                text, re.IGNORECASE,
            ):
                keys.add(m.group(1))
            return sorted(keys)

        keys1 = find_keys(TDB1, phase)
        keys2 = find_keys(TDB2, phase)
        common_keys = sorted(set(keys1) & set(keys2))
        print(f"  共同端元: {len(common_keys)}")

        # Header
        print(f"\n  {'End-member':<18s}  {'max|Δ|(J/mol)':>14s}  {'RMSD(J/mol)':>12s}  "
              f"{'T@max(K)':>8s}  {'A₁':>12s}  {'A₂':>12s}  {'ΔA':>10s}")
        print(f"  {'─' * 18}  {'─' * 14}  {'─' * 12}  {'─' * 8}  "
              f"{'─' * 12}  {'─' * 12}  {'─' * 10}")

        phase_results = []
        y1_curves = {}
        y2_curves = {}
        max_devs = []

        for key in common_keys:
            raw1 = extract_param_raw(TDB1, phase, key)
            raw2 = extract_param_raw(TDB2, phase, key)
            ex1 = strip_function_refs(raw1) if raw1 else None
            ex2 = strip_function_refs(raw2) if raw2 else None

            th1 = expr_to_coeffs(ex1) if ex1 else None
            th2 = expr_to_coeffs(ex2) if ex2 else None

            label = key[:18]
            if th1 is None or th2 is None:
                print(f"  {label:<18s}  {'PARSE ERR':>14s}")
                # Still evaluate pointwise for plotting
                from scripts.etot_fit import evaluate_excess
                y1 = evaluate_excess(ex1, T_EVAL) if ex1 else np.full_like(T_EVAL, np.nan)
                y2 = evaluate_excess(ex2, T_EVAL) if ex2 else np.full_like(T_EVAL, np.nan)
                y1_curves[key] = y1
                y2_curves[key] = y2
                continue

            stats = compare_fits(th1, th2)
            dA = th2[0] - th1[0]

            print(f"  {label:<18s}  {stats['max_dev']:>14.1f}  {stats['rmsd']:>12.1f}  "
                  f"{stats['t_at_max']:>8.0f}  "
                  f"{th1[0]:>12.1f}  {th2[0]:>12.1f}  {dA:>10.1f}")

            # Evaluate full curves for plotting via basis
            X_eval = basis(T_EVAL)
            y1 = X_eval @ th1
            y2 = X_eval @ th2
            y1_curves[key] = y1
            y2_curves[key] = y2

            phase_results.append((label, stats, th1, th2))
            max_devs.append(stats["max_dev"])

        all_results[phase] = phase_results

        # ── Global statistics ──
        if max_devs:
            print(f"\n  全局统计 ({len(max_devs)} 端元):")
            print(f"    mean max|Δ| = {np.mean(max_devs):.0f} J/mol")
            print(f"    median max|Δ| = {np.median(max_devs):.0f} J/mol")
            print(f"    max  max|Δ| = {np.max(max_devs):.0f} J/mol")
            print(f"    min  max|Δ| = {np.min(max_devs):.0f} J/mol")
            p75 = np.percentile(max_devs, 75)
            p90 = np.percentile(max_devs, 90)
            print(f"    P75         = {p75:.0f} J/mol")
            print(f"    P90         = {p90:.0f} J/mol")

        common_keys_plot = [k for k in common_keys if k in y1_curves]

        # ── Per-end-member curve plots ──
        n = len(common_keys_plot)
        if n > 0:
            n_rows = int(np.ceil(n / N_COLS))
            fig, axes = plt.subplots(
                n_rows, N_COLS,
                figsize=(5 * N_COLS, 4 * n_rows),
                squeeze=False,
            )
            fig.suptitle(
                f"End-member excess G(T) — {phase}\n"
                f"{TDB1.name} (blue) vs {TDB2.name} (red)",
                fontsize=12,
            )
            for idx, key in enumerate(common_keys_plot):
                ax = axes[idx // N_COLS][idx % N_COLS]
                label = key[:20]
                y1 = y1_curves[key]
                y2 = y2_curves[key]
                if not np.isnan(y1).all() or not np.isnan(y2).all():
                    ax.plot(T_EVAL, y1, "b-", lw=1.2)
                    ax.plot(T_EVAL, y2, "r--", lw=1.2)
                    # Difference curve faintly
                    diff = y2 - y1
                    ax.fill_between(T_EVAL, 0, diff, alpha=0.15, color='gray',
                                    label=f"Δ max={np.max(np.abs(diff)):.0f}")
                ax.set_title(label, fontsize=7)
                ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, 3))
                ax.tick_params(labelsize=5)
                ax.legend(fontsize=4, loc="best")

            for idx in range(n, n_rows * N_COLS):
                axes[idx // N_COLS][idx % N_COLS].axis("off")

            plt.tight_layout()
            out_path = f"compare_{phase}_endmembers.png"
            plt.savefig(out_path, dpi=150)
            print(f"  [图] 已保存: {out_path}")
            plt.close(fig)

        # ── Histogram of max|Δ| ──
        if max_devs:
            fig2, ax2 = plt.subplots(figsize=(8, 4))
            ax2.hist(max_devs, bins=30, edgecolor='k', alpha=0.7)
            ax2.axvline(np.mean(max_devs), color='r', ls='--',
                        label=f"mean={np.mean(max_devs):.0f}")
            ax2.axvline(np.median(max_devs), color='g', ls='--',
                        label=f"median={np.median(max_devs):.0f}")
            ax2.set_xlabel("max|ΔG| (J/mol)")
            ax2.set_ylabel("Count")
            ax2.set_title(f"Distribution of max|Δ| — {phase} ({len(max_devs)} end-members)")
            ax2.legend()
            ax2.ticklabel_format(axis='x', style='sci', scilimits=(-3, 3))
            fig2.tight_layout()
            hist_path = f"compare_{phase}_histogram.png"
            fig2.savefig(hist_path, dpi=150)
            print(f"  [图] 已保存: {hist_path}")
            plt.close(fig2)

        # ── Ranked bar chart ──
        if phase_results:
            sorted_results = sorted(phase_results, key=lambda r: r[1]["max_dev"], reverse=True)
            top_n = min(20, len(sorted_results))
            labels_top = [r[0] for r in sorted_results[:top_n]]
            maxdev_top = [r[1]["max_dev"] for r in sorted_results[:top_n]]

            fig3, ax3 = plt.subplots(figsize=(10, 5))
            colors = plt.cm.Reds(np.linspace(0.3, 0.9, top_n))
            ax3.barh(range(top_n), maxdev_top, color=colors[::-1], edgecolor='k')
            ax3.set_yticks(range(top_n))
            ax3.set_yticklabels(labels_top, fontsize=6)
            ax3.set_xlabel("max|ΔG| (J/mol)")
            ax3.set_title(f"Top-{top_n} deviating end-members — {phase}")
            ax3.invert_yaxis()
            fig3.tight_layout()
            bar_path = f"compare_{phase}_top_deviations.png"
            fig3.savefig(bar_path, dpi=150)
            print(f"  [图] 已保存: {bar_path}")
            plt.close(fig3)
