"""Compute paired-t + Bonferroni-corrected + BCa-bootstrap statistics for the
GDSF vs LRU comparison on the 30-seed benchmark JSON.

Reads   results/benchmark_results_<timestamp>.json  (or --input)
Writes  results/stats_<timestamp>.json              (or --output)

Every claim in docs/report-draft.md that carries a paired-t or a 95% CI is
required to resolve to a key in this file.

Methodology:

- For each workload, we form paired samples across the 4 cache sizes and 30
  seeds (n = 120 pairs per workload), pairing GDSF and LRU by (cache_size,
  seed). We compute the paired difference d_i = CWHR_GDSF(i) - CWHR_LRU(i).
- Paired t-test:  t = mean(d) / (std(d, ddof=1) / sqrt(n)),
                  p (two-sided) from Student's t with df = n - 1.
- Bonferroni correction: p_adj = min(1.0, p * k) where k = number of
  workloads being tested simultaneously (k = 6 here).
- BCa bootstrap 95% CI (Efron 1987): 10,000 resamples of the paired
  differences, bias-correction z0 from the fraction of resamples below the
  observed mean, acceleration a from the jackknife influence function on the
  paired differences.

The same triple (paired-t, Bonferroni, BCa 10k) is reported by the paper.

Usage
-----
    python -m scripts.compute_statistics
    python scripts/compute_statistics.py --input results/benchmark_results_20260721_191113.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


# Deterministic RNG for reproducible bootstraps
BOOTSTRAP_SEED = 20260721
N_BOOTSTRAP = 10_000
BASELINE = "LRU"
TREATMENT_PATTERN = "GDSF"   # matches "GDSF(a=1.0,b=1.0)"
METRIC = "cost_weighted_hit_rate"


def _find_treatment_policy(policies: Sequence[str]) -> str:
    for p in policies:
        if p.startswith(TREATMENT_PATTERN):
            return p
    raise RuntimeError(
        f"No policy starting with {TREATMENT_PATTERN!r} found. Policies: {list(policies)}"
    )


def _paired_arrays(
    rows: List[dict], workload: str, treatment_policy: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (treatment, baseline, diff) arrays paired by (cache_size, seed)."""
    by_key: Dict[Tuple[int, int], Dict[str, float]] = {}
    for r in rows:
        if r["workload_name"] != workload:
            continue
        policy = r["policy_name"]
        if policy not in (treatment_policy, BASELINE):
            continue
        key = (int(r["cache_size"]), int(r["seed"]))
        by_key.setdefault(key, {})[policy] = float(r[METRIC])

    treatment_vals: List[float] = []
    baseline_vals: List[float] = []
    for key, policies in sorted(by_key.items()):
        if treatment_policy in policies and BASELINE in policies:
            treatment_vals.append(policies[treatment_policy])
            baseline_vals.append(policies[BASELINE])

    t = np.asarray(treatment_vals, dtype=np.float64)
    b = np.asarray(baseline_vals, dtype=np.float64)
    return t, b, t - b


def _paired_t(diff: np.ndarray) -> Tuple[float, float]:
    """Return (t_stat, two_sided_p) for a paired sample of differences."""
    n = diff.size
    mean = float(diff.mean())
    sd = float(diff.std(ddof=1)) if n > 1 else 0.0
    if sd == 0.0:
        t_stat = 0.0 if mean == 0.0 else float("inf") * (1 if mean > 0 else -1)
        p_val = 1.0 if mean == 0.0 else 0.0
        return t_stat, p_val
    t_stat = mean / (sd / np.sqrt(n))
    # two-sided p via scipy.stats.t.sf if available, else survival-function
    # of Student's t implemented via the regularised incomplete beta.
    from math import lgamma, log

    df = n - 1
    # Use scipy if present (more numerically stable at extreme tails)
    try:
        from scipy.stats import t as student_t  # type: ignore

        p_val = 2.0 * float(student_t.sf(abs(t_stat), df))
    except Exception:  # pragma: no cover
        # Fallback: F distribution relation (T^2 ~ F(1, df))
        x = df / (df + t_stat * t_stat)
        # Regularised incomplete beta I_x(df/2, 1/2)
        from math import erf  # noqa: F401

        # Very rough fallback — should never trigger since scipy is a dep.
        # Clip to something visible so a missing scipy is obvious in the JSON.
        p_val = float("nan")
    return float(t_stat), float(p_val)


def _bca_ci(
    diff: np.ndarray,
    n_boot: int = N_BOOTSTRAP,
    alpha: float = 0.05,
    seed: int = BOOTSTRAP_SEED,
) -> Tuple[float, float, float]:
    """Bias-corrected and accelerated (BCa) bootstrap CI for the mean.

    Returns (lower, upper, point_estimate).

    Follows Efron (1987), Section 2, on the mean statistic.
    """
    rng = np.random.default_rng(seed)
    n = diff.size
    theta_hat = float(diff.mean())

    if n < 2 or diff.std(ddof=1) == 0.0:
        return theta_hat, theta_hat, theta_hat

    # Bootstrap resamples
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = diff[idx].mean(axis=1)

    # Bias-correction z0 from proportion of resamples below theta_hat
    from scipy.stats import norm  # type: ignore

    prop_below = float(np.mean(boot_means < theta_hat))
    # Guard against exactly 0 or 1 (Inf z0). Efron recommends the smoothing
    # p -> (p + 0.5/n_boot).
    prop_below = min(max(prop_below, 0.5 / n_boot), 1.0 - 0.5 / n_boot)
    z0 = float(norm.ppf(prop_below))

    # Acceleration by jackknife
    jack = np.empty(n, dtype=np.float64)
    total = diff.sum()
    for i in range(n):
        jack[i] = (total - diff[i]) / (n - 1)
    jack_mean = jack.mean()
    num = float(((jack_mean - jack) ** 3).sum())
    denom = 6.0 * (float(((jack_mean - jack) ** 2).sum()) ** 1.5)
    a = num / denom if denom > 0.0 else 0.0

    z_alpha_lo = float(norm.ppf(alpha / 2.0))
    z_alpha_hi = float(norm.ppf(1.0 - alpha / 2.0))

    def _adjust(z_alpha: float) -> float:
        num = z0 + z_alpha
        return float(norm.cdf(z0 + num / (1.0 - a * num)))

    q_lo = _adjust(z_alpha_lo)
    q_hi = _adjust(z_alpha_hi)
    # Clip quantiles into (0, 1) - required if a or z0 pushes off the edge.
    q_lo = min(max(q_lo, 1.0 / n_boot), 1.0 - 1.0 / n_boot)
    q_hi = min(max(q_hi, 1.0 / n_boot), 1.0 - 1.0 / n_boot)

    lo = float(np.quantile(boot_means, q_lo))
    hi = float(np.quantile(boot_means, q_hi))
    return lo, hi, theta_hat


def compute(
    input_json: Path,
    output_json: Path,
    n_boot: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
) -> Dict:
    rows = json.loads(input_json.read_text())
    policies = sorted({r["policy_name"] for r in rows})
    workloads = sorted({r["workload_name"] for r in rows})
    treatment = _find_treatment_policy(policies)
    k = len(workloads)

    results: Dict[str, Dict] = {}
    for wl in workloads:
        t_arr, b_arr, d_arr = _paired_arrays(rows, wl, treatment)
        n = d_arr.size
        t_stat, p_raw = _paired_t(d_arr)
        p_bonf = float(min(1.0, p_raw * k))
        ci_lo, ci_hi, mean_diff = _bca_ci(d_arr, n_boot=n_boot, seed=seed)

        # Also report percent dollar change on the paired sample for
        # reporting convenience (used in the abstract's headline numbers).
        pct_change = (
            100.0 * float((t_arr - b_arr).sum()) / float(b_arr.sum())
            if b_arr.sum() > 0.0
            else 0.0
        )

        results[wl] = {
            "n_pairs": int(n),
            "treatment_policy": treatment,
            "baseline_policy": BASELINE,
            "metric": METRIC,
            "mean_treatment": float(t_arr.mean()),
            "mean_baseline": float(b_arr.mean()),
            "mean_diff": float(mean_diff),
            "std_diff": float(d_arr.std(ddof=1)) if n > 1 else 0.0,
            "paired_t": float(t_stat),
            "df": int(n - 1),
            "p_raw": float(p_raw),
            "p_bonferroni": float(p_bonf),
            "ci_lower_95_bca": float(ci_lo),
            "ci_upper_95_bca": float(ci_hi),
            "pct_dollar_change_vs_baseline": float(pct_change),
        }

    summary = {
        "input_file": str(input_json.name),
        "n_workloads": k,
        "n_bootstrap": int(n_boot),
        "bootstrap_seed": int(seed),
        "correction": "Bonferroni",
        "ci_method": "BCa (Efron 1987)",
        "alpha": 0.05,
        "results": results,
        "generated_at_unix": int(time.time()),
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2))
    return summary


def _latest_input(results_dir: Path) -> Path:
    candidates = sorted(results_dir.glob("benchmark_results_*.json"))
    if not candidates:
        raise SystemExit(f"No benchmark_results_*.json found under {results_dir}")
    return candidates[-1]


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--input", type=Path, default=None,
                   help="Path to benchmark_results_*.json (default: newest under results/)")
    p.add_argument("--output", type=Path, default=None,
                   help="Path to write stats JSON (default: results/stats_<ts>.json)")
    p.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    p.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    args = p.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    results_dir = repo_root / "results"

    inp = args.input if args.input is not None else _latest_input(results_dir)
    if args.output is None:
        stamp = inp.stem.replace("benchmark_results_", "")
        out = results_dir / f"stats_{stamp}.json"
    else:
        out = args.output

    print(f"Computing statistics from {inp} -> {out} ...")
    summary = compute(inp, out, n_boot=args.n_bootstrap, seed=args.seed)

    print(f"Wrote {out}")
    print(f"Workloads: {list(summary['results'].keys())}")
    for wl, r in summary["results"].items():
        print(
            f"  {wl:24s} n={r['n_pairs']:3d}  "
            f"delta_CWHR={r['mean_diff']:+.4f}  "
            f"CI[{r['ci_lower_95_bca']:+.4f},{r['ci_upper_95_bca']:+.4f}]  "
            f"t={r['paired_t']:+.2f}  p_raw={r['p_raw']:.3g}  "
            f"p_bonf={r['p_bonferroni']:.3g}  "
            f"pct_dollar={r['pct_dollar_change_vs_baseline']:+.1f}%"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
