#!/usr/bin/env python3
"""
Summarize Hyper-LR-PINN run metrics JSONs into CSV tables.

This is intended to post-process existing outputs without re-running training.

Typical usage:
  python scripts/summarize_metrics.py \
    --metrics-dir outputs/allen_cahn_phase1/metrics

It will generate:
  - metrics_runs.csv     (one row per run)
  - metrics_summary.csv  (mean/std/sem across runs, one row per metric)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numbers


@dataclass(frozen=True)
class SummaryPick:
    """A located final summary dict inside a JSON document."""

    path: str
    summary: Dict[str, Any]


def _is_number(x: Any) -> bool:
    return isinstance(x, numbers.Real) and not isinstance(x, bool) and math.isfinite(float(x))


def _find_final_summaries(obj: Any) -> List[SummaryPick]:
    """
    Recursively find dicts keyed by 'final_summary' anywhere in the JSON.
    Returns a list because some files may contain multiple summaries.
    """

    found: List[SummaryPick] = []

    def walk(x: Any, path: str) -> None:
        if isinstance(x, dict):
            if "final_summary" in x and isinstance(x["final_summary"], dict):
                found.append(SummaryPick(path=f"{path}/final_summary", summary=x["final_summary"]))
            for k, v in x.items():
                walk(v, f"{path}/{k}")
        elif isinstance(x, list):
            for i, v in enumerate(x):
                walk(v, f"{path}[{i}]")

    walk(obj, "")
    return found


def _pick_summary(
    picks: Sequence[SummaryPick],
    *,
    preferred_path_suffixes: Sequence[str] = ("/metadata/final_summary",),
    summary_path: Optional[str] = None,
) -> SummaryPick:
    if not picks:
        raise ValueError("No 'final_summary' found in JSON.")

    if summary_path is not None:
        for p in picks:
            if p.path == summary_path:
                return p
        available = ", ".join(p.path for p in picks)
        raise ValueError(f"--summary-path '{summary_path}' not found. Available: {available}")

    if len(picks) == 1:
        return picks[0]

    # Prefer known/expected location first (e.g., metrics_structures RunData JSONs)
    for suffix in preferred_path_suffixes:
        for p in picks:
            if p.path.endswith(suffix):
                return p

    available = ", ".join(p.path for p in picks)
    raise ValueError(
        f"Multiple 'final_summary' blocks found; please pass --summary-path. Available: {available}"
    )


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _mean(xs: Sequence[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else float("nan")


def _std(xs: Sequence[float]) -> float:
    # Sample std (ddof=1) when possible; else 0 for single sample.
    n = len(xs)
    if n <= 1:
        return 0.0
    mu = _mean(xs)
    var = sum((x - mu) ** 2 for x in xs) / (n - 1)
    return float(math.sqrt(var))


def _sem(xs: Sequence[float]) -> float:
    n = len(xs)
    if n <= 1:
        return 0.0
    return float(_std(xs) / math.sqrt(n))


def _format_pm(mu: float, sigma: float, *, sigfig: int = 6) -> str:
    # Keep it parseable and consistent in CSV.
    if not math.isfinite(mu) or not math.isfinite(sigma):
        return ""
    fmt = f"{{:.{sigfig}g}}"
    return f"{fmt.format(mu)} ± {fmt.format(sigma)}"


def summarize_metrics_dir(
    metrics_dir: Path,
    *,
    output_runs_csv: Optional[Path] = None,
    output_summary_csv: Optional[Path] = None,
    summary_path: Optional[str] = None,
    sigfig: int = 6,
) -> Tuple[Path, Path]:
    metrics_dir = metrics_dir.expanduser().resolve()
    if not metrics_dir.exists():
        raise FileNotFoundError(f"metrics dir does not exist: {metrics_dir}")

    json_files = sorted(metrics_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No .json metrics files found in: {metrics_dir}")

    output_runs_csv = output_runs_csv or (metrics_dir / "metrics_runs.csv")
    output_summary_csv = output_summary_csv or (metrics_dir / "metrics_summary.csv")

    runs: List[Dict[str, Any]] = []
    all_metric_keys: List[str] = []
    all_metric_key_set = set()

    for jf in json_files:
        doc = _read_json(jf)
        run_id = str(doc.get("id") or jf.stem)

        picks = _find_final_summaries(doc)
        pick = _pick_summary(picks, summary_path=summary_path)

        row: Dict[str, Any] = {"id": run_id, "_summary_path": pick.path}
        for k, v in pick.summary.items():
            if _is_number(v):
                row[k] = float(v)
                if k not in all_metric_key_set:
                    all_metric_key_set.add(k)
                    all_metric_keys.append(k)
            else:
                # keep non-numerics out of aggregation but still include per-run csv if desired
                # (for now, omit them to keep the runs CSV numeric-focused)
                pass

        runs.append(row)

    # Stable column order: id, summary path, then metrics (in first-seen order)
    run_columns = ["id", "_summary_path", *all_metric_keys]
    os.makedirs(output_runs_csv.parent, exist_ok=True)
    with output_runs_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=run_columns)
        w.writeheader()
        for r in runs:
            w.writerow({c: r.get(c, "") for c in run_columns})

    # Aggregate
    metric_to_vals: Dict[str, List[float]] = {k: [] for k in all_metric_keys}
    for r in runs:
        for k in all_metric_keys:
            v = r.get(k, None)
            if _is_number(v):
                metric_to_vals[k].append(float(v))

    summary_rows: List[Dict[str, Any]] = []
    for k in all_metric_keys:
        vals = metric_to_vals[k]
        n = len(vals)
        mu = _mean(vals) if n else float("nan")
        sd = _std(vals) if n else float("nan")
        se = _sem(vals) if n else float("nan")
        summary_rows.append(
            {
                "metric": k,
                "n": n,
                "mean": mu,
                "std": sd,
                "sem": se,
                "mean_minus_std": (mu - sd) if (math.isfinite(mu) and math.isfinite(sd)) else "",
                "mean_plus_std": (mu + sd) if (math.isfinite(mu) and math.isfinite(sd)) else "",
                "mean_±_std": _format_pm(mu, sd, sigfig=sigfig),
                "mean_±_sem": _format_pm(mu, se, sigfig=sigfig),
            }
        )

    os.makedirs(output_summary_csv.parent, exist_ok=True)
    with output_summary_csv.open("w", newline="", encoding="utf-8") as f:
        cols = [
            "metric",
            "n",
            "mean",
            "std",
            "sem",
            "mean_minus_std",
            "mean_plus_std",
            "mean_±_std",
            "mean_±_sem",
        ]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)

    return output_runs_csv, output_summary_csv


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Summarize metrics JSONs into CSV mean±std tables.")
    p.add_argument(
        "--metrics-dir",
        required=True,
        type=str,
        help="Directory containing per-run *.json metrics files (e.g., outputs/allen_cahn_phase1/metrics).",
    )
    p.add_argument(
        "--output-runs-csv",
        default=None,
        type=str,
        help="Path for per-run CSV (default: <metrics-dir>/metrics_runs.csv).",
    )
    p.add_argument(
        "--output-summary-csv",
        default=None,
        type=str,
        help="Path for aggregated summary CSV (default: <metrics-dir>/metrics_summary.csv).",
    )
    p.add_argument(
        "--summary-path",
        default=None,
        type=str,
        help="If a JSON contains multiple final_summary blocks, pick one by its JSON path (printed in metrics_runs.csv).",
    )
    p.add_argument(
        "--sigfig",
        default=6,
        type=int,
        help="Significant figures to use in the mean±(std/sem) string columns.",
    )

    args = p.parse_args(argv)
    runs_csv, summary_csv = summarize_metrics_dir(
        Path(args.metrics_dir),
        output_runs_csv=Path(args.output_runs_csv) if args.output_runs_csv else None,
        output_summary_csv=Path(args.output_summary_csv) if args.output_summary_csv else None,
        summary_path=args.summary_path,
        sigfig=int(args.sigfig),
    )
    print(f"Wrote per-run metrics CSV: {runs_csv}")
    print(f"Wrote aggregated summary CSV: {summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

