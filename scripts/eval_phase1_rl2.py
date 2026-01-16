#!/usr/bin/env python3
"""
Post-hoc Phase 1 RL2 evaluation over all training samples (PDE conditions).

Purpose:
  - Compute final RL2 statistics from an existing Phase 1 checkpoint (e.g. best.pt)
  - Patch the existing metrics JSON's metadata.final_summary with these RL2 fields
  - Optionally regenerate metrics_runs.csv / metrics_summary.csv

Typical usage:
  uv run python scripts/eval_phase1_rl2.py --config configs/convection_phase1.yaml
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from model import LR_PINN_phase1
from src.config_loader import load_config, validate_config
from src.data_loader import load_parametric_dataset, convert_to_model_format, prepare_batch


def _beta_nu_rho_from_z(z_tensor):
    if z_tensor is None:
        return 1.0, 0.0, 0.0
    params = z_tensor.squeeze().detach().cpu().numpy()
    params = np.atleast_1d(np.array(params))
    beta_val = float(params[0]) if params.size > 0 else 1.0
    nu_val = float(params[1]) if params.size > 1 else 0.0
    rho_val = float(params[2]) if params.size > 2 else 0.0
    return beta_val, nu_val, rho_val


def _compute_phase1_rl2_over_samples(
    *,
    net: torch.nn.Module,
    samples: List[Dict[str, Any]],
    normalization_stats: Dict[str, Any],
    device: torch.device,
    chunk_size: int = 200_000,
) -> Dict[str, float]:
    if not samples:
        return {"rl2_train": float("nan")}

    u_mean = float(normalization_stats.get("u_mean", 0.0))
    u_std = float(normalization_stats.get("u_std", 1.0))
    u_mean_t = torch.tensor(u_mean, device=device)
    u_std_t = torch.tensor(u_std, device=device)

    rl2_per_sample: List[float] = []
    sum_sq_err = 0.0
    sum_sq_true = 0.0

    net.eval()
    with torch.no_grad():
        for sample in samples:
            x, y, z = prepare_batch(sample, include_coords=True, device=str(device))
            beta_val, nu_val, rho_val = _beta_nu_rho_from_z(z)

            if x.shape[1] >= 2:
                x_in = x[:, 0:1]
                t_in = x[:, 1:2]
            else:
                x_in = x[:, 0:1]
                t_in = torch.zeros_like(x_in)

            n = x_in.shape[0]
            beta_in = torch.full((n, 1), float(beta_val), device=device)
            nu_in = torch.full((n, 1), float(nu_val), device=device)
            rho_in = torch.full((n, 1), float(rho_val), device=device)

            preds = []
            cs = int(max(1, chunk_size))
            for s in range(0, n, cs):
                e = min(n, s + cs)
                u_hat, *_ = net(x_in[s:e], t_in[s:e], beta_in[s:e], nu_in[s:e], rho_in[s:e])
                preds.append(u_hat)
            u_hat = torch.cat(preds, dim=0)

            u_hat_phys = u_hat * u_std_t + u_mean_t
            u_true_phys = y.to(device) * u_std_t + u_mean_t

            err = u_hat_phys - u_true_phys
            denom = torch.linalg.norm(u_true_phys)
            if float(denom.item()) > 1e-12:
                rl2 = float((torch.linalg.norm(err) / denom).item())
            else:
                rl2 = float("inf")
            rl2_per_sample.append(rl2)

            sum_sq_err += float((err ** 2).sum().item())
            sum_sq_true += float((u_true_phys ** 2).sum().item())

    if sum_sq_true > 1e-24:
        rl2_global = float(math.sqrt(sum_sq_err / sum_sq_true))
    else:
        rl2_global = float("inf")

    rl2_arr = np.array(rl2_per_sample, dtype=float)
    rl2_mean = float(np.nanmean(rl2_arr)) if rl2_arr.size else float("nan")
    rl2_std = float(np.nanstd(rl2_arr, ddof=1)) if rl2_arr.size > 1 else 0.0
    rl2_sem = float(rl2_std / math.sqrt(rl2_arr.size)) if rl2_arr.size > 1 else 0.0

    return {
        "rl2_train": rl2_global,
        "rl2_train_global": rl2_global,
        "rl2_train_mean_per_sample": rl2_mean,
        "rl2_train_std_per_sample": rl2_std,
        "rl2_train_sem_per_sample": rl2_sem,
        "n_train_samples_for_rl2": float(len(samples)),
    }


def _default_paths_from_config(config: Dict[str, Any]) -> Tuple[str, Path, Path]:
    export_cfg = config.get("export", {})
    dataset_cfg = config.get("dataset", {})
    equation = str(dataset_cfg.get("equation"))
    output_dir = Path(str(export_cfg.get("output_dir", "./outputs")))

    metrics_dir = output_dir / f"{equation}_phase1" / "metrics"
    ckpt = output_dir / f"{equation}_phase1" / "checkpoints" / "best.pt"
    return equation, metrics_dir, ckpt


def _pick_latest_metrics_json(metrics_dir: Path) -> Path:
    jsons = sorted(metrics_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not jsons:
        raise FileNotFoundError(f"No metrics JSONs found in: {metrics_dir}")
    return jsons[0]


def _patch_metrics_json(metrics_json: Path, rl2_stats: Dict[str, float]) -> None:
    data = json.loads(metrics_json.read_text())
    data.setdefault("metadata", {})
    data["metadata"].setdefault("final_summary", {})
    data["metadata"]["final_summary"].update(rl2_stats)
    metrics_json.write_text(json.dumps(data, indent=2, sort_keys=False))


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Post-hoc phase1 RL2 eval + patch metrics JSON.")
    ap.add_argument("--config", required=True, type=str, help="Path to phase1 YAML config.")
    ap.add_argument("--device", default="cpu", type=str, help="Device for eval (default: cpu).")
    ap.add_argument("--checkpoint", default=None, type=str, help="Override checkpoint path (default: outputs/<eq>_phase1/checkpoints/best.pt).")
    ap.add_argument("--metrics-json", default=None, type=str, help="Override metrics JSON to patch (default: latest *.json in metrics dir).")
    ap.add_argument("--chunk-size", default=200000, type=int, help="Inference chunk size to avoid OOM.")
    ap.add_argument("--no-patch", action="store_true", help="Compute and print only; do not modify metrics JSON.")
    ap.add_argument("--regen-csv", action="store_true", help="After patching, regenerate metrics_runs.csv + metrics_summary.csv in metrics dir.")

    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    validate_config(cfg)

    equation, metrics_dir, ckpt_default = _default_paths_from_config(cfg)
    ckpt = Path(args.checkpoint).expanduser().resolve() if args.checkpoint else ckpt_default.resolve()
    metrics_dir = metrics_dir.resolve()

    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    if not metrics_dir.exists():
        raise FileNotFoundError(f"Metrics dir not found: {metrics_dir}")

    device = torch.device(args.device)

    # Load dataset + normalized splits
    ds_cfg = cfg["dataset"]
    splits = load_parametric_dataset(
        equation=ds_cfg["equation"],
        data_dir=ds_cfg["data_dir"],
        seed=ds_cfg.get("seed", 42),
        n_train=ds_cfg.get("n_train", 10),
        cache=ds_cfg.get("cache", True),
        balance=ds_cfg.get("balance", False),
        n_each=ds_cfg.get("n_each", None),
        balance_strategy=ds_cfg.get("balance_strategy", "random"),
        diversify=ds_cfg.get("diversify", False)
    )
    processed = convert_to_model_format(splits, phase="phase1", device=str(device))
    train_samples = list(processed.get("train_few", []))

    # Load model
    hidden_dim = int(cfg["model"]["hidden_dim"])
    net = LR_PINN_phase1(hidden_dim).to(device)
    state = torch.load(str(ckpt), map_location=device)
    net.load_state_dict(state)

    rl2_stats = _compute_phase1_rl2_over_samples(
        net=net,
        samples=train_samples,
        normalization_stats=splits.normalization_stats,
        device=device,
        chunk_size=int(args.chunk_size),
    )

    print(f"equation: {equation}")
    print(f"checkpoint: {ckpt}")
    print("rl2_stats:")
    for k, v in rl2_stats.items():
        print(f"  {k}: {v}")

    if args.no_patch:
        return 0

    metrics_json = Path(args.metrics_json).expanduser().resolve() if args.metrics_json else _pick_latest_metrics_json(metrics_dir)
    _patch_metrics_json(metrics_json, rl2_stats)
    print(f"Patched metrics JSON: {metrics_json}")

    if args.regen_csv:
        from scripts.summarize_metrics import summarize_metrics_dir

        summarize_metrics_dir(metrics_dir)
        print(f"Regenerated CSVs in: {metrics_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

