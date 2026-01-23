#!/usr/bin/env python
"""
Fine-tuning evaluation script for Hyper-LR-PINN.

This script performs fine-tuning on interpolation and extrapolation splits,
evaluates at multiple checkpoints (0, 50, 250, 500, 750, 1000 epochs),
and generates a summary table with L2 Error and Time metrics.
"""

import argparse
import os
import sys
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import random
import torch.backends.cudnn as cudnn
import pandas as pd
from typing import Dict, List, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model import LR_PINN_phase1, LR_PINN_phase2
from utils import get_params
from src.config_loader import load_config, validate_config
from src.pinn_losses import f_cal_phase2_equation_aware
from src.data_loader import (
    load_parametric_dataset,
    normalize_data,
    prepare_batch,
    convert_to_model_format,
    DatasetSplits
)
from scripts.train import convert_dataset_to_training_format
from src.metrics_exporter import compute_metrics, generate_visualizations


def _make_optimizer(model: torch.nn.Module, *, learning_rate: float, optimizer: str = "adamw", weight_decay: float = 0.0, momentum: float = 0.0):
    """Create optimizer from config-like values (kept consistent with scripts/train.py)."""
    lr = float(learning_rate)
    opt_name = str(optimizer or "adamw").lower()
    wd = float(weight_decay or 0.0)
    mom = float(momentum or 0.0)

    if opt_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr)
    if opt_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    if opt_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=mom, weight_decay=wd)
    raise ValueError(f"Unsupported optimizer: {opt_name}")


def _helmholtz_dirichlet_bd_loss(
    *,
    u_pred_phys: torch.Tensor,
    u_true_phys: torch.Tensor,
    x_coords: torch.Tensor,
    y_coords: torch.Tensor,
    mse_cost_function: nn.Module,
    eps_bd: float = 1e-6,
) -> torch.Tensor:
    """Dirichlet boundary loss on all four edges: match prediction to ground truth."""
    x_min = x_coords.min()
    x_max = x_coords.max()
    y_min = y_coords.min()
    y_max = y_coords.max()

    left_mask = (x_coords - x_min).abs() < eps_bd
    right_mask = (x_coords - x_max).abs() < eps_bd
    bottom_mask = (y_coords - y_min).abs() < eps_bd
    top_mask = (y_coords - y_max).abs() < eps_bd

    mse_bd = torch.tensor(0.0, device=u_pred_phys.device)
    for m in (left_mask, right_mask, bottom_mask, top_mask):
        if m.any():
            mse_bd = mse_bd + mse_cost_function(u_pred_phys[m], u_true_phys[m])
    return mse_bd


def _plot_contourf_with_contours(
    ax,
    field2d: np.ndarray,
    *,
    cmap: str,
    vmin: float | None = None,
    vmax: float | None = None,
    n_filled_levels: int = 64,
    n_line_levels: int = 12,
    line_color: str = "k",
    line_width: float = 0.35,
    line_alpha: float = 0.55,
):
    """
    Render a 2D field with contourf + contour overlay.
    Returns a mappable suitable for fig.colorbar().

    Note: We plot in index coordinates (0..W-1, 0..H-1) because the eval script
    doesn't reliably have physical coordinate grids available at this stage.
    """
    field2d = np.asarray(field2d)
    h, w = field2d.shape
    x = np.arange(w)
    y = np.arange(h)
    X, Y = np.meshgrid(x, y)

    if vmin is None:
        vmin = float(np.nanmin(field2d))
    if vmax is None:
        vmax = float(np.nanmax(field2d))

    # Avoid degenerate levels when field is constant
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        norm = None
        filled_levels = n_filled_levels
        line_levels = None
    else:
        norm = Normalize(vmin=vmin, vmax=vmax)
        filled_levels = np.linspace(vmin, vmax, int(max(2, n_filled_levels)))
        line_levels = np.linspace(vmin, vmax, int(max(2, n_line_levels)))

    cf = ax.contourf(X, Y, field2d, levels=filled_levels, cmap=cmap, norm=norm)

    if line_levels is not None:
        ax.contour(
            X,
            Y,
            field2d,
            levels=line_levels,
            colors=line_color,
            linewidths=line_width,
            alpha=line_alpha,
        )

    return cf


def setup_seed(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


def create_phase2_model(phase1_checkpoint: str, hidden_dim: int, target_coeff: torch.Tensor, device: torch.device, use_compile: bool = False):
    """
    Create Phase 2 model from Phase 1 checkpoint.
    
    Args:
        phase1_checkpoint: Path to Phase 1 checkpoint
        hidden_dim: Hidden dimension
        target_coeff: Target parameter coefficients
        device: Device to use
        use_compile: Whether to use torch.compile() for optimization (PyTorch 2.0+)
        
    Returns:
        Phase 2 model initialized from Phase 1
    """
    # Load phase1 model
    net_initial = LR_PINN_phase1(hidden_dim)
    net_initial.load_state_dict(torch.load(phase1_checkpoint, map_location=device))
    net_initial = net_initial.to(device)
    
    # Extract weights
    tanh = nn.Tanh()
    relu = nn.ReLU()
    
    start_w = net_initial.state_dict()['start_layer.weight']
    start_b = net_initial.state_dict()['start_layer.bias']
    end_w = net_initial.state_dict()['end_layer.weight']
    end_b = net_initial.state_dict()['end_layer.bias']
    
    col_0 = net_initial.state_dict()['col_basis_0']
    col_1 = net_initial.state_dict()['col_basis_1']
    col_2 = net_initial.state_dict()['col_basis_2']
    row_0 = net_initial.state_dict()['row_basis_0']
    row_1 = net_initial.state_dict()['row_basis_1']
    row_2 = net_initial.state_dict()['row_basis_2']
    
    # Compute alpha values from meta-network
    meta_layer_1_w = net_initial.state_dict()['meta_layer_1.weight']
    meta_layer_1_b = net_initial.state_dict()['meta_layer_1.bias']
    meta_layer_2_w = net_initial.state_dict()['meta_layer_2.weight']
    meta_layer_2_b = net_initial.state_dict()['meta_layer_2.bias']
    meta_layer_3_w = net_initial.state_dict()['meta_layer_3.weight']
    meta_layer_3_b = net_initial.state_dict()['meta_layer_3.bias']
    
    meta_alpha_0_w = net_initial.state_dict()['meta_alpha_0.weight']
    meta_alpha_0_b = net_initial.state_dict()['meta_alpha_0.bias']
    meta_alpha_1_w = net_initial.state_dict()['meta_alpha_1.weight']
    meta_alpha_1_b = net_initial.state_dict()['meta_alpha_1.bias']
    meta_alpha_2_w = net_initial.state_dict()['meta_alpha_2.weight']
    meta_alpha_2_b = net_initial.state_dict()['meta_alpha_2.bias']
    
    meta_vector = torch.matmul(target_coeff, meta_layer_1_w.T) + meta_layer_1_b
    meta_vector = tanh(meta_vector)
    meta_vector = torch.matmul(meta_vector, meta_layer_2_w.T) + meta_layer_2_b
    meta_vector = tanh(meta_vector)
    meta_vector = torch.matmul(meta_vector, meta_layer_3_w.T) + meta_layer_3_b
    meta_vector = tanh(meta_vector)
    
    alpha_0 = relu(torch.matmul(meta_vector, meta_alpha_0_w.T) + meta_alpha_0_b)
    alpha_1 = relu(torch.matmul(meta_vector, meta_alpha_1_w.T) + meta_alpha_1_b)
    alpha_2 = relu(torch.matmul(meta_vector, meta_alpha_2_w.T) + meta_alpha_2_b)
    
    # Create phase2 model
    model = LR_PINN_phase2(
        hidden_dim, start_w, start_b, end_w, end_b,
        col_0, col_1, col_2, row_0, row_1, row_2,
        alpha_0, alpha_1, alpha_2
    )
    model = model.to(device)
    
    # Apply torch.compile() if requested and available
    if use_compile and hasattr(torch, 'compile'):
        try:
            # Use 'default' mode to avoid CUDAGraphs issues with in-place operations
            # 'reduce-overhead' enables CUDAGraphs which conflicts with in-place ops
            model = torch.compile(model, mode='default')
            print(f"[INFO] Applied torch.compile() to Phase 2 model (mode='default')")
        except Exception as e:
            print(f"[WARN] torch.compile() failed: {e}, continuing without compilation")
    
    return model


def evaluate_model(model, split_data, device, normalization_stats) -> Tuple[float, float]:
    """
    Evaluate model on a data split and return (mean, std) of per-sample relative L2 error.
    
    Args:
        model: Model to evaluate
        split_data: List of data samples
        device: Device to use
        normalization_stats: Normalization statistics
        
    Returns:
        (mean_relative_l2, std_relative_l2) computed across samples in split_data
    """
    model.eval()
    
    per_sample_rel_l2 = []
    
    with torch.no_grad():
        for sample in split_data:
            # Prepare batch
            x, y, z = prepare_batch(sample, include_coords=True, device=device)
            
            # Denormalize ground truth
            u_gt = y.squeeze().cpu().numpy()
            if normalization_stats:
                u_gt = u_gt * normalization_stats['u_std'] + normalization_stats['u_mean']
            
            # Forward pass (phase2 doesn't need parameters)
            if x.shape[1] >= 2:
                x_coords = x[:, 0:1]
                t_coords = x[:, 1:2]
            else:
                x_coords = x[:, 0:1]
                t_coords = torch.zeros_like(x_coords)
            
            u_pred = model(x_coords, t_coords)
            
            # Denormalize prediction
            u_pred_np = u_pred.squeeze().cpu().numpy()
            if normalization_stats:
                u_pred_np = u_pred_np * normalization_stats['u_std'] + normalization_stats['u_mean']
            
            # Per-sample relative L2
            err = float(np.linalg.norm(u_pred_np.flatten() - u_gt.flatten()))
            denom = float(np.linalg.norm(u_gt.flatten()))
            rel = err / denom if denom > 1e-12 else (0.0 if err < 1e-12 else float("inf"))
            per_sample_rel_l2.append(rel)

    if len(per_sample_rel_l2) == 0:
        return float("nan"), float("nan")

    mean_rel = float(np.mean(per_sample_rel_l2))
    # Sample std when possible, otherwise 0.0
    std_rel = float(np.std(per_sample_rel_l2, ddof=1)) if len(per_sample_rel_l2) > 1 else 0.0
    return mean_rel, std_rel


def save_qualitative_plot(
    model,
    split_data,
    split_name: str,
    epoch: int,
    output_dir: str,
    device: torch.device,
    normalization_stats,
    sample_idx: int = 0,
) -> None:
    """
    Save a qualitative plot (ground truth / prediction / abs error) for one sample.

    This produces one image per split per checkpoint epoch.
    """
    if not split_data:
        return
    if sample_idx < 0 or sample_idx >= len(split_data):
        sample_idx = 0

    model.eval()
    sample = split_data[sample_idx]

    with torch.no_grad():
        x, y, _ = prepare_batch(sample, include_coords=True, device=device)

        # Ground truth (denormalized)
        u_gt = y.squeeze().cpu().numpy()
        if normalization_stats:
            u_gt = u_gt * normalization_stats["u_std"] + normalization_stats["u_mean"]

        # Prediction (phase2 model doesn't need params)
        if x.shape[1] >= 2:
            x_coords = x[:, 0:1]
            t_coords = x[:, 1:2]
        else:
            x_coords = x[:, 0:1]
            t_coords = torch.zeros_like(x_coords)

        u_pred = model(x_coords, t_coords)
        u_pred_np = u_pred.squeeze().cpu().numpy()
        if normalization_stats:
            u_pred_np = u_pred_np * normalization_stats["u_std"] + normalization_stats["u_mean"]

    # Restore grid shape for plotting
    u_raw = sample.get("u")
    if isinstance(u_raw, torch.Tensor):
        u_shape = tuple(u_raw.shape)
    else:
        u_shape = tuple(np.array(u_raw).shape)

    gt_reshaped = u_gt.reshape(u_shape)
    pred_reshaped = u_pred_np.reshape(u_shape)

    os.makedirs(output_dir, exist_ok=True)
    split_tag = f"{split_name}_epoch_{epoch}"
    generate_visualizations(
        predictions=np.array([pred_reshaped]),
        ground_truth=np.array([gt_reshaped]),
        output_dir=output_dir,
        split_name=split_tag,
        sample_indices=[0],
        max_samples=1,
        coordinate_grids=None,
    )


def save_qualitative_grid_plot(
    model,
    split_data,
    split_name: str,
    epoch: int,
    output_dir: str,
    device: torch.device,
    normalization_stats,
    num_samples: int = 5,
    sample_start_idx: int = 0,
) -> None:
    """
    Save a single multi-sample figure for one split+epoch.

    Layout: num_samples rows × 3 columns (GT / Pred / |Err|).
    Saves both PNG and SVG.
    """
    if not split_data:
        return

    k = max(1, min(int(num_samples), len(split_data)))
    start = int(max(0, sample_start_idx))
    indices = [(start + i) % len(split_data) for i in range(k)]

    preds = []
    gts = []
    shapes = []

    model.eval()
    with torch.no_grad():
        for idx in indices:
            sample = split_data[idx]
            x, y, _ = prepare_batch(sample, include_coords=True, device=device)

            u_gt = y.squeeze().cpu().numpy()
            if normalization_stats:
                u_gt = u_gt * normalization_stats["u_std"] + normalization_stats["u_mean"]

            if x.shape[1] >= 2:
                x_coords = x[:, 0:1]
                t_coords = x[:, 1:2]
            else:
                x_coords = x[:, 0:1]
                t_coords = torch.zeros_like(x_coords)

            u_pred = model(x_coords, t_coords)
            u_pred_np = u_pred.squeeze().cpu().numpy()
            if normalization_stats:
                u_pred_np = u_pred_np * normalization_stats["u_std"] + normalization_stats["u_mean"]

            u_raw = sample.get("u")
            if isinstance(u_raw, torch.Tensor):
                u_shape = tuple(u_raw.shape)
            else:
                u_shape = tuple(np.array(u_raw).shape)

            preds.append(u_pred_np.reshape(u_shape))
            gts.append(u_gt.reshape(u_shape))
            shapes.append(u_shape)

    # Assume consistent shape across the selected samples; fall back to per-sample if not.
    if len({s for s in shapes}) != 1:
        # Fallback: save separate plots using existing helper
        for idx in indices:
            save_qualitative_plot(
                model=model,
                split_data=split_data,
                split_name=split_name,
                epoch=epoch,
                output_dir=output_dir,
                device=device,
                normalization_stats=normalization_stats,
                sample_idx=idx,
            )
        return

    gt0 = gts[0]
    ndim = gt0.ndim

    fig_h = 3.2 * k
    fig_w = 14
    fig, axes = plt.subplots(k, 3, figsize=(fig_w, fig_h))
    if k == 1:
        axes = np.expand_dims(axes, axis=0)

    for r in range(k):
        gt = gts[r]
        pred = preds[r]
        err = np.abs(pred - gt)

        if ndim == 1:
            xax = np.arange(gt.shape[0])
            axes[r, 0].plot(xax, gt, linewidth=1.5)
            axes[r, 0].set_title(f"GT (sample {indices[r]})")
            axes[r, 1].plot(xax, pred, linewidth=1.5)
            axes[r, 1].set_title("Pred")
            axes[r, 2].plot(xax, err, linewidth=1.5)
            axes[r, 2].set_title("|Err|")
            for c in range(3):
                axes[r, c].grid(True, alpha=0.25)
        else:
            # For 3D, take a middle slice to make it 2D.
            if ndim == 3:
                mid = gt.shape[0] // 2
                gt2 = gt[mid, :, :]
                pred2 = pred[mid, :, :]
                err2 = err[mid, :, :]
            else:
                gt2, pred2, err2 = gt, pred, err

            vmin = min(float(np.min(gt2)), float(np.min(pred2)))
            vmax = max(float(np.max(gt2)), float(np.max(pred2)))

            im0 = _plot_contourf_with_contours(axes[r, 0], gt2, cmap="viridis", vmin=vmin, vmax=vmax)
            axes[r, 0].set_title(f"GT (sample {indices[r]})")
            fig.colorbar(im0, ax=axes[r, 0], fraction=0.046, pad=0.04)

            im1 = _plot_contourf_with_contours(axes[r, 1], pred2, cmap="viridis", vmin=vmin, vmax=vmax)
            axes[r, 1].set_title("Pred")
            fig.colorbar(im1, ax=axes[r, 1], fraction=0.046, pad=0.04)

            im2 = _plot_contourf_with_contours(axes[r, 2], err2, cmap="hot")
            axes[r, 2].set_title("|Err|")
            fig.colorbar(im2, ax=axes[r, 2], fraction=0.046, pad=0.04)

            for c in range(3):
                axes[r, c].set_xticks([])
                axes[r, c].set_yticks([])

    fig.suptitle(f"{split_name} @ epoch {epoch} (k={k})", fontsize=14)
    fig.tight_layout(rect=[0, 0.0, 1, 0.98])

    os.makedirs(output_dir, exist_ok=True)
    base = os.path.join(output_dir, f"{split_name}_epoch_{epoch}_samples_{k}")
    fig.savefig(base + ".png", dpi=200, bbox_inches="tight")
    fig.savefig(base + ".svg", bbox_inches="tight")
    plt.close(fig)


def _extract_target_coeff_from_sample(sample: Dict, device: torch.device) -> torch.Tensor:
    """Extract (beta, nu, rho) (or padded) coeff vector from a dataset sample."""
    _, _, z = prepare_batch(sample, include_coords=True, device=device)
    if z is None:
        return torch.tensor([1.0, 0.0, 0.0], device=device)
    coeff = z.squeeze().to(device)
    if coeff.ndim != 1:
        coeff = coeff.view(-1)
    if coeff.numel() < 3:
        pad = torch.zeros((3 - coeff.numel(),), device=coeff.device, dtype=coeff.dtype)
        coeff = torch.cat([coeff, pad], dim=0)
    return coeff


def _fine_tune_model_on_samples(
    model,
    train_samples: List[Dict],
    normalization_stats,
    equation: str,
    epochs: int,
    learning_rate: float,
    device: torch.device,
    optimizer_name: str = "adamw",
    weight_decay: float = 0.0,
    momentum: float = 0.0,
) -> None:
    """
    In-place fine-tune a phase2 model on provided samples for a fixed number of epochs.
    Uses the same objective as the main fine-tuning loop (mse_u + mse_f + mse_bd).
    """
    # Prepare training data dicts (reuse existing bridge logic)
    temp_splits = DatasetSplits(
        train_few=train_samples,
        interp=[],
        extrap=[],
        normalization_stats=normalization_stats,
    )
    processed_splits = convert_to_model_format(temp_splits, phase="phase2", device=device)
    train_data, train_data_f, train_data_bd = convert_dataset_to_training_format(
        processed_splits, phase="phase2", device=device
    )

    x_initial = Variable(torch.from_numpy(np.array(np.expand_dims(train_data["x_data"], 1))).float(), requires_grad=True).to(device)
    t_initial = Variable(torch.from_numpy(np.array(np.expand_dims(train_data["t_data"], 1))).float(), requires_grad=True).to(device)
    u_initial = Variable(torch.from_numpy(np.array(np.expand_dims(train_data["u_data"], 1))).float(), requires_grad=True).to(device)

    x_collocation = Variable(torch.from_numpy(np.array(np.expand_dims(train_data_f["x_data"], 1))).float(), requires_grad=True).to(device)
    t_collocation = Variable(torch.from_numpy(np.array(np.expand_dims(train_data_f["t_data"], 1))).float(), requires_grad=True).to(device)
    beta_collocation = Variable(torch.from_numpy(np.array(np.expand_dims(train_data_f["beta"], 1))).float(), requires_grad=True).to(device)
    nu_collocation = Variable(torch.from_numpy(np.array(np.expand_dims(train_data_f["nu"], 1))).float(), requires_grad=True).to(device)
    rho_collocation = Variable(torch.from_numpy(np.array(np.expand_dims(train_data_f["rho"], 1))).float(), requires_grad=True).to(device)
    all_zeros = torch.zeros((len(train_data_f["x_data"]), 1), device=device)

    x_lb = Variable(torch.from_numpy(np.array(np.expand_dims(train_data_bd["x_data_lb"], 1))).float(), requires_grad=True).to(device)
    t_lb = Variable(torch.from_numpy(np.array(np.expand_dims(train_data_bd["t_data_lb"], 1))).float(), requires_grad=True).to(device)
    x_ub = Variable(torch.from_numpy(np.array(np.expand_dims(train_data_bd["x_data_ub"], 1))).float(), requires_grad=True).to(device)
    t_ub = Variable(torch.from_numpy(np.array(np.expand_dims(train_data_bd["t_data_ub"], 1))).float(), requires_grad=True).to(device)

    optimizer = _make_optimizer(
        model,
        learning_rate=float(learning_rate),
        optimizer=optimizer_name,
        weight_decay=float(weight_decay),
        momentum=float(momentum),
    )
    mse_cost_function = nn.MSELoss()

    u_mean = float((normalization_stats or {}).get("u_mean", 0.0))
    u_std = float((normalization_stats or {}).get("u_std", 1.0))
    u_mean_t = torch.tensor(u_mean, device=device)
    u_std_t = torch.tensor(u_std, device=device)

    for _ in range(int(epochs)):
        model.train()
        optimizer.zero_grad()

        net_initial_out = model(x_initial, t_initial)
        # Data loss (physical units, to match train_full.py scaling)
        u_pred_phys = net_initial_out * u_std_t + u_mean_t
        u_true_phys = u_initial * u_std_t + u_mean_t
        mse_u = mse_cost_function(u_pred_phys, u_true_phys)

        # Equation-aware PDE residual (critical: utils.f_cal_phase2 is CDR-only)
        f_out = f_cal_phase2_equation_aware(
            x_collocation,
            t_collocation,
            beta_collocation,
            nu_collocation,
            rho_collocation,
            model,
            equation=equation,
            normalization_stats=normalization_stats,
        )
        mse_f = mse_cost_function(f_out, all_zeros)

        # Boundary loss
        if "helmholtz" in str(equation).lower():
            mse_bd = _helmholtz_dirichlet_bd_loss(
                u_pred_phys=u_pred_phys,
                u_true_phys=u_true_phys,
                x_coords=x_initial,
                y_coords=t_initial,
                mse_cost_function=mse_cost_function,
            )
        else:
            u_pred_lb = model(x_lb, t_lb)
            u_pred_ub = model(x_ub, t_ub)
            # Boundary loss (physical units)
            u_lb_phys = u_pred_lb * u_std_t + u_mean_t
            u_ub_phys = u_pred_ub * u_std_t + u_mean_t
            mse_bd = torch.mean((u_lb_phys - u_ub_phys) ** 2)

        loss = mse_u + mse_f + mse_bd
        loss.backward()
        optimizer.step()


def save_qualitative_grid_plot_per_task_adaptation(
    phase1_checkpoint: str,
    hidden_dim: int,
    split_data: List[Dict],
    split_name: str,
    equation: str,
    epoch: int,
    output_dir: str,
    device: torch.device,
    normalization_stats,
    learning_rate: float,
    optimizer_name: str = "adamw",
    weight_decay: float = 0.0,
    momentum: float = 0.0,
    num_samples: int = 5,
    sample_start_idx: int = 0,
) -> None:
    """
    Build *one adapted model per plotted sample*, each conditioned on that sample's PDE params,
    fine-tuned for `epoch` steps on that sample alone, then plot GT/Pred/|Err| for all samples
    in one figure. This matches "each sample optimized for its own PDE parameter".
    """
    if not split_data:
        return

    k = max(1, min(int(num_samples), len(split_data)))
    start = int(max(0, sample_start_idx))
    indices = [(start + i) % len(split_data) for i in range(k)]

    preds = []
    gts = []
    shapes = []

    for idx in indices:
        sample = split_data[idx]
        coeff = _extract_target_coeff_from_sample(sample, device=device)
        # Note: visualization doesn't use compile (not needed for plotting)
        model = create_phase2_model(phase1_checkpoint, hidden_dim, coeff, device, use_compile=False)

        # Fine-tune per task for `epoch` steps on that single sample
        if epoch > 0:
            _fine_tune_model_on_samples(
                model=model,
                train_samples=[sample],
                normalization_stats=normalization_stats,
                equation=equation,
                epochs=epoch,
                learning_rate=learning_rate,
                optimizer_name=optimizer_name,
                weight_decay=weight_decay,
                momentum=momentum,
                device=device,
            )

        # Predict on that sample for plotting
        with torch.no_grad():
            x, y, _ = prepare_batch(sample, include_coords=True, device=device)
            u_gt = y.squeeze().cpu().numpy()
            if normalization_stats:
                u_gt = u_gt * normalization_stats["u_std"] + normalization_stats["u_mean"]

            if x.shape[1] >= 2:
                x_coords = x[:, 0:1]
                t_coords = x[:, 1:2]
            else:
                x_coords = x[:, 0:1]
                t_coords = torch.zeros_like(x_coords)

            u_pred = model(x_coords, t_coords)
            u_pred_np = u_pred.squeeze().cpu().numpy()
            if normalization_stats:
                u_pred_np = u_pred_np * normalization_stats["u_std"] + normalization_stats["u_mean"]

        u_raw = sample.get("u")
        if isinstance(u_raw, torch.Tensor):
            u_shape = tuple(u_raw.shape)
        else:
            u_shape = tuple(np.array(u_raw).shape)

        preds.append(u_pred_np.reshape(u_shape))
        gts.append(u_gt.reshape(u_shape))
        shapes.append(u_shape)

    # Reuse the same renderer used for the fast path
    if len({s for s in shapes}) != 1:
        # If shapes differ, fallback to individual plots (still per-task)
        for i, idx in enumerate(indices):
            base = os.path.join(output_dir, f"{split_name}_epoch_{epoch}_task_{idx}")
            generate_visualizations(
                predictions=np.array([preds[i]]),
                ground_truth=np.array([gts[i]]),
                output_dir=output_dir,
                split_name=f"{split_name}_epoch_{epoch}_task_{idx}",
                sample_indices=[0],
                max_samples=1,
                coordinate_grids=None,
            )
            # Also save SVG by re-loading the png-based helper isn't trivial; keep fallback minimal.
        return

    # Render multi-sample grid and save png+svg
    gt0 = gts[0]
    ndim = gt0.ndim

    fig_h = 3.2 * k
    fig_w = 14
    fig, axes = plt.subplots(k, 3, figsize=(fig_w, fig_h))
    if k == 1:
        axes = np.expand_dims(axes, axis=0)

    for r in range(k):
        gt = gts[r]
        pred = preds[r]
        err = np.abs(pred - gt)

        if ndim == 1:
            xax = np.arange(gt.shape[0])
            axes[r, 0].plot(xax, gt, linewidth=1.5)
            axes[r, 0].set_title(f"GT (task {indices[r]})")
            axes[r, 1].plot(xax, pred, linewidth=1.5)
            axes[r, 1].set_title("Pred")
            axes[r, 2].plot(xax, err, linewidth=1.5)
            axes[r, 2].set_title("|Err|")
            for c in range(3):
                axes[r, c].grid(True, alpha=0.25)
        else:
            if ndim == 3:
                mid = gt.shape[0] // 2
                gt2 = gt[mid, :, :]
                pred2 = pred[mid, :, :]
                err2 = err[mid, :, :]
            else:
                gt2, pred2, err2 = gt, pred, err

            vmin = min(float(np.min(gt2)), float(np.min(pred2)))
            vmax = max(float(np.max(gt2)), float(np.max(pred2)))

            im0 = _plot_contourf_with_contours(axes[r, 0], gt2, cmap="viridis", vmin=vmin, vmax=vmax)
            axes[r, 0].set_title(f"GT (task {indices[r]})")
            fig.colorbar(im0, ax=axes[r, 0], fraction=0.046, pad=0.04)

            im1 = _plot_contourf_with_contours(axes[r, 1], pred2, cmap="viridis", vmin=vmin, vmax=vmax)
            axes[r, 1].set_title("Pred")
            fig.colorbar(im1, ax=axes[r, 1], fraction=0.046, pad=0.04)

            im2 = _plot_contourf_with_contours(axes[r, 2], err2, cmap="hot")
            axes[r, 2].set_title("|Err|")
            fig.colorbar(im2, ax=axes[r, 2], fraction=0.046, pad=0.04)

            for c in range(3):
                axes[r, c].set_xticks([])
                axes[r, c].set_yticks([])

    fig.suptitle(f"{split_name} @ epoch {epoch} (per-task adapted, k={k})", fontsize=14)
    fig.tight_layout(rect=[0, 0.0, 1, 0.98])

    os.makedirs(output_dir, exist_ok=True)
    base = os.path.join(output_dir, f"{split_name}_epoch_{epoch}_per_task_samples_{k}")
    fig.savefig(base + ".png", dpi=200, bbox_inches="tight")
    fig.savefig(base + ".svg", bbox_inches="tight")
    plt.close(fig)


def fine_tune_and_evaluate(
    model,
    train_split,
    eval_split,
    eval_split_name: str,
    equation: str,
    epochs: int,
    learning_rate: float,
    device: torch.device,
    optimizer_name: str = "adamw",
    weight_decay: float = 0.0,
    momentum: float = 0.0,
    normalization_stats=None,
    checkpoint_epochs: List[int] | None = None,
    viz_output_dir: str = None,
    viz_sample_idx: int = 0,
    viz_num_samples: int = 5,
    save_plots: bool = True,
    viz_per_task_adaptation: bool = False,
    phase1_checkpoint: str = None,
    hidden_dim: int = None,
    viz_learning_rate: float = None,
) -> Dict[int, Tuple[float, float]]:
    """
    Fine-tune model and evaluate at specified checkpoints.
    
    Args:
        model: Model to fine-tune
        train_split: Training data split (list of samples)
        eval_split: Evaluation data split (list of samples)
        eval_split_name: Name of evaluation split ('interp' or 'extrap')
        epochs: Total number of epochs
        learning_rate: Learning rate
        device: Device to use
        normalization_stats: Normalization statistics
        checkpoint_epochs: List of epochs at which to evaluate
        
    Returns:
        Dictionary mapping epoch -> (l2_error, elapsed_time)
    """
    checkpoint_epochs = [] if checkpoint_epochs is None else list(checkpoint_epochs)
    results = {}
    
    # Prepare training data - convert list to DatasetSplits-like structure
    from src.data_loader import DatasetSplits
    temp_splits = DatasetSplits(
        train_few=train_split,
        interp=[],
        extrap=[],
        normalization_stats=normalization_stats
    )
    processed_splits = convert_to_model_format(temp_splits, phase='phase2', device=device)
    train_data, train_data_f, train_data_bd = convert_dataset_to_training_format(
        processed_splits, phase='phase2', device=device
    )
    
    # Prepare data tensors
    x_initial = Variable(
        torch.from_numpy(np.array(np.expand_dims(train_data['x_data'], 1))).float(),
        requires_grad=True
    ).to(device)
    t_initial = Variable(
        torch.from_numpy(np.array(np.expand_dims(train_data['t_data'], 1))).float(),
        requires_grad=True
    ).to(device)
    u_initial = Variable(
        torch.from_numpy(np.array(np.expand_dims(train_data['u_data'], 1))).float(),
        requires_grad=True
    ).to(device)
    
    x_collocation = Variable(
        torch.from_numpy(np.array(np.expand_dims(train_data_f['x_data'], 1))).float(),
        requires_grad=True
    ).to(device)
    t_collocation = Variable(
        torch.from_numpy(np.array(np.expand_dims(train_data_f['t_data'], 1))).float(),
        requires_grad=True
    ).to(device)
    beta_collocation = Variable(
        torch.from_numpy(np.array(np.expand_dims(train_data_f['beta'], 1))).float(),
        requires_grad=True
    ).to(device)
    nu_collocation = Variable(
        torch.from_numpy(np.array(np.expand_dims(train_data_f['nu'], 1))).float(),
        requires_grad=True
    ).to(device)
    rho_collocation = Variable(
        torch.from_numpy(np.array(np.expand_dims(train_data_f['rho'], 1))).float(),
        requires_grad=True
    ).to(device)
    
    all_zeros = torch.zeros((len(train_data_f['x_data']), 1), device=device)
    
    x_lb = Variable(
        torch.from_numpy(np.array(np.expand_dims(train_data_bd['x_data_lb'], 1))).float(),
        requires_grad=True
    ).to(device)
    t_lb = Variable(
        torch.from_numpy(np.array(np.expand_dims(train_data_bd['t_data_lb'], 1))).float(),
        requires_grad=True
    ).to(device)
    x_ub = Variable(
        torch.from_numpy(np.array(np.expand_dims(train_data_bd['x_data_ub'], 1))).float(),
        requires_grad=True
    ).to(device)
    t_ub = Variable(
        torch.from_numpy(np.array(np.expand_dims(train_data_bd['t_data_ub'], 1))).float(),
        requires_grad=True
    ).to(device)
    
    # Setup optimizer
    optimizer = _make_optimizer(
        model,
        learning_rate=float(learning_rate),
        optimizer=optimizer_name,
        weight_decay=float(weight_decay),
        momentum=float(momentum),
    )
    mse_cost_function = nn.MSELoss()

    u_mean = float((normalization_stats or {}).get("u_mean", 0.0))
    u_std = float((normalization_stats or {}).get("u_std", 1.0))
    u_mean_t = torch.tensor(u_mean, device=device)
    u_std_t = torch.tensor(u_std, device=device)
    
    # Evaluate zero-shot (epoch 0)
    start_time = time.time()
    l2_mean, l2_std = evaluate_model(model, eval_split, device, normalization_stats)
    elapsed_time = time.time() - start_time
    results[0] = (l2_mean, l2_std, elapsed_time)
    print(f"  Epoch 0 (Zero-shot) | L2 Error: {l2_mean:.6e} ± {l2_std:.6e} | Time: {elapsed_time:.2f}s")
    if save_plots and viz_output_dir:
        if viz_per_task_adaptation:
            if phase1_checkpoint is None or hidden_dim is None:
                raise ValueError("viz_per_task_adaptation=True requires phase1_checkpoint and hidden_dim")
            save_qualitative_grid_plot_per_task_adaptation(
                phase1_checkpoint=phase1_checkpoint,
                hidden_dim=hidden_dim,
                split_data=eval_split,
                split_name=eval_split_name,
                equation=equation,
                epoch=0,
                output_dir=viz_output_dir,
                device=device,
                normalization_stats=normalization_stats,
                learning_rate=float(viz_learning_rate if viz_learning_rate is not None else learning_rate),
                num_samples=viz_num_samples,
                sample_start_idx=viz_sample_idx,
            )
        else:
            save_qualitative_grid_plot(
                model=model,
                split_data=eval_split,
                split_name=eval_split_name,
                epoch=0,
                output_dir=viz_output_dir,
                device=device,
                normalization_stats=normalization_stats,
                num_samples=viz_num_samples,
                sample_start_idx=viz_sample_idx,
            )
    
    # Training loop
    training_start_time = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        net_initial_out = model(x_initial, t_initial)
        # Data loss (physical units)
        u_pred_phys = net_initial_out * u_std_t + u_mean_t
        u_true_phys = u_initial * u_std_t + u_mean_t
        mse_u = mse_cost_function(u_pred_phys, u_true_phys)
        
        # PDE residual loss (equation-aware)
        f_out = f_cal_phase2_equation_aware(
            x_collocation,
            t_collocation,
            beta_collocation,
            nu_collocation,
            rho_collocation,
            model,
            equation=equation,
            normalization_stats=normalization_stats,
        )
        mse_f = mse_cost_function(f_out, all_zeros)
        
        # Boundary loss
        if "helmholtz" in str(equation).lower():
            mse_bd = _helmholtz_dirichlet_bd_loss(
                u_pred_phys=u_pred_phys,
                u_true_phys=u_true_phys,
                x_coords=x_initial,
                y_coords=t_initial,
                mse_cost_function=mse_cost_function,
            )
        else:
            u_pred_lb = model(x_lb, t_lb)
            u_pred_ub = model(x_ub, t_ub)
            # Boundary loss (physical units)
            u_lb_phys = u_pred_lb * u_std_t + u_mean_t
            u_ub_phys = u_pred_ub * u_std_t + u_mean_t
            mse_bd = torch.mean((u_lb_phys - u_ub_phys) ** 2)
        
        # Total loss
        loss = mse_u + mse_f + mse_bd
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Evaluate at checkpoints
        if ep in checkpoint_epochs:
            eval_start_time = time.time()
            l2_mean, l2_std = evaluate_model(model, eval_split, device, normalization_stats)
            elapsed_time = time.time() - eval_start_time
            total_elapsed = time.time() - training_start_time
            results[ep] = (l2_mean, l2_std, elapsed_time)
            print(f"  Epoch {ep} | L2 Error: {l2_mean:.6e} ± {l2_std:.6e} | Eval Time: {elapsed_time:.2f}s | Total Time: {total_elapsed:.2f}s")
            if save_plots and viz_output_dir:
                if viz_per_task_adaptation:
                    if phase1_checkpoint is None or hidden_dim is None:
                        raise ValueError("viz_per_task_adaptation=True requires phase1_checkpoint and hidden_dim")
                    save_qualitative_grid_plot_per_task_adaptation(
                        phase1_checkpoint=phase1_checkpoint,
                        hidden_dim=hidden_dim,
                        split_data=eval_split,
                        split_name=eval_split_name,
                        equation=equation,
                        epoch=ep,
                        output_dir=viz_output_dir,
                        device=device,
                        normalization_stats=normalization_stats,
                        learning_rate=float(viz_learning_rate if viz_learning_rate is not None else learning_rate),
                        num_samples=viz_num_samples,
                        sample_start_idx=viz_sample_idx,
                    )
                else:
                    save_qualitative_grid_plot(
                        model=model,
                        split_data=eval_split,
                        split_name=eval_split_name,
                        epoch=ep,
                        output_dir=viz_output_dir,
                        device=device,
                        normalization_stats=normalization_stats,
                        num_samples=viz_num_samples,
                        sample_start_idx=viz_sample_idx,
                    )
    
    return results


def fine_tune_and_evaluate_per_task(
    *,
    phase1_checkpoint: str,
    hidden_dim: int,
    split_data: List[Dict],
    split_name: str,
    equation: str,
    epochs: int,
    learning_rate: float,
    device: torch.device,
    optimizer_name: str = "adamw",
    weight_decay: float = 0.0,
    momentum: float = 0.0,
    normalization_stats,
    checkpoint_epochs: List[int],
    max_tasks: int = 0,
    task_start_idx: int = 0,
    viz_output_dir: str | None = None,
    viz_num_samples: int = 5,
    save_plots: bool = True,
    viz_learning_rate: float | None = None,
) -> Dict[int, Tuple[float, float, float]]:
    """
    Train/evaluate in a way that mirrors legacy `train_full.py`:

    - One *task* == one PDE parameter vector (one dataset sample here).
    - For each task, build a phase2 model conditioned on that task's params.
    - Fine-tune that model on that task alone for `epochs` steps.
    - Report mean±std relative L2 across tasks at checkpoints.
    """
    if not split_data:
        return {e: (float("nan"), float("nan"), 0.0) for e in checkpoint_epochs}

    n_total = len(split_data)
    start = int(max(0, task_start_idx))
    k = n_total if int(max_tasks) <= 0 else min(int(max_tasks), n_total)
    indices = [(start + i) % n_total for i in range(k)]

    # Prepare aggregator: epoch -> list of rel l2 across tasks
    per_epoch_rel_l2: Dict[int, List[float]] = {int(e): [] for e in checkpoint_epochs}
    per_epoch_time: Dict[int, float] = {int(e): 0.0 for e in checkpoint_epochs}

    # If requested, also emit per-task qualitative grids (this already matches per-task adaptation)
    if save_plots and viz_output_dir:
        for e in checkpoint_epochs:
            save_qualitative_grid_plot_per_task_adaptation(
                phase1_checkpoint=phase1_checkpoint,
                hidden_dim=hidden_dim,
                split_data=[split_data[i] for i in indices],
                split_name=split_name,
                equation=equation,
                epoch=int(e),
                output_dir=viz_output_dir,
                device=device,
                normalization_stats=normalization_stats,
                learning_rate=float(viz_learning_rate if viz_learning_rate is not None else learning_rate),
                optimizer_name=str(optimizer_name),
                weight_decay=float(weight_decay),
                momentum=float(momentum),
                num_samples=min(int(viz_num_samples), len(indices)),
                sample_start_idx=0,
            )

    for idx in indices:
        sample = split_data[idx]
        coeff = _extract_target_coeff_from_sample(sample, device=device)
        # Note: per-task adaptation doesn't use compile for now (can be added if needed)
        model = create_phase2_model(phase1_checkpoint, hidden_dim, coeff, device, use_compile=False)

        # Zero-shot
        if 0 in per_epoch_rel_l2:
            t0 = time.time()
            l2_mean, _l2_std = evaluate_model(model, [sample], device, normalization_stats)
            per_epoch_time[0] += time.time() - t0
            per_epoch_rel_l2[0].append(float(l2_mean))

        # Train once up to max epochs, record checkpoints
        # (more efficient than re-training from scratch at each checkpoint)
        max_ep = int(max(checkpoint_epochs)) if checkpoint_epochs else int(epochs)
        max_ep = min(int(epochs), max_ep)

        # Use the same objective as `_fine_tune_model_on_samples`, but capture metrics as we go.
        # Reuse that helper to keep behavior consistent.
        for ep in range(1, max_ep + 1):
            _fine_tune_model_on_samples(
                model=model,
                train_samples=[sample],
                normalization_stats=normalization_stats,
                equation=equation,
                epochs=1,
                learning_rate=learning_rate,
                optimizer_name=optimizer_name,
                weight_decay=weight_decay,
                momentum=momentum,
                device=device,
            )

            if ep in per_epoch_rel_l2:
                t1 = time.time()
                l2_mean, _l2_std = evaluate_model(model, [sample], device, normalization_stats)
                per_epoch_time[ep] += time.time() - t1
                per_epoch_rel_l2[ep].append(float(l2_mean))

    # Collapse per-epoch lists into mean±std
    results: Dict[int, Tuple[float, float, float]] = {}
    for ep in checkpoint_epochs:
        vals = per_epoch_rel_l2.get(int(ep), [])
        if not vals:
            results[int(ep)] = (float("nan"), float("nan"), float(per_epoch_time.get(int(ep), 0.0)))
        else:
            mean = float(np.mean(vals))
            std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            results[int(ep)] = (mean, std, float(per_epoch_time.get(int(ep), 0.0)))
    return results


def generate_summary_table(
    interp_results: Dict[int, Tuple[float, float]],
    extrap_results: Dict[int, Tuple[float, float]],
    output_path: str
):
    """
    Generate a summary table with fine-tuning results in the format:
    Model | Zero-shot (L2 Error) | ACC Time | FT after (50 epochs) | ACC Time | ...
    
    Args:
        interp_results: Dictionary mapping epoch -> (l2_error, time) for interpolation
        extrap_results: Dictionary mapping epoch -> (l2_error, time) for extrapolation
        output_path: Path to save the table
    """
    # Get all epochs
    all_epochs = sorted(set(list(interp_results.keys()) + list(extrap_results.keys())))
    
    # Build table data in the requested format
    table_data = []
    
    # Interpolation row
    interp_row = {'Split': 'Interpolation'}
    for epoch in all_epochs:
        if epoch == 0:
            if epoch in interp_results:
                l2_mean, l2_std, eval_time = interp_results[epoch]
                interp_row['Zero-shot_L2_Error'] = f"{l2_mean:.6e} ± {l2_std:.6e}"
                interp_row['Zero-shot_Time'] = f"{eval_time:.2f}"
            else:
                interp_row['Zero-shot_L2_Error'] = "N/A"
                interp_row['Zero-shot_Time'] = "N/A"
        else:
            if epoch in interp_results:
                l2_mean, l2_std, eval_time = interp_results[epoch]
                interp_row[f'FT_{epoch}_epochs_L2_Error'] = f"{l2_mean:.6e} ± {l2_std:.6e}"
                interp_row[f'FT_{epoch}_epochs_Time'] = f"{eval_time:.2f}"
            else:
                interp_row[f'FT_{epoch}_epochs_L2_Error'] = "N/A"
                interp_row[f'FT_{epoch}_epochs_Time'] = "N/A"
    table_data.append(interp_row)
    
    # Extrapolation row
    extrap_row = {'Split': 'Extrapolation'}
    for epoch in all_epochs:
        if epoch == 0:
            if epoch in extrap_results:
                l2_mean, l2_std, eval_time = extrap_results[epoch]
                extrap_row['Zero-shot_L2_Error'] = f"{l2_mean:.6e} ± {l2_std:.6e}"
                extrap_row['Zero-shot_Time'] = f"{eval_time:.2f}"
            else:
                extrap_row['Zero-shot_L2_Error'] = "N/A"
                extrap_row['Zero-shot_Time'] = "N/A"
        else:
            if epoch in extrap_results:
                l2_mean, l2_std, eval_time = extrap_results[epoch]
                extrap_row[f'FT_{epoch}_epochs_L2_Error'] = f"{l2_mean:.6e} ± {l2_std:.6e}"
                extrap_row[f'FT_{epoch}_epochs_Time'] = f"{eval_time:.2f}"
            else:
                extrap_row[f'FT_{epoch}_epochs_L2_Error'] = "N/A"
                extrap_row[f'FT_{epoch}_epochs_Time'] = "N/A"
    table_data.append(extrap_row)
    
    # Create DataFrame
    df = pd.DataFrame(table_data)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"\nSummary table saved to {output_path}")
    
    # Print table
    print("\n" + "=" * 100)
    print("Fine-tuning Evaluation Summary")
    print("=" * 100)
    print(df.to_string(index=False))
    print("=" * 100)
    
    # Also create a more compact format matching the user's request
    compact_data = []
    for epoch in all_epochs:
        row = {'Epoch': epoch}
        if epoch == 0:
            row['Stage'] = 'Zero-shot'
        else:
            row['Stage'] = f'FT after {epoch} epochs'
        
        if epoch in interp_results:
            l2_mean, l2_std, eval_time = interp_results[epoch]
            row['Interp_L2_Error'] = f"{l2_mean:.6e} ± {l2_std:.6e}"
            row['Interp_Time'] = f"{eval_time:.2f}"
        else:
            row['Interp_L2_Error'] = "N/A"
            row['Interp_Time'] = "N/A"
        
        if epoch in extrap_results:
            l2_mean, l2_std, eval_time = extrap_results[epoch]
            row['Extrap_L2_Error'] = f"{l2_mean:.6e} ± {l2_std:.6e}"
            row['Extrap_Time'] = f"{eval_time:.2f}"
        else:
            row['Extrap_L2_Error'] = "N/A"
            row['Extrap_Time'] = "N/A"
        
        compact_data.append(row)
    
    compact_df = pd.DataFrame(compact_data)
    compact_path = output_path.replace('.csv', '_compact.csv')
    compact_df.to_csv(compact_path, index=False)
    print(f"\nCompact summary table saved to {compact_path}")
    print("\n" + "=" * 100)
    print("Compact Format:")
    print("=" * 100)
    print(compact_df.to_string(index=False))
    print("=" * 100)


def generate_model_comparison_table(
    equation: str,
    interp_results: Dict[int, Tuple[float, float, float]],
    extrap_results: Dict[int, Tuple[float, float, float]],
    output_path: str
):
    """
    Generate a model comparison summary table in the format:
    Equation | Model | Zero-shot (L2 Error) | ACC Time (ms) | FT after 50 epochs | ACC Time | ...
    
    Args:
        equation: Equation name (e.g., "Allen Cahn")
        interp_results: Dictionary mapping epoch -> (l2_mean, l2_std, time) for interpolation
        extrap_results: Dictionary mapping epoch -> (l2_mean, l2_std, time) for extrapolation
        output_path: Path to save the table
    """
    # Get all epochs (excluding 0, which is zero-shot)
    all_epochs = sorted([e for e in set(list(interp_results.keys()) + list(extrap_results.keys())) if e > 0])
    
    # Get zero-shot results (average of interp and extrap)
    zero_shot_interp = interp_results.get(0, (float('nan'), float('nan'), 0.0))
    zero_shot_extrap = extrap_results.get(0, (float('nan'), float('nan'), 0.0))
    
    # Average zero-shot L2 error (mean of interp and extrap)
    if not (np.isnan(zero_shot_interp[0]) or np.isnan(zero_shot_extrap[0])):
        zero_shot_l2_mean = (zero_shot_interp[0] + zero_shot_extrap[0]) / 2.0
        zero_shot_l2_std = np.sqrt((zero_shot_interp[1]**2 + zero_shot_extrap[1]**2) / 2.0)
    else:
        zero_shot_l2_mean = float('nan')
        zero_shot_l2_std = float('nan')
    
    # Average zero-shot time (sum of interp and extrap times)
    zero_shot_time = zero_shot_interp[2] + zero_shot_extrap[2]
    
    # Build table row
    row_data = {
        'Equation': equation.replace('_', ' ').title(),
        'Model': 'Hyper-LR-PINN',
        'Zero-shot (L2 Error)': f"{zero_shot_l2_mean:.4f} ± {zero_shot_l2_std:.2f}" if not np.isnan(zero_shot_l2_mean) else "N/A",
        'ACC Time (ms)': f"{zero_shot_time * 1000:.2f}" if zero_shot_time > 0 else "N/A"
    }
    
    # Add fine-tuning results for each epoch
    for epoch in all_epochs:
        ft_interp = interp_results.get(epoch, (float('nan'), float('nan'), 0.0))
        ft_extrap = extrap_results.get(epoch, (float('nan'), float('nan'), 0.0))
        
        # Average FT L2 error
        if not (np.isnan(ft_interp[0]) or np.isnan(ft_extrap[0])):
            ft_l2_mean = (ft_interp[0] + ft_extrap[0]) / 2.0
            ft_l2_std = np.sqrt((ft_interp[1]**2 + ft_extrap[1]**2) / 2.0)
            row_data[f'FT after {epoch} epochs'] = f"{ft_l2_mean:.4f} ± {ft_l2_std:.2f}"
        else:
            row_data[f'FT after {epoch} epochs'] = "N/A"
        
        # FT time (sum of interp and extrap)
        ft_time = ft_interp[2] + ft_extrap[2]
        row_data[f'ACC Time ({epoch} epochs)'] = f"{ft_time * 1000:.2f}" if ft_time > 0 else "N/A"
    
    # Create DataFrame
    df = pd.DataFrame([row_data])
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"\nModel comparison table saved to {output_path}")
    
    # Print formatted table
    # Calculate total width based on columns
    col_widths = {
        'Equation': 20,
        'Model': 15,
        'Zero-shot (L2 Error)': 25,
        'ACC Time (ms)': 15,
        'FT epochs': 25,
        'ACC Time': 15
    }
    total_width = col_widths['Equation'] + col_widths['Model'] + col_widths['Zero-shot (L2 Error)'] + col_widths['ACC Time (ms)']
    total_width += len(all_epochs) * (col_widths['FT epochs'] + col_widths['ACC Time'])
    
    print("\n" + "=" * total_width)
    print("Model Comparison Summary")
    print("=" * total_width)
    
    # Build header dynamically
    header_parts = [
        f"{'Equation':<{col_widths['Equation']}}",
        f"{'Model':<{col_widths['Model']}}",
        f"{'Zero-shot (L2 Error)':<{col_widths['Zero-shot (L2 Error)']}}",
        f"{'ACC Time (ms)':<{col_widths['ACC Time (ms)']}}"
    ]
    for epoch in all_epochs:
        header_parts.append(f"{'FT after ' + str(epoch) + ' epochs':<{col_widths['FT epochs']}}")
        header_parts.append(f"{'ACC Time':<{col_widths['ACC Time']}}")
    header = "".join(header_parts)
    print(header)
    print("-" * total_width)
    
    # Build data row
    row_parts = [
        f"{row_data['Equation']:<{col_widths['Equation']}}",
        f"{row_data['Model']:<{col_widths['Model']}}",
        f"{row_data['Zero-shot (L2 Error)']:<{col_widths['Zero-shot (L2 Error)']}}",
        f"{row_data['ACC Time (ms)']:<{col_widths['ACC Time (ms)']}}"
    ]
    for epoch in all_epochs:
        row_parts.append(f"{row_data.get(f'FT after {epoch} epochs', 'N/A'):<{col_widths['FT epochs']}}")
        row_parts.append(f"{row_data.get(f'ACC Time ({epoch} epochs)', 'N/A'):<{col_widths['ACC Time']}}")
    row_str = "".join(row_parts)
    print(row_str)
    print("=" * total_width)


def main():
    parser = argparse.ArgumentParser(description='Fine-tuning evaluation with checkpoint tracking')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to YAML configuration file')
    parser.add_argument('--phase1_checkpoint', type=str, required=True,
                       help='Path to Phase 1 checkpoint')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (overrides config)')
    parser.add_argument('--checkpoint_epochs', type=int, nargs='+', 
                       default=[50, 250, 500, 750, 1000],
                       help='Epochs at which to evaluate (default: 50 250 500 750 1000)')
    parser.add_argument('--max_epochs', type=int, default=1000,
                       help='Maximum number of fine-tuning epochs (default: 1000)')
    parser.add_argument('--viz_sample_idx', type=int, default=0,
                       help='Sample index to visualize for each split (default: 0)')
    parser.add_argument('--viz_num_samples', type=int, default=5,
                       help='Number of samples per figure (default: 5)')
    parser.add_argument('--viz_per_task_adaptation', action='store_true',
                       help='If set, each plotted sample gets its own phase2 model conditioned on its PDE params and fine-tuned for the shown epoch (slower).')
    parser.add_argument('--per_task_adaptation', action='store_true',
                       help='If set, metrics are computed like train_full.py: one phase2 model per task (parameter), fine-tuned per task and averaged across tasks.')
    parser.add_argument('--per_task_max_tasks', type=int, default=0,
                       help='Optional cap on number of tasks used in per-task adaptation (0 = all).')
    parser.add_argument('--per_task_start_idx', type=int, default=0,
                       help='Start index for selecting tasks in per-task adaptation (useful for sharding).')
    parser.add_argument('--viz_learning_rate', type=float, default=None,
                       help='Optional override learning rate used for per-task visualization adaptation.')
    parser.add_argument('--no_save_plots', action='store_true',
                       help='Disable saving qualitative plots (GT / Pred / |Err|)')
    
    args = parser.parse_args()
    
    # Load and validate configuration
    config = load_config(args.config)
    validate_config(config)
    
    # Override device if specified
    if args.device is not None:
        config['device'] = args.device
    
    device = torch.device(config.get('device', 'cuda:0'))
    seed = config.get('seed', 42)
    
    # Setup
    setup_seed(seed)
    
    print("=" * 80)
    print("Hyper-LR-PINN Fine-tuning Evaluation")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Phase 1 Checkpoint: {args.phase1_checkpoint}")
    print(f"Configuration: {args.config}")
    print(f"Checkpoint Epochs: {args.checkpoint_epochs}")
    print(f"Max Epochs: {args.max_epochs}")
    print("=" * 80)
    
    # Load dataset
    dataset_config = config['dataset']
    splits = load_parametric_dataset(
        equation=dataset_config['equation'],
        data_dir=dataset_config['data_dir'],
        seed=dataset_config.get('seed', 42),
        n_train=dataset_config.get('n_train', 10),
        cache=dataset_config.get('cache', True),
        balance=dataset_config.get('balance', False),
        n_each=dataset_config.get('n_each', None),
        balance_strategy=dataset_config.get('balance_strategy', 'random'),
        diversify=dataset_config.get('diversify', False)
    )

    # IMPORTANT: work with normalized samples (model space) throughout evaluation/visualization,
    # then denormalize using `normalization_stats` when computing metrics/plots.
    # Otherwise we'd denormalize raw physical targets and distort magnitudes.
    processed_splits = convert_to_model_format(splits, phase='phase2', device=device)
    splits_train = processed_splits.get("train_few", splits.train_few)
    splits_interp = processed_splits.get("interp", splits.interp)
    splits_extrap = processed_splits.get("extrap", splits.extrap)

    normalization_stats = splits.normalization_stats
    
    # Get model configuration
    model_config = config['model']
    hidden_dim = model_config['hidden_dim']
    
    # Get target parameters from first training sample
    first_sample = splits_train[0] if splits_train else splits.train_few[0]
    _, _, z = prepare_batch(first_sample, include_coords=True, device=device)
    
    if z is not None:
        target_coeff = z.squeeze().to(device)
        if target_coeff.ndim != 1:
            target_coeff = target_coeff.view(-1)
        if target_coeff.numel() < 3:
            pad = torch.zeros(
                (3 - target_coeff.numel(),),
                device=target_coeff.device,
                dtype=target_coeff.dtype,
            )
            target_coeff = torch.cat([target_coeff, pad], dim=0)
    else:
        target_coeff = torch.tensor([1.0, 0.0, 0.0], device=device)
    
    # Training configuration
    training_config = config['training']
    learning_rate = training_config.get('learning_rate', 0.00025)
    optimizer_name = training_config.get("optimizer", "adamw")
    weight_decay = float(training_config.get("weight_decay", 0.0))
    momentum = float(training_config.get("momentum", 0.0))
    
    # Ensure checkpoint_epochs includes 0 and doesn't exceed max_epochs
    checkpoint_epochs = sorted(set([0] + [e for e in args.checkpoint_epochs if e <= args.max_epochs]))
    
    # Fine-tune and evaluate on interpolation split
    print("\n" + "=" * 80)
    print("Fine-tuning on Interpolation Split")
    print("=" * 80)
    equation = dataset_config['equation']
    coeff_interp = _extract_target_coeff_from_sample(splits_interp[0], device=device) if splits_interp else target_coeff
    # Check if torch.compile should be used (can be enabled via config or env var)
    use_compile = config.get('use_torch_compile', False) or os.getenv('USE_TORCH_COMPILE', '0') == '1'
    model_interp = create_phase2_model(args.phase1_checkpoint, hidden_dim, coeff_interp, device, use_compile=use_compile)
    export_config = config['export']
    output_dir = export_config['output_dir']
    viz_dir = os.path.join(output_dir, f"{equation}_phase2", "evaluation", "visualizations")
    if args.per_task_adaptation:
        interp_results = fine_tune_and_evaluate_per_task(
            phase1_checkpoint=args.phase1_checkpoint,
            hidden_dim=hidden_dim,
            split_data=splits_interp,
            split_name="interp",
            equation=equation,
            epochs=args.max_epochs,
            learning_rate=learning_rate,
            optimizer_name=optimizer_name,
            weight_decay=weight_decay,
            momentum=momentum,
            device=device,
            normalization_stats=normalization_stats,
            checkpoint_epochs=checkpoint_epochs,
            max_tasks=int(args.per_task_max_tasks),
            task_start_idx=int(args.per_task_start_idx),
            viz_output_dir=viz_dir,
            viz_num_samples=args.viz_num_samples,
            save_plots=(export_config.get('save_plots', True) and (not args.no_save_plots)),
            viz_learning_rate=args.viz_learning_rate,
        )
    else:
        interp_results = fine_tune_and_evaluate(
            model_interp,
            splits_interp,
            splits_interp,
            'interp',
            equation,
            args.max_epochs,
            learning_rate,
            device,
            optimizer_name=optimizer_name,
            weight_decay=weight_decay,
            momentum=momentum,
            normalization_stats=normalization_stats,
            checkpoint_epochs=checkpoint_epochs,
            viz_output_dir=viz_dir,
            viz_sample_idx=args.viz_sample_idx,
            viz_num_samples=args.viz_num_samples,
            save_plots=(export_config.get('save_plots', True) and (not args.no_save_plots)),
            viz_per_task_adaptation=args.viz_per_task_adaptation,
            phase1_checkpoint=args.phase1_checkpoint,
            hidden_dim=hidden_dim,
            viz_learning_rate=args.viz_learning_rate,
        )
    
    # Fine-tune and evaluate on extrapolation split
    print("\n" + "=" * 80)
    print("Fine-tuning on Extrapolation Split")
    print("=" * 80)
    coeff_extrap = _extract_target_coeff_from_sample(splits_extrap[0], device=device) if splits_extrap else target_coeff
    model_extrap = create_phase2_model(args.phase1_checkpoint, hidden_dim, coeff_extrap, device, use_compile=use_compile)
    if args.per_task_adaptation:
        extrap_results = fine_tune_and_evaluate_per_task(
            phase1_checkpoint=args.phase1_checkpoint,
            hidden_dim=hidden_dim,
            split_data=splits_extrap,
            split_name="extrap",
            equation=equation,
            epochs=args.max_epochs,
            learning_rate=learning_rate,
            optimizer_name=optimizer_name,
            weight_decay=weight_decay,
            momentum=momentum,
            device=device,
            normalization_stats=normalization_stats,
            checkpoint_epochs=checkpoint_epochs,
            max_tasks=int(args.per_task_max_tasks),
            task_start_idx=int(args.per_task_start_idx),
            viz_output_dir=viz_dir,
            viz_num_samples=args.viz_num_samples,
            save_plots=(export_config.get('save_plots', True) and (not args.no_save_plots)),
            viz_learning_rate=args.viz_learning_rate,
        )
    else:
        extrap_results = fine_tune_and_evaluate(
            model_extrap,
            splits_extrap,
            splits_extrap,
            'extrap',
            equation,
            args.max_epochs,
            learning_rate,
            device,
            optimizer_name=optimizer_name,
            weight_decay=weight_decay,
            momentum=momentum,
            normalization_stats=normalization_stats,
            checkpoint_epochs=checkpoint_epochs,
            viz_output_dir=viz_dir,
            viz_sample_idx=args.viz_sample_idx,
            viz_num_samples=args.viz_num_samples,
            save_plots=(export_config.get('save_plots', True) and (not args.no_save_plots)),
            viz_per_task_adaptation=args.viz_per_task_adaptation,
            phase1_checkpoint=args.phase1_checkpoint,
            hidden_dim=hidden_dim,
            viz_learning_rate=args.viz_learning_rate,
        )
    
    # Generate summary table
    summary_dir = os.path.join(output_dir, f"{equation}_phase2", "evaluation")
    os.makedirs(summary_dir, exist_ok=True)
    
    summary_path = os.path.join(summary_dir, "finetune_summary.csv")
    generate_summary_table(interp_results, extrap_results, summary_path)
    
    # Generate model comparison table
    comparison_path = os.path.join(summary_dir, "model_comparison_summary.csv")
    generate_model_comparison_table(equation, interp_results, extrap_results, comparison_path)
    
    print("\n" + "=" * 80)
    print("Fine-tuning evaluation completed!")
    print("=" * 80)


if __name__ == '__main__':
    main()

