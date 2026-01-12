#!/usr/bin/env python
"""
Training script for Hyper-LR-PINN with parametric dataset integration.

This script integrates eff-physics-learn-dataset for data loading and
metrics-structures for metrics export.
"""

import argparse
import os
import sys
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import math
import random
import torch.backends.cudnn as cudnn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from typing import Dict, List, Optional, Sequence, Tuple

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model import LR_PINN_phase1, LR_PINN_phase2
from utils import orthogonality_reg, get_params
from src.config_loader import load_config, validate_config
from src.pinn_losses import get_pinn_loss_pytorch, f_cal_equation_aware, f_cal_phase2_equation_aware
from src.data_loader import (
    load_parametric_dataset,
    normalize_data,
    prepare_batch,
    convert_to_model_format
)
from src.metrics_exporter import (
    create_run_data,
    add_timing_metrics,
    export_metrics
)
from sklearn.metrics import explained_variance_score, max_error


def setup_seed(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


def _make_optimizer(net: torch.nn.Module, training_config: Dict) -> torch.optim.Optimizer:
    """Create optimizer from config."""
    lr = float(training_config["learning_rate"])
    opt_name = str(training_config.get("optimizer", "adam")).lower()
    weight_decay = float(training_config.get("weight_decay", 0.0))

    if opt_name == "adam":
        return torch.optim.Adam(net.parameters(), lr=lr)
    if opt_name == "adamw":
        return torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    if opt_name == "sgd":
        momentum = float(training_config.get("momentum", 0.0))
        return torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    raise ValueError(f"Unsupported optimizer: {opt_name}")


def _maybe_apply_mplstyle(mplstyle: Optional[str]) -> None:
    """
    Best-effort matplotlib styling.

    - Uses `mplstyle` if provided and exists.
    - Otherwise, tries the user's preferred style file if present on disk.
    """
    # User-preferred style (best-effort; safe no-op if missing)
    default_user_style = "/home/etorres/Documents/github/personal/research-project-pinns/style/pitayasmoothie-light.mplstyle"
    style_path = mplstyle or default_user_style
    try:
        if style_path and os.path.exists(style_path):
            plt.style.use(style_path)
    except Exception:
        # Styling should never break training
        pass


def _beta_nu_rho_from_z(z_tensor):
    """
    Robustly extract (beta, nu, rho) from the dataset's parameter tensor.

    Some datasets provide a single scalar parameter (0-d after squeeze),
    which would make `len(params)` fail. We normalize everything to a 1D array.
    """
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
    samples: List[Dict],
    normalization_stats: Dict,
    device: torch.device,
    chunk_size: int = 200_000,
) -> Dict[str, float]:
    """
    Compute RL2 over a list of dataset samples (PDE conditions).

    Returns both:
    - global RL2 (computed over all points across all samples)
    - mean/std RL2 (computed per-sample, then aggregated)
    """
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
            # x: [N, D], y: [N, 1] in normalized solution units
            beta_val, nu_val, rho_val = _beta_nu_rho_from_z(z)

            # Mirror training assumptions: use first two coordinates as (x, t); if only 1D, t=0.
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

            # Inference in chunks to avoid OOM on large grids
            preds = []
            for s in range(0, n, int(max(1, chunk_size))):
                e = min(n, s + int(max(1, chunk_size)))
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

            # Global accumulators
            sum_sq_err += float((err ** 2).sum().item())
            sum_sq_true += float((u_true_phys ** 2).sum().item())

    # Global RL2 across all points (sqrt(sum||err||^2) / sqrt(sum||u||^2))
    if sum_sq_true > 1e-24:
        rl2_global = float(math.sqrt(sum_sq_err / sum_sq_true))
    else:
        rl2_global = float("inf")

    rl2_arr = np.array(rl2_per_sample, dtype=float)
    rl2_mean = float(np.nanmean(rl2_arr)) if rl2_arr.size else float("nan")
    rl2_std = float(np.nanstd(rl2_arr, ddof=1)) if rl2_arr.size > 1 else 0.0
    rl2_sem = float(rl2_std / math.sqrt(rl2_arr.size)) if rl2_arr.size > 1 else 0.0

    return {
        # Keep legacy key name but make sure it's explicitly "global over all points".
        "rl2_train": rl2_global,
        "rl2_train_global": rl2_global,
        "rl2_train_mean_per_sample": rl2_mean,
        "rl2_train_std_per_sample": rl2_std,
        "rl2_train_sem_per_sample": rl2_sem,
        "n_train_samples_for_rl2": float(len(samples)),
    }


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
    """Render a 2D field with contourf + contour overlay; returns a mappable for colorbar()."""
    field2d = np.asarray(field2d)
    h, w = field2d.shape
    x = np.arange(w)
    y = np.arange(h)
    X, Y = np.meshgrid(x, y)

    if vmin is None:
        vmin = float(np.nanmin(field2d))
    if vmax is None:
        vmax = float(np.nanmax(field2d))

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
        ax.contour(X, Y, field2d, levels=line_levels, colors=line_color, linewidths=line_width, alpha=line_alpha)
    return cf


def _save_gt_pred_err_grid(
    *,
    predictions: List[np.ndarray],
    ground_truth: List[np.ndarray],
    split_name: str,
    epoch: int,
    output_dir: str,
) -> None:
    """
    Save a single multi-sample figure for split+epoch.

    Layout: k rows × 3 columns (GT / Pred / |Err|). Saves PNG + SVG.
    """
    os.makedirs(output_dir, exist_ok=True)
    k = min(len(predictions), len(ground_truth))
    if k <= 0:
        return

    preds = predictions[:k]
    gts = ground_truth[:k]
    shapes = [np.asarray(g).shape for g in gts]
    if len({s for s in shapes}) != 1:
        # Fall back to per-sample figures using the metrics_exporter helper
        from src.metrics_exporter import generate_visualizations
        for i in range(k):
            generate_visualizations(
                predictions=np.asarray([preds[i]]),
                ground_truth=np.asarray([gts[i]]),
                output_dir=output_dir,
                split_name=f"{split_name}_epoch_{epoch}",
                sample_indices=[0],
                max_samples=1,
                coordinate_grids=None,
            )
        return

    gt0 = np.asarray(gts[0])
    ndim = gt0.ndim
    fig_h = 3.2 * k
    fig_w = 14
    fig, axes = plt.subplots(k, 3, figsize=(fig_w, fig_h))
    if k == 1:
        axes = np.expand_dims(axes, axis=0)

    for r in range(k):
        gt = np.asarray(gts[r])
        pred = np.asarray(preds[r])
        err = np.abs(pred - gt)

        if ndim == 1:
            xax = np.arange(gt.shape[0])
            axes[r, 0].plot(xax, gt, linewidth=1.5)
            axes[r, 0].set_title(f"GT (sample {r})")
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
            axes[r, 0].set_title(f"GT (sample {r})")
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

    base = os.path.join(output_dir, f"{split_name}_epoch_{epoch}_samples_{k}")
    fig.savefig(base + ".png", dpi=200, bbox_inches="tight")
    fig.savefig(base + ".svg", bbox_inches="tight")
    plt.close(fig)


def _extract_beta_nu_rho_from_z(z_tensor: Optional[torch.Tensor]) -> Tuple[float, float, float]:
    """Robustly extract (beta, nu, rho) scalars from dataset parameter tensor."""
    if z_tensor is None:
        return 1.0, 0.0, 0.0
    params = z_tensor.squeeze().detach().cpu().numpy()
    params = np.atleast_1d(np.array(params))
    beta_val = float(params[0]) if params.size > 0 else 1.0
    nu_val = float(params[1]) if params.size > 1 else 0.0
    rho_val = float(params[2]) if params.size > 2 else 0.0
    return beta_val, nu_val, rho_val


def _save_phase1_qualitative_plots(
    *,
    net: LR_PINN_phase1,
    split_samples: List[Dict],
    split_name: str,
    epoch: int,
    output_dir: str,
    device: torch.device,
    normalization_stats: Dict,
    max_samples: int = 5,
    start_idx: int = 0,
) -> None:
    """Generate GT/Pred/|Err| qualitative grid for a few samples using the phase1 model."""
    if not split_samples:
        return

    k = max(1, min(int(max_samples), len(split_samples)))
    start = int(max(0, start_idx))
    indices = [(start + i) % len(split_samples) for i in range(k)]

    u_mean = float((normalization_stats or {}).get("u_mean", 0.0))
    u_std = float((normalization_stats or {}).get("u_std", 1.0))

    preds: List[np.ndarray] = []
    gts: List[np.ndarray] = []

    net.eval()
    with torch.no_grad():
        for idx in indices:
            sample = split_samples[idx]
            x, y, z = prepare_batch(sample, include_coords=True, device=device)

            # Coordinates
            if x.shape[1] >= 2:
                x_coords = x[:, 0:1]
                t_coords = x[:, 1:2]
            else:
                x_coords = x[:, 0:1]
                t_coords = torch.zeros_like(x_coords)

            # Params (broadcast per point)
            beta_val, nu_val, rho_val = _extract_beta_nu_rho_from_z(z)
            beta = torch.full_like(x_coords, float(beta_val))
            nu = torch.full_like(x_coords, float(nu_val))
            rho = torch.full_like(x_coords, float(rho_val))

            u_pred_norm, *_ = net(x_coords, t_coords, beta, nu, rho)

            # Denormalize both to physical units for plotting
            u_gt = y.squeeze().cpu().numpy() * u_std + u_mean
            u_pred = u_pred_norm.squeeze().cpu().numpy() * u_std + u_mean

            u_raw = sample.get("u")
            if isinstance(u_raw, torch.Tensor):
                u_shape = tuple(u_raw.shape)
            else:
                u_shape = tuple(np.array(u_raw).shape)

            gts.append(u_gt.reshape(u_shape))
            preds.append(u_pred.reshape(u_shape))

    _save_gt_pred_err_grid(
        predictions=preds,
        ground_truth=gts,
        split_name=split_name,
        epoch=int(epoch),
        output_dir=output_dir,
    )


def convert_dataset_to_training_format(splits, phase='phase1', device='cpu'):
    """
    Convert dataset splits to format compatible with existing training code.
    
    This function bridges the gap between eff-physics-learn-dataset format
    and the format expected by the existing model training code.
    """
    train_data = splits['train_few']
    
    # For phase1, we need to aggregate data across all training samples
    # For phase2, we work with a specific parameter value
    
    # Extract all data points
    x_data_list = []
    t_data_list = []
    u_data_list = []
    beta_list = []
    nu_list = []
    rho_list = []
    
    x_collocation_list = []
    t_collocation_list = []
    beta_collocation_list = []
    nu_collocation_list = []
    rho_collocation_list = []
    
    # Boundary points (will be generated if not in dataset)
    x_lb_list = []
    t_lb_list = []
    x_ub_list = []
    t_ub_list = []
    beta_bd_list = []
    nu_bd_list = []
    rho_bd_list = []
    
    for sample in train_data:
        # Prepare batch to get coordinates and solution
        x, y, z = prepare_batch(sample, include_coords=True, device=device)
        
        # Extract parameters
        beta_val, nu_val, rho_val = _beta_nu_rho_from_z(z)
        
        # Extract coordinates (assuming 2D: x, t)
        if x.shape[1] >= 2:
            x_coords = x[:, 0].cpu().numpy()
            t_coords = x[:, 1].cpu().numpy()
        else:
            # If only 1D, create dummy time dimension
            x_coords = x[:, 0].cpu().numpy()
            t_coords = np.zeros_like(x_coords)
        
        u_values = y.squeeze().cpu().numpy()
        
        # Add to lists
        n_points = len(x_coords)
        x_data_list.extend(x_coords)
        t_data_list.extend(t_coords)
        u_data_list.extend(u_values)
        beta_list.extend([beta_val] * n_points)
        nu_list.extend([nu_val] * n_points)
        rho_list.extend([rho_val] * n_points)
        
        # Collocation points (same as data points for now)
        x_collocation_list.extend(x_coords)
        t_collocation_list.extend(t_coords)
        beta_collocation_list.extend([beta_val] * n_points)
        nu_collocation_list.extend([nu_val] * n_points)
        rho_collocation_list.extend([rho_val] * n_points)
        
        # Boundary points (min/max x at t=0 and t=T)
        if len(t_coords) > 0:
            t_min = np.min(t_coords)
            t_max = np.max(t_coords)
            x_min = np.min(x_coords)
            x_max = np.max(x_coords)
            
            # Lower boundary (x_min, t)
            n_bd = len(np.unique(t_coords))
            x_lb_list.extend([x_min] * n_bd)
            t_lb_list.extend(np.unique(t_coords))
            beta_bd_list.extend([beta_val] * n_bd)
            nu_bd_list.extend([nu_val] * n_bd)
            rho_bd_list.extend([rho_val] * n_bd)
            
            # Upper boundary (x_max, t)
            x_ub_list.extend([x_max] * n_bd)
            t_ub_list.extend(np.unique(t_coords))
    
    # Convert to format expected by training code
    train_data_dict = {
        'x_data': np.array(x_data_list),
        't_data': np.array(t_data_list),
        'u_data': np.array(u_data_list),
        'beta': np.array(beta_list),
        'nu': np.array(nu_list),
        'rho': np.array(rho_list)
    }
    
    train_data_f_dict = {
        'x_data': np.array(x_collocation_list),
        't_data': np.array(t_collocation_list),
        'beta': np.array(beta_collocation_list),
        'nu': np.array(nu_collocation_list),
        'rho': np.array(rho_collocation_list)
    }
    
    train_data_bd_dict = {
        'x_data_lb': np.array(x_lb_list),
        't_data_lb': np.array(t_lb_list),
        'x_data_ub': np.array(x_ub_list),
        't_data_ub': np.array(t_ub_list),
        'beta': np.array(beta_bd_list),
        'nu': np.array(nu_bd_list),
        'rho': np.array(rho_bd_list)
    }
    
    return train_data_dict, train_data_f_dict, train_data_bd_dict


def train_phase1(config, device):
    """Train Phase 1 (meta-learning) model."""
    print("=" * 60)
    print("Training Phase 1: Meta-Learning")
    print("=" * 60)
    
    # Load dataset
    dataset_config = config['dataset']
    splits = load_parametric_dataset(
        equation=dataset_config['equation'],
        data_dir=dataset_config['data_dir'],
        seed=dataset_config.get('seed', 42),
        n_train=dataset_config.get('n_train', 10),
        cache=dataset_config.get('cache', True)
    )
    
    # Convert to training format
    processed_splits = convert_to_model_format(splits, phase='phase1', device=device)
    train_data, train_data_f, train_data_bd = convert_dataset_to_training_format(
        processed_splits, phase='phase1', device=device
    )
    
    # Model setup
    model_config = config['model']
    hidden_dim = model_config['hidden_dim']
    net = LR_PINN_phase1(hidden_dim)
    net = net.to(device)
    
    model_size = get_params(net)
    print(f"Model size: {model_size} parameters")
    
    # Get equation-specific loss function (for reporting / future extensions)
    equation = dataset_config['equation']
    pinn_loss_fn = get_pinn_loss_pytorch(equation)
    print(f"Using PINN loss function for equation: {equation}")
    
    # Training setup
    training_config = config['training']
    optimizer = _make_optimizer(net, training_config)
    mse_cost_function = nn.MSELoss()
    ic_weight = float(training_config.get("ic_weight", 0.0))
    
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
    beta_initial = Variable(
        torch.from_numpy(np.array(np.expand_dims(train_data['beta'], 1))).float(),
        requires_grad=True
    ).to(device)
    nu_initial = Variable(
        torch.from_numpy(np.array(np.expand_dims(train_data['nu'], 1))).float(),
        requires_grad=True
    ).to(device)
    rho_initial = Variable(
        torch.from_numpy(np.array(np.expand_dims(train_data['rho'], 1))).float(),
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
    beta_bd = Variable(
        torch.from_numpy(np.array(np.expand_dims(train_data_bd['beta'], 1))).float(),
        requires_grad=True
    ).to(device)
    nu_bd = Variable(
        torch.from_numpy(np.array(np.expand_dims(train_data_bd['nu'], 1))).float(),
        requires_grad=True
    ).to(device)
    rho_bd = Variable(
        torch.from_numpy(np.array(np.expand_dims(train_data_bd['rho'], 1))).float(),
        requires_grad=True
    ).to(device)
    
    # Create metrics tracking
    run_data = create_run_data(config)
    
    # Optional qualitative plots during training (GT / Pred / |Err|)
    export_config = config.get("export", {})
    save_plots = bool(export_config.get("save_plots", False))
    plot_max_samples = int(export_config.get("plot_max_samples", 5))
    plot_start_idx = int(export_config.get("plot_start_idx", 0))
    plot_interval = int(export_config.get("plot_interval", training_config.get("checkpoint_interval", 1000)))
    plot_epochs = export_config.get("plot_epochs", None)  # optional list
    plot_splits = export_config.get("plot_splits", ["train_few"])
    mplstyle = export_config.get("mplstyle", None)

    if save_plots:
        _maybe_apply_mplstyle(mplstyle)

        output_dir = export_config.get("output_dir", "./outputs")
        equation = dataset_config["equation"]
        viz_dir = os.path.join(output_dir, f"{equation}_phase1", "training", "visualizations")

        # Epoch 0 (initial model) plots
        try:
            for sname in plot_splits:
                # IMPORTANT: use the *normalized* splits (model space) for plotting,
                # then denormalize inside the plotting helper. Using raw splits here
                # would "denormalize twice" and distort magnitudes.
                split_samples = None
                if isinstance(processed_splits, dict):
                    split_samples = processed_splits.get(sname, None)
                    if split_samples is None and sname == "train":
                        split_samples = processed_splits.get("train_few", None)
                if split_samples:
                    _save_phase1_qualitative_plots(
                        net=net,
                        split_samples=split_samples,
                        split_name=str(sname),
                        epoch=0,
                        output_dir=viz_dir,
                        device=device,
                        normalization_stats=splits.normalization_stats,
                        max_samples=plot_max_samples,
                        start_idx=plot_start_idx,
                    )
        except Exception as e:
            print(f"[warn] plotting failed at epoch 0: {e}")

    # Training loop
    epochs = training_config['epochs']
    checkpoint_interval = training_config.get('checkpoint_interval', 1000)
    
    start_time = time.time()

    # Use training-split normalization stats to compute losses in *physical* units,
    # matching the original train_meta.py / train_full.py conventions.
    u_mean = float(splits.normalization_stats.get("u_mean", 0.0))
    u_std = float(splits.normalization_stats.get("u_std", 1.0))
    u_mean_t = torch.tensor(u_mean, device=device)
    u_std_t = torch.tensor(u_std, device=device)
    
    # Track last-step stats for final export (always filled, even if epochs < log interval)
    last_stats: Dict[str, float] = {}

    for ep in range(1, epochs + 1):
        net.train()
        optimizer.zero_grad()
        
        # Forward pass
        net_initial_out, col_0_init, col_1_init, col_2_init, row_0_init, row_1_init, row_2_init = net(
            x_initial, t_initial, beta_initial, nu_initial, rho_initial
        )
        
        # Orthogonality regularization
        reg_init_0 = orthogonality_reg(col_0_init, row_0_init, hidden_dim)
        reg_init_1 = orthogonality_reg(col_1_init, row_1_init, hidden_dim)
        reg_init_2 = orthogonality_reg(col_2_init, row_2_init, hidden_dim)
        reg_init = reg_init_0 + reg_init_1 + reg_init_2
        
        # Data loss (physical units)
        u_pred_phys = net_initial_out * u_std_t + u_mean_t
        u_true_phys = u_initial * u_std_t + u_mean_t
        mse_u = mse_cost_function(u_pred_phys, u_true_phys)

        # Optional IC loss for Burgers: u(x, t_min) = A * sin(k*pi*x)
        mse_ic = torch.tensor(0.0, device=device)
        if ic_weight > 0.0 and "burgers" in str(equation).lower():
            t0 = t_initial.min()
            ic_mask = (t_initial - t0).abs() < 1e-6
            if ic_mask.any():
                x_ic = x_initial[ic_mask]
                u_ic_pred = net_initial_out[ic_mask] * u_std_t + u_mean_t
                A_ic = beta_initial[ic_mask]
                k_ic = nu_initial[ic_mask]
                u_ic_gt = A_ic * torch.sin(k_ic * np.pi * x_ic)
                mse_ic = mse_cost_function(u_ic_pred, u_ic_gt)
        
        # PDE residual loss (equation-aware, Hyper-LR-PINN style: residual tensor + reg_f)
        f_out, reg_f = f_cal_equation_aware(
            x_collocation, t_collocation,
            beta_collocation, nu_collocation, rho_collocation,
            net, hidden_dim,
            equation=equation,
            normalization_stats=splits.normalization_stats,
        )
        mse_f = mse_cost_function(f_out, all_zeros)
        
        # Boundary loss
        #
        # Default behavior (matching legacy Hyper-LR-PINN scripts): periodicity in x,
        # enforced by matching predictions at x_min vs x_max for the same second-coordinate.
        #
        # For Helmholtz2D we instead enforce Dirichlet boundary conditions on all four
        # edges using the dataset-provided boundary values (x=min/max and y=min/max).
        if "helmholtz" in str(equation).lower():
            # Treat the second coordinate as y; enforce u_pred(x_edge,y) = u_true(x_edge,y)
            eps_bd = 1e-6
            x_min = x_initial.min()
            x_max = x_initial.max()
            y_min = t_initial.min()
            y_max = t_initial.max()

            left_mask = (x_initial - x_min).abs() < eps_bd
            right_mask = (x_initial - x_max).abs() < eps_bd
            bottom_mask = (t_initial - y_min).abs() < eps_bd
            top_mask = (t_initial - y_max).abs() < eps_bd

            # Boundary loss in physical units (uses normalized u tensors + denorm)
            u_pred_phys = net_initial_out * u_std_t + u_mean_t
            u_true_phys = u_initial * u_std_t + u_mean_t

            mse_bd = torch.tensor(0.0, device=device)
            for m in (left_mask, right_mask, bottom_mask, top_mask):
                if m.any():
                    mse_bd = mse_bd + mse_cost_function(u_pred_phys[m], u_true_phys[m])

            # No additional orthogonality regularization from boundary passes here.
            # (Dirichlet edges are already covered by the supervised data loss; we just
            # upweight boundary agreement for Helmholtz2D.)
            reg_bd = torch.tensor(0.0, device=device)
        else:
            u_pred_lb, col_0_lb, col_1_lb, col_2_lb, row_0_lb, row_1_lb, row_2_lb = net(
                x_lb, t_lb, beta_bd, nu_bd, rho_bd
            )
            u_pred_ub, col_0_ub, col_1_ub, col_2_ub, row_0_ub, row_1_ub, row_2_ub = net(
                x_ub, t_ub, beta_bd, nu_bd, rho_bd
            )
            
            reg_lb_0 = orthogonality_reg(col_0_lb, row_0_lb, hidden_dim)
            reg_lb_1 = orthogonality_reg(col_1_lb, row_1_lb, hidden_dim)
            reg_lb_2 = orthogonality_reg(col_2_lb, row_2_lb, hidden_dim)
            reg_ub_0 = orthogonality_reg(col_0_ub, row_0_ub, hidden_dim)
            reg_ub_1 = orthogonality_reg(col_1_ub, row_1_ub, hidden_dim)
            reg_ub_2 = orthogonality_reg(col_2_ub, row_2_ub, hidden_dim)
            reg_bd = reg_lb_0 + reg_lb_1 + reg_lb_2 + reg_ub_0 + reg_ub_1 + reg_ub_2
            
            # Boundary loss (physical units)
            u_lb_phys = u_pred_lb * u_std_t + u_mean_t
            u_ub_phys = u_pred_ub * u_std_t + u_mean_t
            mse_bd = torch.mean((u_lb_phys - u_ub_phys) ** 2)
        
        # Total loss (Hyper-LR-PINN style)
        loss = mse_u + mse_f + mse_bd + reg_init + reg_f + reg_bd + ic_weight * mse_ic

        # Always keep latest scalar stats (final epoch may not align with logging cadence)
        last_stats = {
            "epoch": float(ep),
            "train_loss": float(loss.item()),
            "mse_u": float(mse_u.item()),
            "mse_f": float(mse_f.item()),
            "mse_bd": float(mse_bd.item()),
        }
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Logging and metrics
        if ep % 10 == 0:
            net.eval()
            with torch.no_grad():
                # Evaluate on training data
                u_out_train, _, _, _, _, _, _ = net(
                    x_initial, t_initial, beta_initial, nu_initial, rho_initial
                )
                mse_train = mse_cost_function(u_out_train, u_initial)

                # RL2 error (relative L2) in physical units
                u_out_train_phys = u_out_train * u_std_t + u_mean_t
                u_true_train_phys = u_initial * u_std_t + u_mean_t
                denom = torch.linalg.norm(u_true_train_phys)
                if float(denom.item()) > 1e-12:
                    rl2_train = (torch.linalg.norm(u_out_train_phys - u_true_train_phys) / denom).item()
                else:
                    rl2_train = float("inf")
                
                # Add metrics to run_data
                # Note: RunData.add_metrics expects specific parameters
                # We'll store training metrics in metadata instead
                if run_data.metadata is None:
                    run_data.metadata = {}
                if 'training_metrics' not in run_data.metadata:
                    run_data.metadata['training_metrics'] = {
                        'epochs': [],
                        'train_loss': [],
                        'val_loss': [],
                        'rl2_train': [],
                        'mse_u': [],
                        'mse_f': [],
                        'mse_bd': [],
                    }
                run_data.metadata['training_metrics']['epochs'].append(ep)
                run_data.metadata['training_metrics']['train_loss'].append(loss.item())
                run_data.metadata['training_metrics']['val_loss'].append(mse_train.item())
                run_data.metadata['training_metrics']['rl2_train'].append(float(rl2_train))
                run_data.metadata['training_metrics']['mse_u'].append(float(mse_u.item()))
                run_data.metadata['training_metrics']['mse_f'].append(float(mse_f.item()))
                run_data.metadata['training_metrics']['mse_bd'].append(float(mse_bd.item()))
                
                # Also add to RunData's built-in metrics if available
                try:
                    run_data.add_metrics(
                        epoch=ep,
                        train_loss=loss.item(),
                        val_loss=mse_train.item()
                    )
                except (AttributeError, TypeError):
                    # If add_metrics doesn't support these parameters, that's okay
                    pass

                # Add richer stats when we compute them
                last_stats.update(
                    {
                        "train_mse_norm": float(mse_train.item()),
                        "rl2_train": float(rl2_train),
                    }
                )
                
                print(f"Epoch {ep}/{epochs} | Loss: {loss.item():.6f} | "
                      f"MSE_u: {mse_u.item():.6f} | MSE_f: {mse_f.item():.6f} | "
                      f"MSE_bd: {mse_bd.item():.6f} | Train MSE: {mse_train.item():.6f} | "
                      f"RL2: {rl2_train:.6f}")
        
        # Save checkpoint
        if ep % checkpoint_interval == 0:
            export_config = config['export']
            output_dir = export_config['output_dir']
            equation = dataset_config['equation']
            checkpoint_dir = os.path.join(output_dir, f"{equation}_phase1", "checkpoints")
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{ep}.pt")
            torch.save(net.state_dict(), checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

        # Qualitative plots at configured epochs/interval
        if save_plots:
            should_plot = False
            if isinstance(plot_epochs, (list, tuple, set)):
                try:
                    should_plot = int(ep) in {int(x) for x in plot_epochs}
                except Exception:
                    should_plot = False
            else:
                should_plot = (plot_interval > 0 and ep % plot_interval == 0) or (ep == epochs)

            if should_plot:
                try:
                    for sname in plot_splits:
                        split_samples = None
                        if isinstance(processed_splits, dict):
                            split_samples = processed_splits.get(sname, None)
                            if split_samples is None and sname == "train":
                                split_samples = processed_splits.get("train_few", None)
                        if split_samples:
                            _save_phase1_qualitative_plots(
                                net=net,
                                split_samples=split_samples,
                                split_name=str(sname),
                                epoch=int(ep),
                                output_dir=viz_dir,
                                device=device,
                                normalization_stats=splits.normalization_stats,
                                max_samples=plot_max_samples,
                                start_idx=plot_start_idx,
                            )
                except Exception as e:
                    print(f"[warn] plotting failed at epoch {ep}: {e}")
    
    # Final metrics
    wall_time = time.time() - start_time
    add_timing_metrics(run_data, wall_time)

    # Store final summary stats for easy access in exported metrics
    # Ensure RL2 is computed at least once (even if epochs < log interval).
    net.eval()
    with torch.no_grad():
        u_out_final, _, _, _, _, _, _ = net(x_initial, t_initial, beta_initial, nu_initial, rho_initial)
        u_out_final_phys = u_out_final * u_std_t + u_mean_t
        u_true_final_phys = u_initial * u_std_t + u_mean_t
        denom = torch.linalg.norm(u_true_final_phys)
        if float(denom.item()) > 1e-12:
            last_stats["rl2_train"] = float((torch.linalg.norm(u_out_final_phys - u_true_final_phys) / denom).item())
        else:
            last_stats["rl2_train"] = float("inf")

    # Also compute per-sample RL2 over all training samples (PDE conditions) and store mean±std.
    # This makes it explicit that "rl2_train" reflects performance across all training samples.
    try:
        rl2_stats = _compute_phase1_rl2_over_samples(
            net=net,
            samples=list(processed_splits.get("train_few", [])) if isinstance(processed_splits, dict) else [],
            normalization_stats=splits.normalization_stats,
            device=device,
            chunk_size=int(training_config.get("eval_chunk_size", 200_000)),
        )
        # Keep legacy key name but overwrite with the more explicit global-over-all-points value.
        last_stats.update(rl2_stats)
    except Exception as e:
        print(f"[warn] could not compute per-sample RL2 stats at end of training: {e}")

    if run_data.metadata is None:
        run_data.metadata = {}
    run_data.metadata["final_summary"] = {
        "wall_time_sec": float(wall_time),
        **(last_stats or {}),
    }
    
    # Export metrics
    export_config = config['export']
    output_dir = export_config['output_dir']
    equation = dataset_config['equation']
    metrics_dir = os.path.join(output_dir, f"{equation}_phase1", "metrics")
    export_metrics(run_data, metrics_dir, format='both')

    # Best-effort: also write CSV summaries (mean±std) across all runs in this metrics_dir.
    # This is useful when you have multiple seeds/runs and want an aggregated table without rerunning.
    try:
        from pathlib import Path as _Path
        from scripts.summarize_metrics import summarize_metrics_dir as _summarize_metrics_dir

        _summarize_metrics_dir(_Path(metrics_dir))
    except Exception as e:
        print(f"[warn] could not generate metrics CSV summaries in '{metrics_dir}': {e}")
    
    # Save final checkpoint
    checkpoint_dir = os.path.join(output_dir, f"{equation}_phase1", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    final_checkpoint = os.path.join(checkpoint_dir, "best.pt")
    torch.save(net.state_dict(), final_checkpoint)
    print(f"Training completed. Final checkpoint saved to {final_checkpoint}")
    
    return net, run_data


def train_phase2(config, device):
    """Train Phase 2 (fine-tuning) model."""
    print("=" * 60)
    print("Training Phase 2: Fine-tuning")
    print("=" * 60)
    
    # Load phase1 checkpoint
    model_config = config['model']
    phase1_checkpoint = model_config.get('phase1_checkpoint')
    if phase1_checkpoint is None:
        raise ValueError("phase1_checkpoint must be specified in config for phase2 training")
    
    if not os.path.exists(phase1_checkpoint):
        raise FileNotFoundError(f"Phase1 checkpoint not found: {phase1_checkpoint}")
    
    # Initialize phase1 model and load weights
    hidden_dim = model_config['hidden_dim']
    net_initial = LR_PINN_phase1(hidden_dim)
    net_initial.load_state_dict(torch.load(phase1_checkpoint))
    net_initial = net_initial.to(device)
    
    # Extract weights for phase2
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
    
    # For phase2, we need target parameters from the dataset
    # This is a simplified version - in practice, you'd extract from dataset
    dataset_config = config['dataset']
    splits = load_parametric_dataset(
        equation=dataset_config['equation'],
        data_dir=dataset_config['data_dir'],
        seed=dataset_config.get('seed', 42),
        n_train=dataset_config.get('n_train', 10),
        cache=dataset_config.get('cache', True)
    )
    
    # Get parameters from first training sample (or specify in config)
    first_sample = splits.train_few[0]
    _, _, z = prepare_batch(first_sample, include_coords=True, device=device)
    
    if z is not None:
        target_coeff = z.squeeze().to(device)
        if target_coeff.ndim != 1:
            target_coeff = target_coeff.view(-1)
        if target_coeff.numel() < 3:
            pad = torch.zeros((3 - target_coeff.numel(),), device=target_coeff.device, dtype=target_coeff.dtype)
            target_coeff = torch.cat([target_coeff, pad], dim=0)
    else:
        # Default parameters
        target_coeff = torch.tensor([1.0, 0.0, 0.0], device=device)
    
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
    net = LR_PINN_phase2(
        hidden_dim, start_w, start_b, end_w, end_b,
        col_0, col_1, col_2, row_0, row_1, row_2,
        alpha_0, alpha_1, alpha_2
    )
    net = net.to(device)
    
    model_size = get_params(net)
    print(f"Model size: {model_size} parameters")
    
    # Get equation-specific loss function (for reporting / future extensions)
    equation = dataset_config['equation']
    pinn_loss_fn = get_pinn_loss_pytorch(equation)
    print(f"Using PINN loss function for equation: {equation}")
    
    # Training setup
    training_config = config['training']
    optimizer = _make_optimizer(net, training_config)
    mse_cost_function = nn.MSELoss()
    ic_weight = float(training_config.get("ic_weight", 0.0))
    
    # Convert dataset to training format
    processed_splits = convert_to_model_format(splits, phase='phase2', device=device)
    train_data, train_data_f, train_data_bd = convert_dataset_to_training_format(
        processed_splits, phase='phase2', device=device
    )
    
    # Prepare data tensors (phase2 doesn't need parameters in forward pass)
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
    # Parameters for optional IC construction (e.g. Burgers uses A,k in the dataset)
    beta_initial = Variable(
        torch.from_numpy(np.array(np.expand_dims(train_data['beta'], 1))).float(),
        requires_grad=False
    ).to(device)
    nu_initial = Variable(
        torch.from_numpy(np.array(np.expand_dims(train_data['nu'], 1))).float(),
        requires_grad=False
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
    
    # Create metrics tracking
    run_data = create_run_data(config)
    
    # Training loop
    epochs = training_config['epochs']
    checkpoint_interval = training_config.get('checkpoint_interval', 1000)
    
    start_time = time.time()

    # Use training-split normalization stats to compute losses in *physical* units,
    # matching the original train_full.py conventions.
    u_mean = float(splits.normalization_stats.get("u_mean", 0.0))
    u_std = float(splits.normalization_stats.get("u_std", 1.0))
    u_mean_t = torch.tensor(u_mean, device=device)
    u_std_t = torch.tensor(u_std, device=device)
    
    for ep in range(1, epochs + 1):
        net.train()
        optimizer.zero_grad()
        
        # Forward pass (phase2 doesn't take parameters)
        net_initial_out = net(x_initial, t_initial)
        # Data loss (physical units)
        u_pred_phys = net_initial_out * u_std_t + u_mean_t
        u_true_phys = u_initial * u_std_t + u_mean_t
        mse_u = mse_cost_function(u_pred_phys, u_true_phys)

        # Optional IC loss for Burgers (same convention as phase1)
        mse_ic = torch.tensor(0.0, device=device)
        if ic_weight > 0.0 and "burgers" in str(equation).lower():
            t0 = t_initial.min()
            ic_mask = (t_initial - t0).abs() < 1e-6
            if ic_mask.any():
                x_ic = x_initial[ic_mask]
                u_ic_pred = net_initial_out[ic_mask] * u_std_t + u_mean_t
                A_ic = beta_initial[ic_mask]
                k_ic = nu_initial[ic_mask]
                u_ic_gt = A_ic * torch.sin(k_ic * np.pi * x_ic)
                mse_ic = mse_cost_function(u_ic_pred, u_ic_gt)
        
        # PDE residual loss (equation-aware)
        f_out = f_cal_phase2_equation_aware(
            x_collocation, t_collocation,
            beta_collocation, nu_collocation, rho_collocation,
            net,
            equation=equation,
            normalization_stats=splits.normalization_stats,
        )
        mse_f = mse_cost_function(f_out, all_zeros)
        
        # Boundary loss
        if "helmholtz" in str(equation).lower():
            # Dirichlet BCs on all four edges using dataset boundary values
            eps_bd = 1e-6
            x_min = x_initial.min()
            x_max = x_initial.max()
            y_min = t_initial.min()
            y_max = t_initial.max()

            left_mask = (x_initial - x_min).abs() < eps_bd
            right_mask = (x_initial - x_max).abs() < eps_bd
            bottom_mask = (t_initial - y_min).abs() < eps_bd
            top_mask = (t_initial - y_max).abs() < eps_bd

            u_pred_phys = net_initial_out * u_std_t + u_mean_t
            u_true_phys = u_initial * u_std_t + u_mean_t

            mse_bd = torch.tensor(0.0, device=device)
            for m in (left_mask, right_mask, bottom_mask, top_mask):
                if m.any():
                    mse_bd = mse_bd + mse_cost_function(u_pred_phys[m], u_true_phys[m])
        else:
            u_pred_lb = net(x_lb, t_lb)
            u_pred_ub = net(x_ub, t_ub)
            # Boundary loss (physical units)
            u_lb_phys = u_pred_lb * u_std_t + u_mean_t
            u_ub_phys = u_pred_ub * u_std_t + u_mean_t
            mse_bd = torch.mean((u_lb_phys - u_ub_phys) ** 2)
        
        # Total loss
        loss = mse_u + mse_f + mse_bd + ic_weight * mse_ic
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Logging and metrics
        if ep % 10 == 0:
            net.eval()
            with torch.no_grad():
                u_out_train = net(x_initial, t_initial)
                mse_train = mse_cost_function(u_out_train, u_initial)
                
                # Store training metrics in metadata
                if run_data.metadata is None:
                    run_data.metadata = {}
                if 'training_metrics' not in run_data.metadata:
                    run_data.metadata['training_metrics'] = {
                        'epochs': [],
                        'train_loss': [],
                        'val_loss': []
                    }
                run_data.metadata['training_metrics']['epochs'].append(ep)
                run_data.metadata['training_metrics']['train_loss'].append(loss.item())
                run_data.metadata['training_metrics']['val_loss'].append(mse_train.item())
                
                # Also add to RunData's built-in metrics if available
                try:
                    run_data.add_metrics(
                        epoch=ep,
                        train_loss=loss.item(),
                        val_loss=mse_train.item()
                    )
                except (AttributeError, TypeError):
                    # If add_metrics doesn't support these parameters, that's okay
                    pass
                
                print(f"Epoch {ep}/{epochs} | Loss: {loss.item():.6f} | "
                      f"MSE_u: {mse_u.item():.6f} | MSE_f: {mse_f.item():.6f} | "
                      f"MSE_bd: {mse_bd.item():.6f} | Train MSE: {mse_train.item():.6f}")
        
        # Save checkpoint
        if ep % checkpoint_interval == 0:
            export_config = config['export']
            output_dir = export_config['output_dir']
            equation = dataset_config['equation']
            checkpoint_dir = os.path.join(output_dir, f"{equation}_phase2", "checkpoints")
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{ep}.pt")
            torch.save(net.state_dict(), checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
    
    # Final metrics
    wall_time = time.time() - start_time
    add_timing_metrics(run_data, wall_time)
    
    # Export metrics
    export_config = config['export']
    output_dir = export_config['output_dir']
    equation = dataset_config['equation']
    metrics_dir = os.path.join(output_dir, f"{equation}_phase2", "metrics")
    export_metrics(run_data, metrics_dir, format='both')

    # Best-effort: also write CSV summaries (mean±std) across all runs in this metrics_dir.
    try:
        from pathlib import Path as _Path
        from scripts.summarize_metrics import summarize_metrics_dir as _summarize_metrics_dir

        _summarize_metrics_dir(_Path(metrics_dir))
    except Exception as e:
        print(f"[warn] could not generate metrics CSV summaries in '{metrics_dir}': {e}")
    
    # Save final checkpoint
    checkpoint_dir = os.path.join(output_dir, f"{equation}_phase2", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    final_checkpoint = os.path.join(checkpoint_dir, "best.pt")
    torch.save(net.state_dict(), final_checkpoint)
    print(f"Training completed. Final checkpoint saved to {final_checkpoint}")
    
    return net, run_data


def main():
    parser = argparse.ArgumentParser(description='Train Hyper-LR-PINN with parametric dataset')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to YAML configuration file')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (overrides config)')
    
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
    
    print("=" * 60)
    print("Hyper-LR-PINN Training")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Seed: {seed}")
    print(f"Configuration: {args.config}")
    print("=" * 60)
    
    # Determine phase and train
    model_config = config['model']
    phase = model_config['phase']
    
    if phase == 'phase1':
        train_phase1(config, device)
    elif phase == 'phase2':
        train_phase2(config, device)
    else:
        raise ValueError(f"Unknown phase: {phase}. Must be 'phase1' or 'phase2'")


if __name__ == '__main__':
    main()

