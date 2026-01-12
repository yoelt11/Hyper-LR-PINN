"""
Metrics export module for standardized evaluation results.

This module provides a wrapper around metrics-structures for exporting
evaluation results in a standardized format.
"""

import os
import time
import json
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

try:
    from metrics_structures import RunData
except ImportError:
    raise ImportError(
        "metrics-structures not found. Please install it using:\n"
        "pip install git+https://github.com/yoelt11/metrics-structures.git"
    )


def create_run_data(
    config: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None
) -> RunData:
    """
    Create a new RunData object for storing metrics.
    
    Args:
        config: Configuration dictionary (will be stored in metadata)
        metadata: Additional metadata to store
        
    Returns:
        RunData object initialized with configuration
    """
    run_data = RunData()
    
    # Store configuration in metadata
    if metadata is None:
        metadata = {}
    
    metadata['config'] = config
    run_data.metadata = metadata
    
    return run_data


def add_evaluation_metrics(
    run_data: RunData,
    split_name: str,
    metrics_dict: Dict[str, float]
) -> None:
    """
    Add aggregate evaluation metrics for a split.
    
    Args:
        run_data: RunData object to update
        split_name: Name of the split ('interp' or 'extrap')
        metrics_dict: Dictionary of metric names to values
            Expected keys: 'mse', 'l2_error', 'relative_l2_error', etc.
    """
    if run_data.metadata is None:
        run_data.metadata = {}
    
    if 'split_metrics' not in run_data.metadata:
        run_data.metadata['split_metrics'] = {}
    
    run_data.metadata['split_metrics'][split_name] = metrics_dict


def add_per_sample_metrics(
    run_data: RunData,
    split_name: str,
    sample_metrics: List[Dict[str, float]]
) -> None:
    """
    Add per-sample metrics for a split.
    
    Args:
        run_data: RunData object to update
        split_name: Name of the split ('interp' or 'extrap')
        sample_metrics: List of dictionaries, each containing metrics for one sample
    """
    if run_data.metadata is None:
        run_data.metadata = {}
    
    if 'per_sample_metrics' not in run_data.metadata:
        run_data.metadata['per_sample_metrics'] = {}
    
    run_data.metadata['per_sample_metrics'][split_name] = sample_metrics


def add_timing_metrics(
    run_data: RunData,
    wall_time: float,
    batches_per_sec: Optional[float] = None,
    total_batches: Optional[int] = None
) -> None:
    """
    Add timing metrics to RunData.
    
    Args:
        run_data: RunData object to update
        wall_time: Total wall clock time in seconds
        batches_per_sec: Batches processed per second (optional)
        total_batches: Total number of batches processed (optional)
    """
    run_data.wall_time = wall_time
    if batches_per_sec is not None:
        run_data.it_per_sec = batches_per_sec
    elif total_batches is not None and wall_time > 0:
        run_data.it_per_sec = total_batches / wall_time


def export_metrics(
    run_data: RunData,
    output_dir: str,
    format: str = 'both',
    filename: Optional[str] = None
) -> None:
    """
    Export metrics to file(s).
    
    Args:
        run_data: RunData object to export
        output_dir: Directory to save metrics
        format: Export format - 'pickle', 'json', or 'both'
        filename: Base filename (without extension). If None, uses run ID.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if filename is None:
        filename = run_data.id
    
    if format in ('pickle', 'both'):
        pickle_path = os.path.join(output_dir, f"{filename}.pkl")
        run_data.save(pickle_path)
        print(f"Saved metrics to {pickle_path}")
    
    if format in ('json', 'both'):
        json_path = os.path.join(output_dir, f"{filename}.json")
        run_data.save_json(json_path)
        print(f"Saved metrics to {json_path}")


def generate_visualizations(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    output_dir: str,
    split_name: str,
    sample_indices: Optional[List[int]] = None,
    max_samples: int = 5,
    coordinate_grids: Optional[Dict[str, np.ndarray]] = None
) -> None:
    """
    Generate visualization plots for ground truth, predictions, and errors.
    
    Args:
        predictions: Predicted solution fields, shape [N_samples, ...]
        ground_truth: Ground truth solution fields, shape [N_samples, ...]
        output_dir: Directory to save plots
        split_name: Name of the split ('interp' or 'extrap')
        sample_indices: Which samples to plot (if None, plots first max_samples)
        max_samples: Maximum number of samples to plot
        coordinate_grids: Optional dictionary with coordinate grids for plotting
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine which samples to plot
    n_samples = len(predictions)
    if sample_indices is None:
        sample_indices = list(range(min(max_samples, n_samples)))
    
    # Determine spatial dimensions from first sample
    pred_shape = predictions[0].shape
    gt_shape = ground_truth[0].shape
    
    if pred_shape != gt_shape:
        raise ValueError(f"Shape mismatch: predictions {pred_shape} vs ground_truth {gt_shape}")
    
    # Plot each selected sample
    for idx in sample_indices:
        if idx >= n_samples:
            continue
        
        pred = predictions[idx]
        gt = ground_truth[idx]
        error = np.abs(pred - gt)
        
        # Determine if 1D, 2D, or 3D
        if len(pred_shape) == 1:
            _plot_1d(gt, pred, error, output_dir, split_name, idx)
        elif len(pred_shape) == 2:
            _plot_2d(gt, pred, error, output_dir, split_name, idx, coordinate_grids)
        elif len(pred_shape) == 3:
            # For 3D, plot a slice
            _plot_3d_slice(gt, pred, error, output_dir, split_name, idx, coordinate_grids)
        else:
            print(f"Warning: Cannot plot {len(pred_shape)}D data for sample {idx}")


def _plot_1d(
    gt: np.ndarray,
    pred: np.ndarray,
    error: np.ndarray,
    output_dir: str,
    split_name: str,
    sample_idx: int
) -> None:
    """Plot 1D solution fields."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    
    x = np.arange(len(gt))
    
    axes[0].plot(x, gt, 'b-', label='Ground Truth', linewidth=2)
    axes[0].plot(x, pred, 'r--', label='Prediction', linewidth=2)
    axes[0].set_title(f'{split_name} - Sample {sample_idx} - Ground Truth vs Prediction')
    axes[0].set_xlabel('Spatial Coordinate')
    axes[0].set_ylabel('Solution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(x, error, 'g-', linewidth=2)
    axes[1].set_title(f'Absolute Error')
    axes[1].set_xlabel('Spatial Coordinate')
    axes[1].set_ylabel('|Error|')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(x, gt, 'b-', alpha=0.5, label='Ground Truth')
    axes[2].plot(x, pred, 'r--', alpha=0.5, label='Prediction')
    axes[2].fill_between(x, gt, pred, alpha=0.3, color='red')
    axes[2].set_title('Overlay')
    axes[2].set_xlabel('Spatial Coordinate')
    axes[2].set_ylabel('Solution')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'{split_name}_sample_{sample_idx}_1d.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def _plot_2d(
    gt: np.ndarray,
    pred: np.ndarray,
    error: np.ndarray,
    output_dir: str,
    split_name: str,
    sample_idx: int,
    coordinate_grids: Optional[Dict[str, np.ndarray]] = None
) -> None:
    """Plot 2D solution fields."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Create coordinate grids if not provided
    if coordinate_grids is not None:
        x = coordinate_grids.get('x', np.arange(gt.shape[0]))
        y = coordinate_grids.get('y', np.arange(gt.shape[1]))
    else:
        x = np.arange(gt.shape[0])
        y = np.arange(gt.shape[1])
    
    X, Y = np.meshgrid(y, x)
    
    vmin = min(gt.min(), pred.min())
    vmax = max(gt.max(), pred.max())
    
    im1 = axes[0].contourf(X, Y, gt, levels=50, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[0].set_title('Ground Truth')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    plt.colorbar(im1, ax=axes[0])
    
    im2 = axes[1].contourf(X, Y, pred, levels=50, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[1].set_title('Prediction')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')
    plt.colorbar(im2, ax=axes[1])
    
    im3 = axes[2].contourf(X, Y, error, levels=50, cmap='hot')
    axes[2].set_title('Absolute Error')
    axes[2].set_xlabel('X')
    axes[2].set_ylabel('Y')
    plt.colorbar(im3, ax=axes[2])
    
    plt.suptitle(f'{split_name} - Sample {sample_idx}', fontsize=14)
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'{split_name}_sample_{sample_idx}_2d.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def _plot_3d_slice(
    gt: np.ndarray,
    pred: np.ndarray,
    error: np.ndarray,
    output_dir: str,
    split_name: str,
    sample_idx: int,
    coordinate_grids: Optional[Dict[str, np.ndarray]] = None
) -> None:
    """Plot a slice of 3D solution fields (middle slice along first dimension)."""
    # Take middle slice
    mid_slice = gt.shape[0] // 2
    gt_slice = gt[mid_slice, :, :]
    pred_slice = pred[mid_slice, :, :]
    error_slice = error[mid_slice, :, :]
    
    _plot_2d(gt_slice, pred_slice, error_slice, output_dir, split_name, sample_idx, coordinate_grids)
    
    # Update filename to indicate it's a slice
    old_path = os.path.join(output_dir, f'{split_name}_sample_{sample_idx}_2d.png')
    new_path = os.path.join(output_dir, f'{split_name}_sample_{sample_idx}_3d_slice_{mid_slice}.png')
    if os.path.exists(old_path):
        os.rename(old_path, new_path)


def compute_metrics(
    predictions: np.ndarray,
    ground_truth: np.ndarray
) -> Dict[str, float]:
    """
    Compute aggregate metrics from predictions and ground truth.
    
    Args:
        predictions: Predicted values, shape [N_samples, ...]
        ground_truth: Ground truth values, shape [N_samples, ...]
        
    Returns:
        Dictionary with metrics: 'mse', 'l2_error', 'relative_l2_error', 'max_error'
    """
    # Flatten for computation
    pred_flat = predictions.flatten()
    gt_flat = ground_truth.flatten()
    
    # MSE
    mse = float(np.mean((pred_flat - gt_flat) ** 2))
    
    # L2 error
    l2_error = float(np.linalg.norm(pred_flat - gt_flat))
    
    # Relative L2 error
    gt_norm = float(np.linalg.norm(gt_flat))
    if gt_norm > 1e-10:
        relative_l2_error = l2_error / gt_norm
    else:
        relative_l2_error = float('inf') if l2_error > 1e-10 else 0.0
    
    # Max error
    max_error = float(np.max(np.abs(pred_flat - gt_flat)))
    
    # Mean absolute error
    mae = float(np.mean(np.abs(pred_flat - gt_flat)))
    
    return {
        'mse': mse,
        'l2_error': l2_error,
        'relative_l2_error': relative_l2_error,
        'max_error': max_error,
        'mae': mae
    }


def compute_per_sample_metrics(
    predictions: np.ndarray,
    ground_truth: np.ndarray
) -> List[Dict[str, float]]:
    """
    Compute metrics for each sample individually.
    
    Args:
        predictions: Predicted values, shape [N_samples, ...]
        ground_truth: Ground truth values, shape [N_samples, ...]
        
    Returns:
        List of dictionaries, each containing metrics for one sample
    """
    n_samples = len(predictions)
    sample_metrics = []
    
    for i in range(n_samples):
        metrics = compute_metrics(predictions[i:i+1], ground_truth[i:i+1])
        sample_metrics.append(metrics)
    
    return sample_metrics

