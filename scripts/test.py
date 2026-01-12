#!/usr/bin/env python
"""
Testing script for Hyper-LR-PINN with parametric dataset evaluation.

This script evaluates trained models on interpolation and extrapolation splits
and exports metrics using metrics-structures.
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

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model import LR_PINN_phase1, LR_PINN_phase2
from utils import get_params
from src.config_loader import load_config, validate_config
from src.data_loader import (
    load_parametric_dataset,
    normalize_data,
    prepare_batch,
    convert_to_model_format
)
from src.metrics_exporter import (
    create_run_data,
    add_evaluation_metrics,
    add_per_sample_metrics,
    add_timing_metrics,
    export_metrics,
    generate_visualizations,
    compute_metrics,
    compute_per_sample_metrics
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


def evaluate_split(
    model,
    split_data,
    split_name,
    phase,
    device,
    normalization_stats
):
    """
    Evaluate model on a data split.
    
    Args:
        model: Trained model
        split_data: List of data samples
        split_name: Name of split ('interp' or 'extrap')
        phase: 'phase1' or 'phase2'
        device: Device to run on
        normalization_stats: Normalization statistics
        
    Returns:
        Dictionary with predictions, ground truth, and metrics
    """
    model.eval()
    
    all_predictions = []
    all_ground_truth = []
    all_params = []
    sample_metrics_list = []
    
    with torch.no_grad():
        for i, sample in enumerate(split_data):
            # Prepare batch
            x, y, z = prepare_batch(sample, include_coords=True, device=device)
            
            # Denormalize ground truth for metrics computation
            u_gt = y.squeeze().cpu().numpy()
            if normalization_stats:
                u_gt = u_gt * normalization_stats['u_std'] + normalization_stats['u_mean']
            
            # Forward pass
            if phase == 'phase1':
                # Phase1 needs parameters
                if z is not None:
                    # Extract parameters
                    params = z.squeeze()
                    if len(params) >= 3:
                        beta = params[0:1].unsqueeze(0)
                        nu = params[1:2].unsqueeze(0) if len(params) > 1 else torch.zeros(1, 1, device=device)
                        rho = params[2:3].unsqueeze(0) if len(params) > 2 else torch.zeros(1, 1, device=device)
                    else:
                        beta = params[0:1].unsqueeze(0) if len(params) > 0 else torch.ones(1, 1, device=device)
                        nu = torch.zeros(1, 1, device=device)
                        rho = torch.zeros(1, 1, device=device)
                else:
                    beta = torch.ones(1, 1, device=device)
                    nu = torch.zeros(1, 1, device=device)
                    rho = torch.zeros(1, 1, device=device)
                
                # Extract coordinates
                if x.shape[1] >= 2:
                    x_coords = x[:, 0:1]
                    t_coords = x[:, 1:2]
                else:
                    x_coords = x[:, 0:1]
                    t_coords = torch.zeros_like(x_coords)
                
                # Expand parameters to match batch size
                batch_size = x_coords.shape[0]
                beta = beta.expand(batch_size, -1)
                nu = nu.expand(batch_size, -1)
                rho = rho.expand(batch_size, -1)
                
                u_pred, _, _, _, _, _, _ = model(x_coords, t_coords, beta, nu, rho)
            else:
                # Phase2 doesn't need parameters
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
            
            # Store results
            all_predictions.append(u_pred_np)
            all_ground_truth.append(u_gt)
            
            if z is not None:
                all_params.append(z.squeeze().cpu().numpy())
            
            # Compute per-sample metrics
            pred_reshaped = u_pred_np.reshape(sample.get('u', u_pred_np).shape) if 'u' in sample else u_pred_np
            gt_reshaped = u_gt.reshape(sample.get('u', u_gt).shape) if 'u' in sample else u_gt
            
            sample_metrics = compute_metrics(
                pred_reshaped[np.newaxis, ...],
                gt_reshaped[np.newaxis, ...]
            )
            sample_metrics_list.append(sample_metrics)
    
    # Convert to numpy arrays
    predictions_array = np.array(all_predictions)
    ground_truth_array = np.array(all_ground_truth)
    
    # Compute aggregate metrics
    aggregate_metrics = compute_metrics(predictions_array, ground_truth_array)
    
    return {
        'predictions': predictions_array,
        'ground_truth': ground_truth_array,
        'params': all_params,
        'aggregate_metrics': aggregate_metrics,
        'per_sample_metrics': sample_metrics_list
    }


def main():
    parser = argparse.ArgumentParser(description='Test Hyper-LR-PINN with parametric dataset')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to YAML configuration file')
    parser.add_argument('--ckpt_path', type=str, required=True,
                       help='Path to model checkpoint')
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
    print("Hyper-LR-PINN Evaluation")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Checkpoint: {args.ckpt_path}")
    print(f"Configuration: {args.config}")
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
    
    # Get normalization stats
    normalization_stats = splits.normalization_stats
    
    # Determine phase
    model_config = config['model']
    phase = model_config['phase']
    hidden_dim = model_config['hidden_dim']
    
    # Load model
    if phase == 'phase1':
        model = LR_PINN_phase1(hidden_dim)
    elif phase == 'phase2':
        # For phase2, we need to load phase1 checkpoint first
        phase1_checkpoint = model_config.get('phase1_checkpoint')
        if phase1_checkpoint is None:
            raise ValueError("phase1_checkpoint must be specified in config for phase2 evaluation")
        
        if not os.path.exists(phase1_checkpoint):
            raise FileNotFoundError(f"Phase1 checkpoint not found: {phase1_checkpoint}")
        
        # Load phase1 model
        net_initial = LR_PINN_phase1(hidden_dim)
        net_initial.load_state_dict(torch.load(phase1_checkpoint))
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
        
        # Get target parameters from first training sample
        first_sample = splits.train_few[0]
        _, _, z = prepare_batch(first_sample, include_coords=True, device=device)
        
        if z is not None:
            target_coeff = z.squeeze()
            if len(target_coeff) < 3:
                target_coeff = torch.cat([target_coeff, torch.zeros(3 - len(target_coeff), device=device)])
        else:
            target_coeff = torch.tensor([1.0, 0.0, 0.0], device=device)
        
        # Compute alpha values
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
    else:
        raise ValueError(f"Unknown phase: {phase}")
    
    # Load checkpoint
    if not os.path.exists(args.ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt_path}")
    
    model.load_state_dict(torch.load(args.ckpt_path))
    model = model.to(device)
    
    model_size = get_params(model)
    print(f"Model size: {model_size} parameters")
    print("=" * 60)
    
    # Create metrics tracking
    run_data = create_run_data(config)
    
    # Evaluate on interpolation split
    print("\nEvaluating on interpolation split...")
    start_time = time.time()
    
    interp_results = evaluate_split(
        model,
        splits.interp,
        'interp',
        phase,
        device,
        normalization_stats
    )
    
    interp_time = time.time() - start_time
    
    # Add interpolation metrics
    add_evaluation_metrics(run_data, 'interp', interp_results['aggregate_metrics'])
    add_per_sample_metrics(run_data, 'interp', interp_results['per_sample_metrics'])
    
    print(f"Interpolation Metrics:")
    for key, value in interp_results['aggregate_metrics'].items():
        print(f"  {key}: {value:.6e}")
    print(f"  Time: {interp_time:.2f}s")
    
    # Evaluate on extrapolation split
    print("\nEvaluating on extrapolation split...")
    start_time = time.time()
    
    extrap_results = evaluate_split(
        model,
        splits.extrap,
        'extrap',
        phase,
        device,
        normalization_stats
    )
    
    extrap_time = time.time() - start_time
    
    # Add extrapolation metrics
    add_evaluation_metrics(run_data, 'extrap', extrap_results['aggregate_metrics'])
    add_per_sample_metrics(run_data, 'extrap', extrap_results['per_sample_metrics'])
    
    print(f"\nExtrapolation Metrics:")
    for key, value in extrap_results['aggregate_metrics'].items():
        print(f"  {key}: {value:.6e}")
    print(f"  Time: {extrap_time:.2f}s")
    
    # Add timing metrics
    total_time = interp_time + extrap_time
    add_timing_metrics(run_data, total_time)
    
    # Generate visualizations
    export_config = config['export']
    if export_config.get('save_plots', True):
        output_dir = export_config['output_dir']
        equation = dataset_config['equation']
        plots_dir = os.path.join(output_dir, f"{equation}_{phase}", "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        max_samples = export_config.get('plot_max_samples', 5)
        
        print(f"\nGenerating visualizations...")
        generate_visualizations(
            interp_results['predictions'],
            interp_results['ground_truth'],
            plots_dir,
            'interp',
            max_samples=max_samples
        )
        
        generate_visualizations(
            extrap_results['predictions'],
            extrap_results['ground_truth'],
            plots_dir,
            'extrap',
            max_samples=max_samples
        )
        print(f"Visualizations saved to {plots_dir}")
    
    # Export metrics
    output_dir = export_config['output_dir']
    equation = dataset_config['equation']
    metrics_dir = os.path.join(output_dir, f"{equation}_{phase}", "metrics")
    export_metrics(run_data, metrics_dir, format='both', filename='evaluation')
    
    print("\n" + "=" * 60)
    print("Evaluation completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()

