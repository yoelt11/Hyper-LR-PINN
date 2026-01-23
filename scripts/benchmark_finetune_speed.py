#!/usr/bin/env python
"""
Benchmark script to measure Phase 2 fine-tuning speed (it/s) with torch.compile().

This script runs short fine-tuning runs to measure iterations per second,
allowing us to update it/s values in tables without re-running full training.

Usage:
    python scripts/benchmark_finetune_speed.py --config configs/allen_cahn_phase2.yaml --phase1_checkpoint outputs/seed_0/allen_cahn_phase1/checkpoints/best.pt --benchmark_epochs 200
"""

import argparse
import os
import sys
import time
import torch
import numpy as np
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.config_loader import load_config, validate_config
from src.data_loader import (
    load_parametric_dataset,
    convert_to_model_format,
    prepare_batch,
    DatasetSplits
)
from scripts.finetune_eval import (
    setup_seed,
    create_phase2_model,
    _extract_target_coeff_from_sample,
    fine_tune_and_evaluate
)
from scripts.train import convert_dataset_to_training_format


def benchmark_finetuning_speed(
    config_path: str,
    phase1_checkpoint: str,
    benchmark_epochs: int = 200,
    use_compile: bool = True,
    device: str = None,
    num_runs: int = 3,
):
    """
    Benchmark fine-tuning speed to measure it/s.
    
    Args:
        config_path: Path to config file
        phase1_checkpoint: Path to Phase 1 checkpoint
        benchmark_epochs: Number of epochs to run for benchmarking
        use_compile: Whether to use torch.compile()
        device: Device to use (overrides config)
        num_runs: Number of benchmark runs to average
        
    Returns:
        Dictionary with it/s and wall_time measurements
    """
    # Load config
    config = load_config(config_path)
    validate_config(config)
    
    if device is None:
        device = torch.device(config.get('device', 'cuda:0'))
    else:
        device = torch.device(device)
        config['device'] = str(device)
    
    seed = config.get('seed', 42)
    setup_seed(seed)
    
    # Enable cuDNN benchmarking for better performance
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    
    print("=" * 80)
    print("Fine-tuning Speed Benchmark")
    print("=" * 80)
    print(f"Config: {config_path}")
    print(f"Phase 1 Checkpoint: {phase1_checkpoint}")
    print(f"Device: {device}")
    print(f"Benchmark Epochs: {benchmark_epochs}")
    print(f"Use torch.compile(): {use_compile}")
    print(f"Number of runs: {num_runs}")
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
    
    processed_splits = convert_to_model_format(splits, phase='phase2', device=device)
    splits_train = processed_splits.get("train_few", splits.train_few)
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
    
    equation = dataset_config['equation']
    
    # Run benchmarks
    wall_times = []
    
    for run_idx in range(num_runs):
        print(f"\nRun {run_idx + 1}/{num_runs}...")
        
        # Create model
        coeff_extrap = _extract_target_coeff_from_sample(splits_extrap[0], device=device) if splits_extrap else target_coeff
        model = create_phase2_model(phase1_checkpoint, hidden_dim, coeff_extrap, device, use_compile=use_compile)
        
        # Warm-up run (let torch.compile optimize)
        if use_compile and run_idx == 0:
            print("  Warming up (torch.compile optimization)...")
            _ = fine_tune_and_evaluate(
                model=model,
                train_split=splits_extrap[:1] if splits_extrap else splits_train[:1],
                eval_split=splits_extrap[:1] if splits_extrap else splits_train[:1],
                eval_split_name='extrap',
                equation=equation,
                epochs=10,
                learning_rate=learning_rate,
                device=device,
                optimizer_name=optimizer_name,
                weight_decay=weight_decay,
                momentum=momentum,
                normalization_stats=normalization_stats,
                checkpoint_epochs=[],
                viz_output_dir=None,
                save_plots=False,
            )
        
        # Benchmark run
        print(f"  Benchmarking {benchmark_epochs} epochs...")
        start_time = time.time()
        
        _ = fine_tune_and_evaluate(
            model=model,
            train_split=splits_extrap if splits_extrap else splits_train,
            eval_split=splits_extrap[:1] if splits_extrap else splits_train[:1],
            eval_split_name='extrap',
            equation=equation,
            epochs=benchmark_epochs,
            learning_rate=learning_rate,
            device=device,
            optimizer_name=optimizer_name,
            weight_decay=weight_decay,
            momentum=momentum,
            normalization_stats=normalization_stats,
            checkpoint_epochs=[],
            viz_output_dir=None,
            save_plots=False,
        )
        
        elapsed_time = time.time() - start_time
        wall_times.append(elapsed_time)
        
        it_per_sec = benchmark_epochs / elapsed_time
        print(f"  Wall time: {elapsed_time:.2f}s, it/s: {it_per_sec:.2f}")
    
    # Calculate statistics
    wall_times = np.array(wall_times)
    mean_wall_time = np.mean(wall_times)
    std_wall_time = np.std(wall_times)
    
    it_per_sec_values = benchmark_epochs / wall_times
    mean_it_per_sec = np.mean(it_per_sec_values)
    std_it_per_sec = np.std(it_per_sec_values)
    
    print("\n" + "=" * 80)
    print("Benchmark Results")
    print("=" * 80)
    print(f"Wall time: {mean_wall_time:.2f} ± {std_wall_time:.2f} s")
    print(f"it/s: {mean_it_per_sec:.2f} ± {std_it_per_sec:.2f}")
    print(f"Wall time for 1000 epochs (estimated): {mean_wall_time * (1000 / benchmark_epochs):.2f} ± {std_wall_time * (1000 / benchmark_epochs):.2f} s")
    print("=" * 80)
    
    return {
        'wall_time_mean': mean_wall_time,
        'wall_time_std': std_wall_time,
        'it_per_sec_mean': mean_it_per_sec,
        'it_per_sec_std': std_it_per_sec,
        'wall_time_1000_epochs_mean': mean_wall_time * (1000 / benchmark_epochs),
        'wall_time_1000_epochs_std': std_wall_time * (1000 / benchmark_epochs),
    }


def main():
    parser = argparse.ArgumentParser(description='Benchmark fine-tuning speed')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to YAML configuration file')
    parser.add_argument('--phase1_checkpoint', type=str, required=True,
                       help='Path to Phase 1 checkpoint')
    parser.add_argument('--benchmark_epochs', type=int, default=200,
                       help='Number of epochs to run for benchmarking (default: 200)')
    parser.add_argument('--use_compile', action='store_true', default=True,
                       help='Use torch.compile() for optimization')
    parser.add_argument('--no_compile', action='store_true',
                       help='Disable torch.compile()')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (overrides config)')
    parser.add_argument('--num_runs', type=int, default=3,
                       help='Number of benchmark runs to average (default: 3)')
    
    args = parser.parse_args()
    
    use_compile = args.use_compile and not args.no_compile
    
    results = benchmark_finetuning_speed(
        config_path=args.config,
        phase1_checkpoint=args.phase1_checkpoint,
        benchmark_epochs=args.benchmark_epochs,
        use_compile=use_compile,
        device=args.device,
        num_runs=args.num_runs,
    )
    
    print("\nResults summary:")
    print(f"  it/s: {results['it_per_sec_mean']:.2f} ± {results['it_per_sec_std']:.2f}")
    print(f"  Wall time (1000 epochs): {results['wall_time_1000_epochs_mean']:.2f} ± {results['wall_time_1000_epochs_std']:.2f} s")


if __name__ == '__main__':
    main()
