#!/usr/bin/env python
"""
Run benchmarks for all equations to collect it/s and wall-time measurements.

This script runs short benchmark runs for all equations and seeds,
then updates the LaTeX tables with optimized it/s values while keeping
error values unchanged.

Usage:
    python scripts/run_all_benchmarks.py --seeds 0,1,2 --benchmark_epochs 200
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path
import json

EQUATIONS = ['allen_cahn', 'burgers', 'convection', 'helmholtz2D']


def run_benchmark(equation: str, seed: int, benchmark_epochs: int = 200, use_compile: bool = True):
    """Run benchmark for a single equation and seed."""
    config_path = f"configs/{equation}_phase2.yaml"
    checkpoint_dir = f"outputs/seed_{seed}/{equation}_phase1/checkpoints"
    checkpoint_path = f"{checkpoint_dir}/best.pt"
    
    if not os.path.exists(config_path):
        print(f"  [SKIP] Config not found: {config_path}")
        return None
    
    # Try to find checkpoint - check for best.pt first, then latest checkpoint
    if not os.path.exists(checkpoint_path):
        # Try to find latest checkpoint
        if os.path.exists(checkpoint_dir):
            import glob
            checkpoints = glob.glob(f"{checkpoint_dir}/checkpoint_epoch_*.pt")
            if checkpoints:
                # Sort by epoch number and use the latest
                checkpoints.sort(key=lambda x: int(x.split('_')[-1].replace('.pt', '')))
                checkpoint_path = checkpoints[-1]
                print(f"  [INFO] Using latest checkpoint: {checkpoint_path}")
            else:
                print(f"  [SKIP] No checkpoints found in {checkpoint_dir}")
                return None
        else:
            print(f"  [SKIP] Checkpoint directory not found: {checkpoint_dir}")
            return None
    
    cmd = [
        sys.executable,
        "scripts/benchmark_finetune_speed.py",
        "--config", config_path,
        "--phase1_checkpoint", checkpoint_path,
        "--benchmark_epochs", str(benchmark_epochs),
        "--num_runs", "3",
    ]
    
    if use_compile:
        cmd.append("--use_compile")
    else:
        cmd.append("--no_compile")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(__file__)))
        if result.returncode != 0:
            print(f"  [ERROR] Benchmark failed: {result.stderr}")
            return None
        
        # Parse output to extract it/s and wall_time
        output = result.stdout
        # Look for "it/s: X.XX ± Y.YY" pattern
        import re
        it_per_sec_match = re.search(r'it/s:\s*([\d.]+)\s*±\s*([\d.]+)', output)
        wall_time_match = re.search(r'Wall time \(1000 epochs\):\s*([\d.]+)\s*±\s*([\d.]+)', output)
        
        if it_per_sec_match and wall_time_match:
            return {
                'it_per_sec_mean': float(it_per_sec_match.group(1)),
                'it_per_sec_std': float(it_per_sec_match.group(2)),
                'wall_time_1000_mean': float(wall_time_match.group(1)),
                'wall_time_1000_std': float(wall_time_match.group(2)),
            }
        else:
            print(f"  [WARN] Could not parse output")
            print(output)
            return None
    except Exception as e:
        print(f"  [ERROR] Exception: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Run benchmarks for all equations')
    parser.add_argument('--seeds', type=str, default='0,1,2',
                       help='Comma-separated list of seeds (default: 0,1,2)')
    parser.add_argument('--benchmark_epochs', type=int, default=200,
                       help='Number of epochs for benchmarking (default: 200)')
    parser.add_argument('--use_compile', action='store_true', default=True,
                       help='Use torch.compile()')
    parser.add_argument('--no_compile', action='store_true',
                       help='Disable torch.compile()')
    parser.add_argument('--output', type=str, default='benchmark_results.json',
                       help='Output JSON file for results')
    
    args = parser.parse_args()
    
    seeds = [int(s.strip()) for s in args.seeds.split(',')]
    use_compile = args.use_compile and not args.no_compile
    
    print("=" * 80)
    print("Running Fine-tuning Speed Benchmarks")
    print("=" * 80)
    print(f"Seeds: {seeds}")
    print(f"Equations: {EQUATIONS}")
    print(f"Benchmark epochs: {args.benchmark_epochs}")
    print(f"Use torch.compile(): {use_compile}")
    print("=" * 80)
    
    results = {}
    
    for equation in EQUATIONS:
        print(f"\n{equation.upper()}:")
        equation_results = {}
        
        for seed in seeds:
            print(f"  Seed {seed}...")
            result = run_benchmark(equation, seed, args.benchmark_epochs, use_compile)
            if result:
                equation_results[f'seed_{seed}'] = result
        
        if equation_results:
            # Calculate mean ± std across seeds
            it_per_sec_values = [r['it_per_sec_mean'] for r in equation_results.values()]
            wall_time_values = [r['wall_time_1000_mean'] for r in equation_results.values()]
            
            import numpy as np
            results[equation] = {
                'it_per_sec_mean': np.mean(it_per_sec_values),
                'it_per_sec_std': np.std(it_per_sec_values),
                'wall_time_1000_mean': np.mean(wall_time_values),
                'wall_time_1000_std': np.std(wall_time_values),
                'per_seed': equation_results
            }
            
            print(f"  Mean it/s: {results[equation]['it_per_sec_mean']:.2f} ± {results[equation]['it_per_sec_std']:.2f}")
            print(f"  Mean wall time (1000 epochs): {results[equation]['wall_time_1000_mean']:.2f} ± {results[equation]['wall_time_1000_std']:.2f} s")
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 80)
    print("Benchmark Results Summary")
    print("=" * 80)
    for eq, res in results.items():
        print(f"{eq}:")
        print(f"  it/s: {res['it_per_sec_mean']:.2f} ± {res['it_per_sec_std']:.2f}")
        print(f"  Wall time (1000 epochs): {res['wall_time_1000_mean']:.2f} ± {res['wall_time_1000_std']:.2f} s")
    print(f"\nResults saved to: {args.output}")
    print("=" * 80)


if __name__ == '__main__':
    main()
