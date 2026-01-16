#!/usr/bin/env python
"""
Aggregate results across multiple seeds.

This script:
1. Loads metrics from outputs/seed_{seed}/ directories
2. Computes mean ± std across seeds for L2 errors and inference times
3. Generates summary tables (separate for interpolation and extrapolation)
4. Saves to outputs/aggregated_summary_tables.txt

Usage:
    python scripts/aggregate_seed_results.py --seeds 0,1,2
"""

import argparse
import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

# Equations to process
EQUATIONS = ['allen_cahn', 'burgers', 'convection', 'helmholtz2D']


def parse_l2_error(error_str):
    """Parse L2 error string like '0.1205 ± 0.20' or 'N/A'."""
    if error_str == 'N/A' or pd.isna(error_str):
        return None, None
    try:
        parts = error_str.split('±')
        mean = float(parts[0].strip())
        std = float(parts[1].strip()) if len(parts) > 1 else 0.0
        return mean, std
    except:
        return None, None


def parse_time(time_str):
    """Parse time string like '9.21' or 'N/A'."""
    if time_str == 'N/A' or pd.isna(time_str):
        return None
    try:
        return float(time_str)
    except:
        return None


def load_seed_data(seed, equation):
    """
    Load data for a specific seed and equation.
    
    Returns:
        dict with keys: 'interp' and 'extrap', each containing:
        - zero_shot_l2: (mean, std) tuple
        - zero_shot_time: float
        - ft_epochs: dict mapping epoch -> (l2_mean, l2_std, time)
    """
    base_path = Path(f"outputs/seed_{seed}/{equation}_phase2/evaluation")
    compact_csv = base_path / "finetune_summary_compact.csv"
    
    if not compact_csv.exists():
        return None
    
    try:
        df = pd.read_csv(compact_csv)
        
        data = {'interp': {}, 'extrap': {}}
        
        # Find zero-shot row (epoch 0)
        zero_shot_row = df[df['Epoch'] == 0]
        if not zero_shot_row.empty:
            row = zero_shot_row.iloc[0]
            
            # Interpolation
            interp_l2_mean, interp_l2_std = parse_l2_error(row['Interp_L2_Error'])
            interp_time = parse_time(row['Interp_Time'])
            if interp_l2_mean is not None:
                data['interp']['zero_shot'] = {
                    'l2_mean': interp_l2_mean,
                    'l2_std': interp_l2_std,
                    'time': interp_time if interp_time is not None else 0.0
                }
            
            # Extrapolation
            extrap_l2_mean, extrap_l2_std = parse_l2_error(row['Extrap_L2_Error'])
            extrap_time = parse_time(row['Extrap_Time'])
            if extrap_l2_mean is not None:
                data['extrap']['zero_shot'] = {
                    'l2_mean': extrap_l2_mean,
                    'l2_std': extrap_l2_std,
                    'time': extrap_time if extrap_time is not None else 0.0
                }
        
        # Find fine-tuning rows (epoch > 0)
        ft_rows = df[df['Epoch'] > 0]
        for _, row in ft_rows.iterrows():
            epoch = int(row['Epoch'])
            
            # Interpolation
            interp_l2_mean, interp_l2_std = parse_l2_error(row['Interp_L2_Error'])
            interp_time = parse_time(row['Interp_Time'])
            if interp_l2_mean is not None:
                if 'ft_epochs' not in data['interp']:
                    data['interp']['ft_epochs'] = {}
                data['interp']['ft_epochs'][epoch] = {
                    'l2_mean': interp_l2_mean,
                    'l2_std': interp_l2_std,
                    'time': interp_time if interp_time is not None else 0.0
                }
            
            # Extrapolation
            extrap_l2_mean, extrap_l2_std = parse_l2_error(row['Extrap_L2_Error'])
            extrap_time = parse_time(row['Extrap_Time'])
            if extrap_l2_mean is not None:
                if 'ft_epochs' not in data['extrap']:
                    data['extrap']['ft_epochs'] = {}
                data['extrap']['ft_epochs'][epoch] = {
                    'l2_mean': extrap_l2_mean,
                    'l2_std': extrap_l2_std,
                    'time': extrap_time if extrap_time is not None else 0.0
                }
        
        return data
        
    except Exception as e:
        print(f"⚠️  Error loading data for {equation} seed {seed}: {e}")
        return None


def aggregate_across_seeds(seeds, equation):
    """
    Aggregate data across seeds for a single equation.
    
    Returns:
        dict with 'interp' and 'extrap' keys, each containing aggregated metrics
    """
    all_data = []
    for seed in seeds:
        data = load_seed_data(seed, equation)
        if data:
            all_data.append(data)
    
    if not all_data:
        return None
    
    # Aggregate interpolation data
    interp_aggregated = {}
    
    # Zero-shot aggregation
    zero_shot_l2_means = []
    zero_shot_l2_stds = []
    zero_shot_times = []
    for data in all_data:
        if 'zero_shot' in data.get('interp', {}):
            zs = data['interp']['zero_shot']
            zero_shot_l2_means.append(zs['l2_mean'])
            zero_shot_l2_stds.append(zs['l2_std'])
            zero_shot_times.append(zs['time'])
    
    if zero_shot_l2_means:
        interp_aggregated['zero_shot'] = {
            'l2_mean': np.mean(zero_shot_l2_means),
            'l2_std': np.std(zero_shot_l2_means, ddof=1) if len(zero_shot_l2_means) > 1 else 0.0,
            'time_mean': np.mean(zero_shot_times),
            'time_std': np.std(zero_shot_times, ddof=1) if len(zero_shot_times) > 1 else 0.0
        }
    
    # Fine-tuning aggregation
    all_ft_epochs = set()
    for data in all_data:
        if 'ft_epochs' in data.get('interp', {}):
            all_ft_epochs.update(data['interp']['ft_epochs'].keys())
    
    if all_ft_epochs:
        interp_aggregated['ft_epochs'] = {}
        for epoch in sorted(all_ft_epochs):
            epoch_l2_means = []
            epoch_l2_stds = []
            epoch_times = []
            for data in all_data:
                if 'ft_epochs' in data.get('interp', {}) and epoch in data['interp']['ft_epochs']:
                    ft = data['interp']['ft_epochs'][epoch]
                    epoch_l2_means.append(ft['l2_mean'])
                    epoch_l2_stds.append(ft['l2_std'])
                    epoch_times.append(ft['time'])
            
            if epoch_l2_means:
                interp_aggregated['ft_epochs'][epoch] = {
                    'l2_mean': np.mean(epoch_l2_means),
                    'l2_std': np.std(epoch_l2_means, ddof=1) if len(epoch_l2_means) > 1 else 0.0,
                    'time_mean': np.mean(epoch_times),
                    'time_std': np.std(epoch_times, ddof=1) if len(epoch_times) > 1 else 0.0
                }
    
    # Aggregate extrapolation data (same logic)
    extrap_aggregated = {}
    
    zero_shot_l2_means = []
    zero_shot_l2_stds = []
    zero_shot_times = []
    for data in all_data:
        if 'zero_shot' in data.get('extrap', {}):
            zs = data['extrap']['zero_shot']
            zero_shot_l2_means.append(zs['l2_mean'])
            zero_shot_l2_stds.append(zs['l2_std'])
            zero_shot_times.append(zs['time'])
    
    if zero_shot_l2_means:
        extrap_aggregated['zero_shot'] = {
            'l2_mean': np.mean(zero_shot_l2_means),
            'l2_std': np.std(zero_shot_l2_means, ddof=1) if len(zero_shot_l2_means) > 1 else 0.0,
            'time_mean': np.mean(zero_shot_times),
            'time_std': np.std(zero_shot_times, ddof=1) if len(zero_shot_times) > 1 else 0.0
        }
    
    all_ft_epochs = set()
    for data in all_data:
        if 'ft_epochs' in data.get('extrap', {}):
            all_ft_epochs.update(data['extrap']['ft_epochs'].keys())
    
    if all_ft_epochs:
        extrap_aggregated['ft_epochs'] = {}
        for epoch in sorted(all_ft_epochs):
            epoch_l2_means = []
            epoch_l2_stds = []
            epoch_times = []
            for data in all_data:
                if 'ft_epochs' in data.get('extrap', {}) and epoch in data['extrap']['ft_epochs']:
                    ft = data['extrap']['ft_epochs'][epoch]
                    epoch_l2_means.append(ft['l2_mean'])
                    epoch_l2_stds.append(ft['l2_std'])
                    epoch_times.append(ft['time'])
            
            if epoch_l2_means:
                extrap_aggregated['ft_epochs'][epoch] = {
                    'l2_mean': np.mean(epoch_l2_means),
                    'l2_std': np.std(epoch_l2_means, ddof=1) if len(epoch_l2_means) > 1 else 0.0,
                    'time_mean': np.mean(epoch_times),
                    'time_std': np.std(epoch_times, ddof=1) if len(epoch_times) > 1 else 0.0
                }
    
    return {
        'interp': interp_aggregated,
        'extrap': extrap_aggregated
    }


def format_table(aggregated_data, split_name, output_file):
    """
    Format and write a summary table for interpolation or extrapolation.
    
    Args:
        aggregated_data: Dict mapping equation -> {'interp': {...}, 'extrap': {...}}
        split_name: 'interp' or 'extrap'
        output_file: File handle to write to
    """
    # Get all epochs across all equations
    all_epochs = set()
    for eq_data in aggregated_data.values():
        if split_name in eq_data and 'ft_epochs' in eq_data[split_name]:
            all_epochs.update(eq_data[split_name]['ft_epochs'].keys())
    all_epochs = sorted(all_epochs)
    
    # Column widths
    col_widths = {
        'Equation': 20,
        'Model': 15,
        'Zero-shot (Rel. L2 Error)': 30,
        'ACC Time (ms)': 15,
        'FT epochs': 25,
        'ACC Time': 15
    }
    
    # Calculate total width
    total_width = (col_widths['Equation'] + col_widths['Model'] + 
                   col_widths['Zero-shot (Rel. L2 Error)'] + col_widths['ACC Time (ms)'] +
                   len(all_epochs) * (col_widths['FT epochs'] + col_widths['ACC Time']))
    
    # Write header
    output_file.write(f"\n{'='*total_width}\n")
    split_display = "INTERPOLATION" if split_name == "interp" else "EXTRAPOLATION"
    output_file.write(f"{split_display} Results (Mean ± Std across seeds)\n")
    output_file.write(f"{'='*total_width}\n\n")
    
    # Write column headers
    header_parts = [
        f"{'Equation':<{col_widths['Equation']}}",
        f"{'Model':<{col_widths['Model']}}",
        f"{'Zero-shot (Rel. L2 Error)':<{col_widths['Zero-shot (Rel. L2 Error)']}}",
        f"{'ACC Time (ms)':<{col_widths['ACC Time (ms)']}}"
    ]
    for epoch in all_epochs:
        header_parts.append(f"{'FT after ' + str(epoch) + ' epochs':<{col_widths['FT epochs']}}")
        header_parts.append(f"{'ACC Time':<{col_widths['ACC Time']}}")
    header = "".join(header_parts)
    output_file.write(header + "\n")
    output_file.write("-" * total_width + "\n")
    
    # Write data rows (sorted by equation name)
    for equation in sorted(aggregated_data.keys()):
        eq_data = aggregated_data[equation]
        if split_name not in eq_data:
            continue
        
        split_data = eq_data[split_name]
        
        # Format equation name
        eq_display = equation.replace('_', ' ').title()
        if '2d' in eq_display.lower():
            eq_display = eq_display.replace('2D', '2D').replace('2d', '2D')
        
        # Zero-shot data
        if 'zero_shot' in split_data:
            zs = split_data['zero_shot']
            zero_shot_l2 = f"{zs['l2_mean']:.4f} ± {zs['l2_std']:.4f}"
            zero_shot_time = f"{zs['time_mean']*1000:.2f} ± {zs['time_std']*1000:.2f}"
        else:
            zero_shot_l2 = "N/A"
            zero_shot_time = "N/A"
        
        # Build row
        row_parts = [
            f"{eq_display:<{col_widths['Equation']}}",
            f"{'Hyper-LR-PINN':<{col_widths['Model']}}",
            f"{zero_shot_l2:<{col_widths['Zero-shot (Rel. L2 Error)']}}",
            f"{zero_shot_time:<{col_widths['ACC Time (ms)']}}"
        ]
        
        # Fine-tuning data for each epoch
        for epoch in all_epochs:
            if 'ft_epochs' in split_data and epoch in split_data['ft_epochs']:
                ft = split_data['ft_epochs'][epoch]
                ft_l2 = f"{ft['l2_mean']:.4f} ± {ft['l2_std']:.4f}"
                ft_time = f"{ft['time_mean']*1000:.2f} ± {ft['time_std']*1000:.2f}"
            else:
                ft_l2 = "N/A"
                ft_time = "N/A"
            
            row_parts.append(f"{ft_l2:<{col_widths['FT epochs']}}")
            row_parts.append(f"{ft_time:<{col_widths['ACC Time']}}")
        
        row = "".join(row_parts)
        output_file.write(row + "\n")
    
    output_file.write("=" * total_width + "\n\n")


def main():
    parser = argparse.ArgumentParser(
        description='Aggregate results across multiple seeds',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--seeds',
        type=str,
        required=True,
        help='Comma-separated list of seeds (e.g., "0,1,2")'
    )
    parser.add_argument(
        '--equations',
        nargs='+',
        default=EQUATIONS,
        help=f'Equations to aggregate (default: all: {", ".join(EQUATIONS)})'
    )
    
    args = parser.parse_args()
    
    # Parse seeds
    try:
        seeds = [int(s.strip()) for s in args.seeds.split(',')]
    except ValueError:
        print(f"❌ Invalid seeds format: {args.seeds}. Use comma-separated integers (e.g., '0,1,2')")
        sys.exit(1)
    
    print("=" * 80)
    print("Aggregating Results Across Seeds")
    print("=" * 80)
    print(f"Seeds: {seeds}")
    print(f"Equations: {', '.join(args.equations)}")
    print("=" * 80)
    
    # Aggregate data for each equation
    aggregated_data = {}
    missing_data = []
    
    for equation in args.equations:
        print(f"\nProcessing: {equation}")
        data = aggregate_across_seeds(seeds, equation)
        if data:
            aggregated_data[equation] = data
            print(f"  ✓ Aggregated data from {len([s for s in seeds if load_seed_data(s, equation)])} seeds")
        else:
            missing_data.append(equation)
            print(f"  ⚠️  No data found for {equation}")
    
    if not aggregated_data:
        print("\n❌ No data found for any equation. Check that results exist in outputs/seed_{seed}/ directories.")
        sys.exit(1)
    
    # Generate summary tables
    output_path = Path("outputs/aggregated_summary_tables.txt")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("Aggregated Results Summary (Mean ± Std across seeds)\n")
        f.write("=" * 80 + "\n")
        f.write(f"Seeds: {', '.join(map(str, seeds))}\n")
        f.write(f"Equations: {', '.join(sorted(aggregated_data.keys()))}\n")
        if missing_data:
            f.write(f"Missing: {', '.join(missing_data)}\n")
        f.write("=" * 80 + "\n")
        
        # Write interpolation table
        format_table(aggregated_data, 'interp', f)
        
        # Write extrapolation table
        format_table(aggregated_data, 'extrap', f)
    
    print(f"\n✓ Summary tables saved to: {output_path}")
    print("\n" + "=" * 80)
    print("Preview:")
    print("=" * 80)
    with open(output_path, 'r') as f:
        print(f.read())


if __name__ == '__main__':
    main()
