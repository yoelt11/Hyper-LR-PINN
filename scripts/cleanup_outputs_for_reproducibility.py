#!/usr/bin/env python
"""
Clean up outputs directory for reproducibility commit.

This script:
1. Keeps only the last epoch PNG from each visualization directory
2. Keeps summary CSV files (model_comparison_summary.csv, finetune_summary*.csv)
3. Keeps aggregated_summary_tables.txt
4. Removes all other intermediate files (checkpoints, intermediate visualizations, etc.)
"""

import os
import re
from pathlib import Path
from collections import defaultdict


def find_last_epoch_pngs(viz_dir):
    """Find the PNG file with the highest epoch number in a visualization directory."""
    if not os.path.exists(viz_dir):
        return []
    
    png_files = []
    for file in os.listdir(viz_dir):
        if file.endswith('.png'):
            # Extract epoch number from filename
            # Pattern: {split}_epoch_{epoch}_samples_{n}.png
            match = re.search(r'epoch_(\d+)', file)
            if match:
                epoch = int(match.group(1))
                png_files.append((epoch, file))
    
    if not png_files:
        return []
    
    # Sort by epoch and get the last one
    png_files.sort(key=lambda x: x[0])
    last_epoch = png_files[-1][0]
    
    # Return all PNGs with the last epoch number (in case there are multiple splits)
    return [f for epoch, f in png_files if epoch == last_epoch]


def cleanup_outputs(outputs_dir='outputs'):
    """Clean up outputs directory, keeping only essential files for reproducibility."""
    outputs_path = Path(outputs_dir)
    if not outputs_path.exists():
        print(f"Outputs directory {outputs_dir} does not exist.")
        return
    
    files_to_keep = []
    files_to_remove = []
    
    # Walk through all files
    for root, dirs, files in os.walk(outputs_path):
        root_path = Path(root)
        
        # Skip if this is a visualization directory - we'll handle it specially
        if root_path.name == 'visualizations':
            # Find last epoch PNGs
            last_epoch_pngs = find_last_epoch_pngs(str(root_path))
            for file in files:
                file_path = root_path / file
                if file.endswith('.png') and file in last_epoch_pngs:
                    files_to_keep.append(file_path)
                    print(f"  KEEP: {file_path}")
                elif file.endswith('.png') or file.endswith('.svg'):
                    files_to_remove.append(file_path)
                # Keep other files in visualization dir (if any)
                elif not (file.endswith('.png') or file.endswith('.svg')):
                    files_to_keep.append(file_path)
            continue
        
        # For other directories, keep summary CSVs and aggregated tables
        for file in files:
            file_path = root_path / file
            if file in ['model_comparison_summary.csv', 'finetune_summary.csv', 
                       'finetune_summary_compact.csv', 'aggregated_summary_tables.txt']:
                files_to_keep.append(file_path)
                print(f"  KEEP: {file_path}")
            elif file.endswith('.csv') or file.endswith('.txt'):
                # Keep other CSV/txt files (metrics, etc.)
                files_to_keep.append(file_path)
            elif file.endswith('.png') or file.endswith('.svg'):
                # Remove all other PNGs/SVGs (not in visualization dirs)
                files_to_remove.append(file_path)
            elif file.endswith('.pt') or file.endswith('.pth'):
                # Remove checkpoints
                files_to_remove.append(file_path)
            elif file.endswith('.pkl') or file.endswith('.json'):
                # Keep metrics files
                files_to_keep.append(file_path)
            else:
                # Keep other files by default
                files_to_keep.append(file_path)
    
    # Remove files
    print(f"\nRemoving {len(files_to_remove)} files...")
    for file_path in files_to_remove:
        try:
            file_path.unlink()
            print(f"  REMOVED: {file_path}")
        except Exception as e:
            print(f"  ERROR removing {file_path}: {e}")
    
    print(f"\nCleanup complete!")
    print(f"  Kept: {len(files_to_keep)} files")
    print(f"  Removed: {len(files_to_remove)} files")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Clean up outputs for reproducibility')
    parser.add_argument('--outputs-dir', default='outputs', help='Outputs directory to clean')
    args = parser.parse_args()
    
    cleanup_outputs(args.outputs_dir)
