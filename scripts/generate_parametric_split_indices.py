#!/usr/bin/env python3
"""
Generate parametric split indices using the same strategy as eff-physics-learn-dataset.

This script uses the exact configuration from the eff-physics-learn-dataset project:
- method="solution_percentile"
- n_train=10
- n_each=20
- balance=True
- balance_strategy="random"
- percentile=50.0 (default)
- diversify=False (default)
- replace=False (default)
- on_insufficient="cap" (default)
"""

import sys
import os
from pathlib import Path
import csv

# Import directly from eff-physics-learn-dataset
try:
    from eff_physics_learn_dataset import load_pde_dataset
except ImportError:
    try:
        from eff_physics_learn_dataset.dataset import load_pde_dataset
    except ImportError:
        try:
            import eff_physics_learn_dataset as epld
            load_pde_dataset = epld.load_pde_dataset
        except (ImportError, AttributeError):
            raise ImportError(
                "eff-physics-learn-dataset not found. Please install it using:\n"
                "pip install git+ssh://git@github.com/yoelt11/eff-physics-learn-dataset.git\n"
                "or\n"
                "uv pip install git+ssh://git@github.com/yoelt11/eff-physics-learn-dataset.git"
            )

def generate_indices_for_equation(equation: str, data_dir: str, seed: int, output_file: Path):
    """Generate indices for a single equation and seed."""
    print(f"\n{'='*80}")
    print(f"Generating indices for {equation} (seed {seed})")
    print(f"{'='*80}")
    
    # Load dataset
    dataset = load_pde_dataset(equation, data_dir, cache=True)
    
    # Use the exact same parameters as eff-physics-learn-dataset
    splits = dataset.parametric_splits(
        seed=seed,
        n_train=10,
        method="solution_percentile",  # Explicitly set method
        balance=True,
        n_each=20,
        balance_strategy="random",
        diversify=False,
        percentile=50.0,  # Default percentile for 50/50 interp/extrap split
    )
    
    # Extract indices
    train_indices = [sample.get('index', i) for i, sample in enumerate(splits['train_few'])]
    interp_indices = [sample.get('index', i) for i, sample in enumerate(splits['interp'])]
    extrap_indices = [sample.get('index', i) for i, sample in enumerate(splits['extrap'])]
    
    # Sort for consistency
    train_indices = sorted(train_indices)
    interp_indices = sorted(interp_indices)
    extrap_indices = sorted(extrap_indices)
    
    print(f"Train: {len(train_indices)} samples")
    print(f"  Indices: {train_indices}")
    print(f"Interp: {len(interp_indices)} samples")
    print(f"  Indices: {interp_indices[:5]}... (showing first 5)")
    print(f"Extrap: {len(extrap_indices)} samples")
    print(f"  Indices: {extrap_indices[:5]}... (showing first 5)")
    
    # Format as space-separated strings
    train_str = ' '.join(map(str, train_indices))
    interp_str = ' '.join(map(str, interp_indices))
    extrap_str = ' '.join(map(str, extrap_indices))
    
    return {
        'equation': equation,
        'seed': seed,
        'train_indices': train_str,
        'interp_indices': interp_str,
        'extrap_indices': extrap_str
    }

def main():
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / 'data_gen' / 'dataset'
    
    # Check if data directory exists
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        print("Please ensure the dataset is available at this location.")
        return 1
    
    # Equations and seeds to process
    equations = ["allen_cahn", "burgers", "convection", "helmholtz2D"]
    seeds = [0, 1]  # Matching the user's CSV file
    
    # Output file
    output_file = base_dir / 'outputs' / 'parametric_split_indices.csv'
    
    print("="*80)
    print("Generating Parametric Split Indices")
    print("="*80)
    print(f"Data directory: {data_dir}")
    print(f"Output file: {output_file}")
    print(f"Equations: {equations}")
    print(f"Seeds: {seeds}")
    print("\nUsing configuration:")
    print("  method='solution_percentile'")
    print("  n_train=10")
    print("  n_each=20")
    print("  balance=True")
    print("  balance_strategy='random'")
    print("  percentile=50.0")
    print("  diversify=False")
    
    # Generate indices for all combinations
    results = []
    for equation in equations:
        for seed in seeds:
            try:
                result = generate_indices_for_equation(equation, str(data_dir), seed, output_file)
                results.append(result)
            except Exception as e:
                print(f"\n❌ Error processing {equation} (seed {seed}): {e}")
                import traceback
                traceback.print_exc()
                return 1
    
    # Write to CSV
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['equation', 'seed', 'train_indices', 'interp_indices', 'extrap_indices'])
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    
    print(f"\n{'='*80}")
    print(f"✓ Successfully generated indices file: {output_file}")
    print(f"{'='*80}")
    
    # Display summary
    print("\nSummary:")
    for result in results:
        train_count = len(result['train_indices'].split())
        interp_count = len(result['interp_indices'].split())
        extrap_count = len(result['extrap_indices'].split())
        print(f"  {result['equation']:15s} seed {result['seed']}: "
              f"train={train_count}, interp={interp_count}, extrap={extrap_count}")
    
    return 0

if __name__ == '__main__':
    exit(main())
