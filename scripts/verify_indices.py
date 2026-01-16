#!/usr/bin/env python
"""
Verify that the current configs produce the expected dataset indices.

This script loads each equation with the current config settings and compares
the resulting indices with the expected values.
"""

import sys
import os
import yaml
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_loader import load_parametric_dataset
from src.config_loader import load_config

# Expected indices (from user)
EXPECTED_INDICES = {
    'allen_cahn': {
        'train': [159, 160, 121, 99, 54, 9, 3, 61, 38, 16],
        'interp': [22, 45, 97, 186, 130, 119, 169, 120, 104, 78, 17, 4, 66, 35, 105, 189, 11, 192, 47, 108],
        'extrap': [196, 96, 15, 153, 28, 94, 41, 187, 164, 33, 84, 180, 193, 23, 143, 151, 166, 191, 65, 147]
    },
    'burgers': {
        'train': [160, 161, 118, 95, 50, 7, 2, 57, 34, 13],
        'interp': [22, 37, 97, 153, 90, 105, 128, 137, 124, 106, 96, 0, 79, 32, 43, 169, 3, 185, 47, 81],
        'extrap': [142, 102, 11, 171, 18, 100, 31, 10, 177, 26, 73, 178, 146, 12, 150, 188, 183, 17, 56, 157]
    },
    'convection': {
        'train': [160, 161, 118, 95, 50, 7, 2, 57, 34, 13],
        'interp': [36, 59, 104, 193, 143, 123, 172, 126, 106, 92, 28, 6, 78, 47, 112, 134, 18, 58, 61, 114],
        'extrap': [190, 163, 4, 108, 5, 85, 11, 173, 149, 128, 187, 142, 136, 169, 122, 139, 131, 175, 17, 99]
    },
    'helmholtz2D': {
        'train': [162, 163, 123, 99, 50, 8, 3, 57, 33, 14],
        'interp': [39, 40, 78, 178, 110, 136, 166, 172, 149, 127, 5, 199, 182, 173, 43, 183, 0, 82, 196, 97],
        'extrap': [134, 96, 10, 161, 18, 94, 29, 9, 171, 25, 71, 174, 139, 11, 143, 186, 181, 16, 54, 151]
    }
}


def extract_indices_from_splits(splits):
    """Extract indices from dataset splits."""
    train_indices = [sample.get('index', i) for i, sample in enumerate(splits.train_few)]
    interp_indices = [sample.get('index', i) for i, sample in enumerate(splits.interp)]
    extrap_indices = [sample.get('index', i) for i, sample in enumerate(splits.extrap)]
    return {
        'train': sorted(train_indices),  # Sort for comparison
        'interp': sorted(interp_indices),
        'extrap': sorted(extrap_indices)
    }


def compare_indices(actual, expected, split_name):
    """Compare actual and expected indices for a split."""
    actual_sorted = sorted(actual)
    expected_sorted = sorted(expected)
    
    if actual_sorted == expected_sorted:
        return True, None
    else:
        missing = set(expected_sorted) - set(actual_sorted)
        extra = set(actual_sorted) - set(expected_sorted)
        return False, {'missing': sorted(missing), 'extra': sorted(extra)}


def verify_equation(equation_name, config_path):
    """Verify indices for a single equation."""
    print(f"\n{'='*80}")
    print(f"Verifying: {equation_name}")
    print(f"{'='*80}")
    
    try:
        # Load config
        config = load_config(config_path)
        dataset_config = config['dataset']
        
        # Load dataset with config settings
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
        
        # Extract actual indices
        actual_indices = extract_indices_from_splits(splits)
        
        # Get expected indices
        if equation_name not in EXPECTED_INDICES:
            print(f"⚠️  No expected indices defined for {equation_name}")
            return False
        
        expected_indices = EXPECTED_INDICES[equation_name]
        
        # Compare each split
        all_match = True
        for split_name in ['train', 'interp', 'extrap']:
            actual = actual_indices[split_name]
            expected = expected_indices[split_name]
            
            match, diff = compare_indices(actual, expected, split_name)
            
            if match:
                print(f"  ✓ {split_name:8} ({len(actual):2} samples): Match")
            else:
                all_match = False
                print(f"  ✗ {split_name:8} ({len(actual):2} samples): MISMATCH")
                print(f"    Expected: {expected}")
                print(f"    Actual:   {actual}")
                if diff:
                    if diff['missing']:
                        print(f"    Missing:  {diff['missing']}")
                    if diff['extra']:
                        print(f"    Extra:    {diff['extra']}")
        
        return all_match
        
    except Exception as e:
        print(f"  ✗ Error loading {equation_name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Verify all equations."""
    print("="*80)
    print("Dataset Indices Verification")
    print("="*80)
    print("Comparing actual indices from configs with expected values...")
    
    equations = {
        'allen_cahn': 'configs/allen_cahn_phase1.yaml',
        'burgers': 'configs/burgers_phase1.yaml',
        'convection': 'configs/convection_phase1.yaml',
        'helmholtz2D': 'configs/helmholtz2D_phase1.yaml',
    }
    
    results = {}
    for eq_name, config_path in equations.items():
        if not os.path.exists(config_path):
            print(f"\n⚠️  Config not found: {config_path}")
            results[eq_name] = False
            continue
        
        results[eq_name] = verify_equation(eq_name, config_path)
    
    # Summary
    print("\n" + "="*80)
    print("Verification Summary")
    print("="*80)
    
    all_passed = True
    for eq_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {eq_name:15} {status}")
        if not passed:
            all_passed = False
    
    print("="*80)
    
    if all_passed:
        print("\n✓ All indices match expected values!")
        return 0
    else:
        print("\n✗ Some indices do not match. Please check the output above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
