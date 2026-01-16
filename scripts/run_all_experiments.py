#!/usr/bin/env python
"""
Run all experiments with multi-seed support.

This script:
1. Trains Phase 1 (meta-learning) for each equation and seed
2. Runs Phase 2 finetune evaluation for each equation and seed

For each seed, it creates temporary config files with updated seed values
and organizes outputs in outputs/seed_{seed}/ directories.

Usage:
    python scripts/run_all_experiments.py --seeds 0,1,2 [OPTIONS]

Options:
    --seeds SEEDS              Comma-separated list of seeds (e.g., "0,1,2")
    --equations EQUATION ...    Specific equations to run (default: all)
    --skip_phase1              Skip Phase 1 training (assumes checkpoints exist)
    --skip_phase2              Skip Phase 2 finetune evaluation
    --device DEVICE            Device to use (overrides config)
    --max_epochs N             Max epochs for finetune evaluation (default: 1000)
"""

import argparse
import subprocess
import sys
import os
import yaml
import tempfile
import shutil
from pathlib import Path

# Equations and their config files
EQUATIONS = {
    'allen_cahn': {
        'phase1': 'configs/allen_cahn_phase1.yaml',
        'phase2': 'configs/allen_cahn_phase2.yaml',
    },
    'convection': {
        'phase1': 'configs/convection_phase1.yaml',
        'phase2': 'configs/convection_phase2.yaml',
    },
    'burgers': {
        'phase1': 'configs/burgers_phase1.yaml',
        'phase2': 'configs/burgers_phase2.yaml',
    },
    'helmholtz2D': {
        'phase1': 'configs/helmholtz2D_phase1.yaml',
        'phase2': 'configs/helmholtz2D_phase2.yaml',
    },
}


def create_temp_config(original_config_path, seed, output_dir_base):
    """
    Create a temporary config file with updated seed and output directory.
    
    Args:
        original_config_path: Path to original config file
        seed: Seed value to use
        output_dir_base: Base output directory (will be updated to outputs/seed_{seed}/...)
        
    Returns:
        Path to temporary config file
    """
    # Load original config
    with open(original_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update seed in dataset section
    if 'dataset' in config:
        config['dataset']['seed'] = seed
    
    # Update output directory to include seed
    if 'export' in config and 'output_dir' in config['export']:
        original_output = config['export']['output_dir']
        # Normalize path (handle ./outputs, outputs/, etc.)
        original_output = original_output.replace('./', '').strip('/')
        # Replace outputs with outputs/seed_{seed}
        if original_output == 'outputs' or original_output.startswith('outputs/'):
            # Remove 'outputs' prefix and any leading slashes
            path_after_outputs = original_output.replace('outputs', '').strip('/')
            if path_after_outputs:
                config['export']['output_dir'] = f"outputs/seed_{seed}/{path_after_outputs}"
            else:
                config['export']['output_dir'] = f"outputs/seed_{seed}"
        else:
            config['export']['output_dir'] = f"outputs/seed_{seed}/{original_output}"
    
    # Update phase1_checkpoint path in phase2 configs
    if 'model' in config and 'phase1_checkpoint' in config['model']:
        original_checkpoint = config['model']['phase1_checkpoint']
        if original_checkpoint and original_checkpoint.startswith('outputs/'):
            # Extract equation name from checkpoint path
            # e.g., outputs/allen_cahn_phase1/checkpoints/best.pt
            parts = original_checkpoint.split('/')
            if len(parts) >= 3:
                equation_phase = parts[1]  # e.g., "allen_cahn_phase1"
                checkpoint_file = '/'.join(parts[2:])  # e.g., "checkpoints/best.pt"
                config['model']['phase1_checkpoint'] = f"outputs/seed_{seed}/{equation_phase}/{checkpoint_file}"
    
    # Create temporary file
    temp_dir = tempfile.mkdtemp(prefix='hyper_lr_pinn_config_')
    temp_config_path = os.path.join(temp_dir, os.path.basename(original_config_path))
    
    with open(temp_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    return temp_config_path, temp_dir


def run_command(cmd, description):
    """Run a command and handle errors."""
    print("\n" + "=" * 80)
    print(f"Running: {description}")
    print("=" * 80)
    print(f"Command: {' '.join(cmd)}")
    print("=" * 80)
    
    result = subprocess.run(cmd, cwd=os.getcwd())
    
    if result.returncode != 0:
        print(f"\n❌ Error: {description} failed with exit code {result.returncode}")
        return False
    else:
        print(f"\n✓ Success: {description} completed")
        return True


def check_phase1_checkpoint(equation, seed):
    """Check if Phase 1 checkpoint exists for a given seed."""
    checkpoint_path = f"outputs/seed_{seed}/{equation}_phase1/checkpoints/best.pt"
    if os.path.exists(checkpoint_path):
        return True, checkpoint_path
    return False, checkpoint_path


def main():
    parser = argparse.ArgumentParser(
        description='Run all experiments with multi-seed support',
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
        choices=list(EQUATIONS.keys()) + ['all'],
        default=['all'],
        help='Equations to run (default: all)'
    )
    parser.add_argument(
        '--skip_phase1',
        action='store_true',
        help='Skip Phase 1 training (assumes checkpoints exist)'
    )
    parser.add_argument(
        '--skip_phase2',
        action='store_true',
        help='Skip Phase 2 finetune evaluation'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (overrides config: cuda:0, cpu, etc.)'
    )
    parser.add_argument(
        '--max_epochs',
        type=int,
        default=1000,
        help='Max epochs for finetune evaluation (default: 1000)'
    )
    
    args = parser.parse_args()
    
    # Parse seeds
    try:
        seeds = [int(s.strip()) for s in args.seeds.split(',')]
    except ValueError:
        print(f"❌ Invalid seeds format: {args.seeds}. Use comma-separated integers (e.g., '0,1,2')")
        sys.exit(1)
    
    # Determine which equations to run
    if 'all' in args.equations:
        equations_to_run = list(EQUATIONS.keys())
    else:
        equations_to_run = [eq for eq in args.equations if eq in EQUATIONS]
    
    if not equations_to_run:
        print("❌ No valid equations specified")
        sys.exit(1)
    
    print("=" * 80)
    print("Multi-Seed Experiments: Phase 1 Training + Phase 2 Finetune Evaluation")
    print("=" * 80)
    print(f"Seeds: {seeds}")
    print(f"Equations: {', '.join(equations_to_run)}")
    print(f"Skip Phase 1: {args.skip_phase1}")
    print(f"Skip Phase 2: {args.skip_phase2}")
    print(f"Max epochs (Phase 2): {args.max_epochs}")
    if args.device:
        print(f"Device override: {args.device}")
    print("=" * 80)
    
    results = {}
    temp_dirs = []  # Track temp directories for cleanup
    
    try:
        for seed in seeds:
            print(f"\n{'='*80}")
            print(f"Processing Seed: {seed}")
            print(f"{'='*80}")
            
            results[seed] = {}
            
            for equation in equations_to_run:
                print(f"\n{'='*80}")
                print(f"Seed {seed} | Equation: {equation}")
                print(f"{'='*80}")
                
                configs = EQUATIONS[equation]
                results[seed][equation] = {'phase1': None, 'phase2': None}
                
                # Create temporary configs for this seed
                phase1_temp, temp_dir1 = create_temp_config(configs['phase1'], seed, f"outputs/seed_{seed}")
                phase2_temp, temp_dir2 = create_temp_config(configs['phase2'], seed, f"outputs/seed_{seed}")
                temp_dirs.extend([temp_dir1, temp_dir2])
                
                # Phase 1: Training
                if not args.skip_phase1:
                    phase1_cmd = ['python', 'scripts/train.py', '--config', phase1_temp]
                    if args.device:
                        phase1_cmd.extend(['--device', args.device])
                    
                    success = run_command(phase1_cmd, f"Phase 1 Training: {equation} (seed {seed})")
                    results[seed][equation]['phase1'] = success
                    
                    if not success:
                        print(f"⚠️  Phase 1 failed for {equation} (seed {seed}), skipping Phase 2")
                        continue
                    
                    # Check that checkpoint was created
                    checkpoint_exists, checkpoint_path = check_phase1_checkpoint(equation, seed)
                    if not checkpoint_exists:
                        print(f"⚠️  Phase 1 checkpoint not found at {checkpoint_path}")
                        print(f"⚠️  Skipping Phase 2 for {equation} (seed {seed})")
                        continue
                else:
                    # Verify checkpoint exists if skipping Phase 1
                    checkpoint_exists, checkpoint_path = check_phase1_checkpoint(equation, seed)
                    if not checkpoint_exists:
                        print(f"⚠️  Phase 1 checkpoint not found at {checkpoint_path}")
                        print(f"⚠️  Skipping Phase 2 for {equation} (seed {seed})")
                        continue
                
                # Phase 2: Finetune Evaluation
                if not args.skip_phase2:
                    checkpoint_path = f"outputs/seed_{seed}/{equation}_phase1/checkpoints/best.pt"
                    phase2_cmd = [
                        'python', 'scripts/finetune_eval.py',
                        '--config', phase2_temp,
                        '--phase1_checkpoint', checkpoint_path,
                        '--max_epochs', str(args.max_epochs)
                    ]
                    if args.device:
                        phase2_cmd.extend(['--device', args.device])
                    
                    success = run_command(phase2_cmd, f"Phase 2 Finetune Evaluation: {equation} (seed {seed})")
                    results[seed][equation]['phase2'] = success
        
        # Print summary
        print("\n" + "=" * 80)
        print("Pipeline Summary")
        print("=" * 80)
        for seed in seeds:
            print(f"\nSeed {seed}:")
            for equation in equations_to_run:
                if equation in results.get(seed, {}):
                    result = results[seed][equation]
                    phase1_status = "✓" if result['phase1'] else ("⏭️ " if args.skip_phase1 else "❌")
                    phase2_status = "✓" if result['phase2'] else ("⏭️ " if args.skip_phase2 else "❌")
                    print(f"  {equation:15} | Phase 1: {phase1_status:3} | Phase 2: {phase2_status:3}")
        print("=" * 80)
        
        # Exit with error if any phase failed
        all_success = True
        for seed in seeds:
            for equation in equations_to_run:
                if equation in results.get(seed, {}):
                    result = results[seed][equation]
                    if not ((result['phase1'] or args.skip_phase1) and (result['phase2'] or args.skip_phase2)):
                        all_success = False
        
        if all_success:
            print("\n✓ All phases completed successfully!")
            sys.exit(0)
        else:
            print("\n⚠️  Some phases failed. Check the output above for details.")
            sys.exit(1)
    
    finally:
        # Cleanup temporary config files
        for temp_dir in temp_dirs:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)


if __name__ == '__main__':
    main()
