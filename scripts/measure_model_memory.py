#!/usr/bin/env python
"""
Measure Hyper-LR-PINN model memory footprint.

This script calculates:
1. Number of parameters
2. Model size in MB
3. Peak GPU memory during inference and fine-tuning

Usage:
    python scripts/measure_model_memory.py --hidden_dim 50
"""

import argparse
import torch
import torch.nn as nn
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model import LR_PINN_phase1, LR_PINN_phase2
from utils import get_params


def count_parameters(model):
    """Count total number of parameters in model."""
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size_mb(model):
    """Calculate model size in MB (assuming float32 = 4 bytes per parameter)."""
    total_params = count_parameters(model)
    # Float32 = 4 bytes per parameter
    size_bytes = total_params * 4
    size_mb = size_bytes / (1024 * 1024)
    return size_mb


def measure_peak_gpu_memory(model, device, phase='phase2', hidden_dim=50):
    """Measure peak GPU memory during inference and fine-tuning."""
    if device.type != 'cuda':
        return None, None
    
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
    # Measure inference memory
    model.eval()
    with torch.no_grad():
        # Create dummy input
        batch_size = 1000  # Typical batch size
        x = torch.randn(batch_size, 1, device=device, requires_grad=False)
        t = torch.randn(batch_size, 1, device=device, requires_grad=False)
        
        if phase == 'phase1':
            beta = torch.ones(batch_size, 1, device=device)
            nu = torch.zeros(batch_size, 1, device=device)
            rho = torch.zeros(batch_size, 1, device=device)
            _ = model(x, t, beta, nu, rho)
        else:
            _ = model(x, t)
    
    inference_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
    # Measure fine-tuning memory (with gradients)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    
    x.requires_grad_(True)
    t.requires_grad_(True)
    
    if phase == 'phase1':
        output, _, _, _, _, _, _ = model(x, t, beta, nu, rho)
    else:
        output = model(x, t)
    
    loss = output.mean()
    loss.backward()
    optimizer.step()
    
    training_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
    
    return inference_memory, training_memory


def main():
    parser = argparse.ArgumentParser(description='Measure model memory footprint')
    parser.add_argument('--hidden_dim', type=int, default=50,
                       help='Hidden dimension (default: 50)')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to use (default: cuda:0)')
    parser.add_argument('--measure_gpu', action='store_true',
                       help='Measure peak GPU memory (requires CUDA)')
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    hidden_dim = args.hidden_dim
    
    print("=" * 80)
    print("Hyper-LR-PINN Memory Footprint Analysis")
    print("=" * 80)
    print(f"Hidden dimension: {hidden_dim}")
    print(f"Device: {device}")
    print("=" * 80)
    
    # Phase 1 Model
    print("\nPhase 1 Model (Meta-learning):")
    print("-" * 80)
    model_phase1 = LR_PINN_phase1(hidden_dim)
    model_phase1 = model_phase1.to(device)
    
    total_params_phase1 = count_parameters(model_phase1)
    trainable_params_phase1 = count_trainable_parameters(model_phase1)
    model_size_phase1 = get_model_size_mb(model_phase1)
    
    print(f"Total parameters: {total_params_phase1:,}")
    print(f"Trainable parameters: {trainable_params_phase1:,}")
    print(f"Model size: {model_size_phase1:.2f} MB")
    
    if args.measure_gpu and device.type == 'cuda':
        inference_mem, training_mem = measure_peak_gpu_memory(
            model_phase1, device, phase='phase1', hidden_dim=hidden_dim
        )
        if inference_mem:
            print(f"Peak GPU memory (inference): {inference_mem:.2f} MB")
        if training_mem:
            print(f"Peak GPU memory (training): {training_mem:.2f} MB")
    
    # Phase 2 Model
    print("\nPhase 2 Model (Fine-tuning):")
    print("-" * 80)
    
    # Create Phase 2 model from Phase 1 weights
    start_w = model_phase1.state_dict()['start_layer.weight']
    start_b = model_phase1.state_dict()['start_layer.bias']
    end_w = model_phase1.state_dict()['end_layer.weight']
    end_b = model_phase1.state_dict()['end_layer.bias']
    
    col_0 = model_phase1.state_dict()['col_basis_0']
    col_1 = model_phase1.state_dict()['col_basis_1']
    col_2 = model_phase1.state_dict()['col_basis_2']
    row_0 = model_phase1.state_dict()['row_basis_0']
    row_1 = model_phase1.state_dict()['row_basis_1']
    row_2 = model_phase1.state_dict()['row_basis_2']
    
    # Dummy alpha values
    alpha_0 = torch.ones(hidden_dim, device=device)
    alpha_1 = torch.ones(hidden_dim, device=device)
    alpha_2 = torch.ones(hidden_dim, device=device)
    
    model_phase2 = LR_PINN_phase2(
        hidden_dim, start_w, start_b, end_w, end_b,
        col_0, col_1, col_2, row_0, row_1, row_2,
        alpha_0, alpha_1, alpha_2
    )
    model_phase2 = model_phase2.to(device)
    
    total_params_phase2 = count_parameters(model_phase2)
    trainable_params_phase2 = count_trainable_parameters(model_phase2)
    model_size_phase2 = get_model_size_mb(model_phase2)
    
    print(f"Total parameters: {total_params_phase2:,}")
    print(f"Trainable parameters: {trainable_params_phase2:,}")
    print(f"Model size: {model_size_phase2:.2f} MB")
    
    if args.measure_gpu and device.type == 'cuda':
        inference_mem, training_mem = measure_peak_gpu_memory(
            model_phase2, device, phase='phase2', hidden_dim=hidden_dim
        )
        if inference_mem:
            print(f"Peak GPU memory (inference): {inference_mem:.2f} MB")
        if training_mem:
            print(f"Peak GPU memory (fine-tuning): {training_mem:.2f} MB")
    
    print("\n" + "=" * 80)
    print("Summary for LaTeX Table")
    print("=" * 80)
    print(f"Hyper-LR-PINN (Phase 2 - used for inference/fine-tuning):")
    print(f"  # Parameters: {total_params_phase2:,}")
    print(f"  Model Size: {model_size_phase2:.2f} MB")
    if args.measure_gpu and device.type == 'cuda':
        if training_mem:
            print(f"  Peak GPU Memory (fine-tuning): {training_mem:.2f} MB")
    print("=" * 80)
    
    # Also calculate theoretical parameter count
    print("\nTheoretical Parameter Count (for verification):")
    print("-" * 80)
    # Phase 1: start(2*h+1) + end(h*1+1) + 6 basis matrices(h*h each) + meta layers
    # start_layer: 2*hidden_dim + hidden_dim = 3*hidden_dim
    # end_layer: hidden_dim*1 + 1 = hidden_dim + 1
    # col_basis: 3 * hidden_dim^2
    # row_basis: 3 * hidden_dim^2
    # meta_layer_1: 3*hidden_dim + hidden_dim = 4*hidden_dim
    # meta_layer_2: hidden_dim^2 + hidden_dim
    # meta_layer_3: hidden_dim^2 + hidden_dim
    # meta_alpha_0/1/2: 3 * (hidden_dim^2 + hidden_dim)
    
    theoretical_phase1 = (
        3 * hidden_dim +  # start_layer
        hidden_dim + 1 +  # end_layer
        6 * hidden_dim * hidden_dim +  # col and row basis (3 each)
        4 * hidden_dim +  # meta_layer_1
        2 * (hidden_dim * hidden_dim + hidden_dim) +  # meta_layer_2,3
        3 * (hidden_dim * hidden_dim + hidden_dim)  # meta_alpha_0,1,2
    )
    
    # Phase 2: start(2*h+1) + end(h*1+1) + 6 basis matrices(h*h, requires_grad=False) + 3 alphas(h)
    theoretical_phase2 = (
        3 * hidden_dim +  # start_layer
        hidden_dim + 1 +  # end_layer
        6 * hidden_dim * hidden_dim +  # col and row basis (frozen)
        3 * hidden_dim  # alpha_0, alpha_1, alpha_2 (trainable)
    )
    
    print(f"Phase 1 theoretical: {theoretical_phase1:,}")
    print(f"Phase 1 actual: {total_params_phase1:,}")
    print(f"Phase 2 theoretical: {theoretical_phase2:,}")
    print(f"Phase 2 actual: {total_params_phase2:,}")


if __name__ == '__main__':
    main()
