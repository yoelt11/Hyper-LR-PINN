"""
Data loading module for parametric PDE datasets.

This module provides a wrapper around eff-physics-learn-dataset for loading
parametric PDE datasets with proper splits, normalization, and coordinate grid handling.
"""

import numpy as np
import torch
import json
import os
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Any
from collections import namedtuple

try:
    from eff_physics_learn_dataset import load_pde_dataset
except ImportError:
    # Fallback if import fails - will need to check actual import path
    try:
        from eff_physics_learn_dataset.dataset import load_pde_dataset
    except ImportError:
        try:
            # Try alternative import paths
            import eff_physics_learn_dataset as epld
            load_pde_dataset = epld.load_pde_dataset
        except (ImportError, AttributeError):
            raise ImportError(
                "eff-physics-learn-dataset not found. Please install it using:\n"
                "pip install git+ssh://git@github.com/yoelt11/eff-physics-learn-dataset.git\n"
                "or\n"
                "uv pip install git+ssh://git@github.com/yoelt11/eff-physics-learn-dataset.git"
            )

# Data structure for splits
DatasetSplits = namedtuple('DatasetSplits', ['train_few', 'interp', 'extrap', 'normalization_stats'])


def load_parametric_dataset(
    equation: str,
    data_dir: str,
    seed: int = 42,
    n_train: int = 10,
    cache: bool = True,
    balance: bool = True,
    n_each: Optional[int] = 20,
    balance_strategy: str = "random",
    diversify: bool = False,
    **kwargs
) -> DatasetSplits:
    """
    Load parametric PDE dataset and generate deterministic splits.
    
    The eff-physics-learn-dataset repository provides two modalities:
    1. 'parametric': Solutions across a range of parameter values (enables parametric generalization)
    2. 'non_parametric': Solutions for fixed parameter values (single parameter setting)
    
    This function uses the PARAMETRIC modality, which is required for
    the parametric splits (train_few, interp, extrap) used in this framework.
    
    Args:
        equation: Name of the PDE equation (e.g., 'convection', 'diffusion', 'allen_cahn')
        data_dir: Directory containing the dataset
        seed: Random seed for deterministic splits (deterministic when fixed)
        n_train: Number of training samples (few-shot scenario)
        cache: Whether to use cached dataset if available
        balance: If True, create balanced splits with equal-sized test sets
        n_each: Number of samples per test split (interp/extrap) when balance=True
        balance_strategy: Strategy for balancing ('random' or other strategies)
        diversify: Whether to diversify the selection
        **kwargs: Additional parameters passed to parametric_splits()
        
    Returns:
        DatasetSplits containing train_few, interp, extrap splits and normalization stats
        
    Note:
        This function requires the PARAMETRIC dataset modality. The non-parametric
        modality does not support parametric_splits() and cannot be used with this framework.
        
        With a fixed seed, splits are deterministic and reproducible across runs.
        Example: seed=0 with balance=True, n_each=20 gives 20 interp + 20 extrap = 40 total test samples.

        Note: The underlying eff-physics-learn-dataset package now uses the
        solution_percentile-based method by default when creating interp/extrap
        splits for parametric datasets.
    """
    # Load the full dataset (parametric modality)
    # The load_pde_dataset function loads the parametric dataset by default
    # which contains solutions across a range of parameter values
    dataset = load_pde_dataset(equation, data_dir, cache=cache)
    
    # Prepare parameters for parametric_splits
    split_params = {
        'seed': seed,
        'n_train': n_train,
        'balance': balance,
        'balance_strategy': balance_strategy,
        'diversify': diversify,
    }
    
    # Add optional parameters if provided
    if n_each is not None:
        split_params['n_each'] = n_each
    
    # Add any additional kwargs
    split_params.update(kwargs)
    
    # Generate parametric splits (only available for parametric modality)
    # This creates train_few, interp, and extrap splits based on parameter ranges
    splits = dataset.parametric_splits(**split_params)
    
    train_few = splits['train_few']
    interp = splits['interp']
    extrap = splits['extrap']

    # Extract indices for reference (which samples were used)
    train_indices = [sample.get('index', i) for i, sample in enumerate(train_few)]
    interp_indices = [sample.get('index', i) for i, sample in enumerate(interp)]
    extrap_indices = [sample.get('index', i) for i, sample in enumerate(extrap)]
    
    # Save indices to a reference file
    indices_info = {
        'equation': equation,
        'seed': seed,
        'n_train': n_train,
        'balance': balance,
        'n_each': n_each,
        'balance_strategy': balance_strategy,
        'diversify': diversify,
        'indices': {
            'train': train_indices,
            'interp': interp_indices,
            'extrap': extrap_indices
        },
        'counts': {
            'train': len(train_indices),
            'interp': len(interp_indices),
            'extrap': len(extrap_indices),
            'total_test': len(interp_indices) + len(extrap_indices)
        }
    }
    
    # Save to a reference directory (create if doesn't exist)
    ref_dir = Path(data_dir).parent / 'dataset_indices'
    ref_dir.mkdir(parents=True, exist_ok=True)
    ref_file = ref_dir / f'{equation}_seed{seed}_indices.json'
    
    with open(ref_file, 'w') as f:
        json.dump(indices_info, f, indent=2)
    
    print(f"Dataset indices saved to: {ref_file}")
    print(f"  Train: {len(train_indices)} samples (indices: {train_indices[:5]}{'...' if len(train_indices) > 5 else ''})")
    print(f"  Interp: {len(interp_indices)} samples (indices: {interp_indices[:5]}{'...' if len(interp_indices) > 5 else ''})")
    print(f"  Extrap: {len(extrap_indices)} samples (indices: {extrap_indices[:5]}{'...' if len(extrap_indices) > 5 else ''})")

    # Note: Some datasets (notably "convection") omit coordinate grids at the per-sample
    # level. We handle that in `extract_coordinate_grids()` by using per-equation domain
    # defaults (or `sample["grid_info"]` if present).
    
    # Compute normalization statistics from train_few only
    normalization_stats = compute_normalization_stats(train_few)
    
    return DatasetSplits(
        train_few=train_few,
        interp=interp,
        extrap=extrap,
        normalization_stats=normalization_stats
    )


def compute_normalization_stats(data: List[Dict[str, Any]]) -> Dict[str, Tuple[float, float]]:
    """
    Compute mean and std statistics from training data.
    
    Args:
        data: List of data samples, each containing 'u' (solution field)
        
    Returns:
        Dictionary with 'u_mean' and 'u_std' statistics
    """
    if not data:
        raise ValueError("Cannot compute normalization stats from empty data")
    
    # Collect all solution fields
    u_values = []
    for sample in data:
        u = sample.get('u')
        if u is not None:
            if isinstance(u, torch.Tensor):
                u_values.append(u.numpy().flatten())
            else:
                u_values.append(np.array(u).flatten())
    
    if not u_values:
        raise ValueError("No solution fields found in data")
    
    # Compute statistics
    all_u = np.concatenate(u_values)
    u_mean = float(np.mean(all_u))
    u_std = float(np.std(all_u))
    
    # Avoid division by zero
    if u_std < 1e-10:
        u_std = 1.0
    
    return {
        'u_mean': u_mean,
        'u_std': u_std
    }


def normalize_data(
    data: List[Dict[str, Any]],
    stats: Optional[Dict[str, Tuple[float, float]]] = None
) -> List[Dict[str, Any]]:
    """
    Normalize solution fields using provided statistics.
    
    Args:
        data: List of data samples
        stats: Normalization statistics (mean, std). If None, computes from data.
        
    Returns:
        List of normalized data samples
    """
    if stats is None:
        stats = compute_normalization_stats(data)
    
    normalized_data: List[Dict[str, Any]] = []
    for sample in data:
        # If data has already been normalized upstream, keep it idempotent.
        # This prevents accidental double-normalization when a caller mixes
        # raw splits with `convert_to_model_format()` outputs.
        if bool(sample.get("_u_is_normalized", False)):
            normalized_data.append(sample)
            continue

        normalized_sample = sample.copy()
        u = sample.get('u')
        
        if u is not None:
            if isinstance(u, torch.Tensor):
                u_norm = (u - stats['u_mean']) / stats['u_std']
            else:
                u_norm = (np.array(u) - stats['u_mean']) / stats['u_std']
            normalized_sample['u'] = u_norm

        # Marker to avoid double-normalization
        normalized_sample["_u_is_normalized"] = True
        
        normalized_data.append(normalized_sample)
    
    return normalized_data


def prepare_batch(
    sample: Dict[str, Any],
    include_coords: bool = True,
    device: str = 'cpu'
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Prepare a single sample for model input.
    
    Args:
        sample: Data sample dictionary containing:
            - 'u': Solution field (2D or 3D spatial grid)
            - 'params': Parameter vector (P parameters per sample)
            - Optional: 'X_grid', 'Y_grid', 'Z_grid', 'T_grid' coordinate grids
        include_coords: Whether to include coordinate grids in input
        device: Device to place tensors on
        
    Returns:
        Tuple of (x, y, z) where:
            x: Input features (coordinate grids or other inputs)
            y: Target solution field
            z: Parameter conditioning vector (None if not available)
    """
    # Extract solution field
    u = sample.get('u')
    if u is None:
        raise ValueError("Sample must contain 'u' (solution field)")
    
    # Convert to tensor if needed
    if not isinstance(u, torch.Tensor):
        u = torch.from_numpy(np.array(u)).float()
    else:
        u = u.float()
    
    # Flatten solution field for target
    y = u.flatten().unsqueeze(-1)  # Shape: [N, 1] where N is number of grid points
    
    # Extract or construct coordinate grids
    if include_coords:
        x = extract_coordinate_grids(sample, device=device)
    else:
        # If no coords, use dummy input (model should handle this)
        x = torch.zeros((y.shape[0], 2), device=device)  # Default: (x, t) for 2D
    
    # Extract parameter vector
    z = None
    params = sample.get('params')
    if params is not None:
        if not isinstance(params, torch.Tensor):
            z = torch.from_numpy(np.array(params)).float()
        else:
            z = params.float()
        z = z.unsqueeze(0) if z.dim() == 1 else z  # Ensure 2D: [1, P] or [B, P]
    
    return x, y, z


def extract_coordinate_grids(
    sample: Dict[str, Any],
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Extract or auto-construct coordinate grids from sample.
    
    Args:
        sample: Data sample dictionary
        device: Device to place tensors on
        
    Returns:
        Coordinate grid tensor of shape [N, D] where N is number of points
        and D is number of spatial dimensions (2 for X,T or 3 for X,Y,Z)
    """
    # Check for existing coordinate grids
    has_x = 'X_grid' in sample or 'x' in sample
    has_y = 'Y_grid' in sample or 'y' in sample
    has_z = 'Z_grid' in sample or 'z' in sample
    has_t = 'T_grid' in sample or 't' in sample
    
    # Get solution field shape to determine grid dimensions
    u = sample.get('u')
    if u is None:
        raise ValueError("Cannot determine grid shape without solution field")
    
    if isinstance(u, torch.Tensor):
        u_shape = u.shape
    else:
        u_shape = np.array(u).shape

    def _stack_if_full_mesh(grids: List[torch.Tensor], expected_shape: Tuple[int, ...]) -> Optional[torch.Tensor]:
        """
        If the dataset provides *full* coordinate meshes (e.g. X_grid and T_grid
        already have shape == u_shape), stack+flatten them directly.
        """
        if not grids:
            return None
        # All provided grids must match shape and be at least 2D to be considered "full mesh".
        try:
            exp = torch.Size(expected_shape)
        except Exception:
            exp = None
        if exp is None:
            return None
        if all(isinstance(g, torch.Tensor) and g.ndim >= 2 and g.shape == exp for g in grids):
            return torch.stack([g.flatten() for g in grids], dim=1)
        return None
    
    # Heuristic: if no coordinate keys are provided, infer coordinate system from equation + u dimensionality.
    # Many datasets (e.g. convection/burgers/allen_cahn) are time-dependent 1D PDEs where `u` is stored as [Nt, Nx].
    # If we don't infer this, we'd incorrectly return a 1D grid of length Nt (mismatch with u.flatten()).
    equation_name = str(sample.get("equation", "")).lower()
    time_dependent_equations = {
        "allen_cahn",
        "burgers",
        "convection",
        "hlrp_cdr",
        "hlrp_convection",
        "hlrp_diffusion",
        "hlrp_reaction",
    }
    if (not has_x) and (not has_y) and (not has_z) and (not has_t):
        if len(u_shape) == 2:
            if equation_name in time_dependent_equations:
                # Assume u is [Nt, Nx]
                nt, nx = int(u_shape[0]), int(u_shape[1])
                # Try to use domain metadata if provided.
                grid_info = sample.get("grid_info", None)
                x0, x1 = 0.0, 1.0
                t0, t1 = 0.0, 1.0
                if isinstance(grid_info, dict):
                    try:
                        if "x_domain" in grid_info:
                            x0, x1 = float(grid_info["x_domain"][0]), float(grid_info["x_domain"][1])
                        if "t_domain" in grid_info:
                            t0, t1 = float(grid_info["t_domain"][0]), float(grid_info["t_domain"][1])
                    except Exception:
                        # fall back to defaults
                        x0, x1, t0, t1 = 0.0, 1.0, 0.0, 1.0

                # Equation-specific defaults (important for convection: x ∈ [0,6], t ∈ [0,1])
                # These match `eff-physics-learn-dataset`'s dataset-level grid_info.
                domain_defaults = {
                    "convection": ((0.0, 6.0), (0.0, 1.0)),
                    "hlrp_convection": ((0.0, 6.0), (0.0, 1.0)),
                    "burgers": ((-1.0, 1.0), (0.0, 1.0)),
                    "allen_cahn": ((-1.0, 1.0), (0.0, 1.0)),
                }
                if equation_name in domain_defaults:
                    (x0, x1), (t0, t1) = domain_defaults[equation_name]

                t_grid = torch.linspace(float(t0), float(t1), nt, device=device)
                x_grid = torch.linspace(float(x0), float(x1), nx, device=device)
                T, X = torch.meshgrid(t_grid, x_grid, indexing="ij")  # [Nt, Nx]
                coords = torch.stack([X.flatten(), T.flatten()], dim=1)  # [Nt*Nx, 2] with (x,t)
                return coords.to(device)
            else:
                # Assume 2D spatial: u is [Nx, Ny]
                nx, ny = int(u_shape[0]), int(u_shape[1])
                x_grid = torch.linspace(0, 1, nx, device=device)
                y_grid = torch.linspace(0, 1, ny, device=device)
                X, Y = torch.meshgrid(x_grid, y_grid, indexing="ij")
                coords = torch.stack([X.flatten(), Y.flatten()], dim=1)
                return coords.to(device)

    # Determine spatial dimensions
    if has_t:
        # 2D problem: (X, T) or (X, Y, T)
        if has_y:
            # 3D spatial: (X, Y, T)
            if has_x and has_t:
                x_grid = get_grid(sample, 'X_grid', 'x', u_shape[0], device)
                y_grid = get_grid(sample, 'Y_grid', 'y', u_shape[1], device)
                t_grid = get_grid(sample, 'T_grid', 't', u_shape[2], device)
                # If grids are already full meshes (same shape as u), stack directly.
                coords = _stack_if_full_mesh([x_grid, y_grid, t_grid], u_shape)
                if coords is None:
                    # Otherwise assume 1D axes and create meshgrid.
                    X, Y, T = torch.meshgrid(x_grid, y_grid, t_grid, indexing='ij')
                    coords = torch.stack([X.flatten(), Y.flatten(), T.flatten()], dim=1)
            else:
                # Auto-construct normalized grids
                coords = create_normalized_grids(u_shape, spatial_dims=3, device=device)
        else:
            # 2D spatial: (X, T)
            if has_x and has_t:
                x_grid = get_grid(sample, 'X_grid', 'x', u_shape[0], device)
                t_grid = get_grid(sample, 'T_grid', 't', u_shape[1], device)
                # If grids are already full meshes (same shape as u), stack directly.
                coords = _stack_if_full_mesh([x_grid, t_grid], u_shape)
                if coords is None:
                    # Otherwise assume 1D axes and create meshgrid.
                    X, T = torch.meshgrid(x_grid, t_grid, indexing='ij')
                    coords = torch.stack([X.flatten(), T.flatten()], dim=1)
            else:
                # Auto-construct normalized grids
                coords = create_normalized_grids(u_shape, spatial_dims=2, device=device)
    else:
        # No time dimension - spatial only
        if has_y:
            # 2D spatial: (X, Y)
            if has_x and has_y:
                x_grid = get_grid(sample, 'X_grid', 'x', u_shape[0], device)
                y_grid = get_grid(sample, 'Y_grid', 'y', u_shape[1], device)
                coords = _stack_if_full_mesh([x_grid, y_grid], u_shape)
                if coords is None:
                    X, Y = torch.meshgrid(x_grid, y_grid, indexing='ij')
                    coords = torch.stack([X.flatten(), Y.flatten()], dim=1)
            else:
                coords = create_normalized_grids(u_shape, spatial_dims=2, device=device)
        else:
            # 1D spatial: (X,)
            if has_x:
                x_grid = get_grid(sample, 'X_grid', 'x', u_shape[0], device)
                coords = x_grid.unsqueeze(-1)
            else:
                coords = create_normalized_grids(u_shape, spatial_dims=1, device=device)
    
    return coords.to(device)


def get_grid(
    sample: Dict[str, Any],
    grid_key: str,
    alt_key: str,
    size: int,
    device: str = 'cpu'
) -> torch.Tensor:
    """Extract grid from sample or create default."""
    # NOTE: Don't use `or` here — grids are often numpy arrays / torch tensors,
    # and their truth-value is ambiguous (raises ValueError).
    grid = sample.get(grid_key, None)
    if grid is None:
        grid = sample.get(alt_key, None)
    if grid is not None:
        if not isinstance(grid, torch.Tensor):
            grid = torch.from_numpy(np.array(grid)).float()
        return grid.to(device)
    else:
        # Create normalized grid [0, 1]
        return torch.linspace(0, 1, size, device=device)


def create_normalized_grids(
    shape: Tuple[int, ...],
    spatial_dims: int = 2,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Auto-construct normalized coordinate grids [0, 1].
    
    Args:
        shape: Shape of solution field
        spatial_dims: Number of spatial dimensions (1, 2, or 3)
        device: Device to place tensors on
        
    Returns:
        Coordinate tensor of shape [N, spatial_dims]
    """
    grids = []
    for i in range(min(spatial_dims, len(shape))):
        size = shape[i]
        grid = torch.linspace(0, 1, size, device=device)
        grids.append(grid)
    
    # Create meshgrid
    if len(grids) == 1:
        coords = grids[0].unsqueeze(-1)
    elif len(grids) == 2:
        X, Y = torch.meshgrid(grids[0], grids[1], indexing='ij')
        coords = torch.stack([X.flatten(), Y.flatten()], dim=1)
    elif len(grids) == 3:
        X, Y, Z = torch.meshgrid(grids[0], grids[1], grids[2], indexing='ij')
        coords = torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=1)
    else:
        raise ValueError(f"Unsupported number of spatial dimensions: {spatial_dims}")
    
    return coords


def convert_to_model_format(
    splits: DatasetSplits,
    phase: str = 'phase1',
    device: str = 'cpu'
) -> Dict[str, Any]:
    """
    Convert dataset splits to format compatible with existing model code.
    
    This function bridges the gap between eff-physics-learn-dataset format
    and the CSV-based format expected by existing train_meta.py and train_full.py.
    
    Args:
        splits: DatasetSplits from load_parametric_dataset
        phase: 'phase1' or 'phase2'
        device: Device to place tensors on
        
    Returns:
        Dictionary with keys:
            - 'train_few': Processed training data
            - 'interp': Processed interpolation test data
            - 'extrap': Processed extrapolation test data
            - 'normalization_stats': Normalization statistics
    """
    # Normalize all splits using training statistics
    train_few_norm = normalize_data(splits.train_few, splits.normalization_stats)
    interp_norm = normalize_data(splits.interp, splits.normalization_stats)
    extrap_norm = normalize_data(splits.extrap, splits.normalization_stats)
    
    # Convert to model-compatible format
    # For phase1, we need parameter vectors; for phase2, parameters are extracted per sample
    processed_data = {
        'train_few': train_few_norm,
        'interp': interp_norm,
        'extrap': extrap_norm,
        'normalization_stats': splits.normalization_stats
    }
    
    return processed_data

