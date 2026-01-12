"""
PyTorch-based PINN loss functions for various PDE equations.

This module provides PINN loss formulations compatible with PyTorch,
adapted from JAX implementations to work with the Hyper-LR-PINN framework.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Any
import numpy as np


def compute_derivatives_pytorch(
    u: torch.Tensor,
    x: torch.Tensor,
    t: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """
    Compute derivatives using PyTorch autograd.
    
    Args:
        u: Solution field (N, 1)
        x: Spatial coordinates (N, 1)
        t: Time coordinates (N, 1)
        
    Returns:
        Dictionary containing u, u_x, u_t, u_xx, etc.
    """
    # Ensure inputs require gradients
    if not x.requires_grad:
        x.requires_grad_(True)
    if not t.requires_grad:
        t.requires_grad_(True)
    
    def _grad(outputs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        """
        Safe autograd helper.

        Returns zeros when the derivative is structurally zero / unused, which happens for:
        - Models that do not depend on `inputs` at all
        - Linear dependence when requesting higher-order derivatives
        """
        if not outputs.requires_grad:
            return torch.zeros_like(inputs)
        g = torch.autograd.grad(
            outputs,
            inputs,
            grad_outputs=torch.ones_like(outputs),
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )[0]
        if g is None:
            return torch.zeros_like(inputs)
        return g

    # First derivatives
    u_x = _grad(u, x)
    u_t = _grad(u, t)

    # Second derivatives
    u_xx = _grad(u_x, x)
    # For 2D spatial problems we can reinterpret `t` as a second spatial coordinate
    # and compute the second derivative w.r.t. `t` as well.
    u_tt = _grad(u_t, t)
    
    return {
        'u': u,
        'u_x': u_x,
        'u_t': u_t,
        'u_xx': u_xx,
        'u_tt': u_tt
    }


def burgers_loss_pytorch(
    u: torch.Tensor,
    x: torch.Tensor,
    t: torch.Tensor,
    nu: torch.Tensor | float = 0.01,
    bc_weight: float = 1.0,
    ic_weight: float = 1.0,
    pde_weight: float = 1.0,
    return_components: bool = False
) -> torch.Tensor:
    """
    Compute the loss for the 1D viscous Burgers equation (PyTorch).

    PDE: u_t + u u_x - nu u_xx = 0

    Notes:
    - `nu` can be a float or a tensor broadcastable to u's shape.
    - BC/IC here are lightweight defaults; the Hyper-LR-PINN training loop
      typically handles BC via its periodic boundary term and "IC" implicitly
      via data points at t=0.
    """
    derivs = compute_derivatives_pytorch(u, x, t)
    u_t = derivs["u_t"]
    u_x = derivs["u_x"]
    u_xx = derivs["u_xx"]

    # Ensure nu is tensor on correct device for broadcasting
    if not torch.is_tensor(nu):
        nu_t = torch.tensor(float(nu), device=u.device, dtype=u.dtype)
    else:
        nu_t = nu.to(device=u.device, dtype=u.dtype)

    pde_residual = u_t + u * u_x - nu_t * u_xx
    pde_loss = torch.mean(pde_residual ** 2)

    # Defaults (kept for API symmetry; typically not used by our training scripts)
    bc_loss = torch.tensor(0.0, device=u.device, dtype=u.dtype)
    ic_loss = torch.tensor(0.0, device=u.device, dtype=u.dtype)

    weighted_pde_loss = pde_weight * pde_loss
    weighted_bc_loss = bc_weight * bc_loss
    weighted_ic_loss = ic_weight * ic_loss
    total_loss = weighted_pde_loss + weighted_bc_loss + weighted_ic_loss

    if return_components:
        return {
            "total_loss": total_loss,
            "pde_loss": weighted_pde_loss,
            "bc_loss": weighted_bc_loss,
            "ic_loss": weighted_ic_loss,
            "unweighted_pde_loss": pde_loss,
            "unweighted_bc_loss": bc_loss,
            "unweighted_ic_loss": ic_loss,
        }

    return total_loss


def allen_cahn_loss_pytorch(
    u: torch.Tensor,
    x: torch.Tensor,
    t: torch.Tensor,
    eps: float = 0.01,
    lam: float = 1.0,
    bc_weight: float = 1.0,
    ic_weight: float = 100.0,
    pde_weight: float = 1.0,
    return_components: bool = False
) -> torch.Tensor:
    """
    Compute the loss for the Allen-Cahn equation (PyTorch version).
    
    PDE: u_t - eps^2 * u_xx + lam * (u^3 - u) = 0
    
    Args:
        u: Solution predictions (N, 1)
        x: Spatial coordinates (N, 1)
        t: Time coordinates (N, 1)
        eps: Interface parameter (epsilon)
        lam: Diffusion parameter (lambda)
        bc_weight: Weight for boundary condition loss
        ic_weight: Weight for initial condition loss
        pde_weight: Weight for PDE residual loss
        return_components: If True, return dict with individual loss components
        
    Returns:
        Total loss value or dict with loss components
    """
    # Compute derivatives
    derivs = compute_derivatives_pytorch(u, x, t)
    u_t = derivs['u_t']
    u_xx = derivs['u_xx']
    
    # Allen-Cahn PDE residual: ∂u/∂t - ε² ∂²u/∂x² + λ(u³ - u) = 0
    pde_residual = u_t - eps**2 * u_xx + lam * (u**3 - u)
    pde_loss = torch.mean(pde_residual**2)
    
    # Boundary conditions: u = -1 at spatial boundaries
    # Extract boundary points (assuming x is sorted)
    x_min = x.min()
    x_max = x.max()
    
    # Left boundary
    left_mask = (x - x_min).abs() < 1e-6
    u_left = u[left_mask]
    bc_left_loss = torch.mean((u_left + 1.0)**2) if u_left.numel() > 0 else torch.tensor(0.0, device=u.device)
    
    # Right boundary
    right_mask = (x - x_max).abs() < 1e-6
    u_right = u[right_mask]
    bc_right_loss = torch.mean((u_right + 1.0)**2) if u_right.numel() > 0 else torch.tensor(0.0, device=u.device)
    
    bc_loss = bc_left_loss + bc_right_loss
    
    # Initial condition: u(x, t=0) = x^2 * cos(π*x)
    t_min = t.min()
    ic_mask = (t - t_min).abs() < 1e-6
    if ic_mask.sum() > 0:
        x_ic = x[ic_mask]
        u_ic_pred = u[ic_mask]
        u_ic_gt = (x_ic**2 * torch.cos(np.pi * x_ic))
        ic_loss = torch.mean((u_ic_pred - u_ic_gt)**2)
    else:
        ic_loss = torch.tensor(0.0, device=u.device)
    
    # Compute weighted losses
    weighted_pde_loss = pde_weight * pde_loss
    weighted_bc_loss = bc_weight * bc_loss
    weighted_ic_loss = ic_weight * ic_loss
    
    total_loss = weighted_pde_loss + weighted_bc_loss + weighted_ic_loss
    
    if return_components:
        return {
            'total_loss': total_loss,
            'pde_loss': weighted_pde_loss,
            'bc_loss': weighted_bc_loss,
            'ic_loss': weighted_ic_loss,
            'unweighted_pde_loss': pde_loss,
            'unweighted_bc_loss': bc_loss,
            'unweighted_ic_loss': ic_loss
        }
    
    return total_loss


def convection_loss_pytorch(
    u: torch.Tensor,
    x: torch.Tensor,
    t: torch.Tensor,
    beta: float = 1.0,
    bc_weight: float = 1.0,
    ic_weight: float = 1.0,
    pde_weight: float = 1.0,
    return_components: bool = False
) -> torch.Tensor:
    """
    Compute the loss for the pure Convection equation (PyTorch version).
    
    PDE: ∂u/∂t + β ∂u/∂x = 0
    
    Args:
        u: Solution predictions (N, 1)
        x: Spatial coordinates (N, 1)
        t: Time coordinates (N, 1)
        beta: Convection parameter (advection speed)
        bc_weight: Weight for boundary condition loss
        ic_weight: Weight for initial condition loss
        pde_weight: Weight for PDE residual loss
        return_components: If True, return dict with individual loss components
        
    Returns:
        Total loss value or dict with loss components
    """
    # Compute derivatives
    derivs = compute_derivatives_pytorch(u, x, t)
    u_t = derivs['u_t']
    u_x = derivs['u_x']
    
    # Pure convection PDE residual: ∂u/∂t + β ∂u/∂x = 0
    pde_residual = u_t + beta * u_x
    pde_loss = torch.mean(pde_residual**2)
    
    # Periodic boundary conditions: u(x=0,t) = u(x=L,t)
    x_min = x.min()
    x_max = x.max()
    L = x_max - x_min
    
    # Left boundary
    left_mask = (x - x_min).abs() < 1e-6
    u_left = u[left_mask]
    
    # Right boundary
    right_mask = (x - x_max).abs() < 1e-6
    u_right = u[right_mask]
    
    # Periodic BC: match left and right at same time points
    if u_left.numel() > 0 and u_right.numel() > 0:
        # Simple approach: mean difference
        bc_loss = torch.mean((u_left.mean() - u_right.mean())**2)
    else:
        bc_loss = torch.tensor(0.0, device=u.device)
    
    # Initial condition: u(x, t=0) = 1 + sin(2πx/L)
    t_min = t.min()
    ic_mask = (t - t_min).abs() < 1e-6
    if ic_mask.sum() > 0:
        x_ic = x[ic_mask]
        u_ic_pred = u[ic_mask]
        u_ic_gt = 1.0 + torch.sin(2 * np.pi * x_ic / L)
        ic_loss = torch.mean((u_ic_pred - u_ic_gt)**2)
    else:
        ic_loss = torch.tensor(0.0, device=u.device)
    
    # Compute weighted losses
    weighted_pde_loss = pde_weight * pde_loss
    weighted_bc_loss = bc_weight * bc_loss
    weighted_ic_loss = ic_weight * ic_loss
    
    total_loss = weighted_pde_loss + weighted_bc_loss + weighted_ic_loss
    
    if return_components:
        return {
            'total_loss': total_loss,
            'pde_loss': weighted_pde_loss,
            'bc_loss': weighted_bc_loss,
            'ic_loss': weighted_ic_loss,
            'unweighted_pde_loss': pde_loss,
            'unweighted_bc_loss': bc_loss,
            'unweighted_ic_loss': ic_loss
        }
    
    return total_loss


def cdr_loss_pytorch(
    u: torch.Tensor,
    x: torch.Tensor,
    t: torch.Tensor,
    beta: float = 0.01,
    nu: float = 1.0,
    rho: float = 1.0,
    bc_weight: float = 1.0,
    ic_weight: float = 1.0,
    pde_weight: float = 1.0,
    return_components: bool = False
) -> torch.Tensor:
    """
    Compute the loss for the Convection-Diffusion-Reaction (CDR) equation (PyTorch version).
    
    PDE: ∂u/∂t + β ∂u/∂x - ν ∂²u/∂x² - ρ u(1-u) = 0
    
    This matches the existing f_cal and f_cal_phase2 functions in utils.py.
    
    Args:
        u: Solution predictions (N, 1)
        x: Spatial coordinates (N, 1)
        t: Time coordinates (N, 1)
        beta: Convection coefficient
        nu: Diffusion coefficient
        rho: Reaction coefficient
        bc_weight: Weight for boundary condition loss
        ic_weight: Weight for initial condition loss
        pde_weight: Weight for PDE residual loss
        return_components: If True, return dict with individual loss components
        
    Returns:
        Total loss value or dict with loss components
    """
    # Compute derivatives
    derivs = compute_derivatives_pytorch(u, x, t)
    u_t = derivs['u_t']
    u_x = derivs['u_x']
    u_xx = derivs['u_xx']
    
    # CDR PDE residual (matches the JAX reference implementation shared by the user):
    #   ∂u/∂t + v ∂u/∂x = D ∂²u/∂x² - R u
    # Rearranged as:
    #   ∂u/∂t + v ∂u/∂x - D ∂²u/∂x² + R u = 0
    pde_residual = u_t + beta * u_x - nu * u_xx + rho * u
    pde_loss = torch.mean(pde_residual**2)
    
    # Boundary conditions: Periodic u(x=0,t) = u(x=L,t)
    x_min = x.min()
    x_max = x.max()
    
    left_mask = (x - x_min).abs() < 1e-6
    right_mask = (x - x_max).abs() < 1e-6
    u_left = u[left_mask] if left_mask.sum() > 0 else None
    u_right = u[right_mask] if right_mask.sum() > 0 else None
    
    if u_left is not None and u_right is not None:
        bc_loss = torch.mean((u_left.mean() - u_right.mean())**2)
    else:
        bc_loss = torch.tensor(0.0, device=u.device)
    
    # Initial condition (simplified - would need actual IC from data)
    ic_loss = torch.tensor(0.0, device=u.device)
    
    # Compute weighted losses
    weighted_pde_loss = pde_weight * pde_loss
    weighted_bc_loss = bc_weight * bc_loss
    weighted_ic_loss = ic_weight * ic_loss
    
    total_loss = weighted_pde_loss + weighted_bc_loss + weighted_ic_loss
    
    if return_components:
        return {
            'total_loss': total_loss,
            'pde_loss': weighted_pde_loss,
            'bc_loss': weighted_bc_loss,
            'ic_loss': weighted_ic_loss,
            'unweighted_pde_loss': pde_loss,
            'unweighted_bc_loss': bc_loss,
            'unweighted_ic_loss': ic_loss
        }
    
    return total_loss


def helmholtz2d_loss_pytorch(
    u: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    k: torch.Tensor | float,
    pde_weight: float = 1.0,
    return_components: bool = False,
) -> torch.Tensor:
    """
    Compute the loss for the 2D Helmholtz equation (PyTorch version).

    We follow the common convention:
        Δu + k^2 u = 0

    In this codebase, many pipelines represent 2D spatial coordinates as (x, t);
    for Helmholtz2D we interpret the second coordinate as y.
    """
    derivs = compute_derivatives_pytorch(u, x, y)
    u_xx = derivs["u_xx"]
    u_yy = derivs["u_tt"]  # second coordinate second-derivative

    if not torch.is_tensor(k):
        k_t = torch.tensor(float(k), device=u.device, dtype=u.dtype)
    else:
        k_t = k.to(device=u.device, dtype=u.dtype)

    pde_residual = u_xx + u_yy + (k_t ** 2) * u
    pde_loss = torch.mean(pde_residual ** 2)
    total_loss = pde_weight * pde_loss

    if return_components:
        return {
            "total_loss": total_loss,
            "pde_loss": total_loss,
            "unweighted_pde_loss": pde_loss,
        }

    return total_loss


def get_pinn_loss_pytorch(equation: str):
    """
    Get the appropriate PINN loss function for a given equation (PyTorch version).
    
    Args:
        equation: Name of the equation ('allen_cahn', 'convection', 'cdr', etc.)
        
    Returns:
        Loss function for the specified equation
    """
    loss_functions = {
        'allen_cahn': allen_cahn_loss_pytorch,
        'convection': convection_loss_pytorch,
        'burgers': burgers_loss_pytorch,
        'cdr': cdr_loss_pytorch,
        'convection_diffusion_reaction': cdr_loss_pytorch,
        'helmholtz2d': helmholtz2d_loss_pytorch,
    }
    
    # Handle exact matches first
    if equation in loss_functions:
        return loss_functions[equation]
    
    # Handle partial matches
    equation_lower = equation.lower()
    for key, loss_fn in loss_functions.items():
        if key in equation_lower:
            return loss_fn
    
    # Default to CDR (matches existing behavior)
    print(f"Warning: Unknown equation '{equation}', using CDR loss")
    return cdr_loss_pytorch


def _equation_to_key(equation: str) -> str:
    eq = (equation or "").lower()
    if "burgers" in eq:
        return "burgers"
    if "allen" in eq and "cahn" in eq:
        return "allen_cahn"
    if "helmholtz" in eq and ("2d" in eq or "hlrp_helmholtz" in eq):
        return "helmholtz2d"
    if "convection" in eq and "diffusion" in eq:
        return "cdr"
    if eq in {"cdr", "convection_diffusion_reaction"}:
        return "cdr"
    # In the original Hyper-LR-PINN codebase (see `utils.f_cal`) the dataset/task
    # named "convection" is actually the convection-diffusion-reaction residual:
    #   u_t + beta*u_x - nu*u_xx - rho*u*(1-u) = 0
    #
    # Using "cdr" here is strictly more general than pure advection: if nu=rho=0
    # it reduces to u_t + beta*u_x = 0.
    if "convection" in eq or "hlrp_convection" in eq:
        return "cdr"
    return eq


def _pde_residual_hyper_lr(
    equation: str,
    u: torch.Tensor,
    x: torch.Tensor,
    t: torch.Tensor,
    beta: torch.Tensor,
    nu: torch.Tensor,
    rho: torch.Tensor,
    *,
    u_mean: float | torch.Tensor | None = None,
    u_std: float | torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Return per-point PDE residual tensor matching the Hyper-LR-PINN training style.

    Important: In our pipeline the dataset parameters are routed into (beta, nu, rho)
    purely as a 3-slot conditioning vector. For Burgers, the dataset typically has a
    single parameter (viscosity), which we store in `beta` (slot-1) and set the others to 0.
    """
    derivs = compute_derivatives_pytorch(u, x, t)
    u_t = derivs["u_t"]
    u_x = derivs["u_x"]
    u_xx = derivs["u_xx"]
    u_tt = derivs["u_tt"]

    # If the dataset normalized the target field as:
    #   u_norm = (u_phys - mean) / std
    # then the physically-correct residual should be computed on u_phys.
    #
    # Derivatives scale linearly with std: d(u_phys)/dx = std * d(u_norm)/dx.
    if u_mean is not None and u_std is not None:
        if not torch.is_tensor(u_mean):
            u_mean_t = torch.tensor(float(u_mean), device=u.device, dtype=u.dtype)
        else:
            u_mean_t = u_mean.to(device=u.device, dtype=u.dtype)
        if not torch.is_tensor(u_std):
            u_std_t = torch.tensor(float(u_std), device=u.device, dtype=u.dtype)
        else:
            u_std_t = u_std.to(device=u.device, dtype=u.dtype)

        u_phys = u * u_std_t + u_mean_t
        u_t = u_t * u_std_t
        u_x = u_x * u_std_t
        u_xx = u_xx * u_std_t
        u_tt = u_tt * u_std_t
    else:
        u_phys = u

    eq_key = _equation_to_key(equation)

    if eq_key == "burgers":
        # eff-physics-learn-dataset burgers params are ordered as [A, k, nu].
        # Our pipeline maps params -> (beta, nu, rho) = (A, k, nu_viscosity).
        # Therefore viscosity is the *third* slot: `rho`.
        nu_visc = rho
        return u_t + u_phys * u_x - nu_visc * u_xx

    if eq_key == "convection":
        # Pure advection: u_t + beta u_x = 0
        return u_t + beta * u_x

    if eq_key == "allen_cahn":
        # Allen-Cahn: u_t - eps^2 u_xx + lam (u^3 - u) = 0
        # Map conditioning slots: eps <- beta, lam <- nu
        eps = beta
        lam = nu
        return u_t - (eps ** 2) * u_xx + lam * (u_phys ** 3 - u_phys)

    if eq_key == "helmholtz2d":
        # Helmholtz2D in eff-physics-learn-dataset is a manufactured-solution problem:
        #   u(x,y) = sin(a1*pi*x) * sin(a2*pi*y)
        #
        # and the PDE is:
        #   Δu + k^2 u = q(x,y)
        # where:
        #   q(x,y) = (-(a1*pi)^2 - (a2*pi)^2 + k^2) * sin(a1*pi*x) * sin(a2*pi*y)
        #
        # Our training pipeline uses (x, t) tensors; for Helmholtz2D interpret `t` as y.
        # Parameter convention: (beta, nu, rho) store the first 3 dataset params => (a1, a2, k).
        a1 = beta
        a2 = nu
        k = rho

        lap = u_xx + u_tt

        pi = torch.tensor(np.pi, device=u.device, dtype=u.dtype)
        q_coeff = -((a1 * pi) ** 2) - ((a2 * pi) ** 2) + (k ** 2)
        q = q_coeff * torch.sin(a1 * pi * x) * torch.sin(a2 * pi * t)

        # Residual: Δu + k^2 u - q = 0
        return lap + (k ** 2) * u_phys - q

    # Default / shared "convection" behavior in this repo is the CDR residual.
    # Match the user's JAX implementation:
    #   u_t + beta*u_x - nu*u_xx + rho*u = 0
    return u_t + beta * u_x - nu * u_xx + rho * u_phys


# Backward compatibility: wrapper functions matching utils.py interface
def f_cal_equation_aware(
    x,
    t,
    beta,
    nu,
    rho,
    net,
    hidden_dim,
    equation: str = 'cdr',
    *,
    normalization_stats: dict | None = None,
):
    """
    Equation-aware version of utils.f_cal.

    Returns:
      - pde_residual: tensor shaped like u (N,1) at collocation points
      - reg_f: orthogonality regularization term (scalar)
    """
    u, col_0_f, col_1_f, col_2_f, row_0_f, row_1_f, row_2_f = net(x, t, beta, nu, rho)
    u_mean = None
    u_std = None
    if normalization_stats:
        u_mean = normalization_stats.get("u_mean", None)
        u_std = normalization_stats.get("u_std", None)
    pde_residual = _pde_residual_hyper_lr(
        equation,
        u,
        x,
        t,
        beta,
        nu,
        rho,
        u_mean=u_mean,
        u_std=u_std,
    )

    from utils import orthogonality_reg
    reg_f_0 = orthogonality_reg(col_0_f, row_0_f, hidden_dim, device=col_0_f.device)
    reg_f_1 = orthogonality_reg(col_1_f, row_1_f, hidden_dim, device=col_0_f.device)
    reg_f_2 = orthogonality_reg(col_2_f, row_2_f, hidden_dim, device=col_0_f.device)
    reg_f = reg_f_0 + reg_f_1 + reg_f_2

    return pde_residual, reg_f


def f_cal_phase2_equation_aware(
    x,
    t,
    beta,
    nu,
    rho,
    net,
    equation: str = 'cdr',
    *,
    normalization_stats: dict | None = None,
):
    """
    Equation-aware version of f_cal_phase2 that supports multiple equations.
    
    This is a drop-in replacement for f_cal_phase2() in utils.py but with equation support.
    """
    u = net(x, t)
    u_mean = None
    u_std = None
    if normalization_stats:
        u_mean = normalization_stats.get("u_mean", None)
        u_std = normalization_stats.get("u_std", None)
    pde_residual = _pde_residual_hyper_lr(
        equation,
        u,
        x,
        t,
        beta,
        nu,
        rho,
        u_mean=u_mean,
        u_std=u_std,
    )
    return pde_residual

