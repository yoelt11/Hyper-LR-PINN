#!/usr/bin/env python
"""
Quick sanity checks for PDE residual implementations used by training.

This script validates:
- Convection/"convection" path uses the CDR residual (matches utils.f_cal convention)
  and reduces to pure advection when nu=rho=0.
- Helmholtz2D residual matches Δu + k^2 u = 0 for an analytic separable solution.

Run:
  uv run python scripts/validate_losses.py
"""

import math
import torch

from src.pinn_losses import _pde_residual_hyper_lr


def _make_grid_1d(n: int, device: str) -> torch.Tensor:
    x = torch.linspace(0.0, 1.0, int(n), device=device).view(-1, 1)
    x.requires_grad_(True)
    return x


def _max_abs(x: torch.Tensor) -> float:
    return float(x.detach().abs().max().cpu().item())


def check_convection_reduces_to_advection(*, device: str = "cpu") -> None:
    """
    For nu=rho=0, the "convection" key should behave like advection:
        u_t + beta u_x = 0
    with u(x,t) = sin(2π(x - beta t)).
    """
    n = 200
    x = _make_grid_1d(n, device)
    t = _make_grid_1d(n, device)  # reuse a 1D grid; treat as paired points

    beta = torch.full_like(x, 0.7)
    nu = torch.zeros_like(x)
    rho = torch.zeros_like(x)

    u = torch.sin(2.0 * math.pi * (x - beta * t))
    u = u.view(-1, 1)

    r = _pde_residual_hyper_lr("convection", u, x, t, beta, nu, rho)
    print("[convection] max|residual| (nu=rho=0, analytic advection):", _max_abs(r))

def check_convection_linear_reaction(*, device: str = "cpu") -> None:
    """
    For nu=0, the (convection/CDR) residual used in this repo is:
        u_t + beta u_x + rho u = 0
    Analytic solution: u(x,t) = exp(-rho t) * sin(2π(x - beta t)).
    """
    n = 200
    x = _make_grid_1d(n, device)
    t = _make_grid_1d(n, device)

    beta = torch.full_like(x, 0.9)
    nu = torch.zeros_like(x)
    rho = torch.full_like(x, 1.3)

    u = torch.exp(-rho * t) * torch.sin(2.0 * math.pi * (x - beta * t))
    u = u.view(-1, 1)

    r = _pde_residual_hyper_lr("convection", u, x, t, beta, nu, rho)
    print("[convection] max|residual| (nu=0, linear reaction analytic):", _max_abs(r))

def check_helmholtz2d_analytic(*, device: str = "cpu") -> None:
    """
    Helmholtz2D (dataset-consistent manufactured solution):

      u(x,y) = sin(a1*pi*x) sin(a2*pi*y)

    Dataset PDE is:
      Δu + k^2 u = q(x,y)
      q(x,y) = (-(a1*pi)^2 - (a2*pi)^2 + k^2) sin(a1*pi*x) sin(a2*pi*y)

    In this codebase, the second coordinate is passed as `t`; interpret it as y.
    """
    n = 64
    # Build a full mesh of (x,y) points on the dataset domain [-1, 1].
    # Use float64 to avoid visible float32 second-derivative noise.
    x1 = torch.linspace(-1.0, 1.0, n, device=device, dtype=torch.float64)
    y1 = torch.linspace(-1.0, 1.0, n, device=device, dtype=torch.float64)
    X, Y = torch.meshgrid(x1, y1, indexing="ij")
    x = X.reshape(-1, 1).clone().detach().requires_grad_(True)
    y = Y.reshape(-1, 1).clone().detach().requires_grad_(True)

    a1 = 2.7
    a2 = 3.4
    k = 4.6

    u = torch.sin(a1 * math.pi * x) * torch.sin(a2 * math.pi * y)
    u = u.view(-1, 1)

    # Parameter slot convention: (beta,nu,rho) = (a1,a2,k) for helmholtz2d
    beta = torch.full_like(x, a1)
    nu = torch.full_like(x, a2)
    rho = torch.full_like(x, k)

    r = _pde_residual_hyper_lr("helmholtz2D", u, x, y, beta, nu, rho)
    print("[helmholtz2D] max|residual| (manufactured-solution analytic):", _max_abs(r))


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    check_convection_reduces_to_advection(device=device)
    check_convection_linear_reaction(device=device)
    check_helmholtz2d_analytic(device=device)


if __name__ == "__main__":
    main()


