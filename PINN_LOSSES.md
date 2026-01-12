# PINN Loss Functions Implementation

## Overview

This document explains where PINN (Physics-Informed Neural Network) loss functions are implemented in the codebase and how they work.

## Current Implementation

### Location

PINN losses are implemented in two places:

1. **`utils.py`** (Original, equation-specific):
   - `f_cal()`: Phase 1 PDE residual computation (CDR equation only)
   - `f_cal_phase2()`: Phase 2 PDE residual computation (CDR equation only)
   - Hardcoded for Convection-Diffusion-Reaction equation

2. **`src/pinn_losses.py`** (New, equation-agnostic):
   - `allen_cahn_loss_pytorch()`: Allen-Cahn equation loss
   - `convection_loss_pytorch()`: Pure convection equation loss
   - `cdr_loss_pytorch()`: Convection-Diffusion-Reaction equation loss
   - `get_pinn_loss_pytorch()`: Factory function to get loss by equation name

## Loss Function Structure

All PINN loss functions compute three main components:

1. **PDE Residual Loss**: Measures how well the solution satisfies the PDE
2. **Boundary Condition (BC) Loss**: Enforces boundary conditions
3. **Initial Condition (IC) Loss**: Enforces initial conditions

### Total Loss Formula

```
Total Loss = pde_weight * PDE_Loss + bc_weight * BC_Loss + ic_weight * IC_Loss
```

## Equation-Specific Losses

### 1. Allen-Cahn Equation

**PDE**: `u_t - ε² u_xx + λ(u³ - u) = 0`

**Implementation**: `allen_cahn_loss_pytorch()`

**Parameters**:
- `eps`: Interface parameter (ε)
- `lam`: Diffusion parameter (λ)
- Default weights: `ic_weight=100.0` (strong IC enforcement)

**Boundary Conditions**: `u = -1` at spatial boundaries

**Initial Condition**: `u(x, 0) = x² cos(πx)`

### 2. Convection Equation

**PDE**: `u_t + β u_x = 0`

**Implementation**: `convection_loss_pytorch()`

**Parameters**:
- `beta`: Convection velocity (β)

**Boundary Conditions**: Periodic `u(0, t) = u(L, t)`

**Initial Condition**: `u(x, 0) = 1 + sin(2πx/L)`

### 3. CDR (Convection-Diffusion-Reaction) Equation

**PDE**: `u_t + β u_x - ν u_xx - ρ u(1-u) = 0`

**Implementation**: `cdr_loss_pytorch()`

**Parameters**:
- `beta`: Convection coefficient (β)
- `nu`: Diffusion coefficient (ν)
- `rho`: Reaction coefficient (ρ)

**Boundary Conditions**: Periodic

**Note**: This matches the existing `f_cal()` and `f_cal_phase2()` functions in `utils.py`.

## Usage in Training Scripts

### Current Usage (CDR only)

The training scripts (`scripts/train.py`) currently use:

```python
from utils import f_cal, f_cal_phase2

# Phase 1
f_out, reg_f = f_cal(x, t, beta, nu, rho, net, hidden_dim)
mse_f = mse_cost_function(f_out, all_zeros)

# Phase 2
f_out = f_cal_phase2(x, t, beta, nu, rho, net)
mse_f = mse_cost_function(f_out, all_zeros)
```

### Using Equation-Aware Losses

To use equation-specific losses, you can:

```python
from src.pinn_losses import get_pinn_loss_pytorch

# Get loss function for equation
loss_fn = get_pinn_loss_pytorch('allen_cahn')

# Compute loss
u = net(x, t, beta, nu, rho)  # or net(x, t) for phase2
loss_dict = loss_fn(u, x, t, eps=0.01, lam=1.0, return_components=True)
total_loss = loss_dict['total_loss']
```

## Integration with Training Scripts

The training scripts need to be updated to:

1. **Detect equation from config**: Read `equation` from dataset config
2. **Select appropriate loss function**: Use `get_pinn_loss_pytorch(equation)`
3. **Pass equation-specific parameters**: Extract from dataset or config

### Example Integration

```python
# In scripts/train.py
from src.pinn_losses import get_pinn_loss_pytorch

# Get equation from config
equation = config['dataset']['equation']
loss_fn = get_pinn_loss_pytorch(equation)

# In training loop
u_pred = net(x, t, beta, nu, rho)
loss_dict = loss_fn(
    u_pred, x, t,
    beta=beta, nu=nu, rho=rho,
    return_components=True
)
total_loss = loss_dict['total_loss']
```

## Comparison: JAX vs PyTorch

You provided a JAX-based `pinn_losses.py` file. The PyTorch version in `src/pinn_losses.py`:

- **Same equations supported**: Allen-Cahn, Convection, CDR, etc.
- **Same loss structure**: PDE + BC + IC components
- **Different framework**: PyTorch instead of JAX
- **Compatible with existing code**: Works with current PyTorch models

## Adding New Equations

To add a new equation:

1. **Create loss function** in `src/pinn_losses.py`:
```python
def new_equation_loss_pytorch(
    u, x, t,
    param1=1.0, param2=1.0,
    bc_weight=1.0, ic_weight=1.0, pde_weight=1.0,
    return_components=False
):
    # Compute derivatives
    derivs = compute_derivatives_pytorch(u, x, t)
    
    # PDE residual
    pde_residual = ...  # Your PDE
    pde_loss = torch.mean(pde_residual**2)
    
    # BC and IC losses
    bc_loss = ...
    ic_loss = ...
    
    # Return total or components
    ...
```

2. **Register in factory function**:
```python
def get_pinn_loss_pytorch(equation: str):
    loss_functions = {
        ...
        'new_equation': new_equation_loss_pytorch,
    }
    ...
```

3. **Update config files** to use new equation name

## Notes

- The original `utils.py` functions (`f_cal`, `f_cal_phase2`) are kept for backward compatibility
- The new `src/pinn_losses.py` provides equation-agnostic losses
- Both implementations compute the same PDE residual for CDR equation
- The new implementation supports multiple equations without code changes

## Future Improvements

1. **Causal PINN losses**: Add temporal causality weighting (like in JAX version)
2. **More equations**: Add Burgers, Helmholtz, Flow Mixing, etc.
3. **Automatic parameter extraction**: Extract equation parameters from dataset
4. **Loss weighting strategies**: Adaptive weighting schemes

