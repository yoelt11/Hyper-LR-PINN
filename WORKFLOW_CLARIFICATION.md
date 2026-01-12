# Workflow Clarification

## Three-Phase Workflow

The Hyper-LR-PINN framework uses a three-phase workflow:

### Phase 1: Pre-Training (Meta-Learning)
- **Purpose**: Train the hypernetwork to learn parameter-conditioned weights
- **Model**: `LR_PINN_phase1`
- **Input**: Takes parameters (beta, nu, rho) as input
- **Training**: Learns to generalize across a range of parameter values
- **Script**: `scripts/train.py` with `phase: "phase1"` in config
- **Output**: Phase 1 checkpoint (hypernetwork weights)

**Command**:
```bash
python scripts/train.py --config configs/allen_cahn_phase1.yaml
```

### Phase 2: Adaptation (Fine-Tuning)
- **Purpose**: Adapt the pre-trained model to specific parameter values
- **Model**: `LR_PINN_phase2`
- **Input**: Only takes (x, t) - parameters are baked into weights
- **Training**: Fine-tunes the model for a specific parameter setting
- **Script**: `scripts/train.py` with `phase: "phase2"` in config
- **Requires**: Phase 1 checkpoint
- **Output**: Phase 2 checkpoint (adapted model)

**Command**:
```bash
python scripts/train.py --config configs/allen_cahn_phase2.yaml
```

**Note**: Phase 2 is sometimes called "adaptation" or "testing" because it adapts the model to a specific parameter, but it is still a **training phase** (fine-tuning).

### Phase 3: Testing/Evaluation
- **Purpose**: Evaluate the adapted model on test splits
- **Model**: Uses Phase 2 checkpoint
- **Evaluation**: Tests on interpolation and extrapolation splits
- **Script**: `scripts/test.py`
- **Output**: Metrics and visualizations

**Command**:
```bash
python scripts/test.py \
  --config configs/allen_cahn_phase2.yaml \
  --ckpt_path outputs/allen_cahn_phase2/checkpoints/best.pt
```

## Terminology Clarification

| Phase | Name | Type | Purpose |
|-------|------|------|---------|
| Phase 1 | Pre-Training / Meta-Learning | **Training** | Learn parameter-conditioned weights |
| Phase 2 | Adaptation / Fine-Tuning | **Training** | Adapt to specific parameters |
| Phase 3 | Testing / Evaluation | **Evaluation** | Test parametric capabilities |

## Complete Workflow Example

### Step 1: Pre-Train (Phase 1)
```bash
python scripts/train.py --config configs/allen_cahn_phase1.yaml
```
- Trains hypernetwork
- Output: `outputs/allen_cahn_phase1/checkpoints/best.pt`

### Step 2: Adapt (Phase 2)
```bash
python scripts/train.py --config configs/allen_cahn_phase2.yaml
```
- Adapts model to specific parameters
- Requires: Phase 1 checkpoint
- Output: `outputs/allen_cahn_phase2/checkpoints/best.pt`

### Step 3: Evaluate (Testing)
```bash
python scripts/test.py \
  --config configs/allen_cahn_phase2.yaml \
  --ckpt_path outputs/allen_cahn_phase2/checkpoints/best.pt
```
- Evaluates on interp/extrap splits
- Output: Metrics and plots

## Key Points

1. **Phase 1 = Pre-Training**: Meta-learning across parameter range
2. **Phase 2 = Adaptation**: Fine-tuning for specific parameters (still training)
3. **Testing = Evaluation**: Final evaluation on test splits (not training)

The confusion may arise because:
- Phase 2 adapts the model, which can be seen as "testing" the adaptation capability
- But Phase 2 is still a training phase (fine-tuning)
- The actual testing/evaluation is done by `test.py` script

## Mapping to Original Scripts

- `train_meta.py` → Phase 1 (Pre-training)
- `train_full.py` / `train_adap.py` → Phase 2 (Adaptation)
- `test.py` → Testing/Evaluation

