# Training and Testing Scripts

This directory contains standardized training and testing scripts that integrate with `eff-physics-learn-dataset` and `metrics-structures` repositories.

## Setup

First, ensure you have installed the required dependencies:

```bash
# Using uv (recommended)
uv pip install -e .

# Or using pip
pip install -r requirements.txt
```

## Training

### Phase 1 (Meta-learning)

Train the hypernetwork for meta-learning:

```bash
python scripts/train.py --config configs/convection_phase1.yaml
```

### Phase 2 (Fine-tuning)

Fine-tune the model for specific parameter values:

```bash
python scripts/train.py --config configs/convection_phase2.yaml
```

## Testing

Evaluate a trained model on interpolation and extrapolation splits:

```bash
python scripts/test.py \
  --config configs/convection_phase2.yaml \
  --ckpt_path outputs/convection_phase2/checkpoints/best.pt
```

## Configuration Files

Configuration files are YAML files that specify:

- **Dataset settings**: Equation name, data directory, split parameters
- **Model settings**: Architecture hyperparameters
- **Training settings**: Learning rate, batch size, epochs
- **Export settings**: Paths for metrics export and visualization

Example configuration files are provided in the `configs/` directory.

## Output Structure

The scripts generate output in the following structure:

```
outputs/
└── {equation}_{phase}/
    ├── checkpoints/
    │   ├── checkpoint_epoch_1000.pt
    │   ├── checkpoint_epoch_2000.pt
    │   └── best.pt
    ├── metrics/
    │   ├── {run_id}.pkl
    │   └── {run_id}.json
    └── plots/
        ├── interp_sample_0_2d.png
        ├── extrap_sample_0_2d.png
        └── ...
```

## Metrics Export

All evaluation metrics are exported using the `metrics-structures` format, which includes:

- Aggregate metrics per split (MSE, L2 error, relative L2 error)
- Per-sample metrics
- Timing information
- Full run configuration

Metrics can be loaded and analyzed using the `metrics-structures` library.

