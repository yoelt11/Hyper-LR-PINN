# Usage Guide for Parametric Train/Test Scripts

This guide explains how to use the new training and testing scripts with parametric datasets from `eff-physics-learn-dataset`.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Workflow Overview](#workflow-overview)
4. [Configuration Files](#configuration-files)
5. [Training](#training)
6. [Testing](#testing)
7. [Examples](#examples)
8. [Troubleshooting](#troubleshooting)

## Prerequisites

Before using the scripts, ensure you have:

1. **Python 3.9+** installed
2. **UV package manager** (recommended) or pip
3. **Access to the repositories**:
   - `eff-physics-learn-dataset` (SSH access required)
   - `metrics-structures` (public repository)

## Environment Setup

### Option 1: Using UV (Recommended)

```bash
# Install UV if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv sync

# Or manually:
# uv venv
# source .venv/bin/activate  # On Windows: .venv\Scripts\activate
# uv pip install -e .
```

### Downloading Datasets

Before training, you need to download the datasets. The project includes a download script based on the [eff-physics-learn-dataset repository](https://github.com/yoelt11/eff-physics-learn-dataset):

```bash
# List available datasets
uv run python scripts/download_datasets.py --list

# Download a specific dataset (our case: convection)
uv run python scripts/download_datasets.py -d convection -o ./data_gen/dataset

# Download all datasets
uv run python scripts/download_datasets.py -o ./data_gen/dataset
```

**Available datasets:**
- `allen_cahn` - Allen-Cahn equation
- `convection` - Convection equation
- `burgers` - Burgers equation
- `helmholtz2D`, `helmholtz3D` - Helmholtz equation (2D/3D)
- `flow_mixing` - Flow mixing simulation
- `hlrp_*` - HLRP variants (cdr, convection, diffusion, helmholtz, reaction)

The datasets will be downloaded to `./data_gen/dataset/{equation_name}/` with the required `ground_truth/` directory structure.

### Option 2: Using Pip

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Verify Installation

```bash
# Check that external repositories are installed
python -c "from eff_physics_learn_dataset import load_pde_dataset; print('eff-physics-learn-dataset: OK')"
python -c "from metrics_structures import RunData; print('metrics-structures: OK')"
```

## Workflow Overview

The Hyper-LR-PINN framework uses a **three-phase workflow**:

1. **Phase 1: Pre-Training (Meta-Learning)**
   - Trains the hypernetwork to learn parameter-conditioned weights
   - Model learns to generalize across a range of parameter values
   - Uses `LR_PINN_phase1` model
   - **This is a training phase**

2. **Phase 2: Adaptation (Fine-Tuning)**
   - Adapts the pre-trained model to specific parameter values
   - Fine-tunes the hypernetwork for a particular parameter setting
   - Uses `LR_PINN_phase2` model (initialized from Phase 1)
   - **This is also a training phase** (fine-tuning/adaptation)

3. **Testing/Evaluation**
   - Evaluates the adapted model on test splits
   - Tests interpolation and extrapolation capabilities
   - Uses `scripts/test.py` script
   - **This is the evaluation phase** (not training)

**Important**: Phase 2 is the **adaptation phase** where the model is fine-tuned. While it adapts the model (which might seem like "testing"), it is still a **training phase**. The actual testing/evaluation is done by the `test.py` script.

See `WORKFLOW_CLARIFICATION.md` for more details.

## Dataset Modalities

The `eff-physics-learn-dataset` repository provides **two modalities**:

1. **Parametric Dataset** (default, used by this framework):
   - Contains solutions across a **range of parameter values**
   - Enables parametric generalization (learning how solutions vary with parameters)
   - Supports `parametric_splits()` to generate train_few, interp, and extrap splits
   - **This is the modality used by default in our implementation**

2. **Non-Parametric Dataset**:
   - Contains solutions for **fixed parameter values**
   - Focuses on learning solution patterns for a single parameter setting
   - Does not support parametric splits
   - **Not currently supported** - this framework requires parametric modality

**Important**: This framework uses the **Parametric Dataset** modality by default, which is required for the parametric evaluation capabilities (interpolation/extrapolation splits).

## Configuration Files

Configuration files are YAML files that specify all training and evaluation parameters. They should be placed in the `configs/` directory.

### Configuration Structure

```yaml
dataset:
  equation: "allen_cahn"        # Equation name from eff-physics-learn-dataset
  data_dir: "./data_gen/dataset" # Path to dataset directory
  seed: 42                       # Random seed for deterministic splits
  n_train: 10                    # Number of training samples (few-shot)
  cache: true                    # Use cached dataset if available
  # modality: "parametric"      # Optional: explicitly specify modality (default: parametric)

model:
  hidden_dim: 50                 # Hidden dimension for neural network
  phase: "phase1"                 # "phase1" (meta-learning) or "phase2" (fine-tuning)
  phase1_checkpoint: null         # Path to phase1 checkpoint (required for phase2)

training:
  epochs: 20000                   # Number of training epochs
  learning_rate: 0.00025          # Learning rate
  batch_size: null                 # Batch size (null = full batch)
  optimizer: "adam"               # Optimizer type
  checkpoint_interval: 1000       # Save checkpoint every N epochs

export:
  output_dir: "./outputs"         # Output directory for results
  save_plots: true                 # Generate visualization plots
  plot_max_samples: 5             # Maximum samples to plot

device: "cuda:0"                  # Device to use (cuda:0, cpu, etc.)
seed: 42                          # Global random seed
```

### Supported Equations

The following equations are typically available in `eff-physics-learn-dataset`:

- `allen_cahn` - Allen-Cahn equation
- `convection` - Convection equation
- `diffusion` - Diffusion equation
- `reaction` - Reaction equation
- `convection_diffusion` - Convection-Diffusion equation
- `reaction_diffusion` - Reaction-Diffusion equation
- `convection_diffusion_reaction` - Convection-Diffusion-Reaction equation

**Note**: Check the `eff-physics-learn-dataset` repository for the complete list of available equations.

## Training

### Phase 1: Pre-Training (Meta-Learning)

Phase 1 is the **pre-training phase** that trains the hypernetwork to learn parameter-conditioned weights. This is the meta-learning phase where the model learns to generalize across different parameter values.

#### Step 1: Create / Select Configuration File

For our case (convection), you can use the provided config:

- `configs/convection_phase1.yaml`

If you want to create your own, use the template below (convection shown):

```yaml
dataset:
  equation: "convection"
  data_dir: "./data_gen/dataset"
  seed: 42
  n_train: 10
  cache: true

model:
  hidden_dim: 50
  phase: "phase1"

training:
  epochs: 20000
  learning_rate: 0.00025
  batch_size: null
  optimizer: "adamw"
  checkpoint_interval: 1000

export:
  output_dir: "./outputs"
  save_plots: true
  plot_max_samples: 5

device: "cuda:0"
seed: 42
```

#### Step 2: Run Training

```bash
python scripts/train.py --config configs/convection_phase1.yaml
```

#### Output

Training will create:
- Checkpoints in `outputs/convection_phase1/checkpoints/`
- Training metrics in `outputs/convection_phase1/metrics/`
- Logs printed to console

### Phase 2: Adaptation (Fine-Tuning)

Phase 2 is the **adaptation phase** that fine-tunes the model for specific parameter values using the phase1 checkpoint. This adapts the pre-trained hypernetwork to work with a particular parameter setting.

#### Step 1: Create / Select Configuration File

For our case (convection), you can use the provided config:

- `configs/convection_phase2.yaml`

The only required edit is ensuring `model.phase1_checkpoint` points to your Phase 1 checkpoint (default recommended below).

If you want to create your own, use the template below (convection shown):

```yaml
dataset:
  equation: "convection"
  data_dir: "./data_gen/dataset"
  seed: 42
  n_train: 10
  cache: true

model:
  hidden_dim: 50
  phase: "phase2"
  phase1_checkpoint: "./outputs/convection_phase1/checkpoints/best.pt"

training:
  epochs: 10000
  learning_rate: 0.00025
  batch_size: null
  optimizer: "adamw"
  checkpoint_interval: 1000

export:
  output_dir: "./outputs"
  save_plots: true
  plot_max_samples: 5

device: "cuda:0"
seed: 42
```

**Important**: Update `phase1_checkpoint` to point to your phase1 checkpoint path.

#### Step 2: Run Training

```bash
python scripts/train.py --config configs/convection_phase2.yaml
```

## Testing/Evaluation

After Phase 2 adaptation, evaluate the model on interpolation and extrapolation splits. This is the **evaluation phase** that tests the model's parametric capabilities.

### Step 1: Ensure Configuration File Exists

Use the same configuration file as training (e.g., `configs/convection_phase2.yaml`).

### Step 2: Run Evaluation

```bash
python scripts/test.py \
  --config configs/convection_phase2.yaml \
  --ckpt_path outputs/convection_phase2/checkpoints/best.pt
```

### Output

Testing will create:
- Evaluation metrics in `outputs/convection_phase2/metrics/evaluation.pkl` and `.json`
- Visualization plots in `outputs/convection_phase2/plots/`
- Console output with metrics summary

## Examples

### Complete Workflow: Convection Equation (our case)

#### 1. Phase 1: Pre-Training (Meta-Learning)

```bash
# Run pre-training
python scripts/train.py --config configs/convection_phase1.yaml
```

This trains the hypernetwork to learn parameter-conditioned weights across a range of parameter values.

#### 2. Phase 2: Adaptation (Fine-Tuning)

```bash
# Ensure configs/convection_phase2.yaml points at your phase1 checkpoint:
#   model.phase1_checkpoint: "./outputs/convection_phase1/checkpoints/best.pt"

# Run adaptation
python scripts/train.py --config configs/convection_phase2.yaml
```

This adapts the pre-trained model to work with specific parameter values.

#### 3. Testing/Evaluation

```bash
python scripts/test.py \
  --config configs/convection_phase2.yaml \
  --ckpt_path outputs/convection_phase2/checkpoints/best.pt
```

This evaluates the adapted model on interpolation and extrapolation test splits.

### Using Different Devices

```bash
# Use CPU
python scripts/train.py --config configs/convection_phase1.yaml --device cpu

# Use specific GPU
python scripts/train.py --config configs/convection_phase1.yaml --device cuda:1
```

### Custom Training Parameters

You can override training parameters by editing the configuration file:

```yaml
training:
  epochs: 50000          # Train for more epochs
  learning_rate: 0.001   # Use higher learning rate
  checkpoint_interval: 500 # Save checkpoints more frequently
```

## Command-Line Options

### Training Script (`scripts/train.py`)

```bash
python scripts/train.py [OPTIONS]

Options:
  --config PATH    Path to YAML configuration file (required)
  --device DEVICE  Device to use (overrides config: cuda:0, cpu, etc.)
```

### Testing Script (`scripts/test.py`)

```bash
python scripts/test.py [OPTIONS]

Options:
  --config PATH      Path to YAML configuration file (required)
  --ckpt_path PATH   Path to model checkpoint file (required)
  --device DEVICE    Device to use (overrides config: cuda:0, cpu, etc.)
```

## Output Structure

After running training and testing, you'll have the following structure:

```
outputs/
└── convection_phase1/          # Phase 1 outputs
    ├── checkpoints/
    │   ├── checkpoint_epoch_1000.pt
    │   ├── checkpoint_epoch_2000.pt
    │   └── best.pt
    └── metrics/
        ├── {run_id}.pkl
        └── {run_id}.json
└── convection_phase2/          # Phase 2 outputs
    ├── checkpoints/
    │   ├── checkpoint_epoch_1000.pt
    │   └── best.pt
    ├── metrics/
    │   ├── {run_id}.pkl
    │   ├── {run_id}.json
    │   ├── evaluation.pkl
    │   └── evaluation.json
    └── plots/
        ├── interp_sample_0_2d.png
        ├── interp_sample_1_2d.png
        ├── extrap_sample_0_2d.png
        └── ...
```

## Loading and Analyzing Metrics

Metrics are exported in the `metrics-structures` format. You can load them in Python:

```python
from metrics_structures import RunData

# Load metrics
run_data = RunData.load("outputs/convection_phase2/metrics/evaluation.pkl")

# Access metrics
print("Interpolation MSE:", run_data.metadata['split_metrics']['interp']['mse'])
print("Extrapolation MSE:", run_data.metadata['split_metrics']['extrap']['mse'])

# Convert to DataFrame for analysis
df = run_data.get_dataframe()
print(df)
```

## Troubleshooting

### Issue: Import Error for eff-physics-learn-dataset

**Error**: `ImportError: eff-physics-learn-dataset not found`

**Solution**:
```bash
# Install using SSH (requires SSH key setup)
pip install git+ssh://git@github.com:yoelt11/eff-physics-learn-dataset.git

# Or clone and install locally
git clone git@github.com:yoelt11/eff-physics-learn-dataset.git
cd eff-physics-learn-dataset
pip install -e .
```

### Issue: Import Error for metrics-structures

**Error**: `ImportError: metrics-structures not found`

**Solution**:
```bash
pip install git+https://github.com/yoelt11/metrics-structures.git
```

### Issue: Dataset Not Found

**Error**: Dataset file not found or equation not recognized

**Solution**:
1. Check that the equation name matches what's available in `eff-physics-learn-dataset`
2. Verify the `data_dir` path is correct
3. Ensure the dataset has been generated/downloaded

### Issue: CUDA Out of Memory

**Error**: `RuntimeError: CUDA out of memory`

**Solution**:
1. Reduce batch size in config (if using batches)
2. Use CPU: `--device cpu`
3. Reduce model size (hidden_dim)
4. Process data in smaller chunks

### Issue: Phase1 Checkpoint Not Found (Phase 2)

**Error**: `FileNotFoundError: Phase1 checkpoint not found`

**Solution**:
1. Ensure phase1 training completed successfully
2. Check the path in `phase1_checkpoint` in config file
3. Verify the checkpoint file exists at the specified path

### Issue: Configuration Validation Error

**Error**: `ValueError: Missing required field...`

**Solution**:
1. Check that all required sections are present in YAML file
2. Verify YAML syntax is correct (proper indentation, no tabs)
3. Use the example configs as templates

## Quick Reference

### Workflow Overview

The complete workflow consists of three phases:

1. **Phase 1: Pre-Training** (Meta-learning) - Trains hypernetwork across parameter range
2. **Phase 2: Adaptation** (Fine-tuning) - Adapts model to specific parameter values  
3. **Testing/Evaluation** - Evaluates model on interpolation and extrapolation splits

### Training Commands

```bash
# Phase 1: Pre-Training (Meta-learning)
python scripts/train.py --config configs/convection_phase1.yaml

# Phase 2: Adaptation (Fine-tuning to specific parameters)
python scripts/train.py --config configs/convection_phase2.yaml

# With custom device
python scripts/train.py --config configs/convection_phase1.yaml --device cuda:1
```

### Testing/Evaluation Commands

```bash
# Evaluation after Phase 2 adaptation
python scripts/test.py \
  --config configs/convection_phase2.yaml \
  --ckpt_path outputs/convection_phase2/checkpoints/best.pt

# With custom device
python scripts/test.py \
  --config configs/convection_phase2.yaml \
  --ckpt_path outputs/convection_phase2/checkpoints/best.pt \
  --device cpu
```

**Note**: Phase 2 is the adaptation phase where the model is fine-tuned for specific parameters. The `test.py` script is used for final evaluation/testing on the adapted model.

### Fine-tuning Evaluation with Checkpoint Tracking

The `finetune_eval.py` script performs fine-tuning on both interpolation and extrapolation splits, evaluating at multiple checkpoints (zero-shot, 50, 250, 500, 750, 1000 epochs) and generates a summary table with L2 Error and Time metrics.

```bash
# Fine-tuning evaluation with default checkpoints (50, 250, 500, 750, 1000)
python scripts/finetune_eval.py \
  --config configs/convection_phase2.yaml \
  --phase1_checkpoint outputs/convection_phase1/checkpoints/best.pt \
  --max_epochs 1000

# Custom checkpoint epochs
python scripts/finetune_eval.py \
  --config configs/convection_phase2.yaml \
  --phase1_checkpoint outputs/convection_phase1/checkpoints/best.pt \
  --checkpoint_epochs 50 100 250 500 750 1000 \
  --max_epochs 1000

# With custom device
python scripts/finetune_eval.py \
  --config configs/convection_phase2.yaml \
  --phase1_checkpoint outputs/convection_phase1/checkpoints/best.pt \
  --device cuda:1
```

**What it does:**
1. **Zero-shot Evaluation**: Evaluates the Phase 1 model (before fine-tuning) on both interp and extrap splits
2. **Fine-tuning on Interpolation Split**: Fine-tunes the model on the interpolation split and evaluates at checkpoints
3. **Fine-tuning on Extrapolation Split**: Fine-tunes the model on the extrapolation split and evaluates at checkpoints
4. **Summary Table**: Generates a CSV table with format:
   - `Epoch | Stage | Interp_L2_Error | Interp_Time | Extrap_L2_Error | Extrap_Time`
   - Shows zero-shot (epoch 0) and fine-tuning results at specified epochs

**Output:**
- Summary tables saved to `outputs/{equation}_phase2/evaluation/finetune_summary.csv` and `finetune_summary_compact.csv`
- The compact format matches the requested table structure showing L2 Error and Time for each checkpoint

## PINN Loss Functions

The training scripts use PINN (Physics-Informed Neural Network) loss functions. The implementation includes:

- **Location**: PINN losses are in `utils.py` (original) and `src/pinn_losses.py` (equation-aware)
- **Current Usage**: Training scripts use `f_cal()` and `f_cal_phase2()` from `utils.py` which are optimized for the Hyper-LR-PINN architecture
- **Equation-Specific Losses**: Available in `src/pinn_losses.py` for:
  - Allen-Cahn equation
  - Convection equation
  - CDR (Convection-Diffusion-Reaction) equation
- **See**: `PINN_LOSSES.md` for detailed documentation

## Additional Resources

- **Plan Document**: See `PLAN.md` for implementation details
- **Implementation Summary**: See `IMPLEMENTATION_SUMMARY.md` for component overview
- **Implementation Complete**: See `IMPLEMENTATION_COMPLETE.md` for final status
- **Dry-Run Results**: See `DRY_RUN_RESULTS.md` for test results
- **PINN Losses**: See `PINN_LOSSES.md` for loss function documentation
- **Dataset Modalities**: See `DATASET_MODALITIES.md` for dataset information
- **Scripts README**: See `scripts/README.md` for script-specific documentation

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Verify all dependencies are installed correctly
3. Ensure configuration files are valid YAML
4. Check that dataset paths and checkpoint paths are correct

