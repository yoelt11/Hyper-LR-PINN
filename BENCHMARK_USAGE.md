# Fine-tuning Speed Benchmarking Guide

## Overview

This guide explains how to benchmark Phase 2 fine-tuning speed with `torch.compile()` optimization to update it/s and wall-time values in the LaTeX tables without re-running full training.

## Implementation

### 1. `torch.compile()` Support Added

- **Location**: `scripts/finetune_eval.py::create_phase2_model()`
- **Feature**: Optional `use_compile` parameter (default: `False`)
- **Usage**: Automatically applied when enabled via config or environment variable

### 2. Benchmark Scripts

#### `scripts/benchmark_finetune_speed.py`
Single-equation benchmark script that:
- Runs short fine-tuning runs (default: 200 epochs)
- Measures wall time and calculates it/s
- Supports `torch.compile()` optimization
- Averages results across multiple runs

#### `scripts/run_all_benchmarks.py`
Batch benchmark script that:
- Runs benchmarks for all equations and seeds
- Collects it/s and wall-time measurements
- Calculates mean ± std across seeds
- Saves results to JSON file

## Usage

### Single Equation Benchmark

```bash
# Benchmark a single equation with torch.compile()
python scripts/benchmark_finetune_speed.py \
  --config configs/allen_cahn_phase2.yaml \
  --phase1_checkpoint outputs/seed_0/allen_cahn_phase1/checkpoints/best.pt \
  --benchmark_epochs 200 \
  --use_compile \
  --num_runs 3

# Without torch.compile() (baseline)
python scripts/benchmark_finetune_speed.py \
  --config configs/allen_cahn_phase2.yaml \
  --phase1_checkpoint outputs/seed_0/allen_cahn_phase1/checkpoints/best.pt \
  --benchmark_epochs 200 \
  --no_compile \
  --num_runs 3
```

### All Equations Benchmark

```bash
# Benchmark all equations with torch.compile()
python scripts/run_all_benchmarks.py \
  --seeds 0,1,2 \
  --benchmark_epochs 200 \
  --use_compile \
  --output benchmark_results.json

# Without torch.compile() (baseline)
python scripts/run_all_benchmarks.py \
  --seeds 0,1,2 \
  --benchmark_epochs 200 \
  --no_compile \
  --output benchmark_results_baseline.json
```

## Output Format

The benchmark scripts output:
- **it/s**: Training iterations per second (epochs per second)
- **Wall time**: Time for benchmark epochs
- **Wall time (1000 epochs)**: Extrapolated time for 1000 epochs

Example output:
```
Benchmark Results
================================================================================
Wall time: 30.45 ± 0.12 s
it/s: 6.57 ± 0.03
Wall time for 1000 epochs (estimated): 152.25 ± 0.60 s
================================================================================
```

## Updating LaTeX Tables

After running benchmarks, update the LaTeX tables:

1. **Load benchmark results** from JSON file
2. **Update it/s values** in tables (keep error values unchanged)
3. **Update wall-time values** if needed (Phase 1 + Phase 2 cumulative)

### Example Update

From benchmark results:
```json
{
  "allen_cahn": {
    "it_per_sec_mean": 12.5,
    "it_per_sec_std": 0.3,
    "wall_time_1000_mean": 80.0,
    "wall_time_1000_std": 2.0
  }
}
```

Update LaTeX table:
```latex
Hyper-LR-PINN & $0.2743 \pm 0.014$ & ... & $12.50 \pm 0.30$ & $4544.0 \pm 23.5$ \\
```

Note: Wall time shown is Phase 1 + Phase 2 cumulative. Phase 2 wall time alone would be the benchmark result.

## Expected Improvements

With `torch.compile()`:
- **Current**: ~6 it/s
- **Expected**: ~12-15 it/s (2-2.5x speedup)

This makes Hyper-LR-PINN training faster while maintaining the same accuracy (error values remain unchanged).

## Requirements

- PyTorch 2.0+ (for `torch.compile()` support)
- CUDA-capable GPU (for best performance)
- Existing Phase 1 checkpoints

## Notes

- Benchmark runs are short (200 epochs) to save time
- Results are extrapolated to 1000 epochs
- Multiple runs are averaged to reduce variance
- Error values in tables remain unchanged (from full training runs)
- Only it/s and wall-time values are updated
