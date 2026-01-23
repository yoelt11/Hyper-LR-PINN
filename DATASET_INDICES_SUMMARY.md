# Dataset Indices Summary

This document lists the solution indices used for train, interpolation, and extrapolation splits for each equation and seed in `outputs_1/seed_0`, `outputs_1/seed_1`, and `outputs_1/seed_2`.

## About

These indices are obtained using the `eff-physics-learn-dataset` Python package. The process works as follows:

1. **Load the dataset** using `load_pde_dataset()` from `eff-physics-learn-dataset`:
   ```python
   dataset = load_pde_dataset(equation, data_dir, cache=cache)
   ```

2. **Generate splits** using the package's `parametric_splits()` method:
   ```python
   splits = dataset.parametric_splits(
       seed=seed,
       n_train=n_train,
       balance=balance,
       n_each=n_each,
       balance_strategy=balance_strategy,
       diversify=diversify
   )
   ```

3. **Extract indices** from the returned splits:
   ```python
   train_indices = [sample.get('index', i) for i, sample in enumerate(train_few)]
   interp_indices = [sample.get('index', i) for i, sample in enumerate(interp)]
   extrap_indices = [sample.get('index', i) for i, sample in enumerate(extrap)]
   ```

4. **Save to JSON files** for reference (in `data_gen/dataset_indices/`)

The `parametric_splits()` method from `eff-physics-learn-dataset`:
- Splits based on parameter ranges (interpolation vs extrapolation)
- Uses the provided seed for deterministic/reproducible splits
- Returns samples with their original indices from the full dataset

With the latest dataset package updates, `parametric_splits()` now defaults to
the solution_percentile-based method, which yields clearer interp/extrap
separation without extra arguments.

## Allen-Cahn

### Seed 0
- **Train (10 samples)**: [159, 160, 121, 99, 54, 9, 3, 61, 38, 16]
- **Interp (20 samples)**: [22, 45, 97, 186, 130, 119, 169, 120, 104, 78, 17, 4, 66, 35, 105, 189, 11, 192, 47, 108]
- **Extrap (20 samples)**: [196, 96, 15, 153, 28, 94, 41, 187, 164, 33, 84, 180, 193, 23, 143, 151, 166, 191, 65, 147]

### Seed 1
- **Train (10 samples)**: [64, 88, 160, 8, 185, 99, 31, 144, 182, 52]
- **Interp (20 samples)**: [135, 149, 116, 98, 188, 86, 174, 53, 39, 47, 25, 113, 16, 119, 130, 173, 192, 56, 97, 14]
- **Extrap (20 samples)**: [159, 141, 175, 80, 148, 126, 161, 140, 94, 112, 183, 102, 196, 82, 172, 46, 15, 129, 6, 9]

### Seed 2
- **Train (10 samples)**: [119, 59, 80, 24, 157, 193, 52, 21, 87, 68]
- **Interp (20 samples)**: [131, 5, 20, 48, 73, 14, 95, 57, 81, 56, 136, 155, 18, 67, 41, 167, 192, 105, 36, 158]
- **Extrap (20 samples)**: [177, 147, 178, 182, 151, 31, 107, 166, 32, 113, 134, 108, 180, 174, 160, 111, 26, 93, 112, 184]

## Burgers

### Seed 0
- **Train (10 samples)**: [160, 161, 118, 95, 50, 7, 2, 57, 34, 13]
- **Interp (20 samples)**: [22, 37, 97, 153, 90, 105, 128, 137, 124, 106, 96, 0, 79, 32, 43, 169, 3, 185, 47, 81]
- **Extrap (20 samples)**: [142, 102, 11, 171, 18, 100, 31, 10, 177, 26, 73, 178, 146, 12, 150, 188, 183, 17, 56, 157]

### Seed 1
- **Train (10 samples)**: [60, 87, 160, 6, 188, 95, 27, 143, 185, 48]
- **Interp (20 samples)**: [166, 137, 198, 97, 169, 56, 158, 164, 133, 22, 12, 96, 3, 172, 138, 140, 8, 43, 88, 116]
- **Extrap (20 samples)**: [167, 148, 181, 80, 152, 123, 165, 144, 98, 110, 122, 102, 196, 85, 179, 49, 18, 132, 11, 15]

### Seed 2
- **Train (10 samples)**: [117, 55, 78, 18, 158, 195, 48, 16, 86, 63]
- **Interp (16 samples)**: [8, 15, 21, 32, 39, 41, 96, 105, 124, 128, 138, 140, 181, 185, 187, 196]
- **Extrap (16 samples)**: [5, 93, 42, 66, 102, 156, 24, 177, 20, 126, 70, 81, 26, 180, 60, 77]

## Convection

### Seed 0
- **Train (10 samples)**: [160, 161, 118, 95, 50, 7, 2, 57, 34, 13]
- **Interp (20 samples)**: [36, 59, 104, 193, 143, 123, 172, 126, 106, 92, 28, 6, 78, 47, 112, 134, 18, 58, 61, 114]
- **Extrap (20 samples)**: [190, 163, 4, 108, 5, 85, 11, 173, 149, 128, 187, 142, 136, 169, 122, 139, 131, 175, 17, 99]

### Seed 1
- **Train (10 samples)**: [60, 87, 160, 6, 188, 95, 27, 143, 185, 48]
- **Interp (20 samples)**: [140, 162, 119, 102, 171, 96, 182, 62, 47, 61, 33, 128, 21, 123, 198, 29, 196, 69, 98, 19]
- **Extrap (20 samples)**: [151, 124, 177, 82, 136, 131, 184, 142, 79, 109, 135, 100, 197, 63, 178, 31, 7, 104, 5, 167]

### Seed 2
- **Train (10 samples)**: [117, 55, 78, 18, 158, 195, 48, 16, 86, 63]
- **Interp (20 samples)**: [134, 6, 24, 53, 71, 15, 91, 61, 79, 57, 141, 156, 20, 66, 39, 171, 196, 111, 37, 164]
- **Extrap (20 samples)**: [175, 149, 100, 178, 160, 28, 187, 163, 31, 114, 136, 108, 109, 174, 197, 166, 23, 99, 115, 181]

## Helmholtz2D

### Seed 0
- **Train (10 samples)**: [162, 163, 123, 99, 50, 8, 3, 57, 33, 14]
- **Interp (20 samples)**: [39, 40, 78, 178, 110, 136, 166, 172, 149, 127, 5, 199, 182, 173, 43, 183, 0, 82, 196, 97]
- **Extrap (20 samples)**: [134, 96, 10, 161, 18, 94, 29, 9, 171, 25, 71, 174, 139, 11, 143, 186, 181, 16, 54, 151]

### Seed 1
- **Train (10 samples)**: [61, 87, 162, 7, 188, 99, 27, 146, 184, 48]
- **Interp (20 samples)**: [176, 119, 199, 114, 173, 78, 166, 70, 40, 39, 22, 102, 5, 136, 196, 149, 183, 53, 97, 82]
- **Extrap (20 samples)**: [163, 145, 179, 76, 151, 123, 159, 140, 94, 109, 122, 101, 195, 85, 174, 46, 16, 132, 10, 12]

### Seed 2
- **Train (10 samples)**: [122, 55, 78, 20, 160, 195, 48, 17, 86, 64]
- **Interp (20 samples)**: [196, 5, 9, 166, 167, 142, 82, 97, 40, 155, 31, 178, 149, 53, 173, 148, 177, 16, 146, 96]
- **Extrap (20 samples)**: [171, 144, 99, 32, 198, 33, 192, 175, 36, 118, 125, 105, 187, 172, 153, 112, 27, 103, 108, 191]

## Notes

- All splits use `balance=true`, `n_each=20`, `balance_strategy="random"`, and `diversify=false`
- Most equations have 10 train samples, 20 interp samples, and 20 extrap samples (40 total test samples)
- Burgers seed 2 has 16 interp and 16 extrap samples (32 total test samples) - this appears to be due to dataset size constraints
- The indices are saved in `data_gen/dataset_indices/{equation}_seed{seed}_indices.json`
