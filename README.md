# When Geometry Fails: Stress-Testing Git Re-Basin on Spurious vs Robust Features

This repository contains a complete, reproducible research codebase for investigating how Git Re-Basin (permutation-based model alignment) handles models with different learned mechanisms.

## Core Hypothesis

Permutation alignment (Git Re-Basin / weight matching) can successfully connect independently trained models that rely on the **same** spurious or robust features, but will **fail** to connect a model relying on a spurious feature to one relying on a robust feature, producing a measurable **loss barrier** and a **semantic barrier** (mechanism mismatch) along the interpolation path.

## Project Structure

```
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── src/                      # Reusable Python modules
│   ├── __init__.py
│   ├── config.py            # Global configuration
│   ├── data.py              # CIFAR-10 + spurious patch injection
│   ├── models.py            # ConvNet architecture
│   ├── train.py             # Training and evaluation
│   ├── rebasin.py           # Git Re-Basin implementation
│   ├── interp.py            # Weight interpolation utilities
│   ├── metrics.py           # Spurious Reliance Score, barriers
│   └── plotting.py          # Visualization helpers
├── notebooks/               # Jupyter notebooks (run in order)
│   ├── 00_setup.ipynb
│   ├── 01_data_spurious_envs.ipynb
│   ├── 02_train_models.ipynb
│   ├── 03_mechanism_verification.ipynb
│   ├── 04_rebasin_alignment.ipynb
│   ├── 05_interpolation_and_barriers.ipynb
│   └── 06_summary_report.ipynb
└── results/                 # Generated outputs (created automatically)
    ├── checkpoints/         # Model weights
    ├── figures/             # Visualizations
    └── metrics/             # JSON metrics files
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Notebooks in Order

Navigate to the `notebooks/` directory and run each notebook sequentially:

```
00_setup.ipynb              → Validate environment, set seeds
01_data_spurious_envs.ipynb → Create and visualize spurious environments
02_train_models.ipynb       → Train 4 models (A1, A2, R1, R2)
03_mechanism_verification.ipynb → Quantify spurious reliance
04_rebasin_alignment.ipynb  → Perform Git Re-Basin alignment
05_interpolation_and_barriers.ipynb → Analyze loss barriers
06_summary_report.ipynb     → Generate final report
```

Each notebook is designed to be run top-to-bottom with minimal manual intervention.

## Environments

We create CIFAR-10 variants with spurious colored patches:

| Environment | Description | Patch-Label Alignment |
|-------------|-------------|----------------------|
| **Env A** | Spurious aligned | 95% |
| **Env B** | Spurious flipped | 5% |
| **No Patch** | Clean CIFAR-10 | N/A (OOD test) |

## Models

We train 4 models to compare:

| Model | Training Data | Expected Behavior |
|-------|--------------|-------------------|
| **A1** | Env A only (seed 1) | Relies on spurious patch |
| **A2** | Env A only (seed 2) | Relies on spurious patch |
| **R1** | Mixed Env A+B (seed 1) | More robust features |
| **R2** | Mixed Env A+B (seed 2) | More robust features |

## Key Metrics

### Spurious Reliance Score (SRS)
```
SRS = 0.4 * OOD_drop + 0.3 * CF_accuracy_drop + 0.3 * flip_rate
```
Where:
- **OOD_drop**: ID accuracy - OOD accuracy
- **CF_accuracy_drop**: Accuracy drop when patch color is swapped
- **flip_rate**: Rate of prediction flips on counterfactual images

### Loss Barrier
```
barrier = max_α L(θ_α) - max(L(θ_0), L(θ_1))
```
Where θ_α is the interpolated model at position α ∈ [0, 1].

## Expected Results

1. **Spurious models (A1, A2)** should have:
   - High SRS (~0.2-0.4)
   - Large OOD accuracy drop (>20%)

2. **Robust models (R1, R2)** should have:
   - Low SRS (~0.05-0.1)
   - Small OOD accuracy drop (<10%)

3. **Same-mechanism pairs** (A1↔A2, R1↔R2) should have:
   - Low loss barriers after Re-Basin

4. **Different-mechanism pairs** (A1↔R1) should have:
   - Higher loss barriers even after Re-Basin
   - Significant SRS variation along interpolation path

## Configuration

Key hyperparameters can be modified in `src/config.py`:

```python
CONFIG = {
    "patch": {
        "size": 6,              # Patch size in pixels
        "p_align_env_a": 0.95,  # Env A alignment probability
        "p_align_env_b": 0.05,  # Env B alignment probability
    },
    "training": {
        "batch_size": 128,
        "num_epochs": 30,
        "learning_rate": 0.1,
    },
    "interpolation": {
        "num_alphas": 21,       # Number of interpolation points
    },
}
```

## Hardware Requirements

- **GPU**: NVIDIA GPU with CUDA support (recommended)
- **Memory**: ~8GB VRAM for training
- **Time**: ~30 minutes per model on modern GPU

The code also supports CPU and Apple MPS backends, though training will be slower.

## Outputs

After running all notebooks, you'll find:

### results/checkpoints/
- `model_A1.pt`, `model_A2.pt`, `model_R1.pt`, `model_R2.pt`
- `model_A2_aligned_to_A1.pt`, etc.

### results/figures/
- Dataset visualizations
- Training curves
- Barrier comparison plots
- Interpolation path plots
- Final summary figure

### results/
- `summary.json` - All numerical results
- `final_report.json` - Key findings

## Citation

If you use this code, please cite:

```
@misc{geometry_fails_rebasin,
  title={When Geometry Fails: Stress-Testing Git Re-Basin on Spurious vs Robust Features},
  year={2024},
  note={6.9760 Course Project}
}
```

## References

- Ainsworth et al., "Git Re-Basin: Merging Models modulo Permutation Symmetries"
- Sagawa et al., "Distributionally Robust Neural Networks"

## License

MIT License
