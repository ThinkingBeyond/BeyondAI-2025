# How to Use This Code: UAT Experimentation Framework

## What This Is

This is the experimental pipeline for testing the Universal Approximation Theorem (UAT). You've already read the README and understand *why* this matters. This manual shows you *how* to run your own experiments—from basic setup to custom configurations.

**Current architecture:** Lightweight OOP design with two core classes (`ExperimentConfig`, `SingleHiddenLayerNet`) and modular functions. A full class-based refactor is coming, but this structure gives you everything you need to experiment now.

**What you need:** The notebook (`Empirical Validation of the UAT.ipynb`), this manual for structure, and the inline comments for function-level details.

---

## Quickstart: Your First Experiment

Let's run a real experiment: **Does ReLU approximate Gaussian functions better than tanh?**

### Step 1: Define Your Experiment

In the **"Run all"** section, modify the experimental grid:

```python
# Define what you want to test
target_functions = ['gaussian']  # Focus on Gaussians
activations = ['relu', 'tanh']   # Compare these two

# Shared parameters
base_config = {
    'neuron_counts': [10, 50, 100],  # Test these widths
    'epochs': 3000,
    'threshold': 0.01,
    'learning_rate': 0.01,
    'n_samples': 3000,
    'save_dir': 'results/gaussian_comparison',
    'verbose': True,
    'random_seed': 42,
}
```

### Step 2: Run It

Execute the cell. The pipeline will:
- Train 6 networks (2 activations × 3 widths)
- Track convergence for each
- Generate visualizations and metrics

### Step 3: Interpret Results

Check `results/gaussian_comparison/` for:

**Plots:** 
- Convergence curves showing which activation learns faster
- Approximation quality at each width
- Error distributions

**CSV:**
- One row per configuration
- Key columns: `converged`, `final_mse`, `convergence_time`

**What to look for:**
- Which activation reaches threshold first?
- Does one need fewer neurons to converge?
- Check `final_r2` — closer to 1.0 means better fit

---

## How It Works: Architecture Overview

### The Flow

```
Config → Generate Data → Train Networks → Evaluate → Visualize/Export
```

Each experiment follows this pipeline:

1. **ExperimentConfig** defines parameters (widths, activation, target function)
2. **generate_target_data** creates training samples and test grid
3. **train_network** fits each width configuration using **SingleHiddenLayerNet**
4. **evaluate_network** computes error metrics on independent test data
5. **compute_metrics** aggregates results across widths
6. **plot_experiment_results** generates comprehensive visualizations
7. **save_results** exports to CSV for analysis

### The Two Classes

**`ExperimentConfig`**
- **What it does:** Stores all experiment parameters in one place
- **Why it exists:** Ensures reproducibility and validation (catches bad configs before training)
- **Key attributes:** `neuron_counts`, `activation`, `target_function`, `epochs`, `learning_rate`, `threshold`

**`SingleHiddenLayerNet`**
- **What it does:** Implements the single-hidden-layer architecture
- **Why it exists:** Direct test of UAT's claim that width alone enables approximation
- **Key methods:** `forward()` for predictions, built with `torch.nn`

### Core Functions (Grouped by Purpose)

**Data Generation:**
- `generate_target_data` — Creates both training data and evaluation grid for chosen target function

**Training & Evaluation:**
- `train_network` — Trains one configuration, tracks loss/time/convergence
- `evaluate_network` — Tests trained model on independent data, computes MSE/MAE/R²

**Analysis:**
- `compute_metrics` — Aggregates results: minimum viable width, marginal efficiency, convergence stats

**Output:**
- `plot_experiment_results` — Full visualization panel (convergence curves, approximations, errors)
- `save_results` — Exports CSV with all metrics

---

## Customization Guide

### Change Network Widths

Testing wider networks or finer granularity?

```python
base_config = {
    'neuron_counts': [5, 25, 50, 100, 250, 500, 1000],  # More granular
    # ... rest of config
}
```

**Trade-off:** More widths = better resolution of "minimum viable width", but longer runtime.

### Adjust Training Parameters

**Network not converging?**

```python
'epochs': 5000,           # More training time
'learning_rate': 0.005,   # Slower, more stable
'threshold': 0.05,        # Accept less precision
```

**Training too slow?**

```python
'epochs': 2000,           # Fewer iterations
'n_samples': 1000,        # Less data
'neuron_counts': [10, 50, 100],  # Fewer widths
```

### Control Output

**Disable CSV export:**

In `main()` function, comment out:
```python
# save_results(all_results)
```

**Disable plots:**

In `run_single_experiment()`, comment out:
```python
# plot_experiment_results(results, x_plot, y_plot, label, config, str(save_path))
```

**Disable console output:**

In your config:
```python
'verbose': False
```

---

## Understanding Your Results

### The CSV Output

Each row represents one configuration with these metrics:

| Column | What It Means |
|--------|---------------|
| `n_neurons` | Hidden layer width tested |
| `act_function` | Activation used (relu/tanh/sigmoid) |
| `target_function` | Function being approximated |
| `converged` | Did it reach the MSE threshold? |
| `convergence_time` | Epochs to reach threshold (if converged) |
| `training_time` | Wall-clock time in seconds |
| `final_mse` | Mean squared error on test data |
| `final_mae` | Mean absolute error |
| `max_error` | Worst single prediction |
| `final_r2` | R² score (1.0 = perfect fit) |

### Key Questions to Ask

**Approximation quality:**
- Which configurations converged? (`converged == True`)
- What's the minimum width needed? (smallest `n_neurons` with `converged == True`)
- How tight is the fit? (`final_r2` close to 1.0, `final_mse` low)

**Efficiency:**
- Does doubling neurons give proportional improvement? (marginal efficiency)
- Which activation converges fastest? (compare `convergence_time`)
- Is there a plateau? (when more neurons don't help much)

### Reading the Plots

**Convergence curves (top left):**
- Each line = one width configuration
- Y-axis = MSE (log scale)
- Horizontal line = threshold
- Look for: when lines cross threshold, if they plateau

**Approximation plots (top right):**
- Dotted line = target function
- Colored lines = network predictions at different widths
- Look for: visual closeness to target, where predictions break down

**Error distribution (bottom left):**
- Shows where the network struggles most
- Large errors at domain edges? Consider boundary handling
- Clustered errors? Might need more neurons

**Summary text (bottom right):**
- Aggregated metrics across all widths
- Quick reference for minimum viable width and best performance

---

## Common Patterns

### Comparing Activations for One Function

```python
target_functions = ['sin']
activations = ['relu', 'tanh', 'sigmoid']
```

**Look for:** Which activation gives lowest `final_mse` with fewest neurons?

### Testing One Activation Across Functions

```python
target_functions = ['sin', 'gaussian', 'sin_high_freq']
activations = ['relu']
```

**Look for:** Does ReLU struggle with any function type? Check `converged` rates.

### Finding Minimum Viable Width

```python
'neuron_counts': [1, 2, 5, 10, 20, 50, 100, 200]  # Fine-grained
'threshold': 0.01  # Strict convergence
```

**Look for:** Smallest `n_neurons` where `converged == True` across multiple runs.

---

## Troubleshooting

**"Network not converging even with 1000 neurons"**
- Check if `threshold` is too strict (try 0.05 or 0.1)
- Increase `learning_rate` slightly (0.01 → 0.02)
- Verify target function is smooth enough to approximate

**"Training extremely slow"**
- Reduce `n_samples` (3000 → 1000)
- Test fewer widths first
- Use smaller `epochs` for initial tests

**"Results look unstable across runs"**
- Set consistent `random_seed`
- Increase `n_samples` for more stable gradients
- Try multiple seeds and average results

**"Plots not saving"**
- Check `save_dir` exists and is writable
- Verify `plot_experiment_results()` is not commented out
- Check console for any path-related errors

---

## What's Next

Once you're comfortable with basic experiments:

1. **Experiment with domain boundaries** (`x_min`, `x_max`) to test generalization
2. **Test extreme cases** (very few neurons, very high frequency functions)
3. **Compare against theoretical predictions** from UAT literature

Remember: This framework tests approximation capacity given "enough neurons." The interesting question is always: **how many is enough for your specific case?**

---

## Code Structure Reference

For deeper dives into implementation details, check the inline comments in the notebook. The functions are organized as:

- **Classes:** `ExperimentConfig`, `SingleHiddenLayerNet`
- **Data:** `generate_target_data`
- **Training:** `train_network`
- **Evaluation:** `evaluate_network`, `compute_metrics`
- **Visualization:** `plot_experiment_results`
- **Export:** `save_results`

Each function's docstring explains inputs, outputs, and purpose. Use this manual for the big picture, the code comments for the details.