# Neural ODE for EV Charging Load Forecasting

## Overview

This project evaluates **Neural Ordinary Differential Equations (Neural ODEs)** for forecasting aggregate hourly charging load (AHCL) from real-world EV charging session data. The study systematically compares multiple ODE solvers (Euler, RK4, Dormand-Prince, Adams-Bashforth) across two aggregation strategies (day-of-week and seasonal) to understand how integrator choice affects training stability and prediction accuracy.

**Research Question:**  
How does the performance of a continuous-time Neural ODE model compare to established temporal deep learning benchmarks (e.g., LSTM) in accurately forecasting AHCL for residential locations, considering metrics like RMSE, MAE, and R²?

## Dataset

- **Source:** `Dataset1_charging_reports.csv` — large-scale Norwegian EV charging dataset
- **Scope:** 6,371 charging sessions from location "ASK" (Nov 2018 – Feb 2020)
- **Features per session:** plugin_time, plugout_time, energy_session
- **Target:** Aggregated hourly charging load (kWh/h)

## Project Structure

```
BeyondAI/
├── NODE.py                          # Original single-run Neural ODE pipeline
├── NODE_2.py                        # Enhanced pipeline with multi-solver comparison
├── Dataset1_charging_reports.csv    # Raw session-level data
├── outputs/
│   └── new_perf/
│       └── solver_comparison/       # Results from solver experiments
│           ├── dayofweek/           # Day-of-week aggregation per solver
│           │   ├── euler/, rk4/, dopri5/, adams/
│           │   ├── dayofweek_preds_comparison.png
│           │   └── dayofweek_rmse_comparison.png
│           ├── season/              # Seasonal aggregation per solver
│           │   ├── euler/, rk4/, dopri5/, adams/
│           │   ├── season_preds_comparison.png
│           │   └── season_rmse_comparison.png
│           └── solver_metrics_summary.csv
├── Papers/                          # Reference papers
└── README.md                        # This file
```

## Aggregation Strategies

### Day-of-Week Aggregation
- **Concept:** Group all charging sessions by weekday (Mon–Sun) and compute an average 24-hour AHCL profile per weekday.
- **Energy distribution:** For each session, distribute its energy proportionally across overlapping hourly buckets.
- **Result:** 7 × 24 = **168-hour sequence** (one per weekday)
- **Captures:** Weekday vs. weekend patterns, daily rush hours, behavioral differences across days.
- **Use case:** When weekday/weekend variation dominates; longer training sequence enables richer pattern learning.

**Example:**  
A session running 14:30–15:45 with 10 kWh on Monday:
- Hour 14 overlap: 30 min → 10 × (1800s / 4500s) = 4 kWh
- Hour 15 overlap: 45 min → 10 × (2700s / 4500s) = 6 kWh
- Aggregated across all Monday sessions → Monday's average profile.

### Seasonal Aggregation
- **Concept:** Group sessions by season (Winter, Spring, Summer, Fall) and compute an average 24-hour AHCL profile per season.
- **Energy distribution:** Same proportional overlap method as day-of-week.
- **Result:** 4 × 24 = **96-hour sequence** (one per season)
- **Captures:** Climate/seasonal effects on charging behavior (e.g., winter heating vs. summer cooling).
- **Use case:** When seasonal variation is dominant; compact representation for smoother dynamics.

## Neural ODE Model

### Architecture
- **ODE Function:** `dy/dt = f_θ(y, u)` where:
  - `y` = current hourly load (scalar)
  - `u` = time features: `[sin(hour), cos(hour), day/season_index]`
  - `f_θ` = small MLP (1+3 inputs → 32 hidden → 16 hidden → 1 output)
- **Learnable Parameters:**
  - Initial state: `y0` (parameter)
  - Network weights in `f_θ`
- **Integration:** Fixed time steps (dt=1.0 hour) with switchable solver

### Training Pipeline
1. **Normalize** targets to [0, 1] using min–max scaling.
2. **Train** the Neural ODE with MSE loss on normalized predictions.
3. **Denormalize** predictions back to original kWh/h units.
4. **Evaluate** using RMSE, MAE, and R² metrics.

### Solvers Implemented

| Solver | Type | Substeps | Stability | Notes |
|--------|------|----------|-----------|-------|
| **Euler** | Explicit 1st-order | 1 | ⚠️ Low | Baseline; prone to instability with large steps/gradients |
| **RK4** | Explicit 4th-order | 4 | ✓ Good | Runge-Kutta 4; robust for moderate stiffness |
| **Dopri5** | Explicit 5th-order (fixed) | 6 | ✓ Good | Dormand-Prince approximation; high-order accuracy |
| **Adams-Bashforth** | Explicit multistep | adaptive | ⚠️ Variable | Order 2–3; can diverge without careful tuning |

## Key Results (Smoke Test, EPOCHS=200)

### Day-of-Week Aggregation (168h sequence)
| Solver | RMSE (kWh/h) | MAE (kWh/h) | R² |
|--------|--------------|-------------|----|
| Euler | 2662.24 | 2572.37 | -342.10 ❌ (diverged) |
| RK4 | 197.26 | 140.63 | -0.88 |
| **Dopri5** | **78.83** | **64.83** | **0.699** ✓ |
| **Adams** | **73.19** | **57.86** | **0.741** ✓ |

### Seasonal Aggregation (96h sequence)
| Solver | RMSE (kWh/h) | MAE (kWh/h) | R² |
|--------|--------------|-------------|----|
| **Euler** | **178.28** | **146.75** | **0.716** ✓ |
| RK4 | 208.70 | 159.54 | 0.611 |
| Dopri5 | 201.51 | 135.94 | 0.638 |
| Adams | 494.85 | 359.76 | -1.186 ❌ (diverged) |

### Key Observations
- **Solver sensitivity:** Integrator choice significantly affects training stability and final accuracy.
- **Aggregation effect:** Day-of-week sequence requires stable solvers (Dopri5, Adams); seasonal sequence is more forgiving (Euler performs well).
- **Stability issues:** Euler diverges on day-of-week; Adams diverges on seasonal. Higher-order methods (RK4, Dopri5) are more reliable.
- **Best performers:** Dopri5 + Adams on day-of-week (R² ~ 0.70–0.74); Euler on seasonal (R² ~ 0.72).

## Usage

### Quick Test (200 epochs)
```bash
$env:EPOCHS=200
C:/Users/equbi/anaconda3/envs/dev/python.exe .\NODE_2.py
```

### Full Experiments (2000 epochs)
```bash
$env:EPOCHS=2000
C:/Users/equbi/anaconda3/envs/dev/python.exe .\NODE_2.py
```

### Run Original Single Pipeline
```bash
C:/Users/equbi/anaconda3/envs/dev/python.exe .\NODE.py
```

### Outputs
- Per-solver metrics, plots, and model checkpoints saved under:
  - `outputs/new_perf/solver_comparison/dayofweek/{euler,rk4,dopri5,adams}/`
  - `outputs/new_perf/solver_comparison/season/{euler,rk4,dopri5,adams}/`
- Comparison plots (predictions overlay, RMSE bar charts)
- Summary CSV: `outputs/new_perf/solver_comparison/solver_metrics_summary.csv`

## Code Organization

### Main Functions

**Data Loading & Aggregation:**
- `load_real_data(csv_path)` — load and preprocess session data
- `aggregate_by_day_of_week(df_sessions)` — compute weekday profiles
- `aggregate_by_season(df_sessions)` — compute seasonal profiles
- `load_and_prepare_data()` — orchestrate loading and day-of-week aggregation

**Feature Engineering:**
- `build_features()` — create sin/cos hour + day index features

**Model Definition:**
- `ODEFunc(nn.Module)` — MLP computing dy/dt
- `NeuralODEHourly(nn.Module)` — Neural ODE wrapper with learnable y0 and integrator dispatch

**Integrators:**
- `euler_integrate()` — explicit Euler
- `rk4_integrate()` — Runge-Kutta 4
- `dopri5_integrate()` — Dormand-Prince approximation
- `adams_bashforth_integrate()` — Adams-Bashforth multistep

**Training & Evaluation:**
- `train_model()` — training loop with solver selection
- `run_training_pipeline()` — normalized train/denormalize/evaluate pipeline
- `run_solver_comparison()` — orchestrate multi-solver experiments
- `compute_and_save_metrics()` — RMSE, MAE, R², save results
- `plot_simple_results()` — visualization of predictions & loss
- `predict_and_save_samples()` — sample predictions CSV

**Main Entry:**
- `main()` — load data, prepare features, run day-of-week + seasonal comparisons

## Limitations & Future Work

### Current Limitations
1. **No LSTM/RNN baselines** — Neural ODE comparison incomplete; need discrete RNN benchmarks.
2. **Single data split** — no train/val/test separation; all metrics on training data.
3. **No out-of-sample evaluation** — no held-out weeks or cross-validation.
4. **Aggregation loses variability** — models fit averaged profiles, not per-day forecasting.
5. **Fixed-step integrators** — Dopri5 implemented as fixed-step composite; no true adaptive stepping.
6. **No hyperparameter tuning per solver** — same lr=5e-3, epochs for all.
7. **Limited uncertainty quantification** — point predictions only.

### Recommended Future Research

1. **Add RNN Baselines**
   - Implement LSTM/GRU with comparable architecture.
   - Train on train/val/test splits with early stopping.
   - Compare RMSE/MAE/R² on held-out test sets.

2. **Proper Evaluation Protocol**
   - Implement train/val/test splits by date (e.g., 60%/20%/20%).
   - Use K-fold cross-validation or walk-forward validation.
   - Report metrics on test set only; perform statistical significance tests (paired t-test).

3. **Solver & Hyperparameter Tuning**
   - Grid search: learning rate (1e-4, 5e-4, 1e-3, 5e-3), gradient clipping, RK4 substeps (2–8), Adams order (2–3).
   - Monitor training stability; add early stopping and weight decay if needed.

4. **Irregular Sampling Experiments**
   - Simulate irregular time gaps (remove random hours, merge hours).
   - Compare Neural ODE on irregular timestamps vs. LSTM on imputed sequences.
   - Test whether Neural ODEs truly handle irregularity better.

5. **Advanced Integration Methods**
   - Use `torchdiffeq` library for adjoint-based backprop and true adaptive stepping.
   - Compare computational cost vs. accuracy of adaptive methods.

6. **Feature Engineering**
   - Add external covariates: holidays, weather (temperature, cloud cover), price signals.
   - Use one-hot encoding for categorical features (day, season).

7. **Uncertainty Quantification**
   - Ensemble forecasts with different initializations.
   - Bayesian Neural ODE for credible intervals.
   - MC Dropout for predictive uncertainty.

8. **Ablation Studies**
   - Effect of learnable y0 vs. fixed initialization.
   - Impact of different feature sets.
   - Sensitivity to network capacity and dt.

## Environment Setup

**Python 3.8+, PyTorch 1.9+, NumPy, Pandas, Matplotlib**

```bash
conda create -n dev python=3.9
conda activate dev
pip install torch torchvision torchaudio
pip install pandas numpy matplotlib
```

## References

- Chen, R. T., Rubanova, Y., Bettencourt, J., & Duvenaud, D. K. (2018). Neural ordinary differential equations. NeurIPS.
- Dormand, J., & Prince, P. (1980). A family of embedded Runge-Kutta formulae. Journal of Computational and Applied Mathematics.
- Hairer, E., Nørsett, S. P., & Wanner, G. (1993). Solving ordinary differential equations I: Nonstiff problems.

## Author Notes

This pipeline demonstrates:
- Practical Neural ODE implementation with multiple integrators.
- Systematic comparison of solvers on real data.
- Trade-offs between sequence length, aggregation strategy, and solver stability.
- Path toward rigorous comparison with RNN baselines for continuous-time forecasting.

For questions or contributions, refer to the code comments and docstrings in `NODE_2.py`.

---

**Last Updated:** November 2025  
**Status:** Research prototype; ready for extended experiments with RNN baselines and full hyperparameter tuning.
