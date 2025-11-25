# A Qualitative Study of CNN Optimisers with Weight Decay  
**BeyondAI Research Project • 2025**

*1D linear interpolation in loss landscape for 10 different optimisers (20 epochs on Human Faces Dataset). Blue = Train Loss, Red = Test Loss.*

## Research Question
**How does the introduction of weight decay (L2 regularisation) affect the linearity of the loss barrier between the initialisation and the converged solution across modern CNN optimisers?**  

## Motivation
- Classic works (Garipov et al., 2018) showed SGD solutions often lie in the same wide valley.
- Modern adaptive optimisers (Adam, RMSProp, AdamW, NAdam, etc.) are known to find solutions that are **not linearly connected**.
- Weight decay is ubiquitous, yet its **geometric effect** on the loss landscape is under-explored beyond test accuracy.
- A simple, visual, side-by-side comparison of 10 optimisers on the same task reveals intuitive patterns that tables of numbers cannot.

## Method & Implementation

### Dataset
Human Faces Dataset (Kaggle) – ~10k real vs AI-generated face images (binary classification).

### Model
Lightweight 3-block CNN:  
`3→32 → 64 → 128` channels → ReLU + MaxPool → 512-unit FC + Dropout(0.5) → 2 classes.

### Training (`CNN_train.py`)
- Fully deterministic
- 20 epochs, LR = 0.01, batch size 128, no scheduler
- For each of the **10 optimisers**, train **two models** from the **exact same initial weights**:
  - Without weight decay
  - With weight decay = 0.01
- Save everything (weights, dataset split indices, hyperparams) into `trained_models.pth`

### Interpolation & Visualisation (`CNN_inference.py`)
- Loads the saved bundle
- Performs **three 1D interpolations** per optimiser:
  1. Initial → Trained (no WD)
  2. Trained (no WD) → Trained (with WD)
  3. Initial → Trained (with WD)
- Custom α-ranges per optimiser
- Evaluates exact train & test loss at 100 points
- Generates a clean **10×3 grid plot** with automatic log-scale when loss > 10

### Optimisers Tested
| Optimiser     | Decoupled WD Variant? |
|---------------|-----------------------|
| SGD           | –                     |
| SGD+Momentum  | –                     |
| RMSProp       | –                     |
| Adagrad       | –                     |
| Adadelta      | –                     |
| Adam          | –                     |
| **AdamW**     | Yes                   |
| NAdam         | –                     |
| **NAdamW**    | Yes                   |
| Adamax        | –                     |
