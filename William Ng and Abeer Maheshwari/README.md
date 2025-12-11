# A Qualitative Study of CNN Optimisers with Weight Decay  
**BeyondAI Research Project 9B • 2025**

*1D linear interpolation in loss landscape for 10 different optimisers (20 epochs on Human Faces Dataset). Blue = Train Loss, Red = Test Loss.*

## Research Question
**How does the introduction of weight decay (L2 regularisation) affect the linearity of the loss barrier between the initialisation and the converged solution across modern CNN optimisers?**  

## Overview

Optimisation plays a crucial role in neural network training tasks. We tested ten different gradient-based optimisers: SGD, SGD Momentum, Adagrad, Adadelta, RMSProp, Adam, NAdam, Adamax, and the decoupled weight decay variants of Adam and NAdam, AdamW and NAdamW respectively. For visualisation of the loss landscape, we used the ‘1D Linear Interpolation’ approach that qualitatively compares how the different optimisers behave locally. We also studied the effects of decoupled weight decay on the optimsiers AdamW and NAdamW, and compared them to their coupled weight decay equivalents. Our study was performed on a binary face classification task for which we train a small CNN twice with each optimiser (one with weight decay, one without). We use the Human Faces Dataset publicly available on Kaggle that consists of 5000 real human images and 5000 AI generated images to train our CNN.

## Motivation
- Classic works (Garipov et al., 2018) showed SGD solutions often lie in the same wide valley.
- Modern adaptive optimisers (Adam, RMSProp, AdamW, NAdam, etc.) are known to find solutions that are **not linearly connected**.
- Weight decay is ubiquitous, yet its **geometric effect** on the loss landscape is under-explored beyond test accuracy.
- A simple, visual, side-by-side comparison of 10 optimisers on the same task reveals intuitive patterns that tables of numbers cannot.

## Method & Implementation

### Dataset
Human Faces Dataset (Kaggle) – 5k real and 5k AI-generated face images (binary image classification).

### Model
Lightweight 3-block CNN:  
`3→32 → 64 → 128` channels → ReLU + MaxPool → 512-unit FC + Dropout(0.5) → 2 classes.

### Training (`CNN_train.py`)
- Fully deterministic
- 20 epochs, LR = 0.01, batch size 256, no scheduler
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

## Future Work

1. Repeat this experiment for neural network architechtures of a higher complexity and see if the results still hold or not.
2. Expand on the visualisation techniques, from a 1D linear interpolation to using a 2D contour plot, or filter-wise normalisation.

### Related Work
1. Poster for the BeyondAI 2025 Fair: https://www.canva.com/design/DAG5KQh2MVg/sFYujYUvKuYGZySXvsTSmA/view?utm_content=DAG5KQh2MVg&utm_campaign=designshare&utm_medium=link2&utm_source=uniquelinks&utlId=h1549a82cdf
2. Paper/ Observation Report: https://www.overleaf.com/read/kcpgftbqvrqg#08ad11
3. Google Colab code: https://colab.research.google.com/drive/1v97lhrarRbJnyGTKZrb56wZxL3doER2c?usp=sharing


## References

[1] Ruder, S. (2016). An overview of gradient descent optimization algorithms. *arXiv preprint arXiv:1609.04747*.

[2] Li, H., Xu, Z., Taylor, G., Studer, C., & Goldstein, T. (2018). Visualizing the loss landscape of neural nets. *Advances in neural information processing systems, 31*.

[3] Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization. *arXiv preprint arXiv:1412.6980*.

[4] Loshchilov, I., & Hutter, F. (2017). Decoupled weight decay regularization. *arXiv preprint arXiv:1711.05101*.

[5] Goodfellow, I. J., Vinyals, O., & Saxe, A. M. (2014). Qualitatively characterizing neural network optimization problems. *arXiv preprint arXiv:1412.6544*.

[6] Keskar, N. S., Mudigere, D., Nocedal, J., Smelyanskiy, M., & Tang, P. T. P. (2016). On large-batch training for deep learning: Generalization gap and sharp minima. *arXiv preprint arXiv:1609.04836*.

[7] Kaustubh D. Human Faces Dataset. https://www.kaggle.com/datasets/kaustubhdhote/human-faces-dataset/data
