![BeyondAI Banner for Research Projects](../BeyondAI_Banner_Research_Projects_2025.png)

# Nonlinear Classifiers: How Dataset Conditions Such As Noise And Dimensionality Affect Regularisation


## Research Question
How does the optimal capacity-controlling hyperparameter change across different nonlinear classifiers when feature dimensionality and noise are varied?

In non-linear, nonparametric models, how does the optimal level of regularisation change when dataset conditions, such as dimensions and noise, are changed? Regularisation occurs by tuning the model’s hyperparameters. The optimal settings of these hyperparameters are influenced by the dataset and its properties, for example: 

1-noise  
2-dimensionality  
3-distribution  
4-number of classes

We focused on investigating the first two options: noise and dimensionality. We did so across 3 nonparametric nonlinear ML models:
- kNN
- SVM
- Decision Trees

We concentrated on the most influential hyperparameter for each of these models, and observed how the optimal settings of these hyperparameters change when the noise changes in the make_moons synthetic dataset, and the dimensions in the make_classifications.

## Motivation

Nonparametric, nonlinear models have more freedom when drawing decision boundaries due to their minimal, intuitive underlying assumptions. However, this makes them prone to overfitting, especially as dataset conditions such as noise and dimensionality vary.

Theoretical guidelines are set to bound these modes’ behaviour; however, practically, their behaviour highly depends on specific dataset factors. That’s why our motivation was to explore the approximate trends in the behaviours of these models by changing these two dataset factors one at a time.

## Methods

### Models:
#### 1. kNN (k‑Nearest Neighbors)

Choosing k is a form of implicit regularisation: it controls the bias–variance trade‑off so the model generalises better to unseen data. Small k (e.g., k=1) yields low bias and high variance, while large k increases bias and reduces variance. Because kNN is a non‑parametric, instance‑based method, it is less interpretable than parametric models — we rely on theoretical benchmarks and empirical tuning to choose k.

Benchmarks used to frame k selection:
- Cover–Hart theorem (1‑NN): when k = 1, the asymptotic error rate of 1‑NN is at most twice the Bayes error rate. This shows 1‑NN can be effective but is noisy.
- n‑NN (k = n, the training set size): the set of nearest neighbours is the entire training set, so predictions always equal the majority class (a constant predictor), which has maximal smoothing and high bias.


scikit-learn classifier signature:

```python
class sklearn.neighbors.KNeighborsClassifier(
    n_neighbors=5, *, weights='uniform', algorithm='auto', leaf_size=30,
    p=2, metric='minkowski', metric_params=None, n_jobs=None
)
```

We focused mainly on n_neighbors as the most influential hyperparameter, but weights and the distance metric can also meaningfully affect performance depending on the dataset.

<p align="center">╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌</p>

#### 2. SVM (Support Vector Machine)

The SVM we used optimises the following objective (soft-margin C‑SVM):

$$
\min_{w,\,b,\,\xi} \\ \frac{1}{2}\|w\|^2 \+\ C\sum_{i=1}^{n}\xi_i
$$

subject to the usual constraints

$$
y_i\,(w^\top x_i + b) \ge 1 - \xi_i,\qquad \xi_i \ge 0 \quad \text{for } i=1,\dots,n.
$$

Notes:
- C is an *explicit* regularisation hyperparameter. It acts as an inverse regularisation: larger C enforces a harder margin (stronger penalty on slack variables ξ), which aims to reduce misclassification on the training set but can increase the risk of overfitting; smaller C yields a softer margin and stronger regularisation.
- It is recommended to scale features (e.g., StandardScaler) before fitting an SVM, especially when using the RBF kernel.

We used the C‑Support Vector Classification implementation from scikit‑learn. The raw class signature is:

```python
class sklearn.svm.SVC(*, C=1.0, kernel='rbf', degree=3, gamma='scale',
                     coef0=0.0, shrinking=True, probability=False,
                     tol=0.001, cache_size=200, class_weight=None,
                     verbose=False, max_iter=-1,
                     decision_function_shape='ovr', break_ties=False,
                     random_state=None)
```

In our experiments we only changed the C value; all other hyperparameters remained at their defaults (kernel='rbf', etc.).


<p align="center">╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌</p>

#### 3. Decision Trees

Decision trees partition the feature space using a sequence of splits, producing a tree of decision rules that assign labels to inputs. Deep trees can memorise training examples and overfit, while shallow trees are more biased but generalise better. The primary regularisation hyperparameter is max_depth, which limits how deep the tree can grow and therefore controls model complexity.

Key regularisation and control parameter
- max_depth: maximum depth of the tree (limits complexity; smaller → more regularisation).

Practical notes
- Decision trees are scale-invariant (no need for StandardScaler) but are sensitive to how categorical features are encoded.
- Trees are interpretable: use feature_importances_ and sklearn.tree.plot_tree to inspect learned structure.
- For robust performance, control depth and/or use pruning (ccp_alpha) and validate hyperparameters via cross-validation.

scikit-learn classifier signature:

```python
class sklearn.tree.DecisionTreeClassifier(
    criterion='gini', splitter='best', max_depth=None, min_samples_split=2,
    min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None,
    random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0,
    class_weight=None, ccp_alpha=0.0
)
```
<p align="center">╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌</p>

### Datasets

Our goal is to study how regularisation behaviour changes across dataset conditions (noise level, dimensionality, class overlap). For precise, repeatable control over these factors we use synthetic datasets from scikit‑learn: make_moons and make_classification.

Rationale:

Synthetic datasets allow controlled experiments: you can vary noise, number of samples, number of features, informative vs redundant features, and class overlap deterministically.

1) make_moons
- Description: two interleaving half-circles ("moons") — not linearly separable, useful for testing non-linear classifiers (e.g., kernel SVM, kNN, tree ensembles).
- Key parameters to vary:
  - n_samples: dataset size (controls statistical power / variance of estimates).
  - noise: standard deviation of Gaussian noise added to the coordinates (controls label/feature noise).
  - random_state: make experiments reproducible.
- Signature (scikit-learn):

```python
from sklearn.datasets import make_moons

# Create a 2D two-moons dataset with controllable noise
X, y = make_moons(n_samples=500, noise=0.2, random_state=42)
```
<p align="center">╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌</p>

2) make_classification
- Description: flexible synthetic dataset generator: control number of samples, features, informative/redundant features, class separation, label noise, and more — ideal for experiments that require changing dimensionality and signal-to-noise.
- Important parameters to control:
  - n_samples, n_features: dataset size and dimensionality.
  - n_informative: number of informative features (signal).

- Signature (scikit-learn):

```python
# signature (from sklearn.datasets)
sklearn.datasets.make_classification(
    n_samples=100, n_features=20, *, n_informative=2, n_redundant=2,
    n_repeated=0, n_classes=2, n_clusters_per_class=2, weights=None,
    flip_y=0.01, class_sep=1.0, hypercube=True, shift=0.0, scale=1.0,
    shuffle=True, random_state=None, return_X_y=True
)
```

## Implementation:

Each script implementation can vary slightly; however, overall, they do follow these procedures:

First, import the necessary libraries such as sklearn, pandas, matplotlib and numpy.
Identify a function that takes two parameters: the dataset noise/dimensions and the range of the model’s hyperparameter
Function building steps:

1. Generate the dataset
2. Divide the dataset into training (80%) and combined (20%)
3. Divide the combined dataset into validation (50%) and testing (50%)
4. Merging training and validation datasets for hyperparameter selection
5. Identifying the model and parameter
6. Performing a grid search with cross-validation to find the optimal value for each hyperparameter
7. Accuracy evaluation
The function returns the best hyperparameter value for this amount of noise/dimensions, in addition to the test_score

An array is made to hold the range of noises we’d like to test the model across, and the function is called in iterations with each noise/dimension level.
The results are saved in a dataframe, then displayed in plots for visualisation.


## Results:

* Noise pushed models toward more regularisation: larger k, smaller max_depth, lower C.
* Higher dimensions required more flexibility and were more model-dependent: k increased, trees deepened, and SVMs shifted to lower C values.
* Noise simplifies boundaries, while dimensions make them more complex.
* Synthetic data may not represent real-world feature noise or correlations, so results may not generalise to noisy or correlated features.
* We tune only one hyperparameter per model, so other settings (e.g., tree splitting rules or SVM kernel parameters) are fixed.
* Evaluation relies on accuracy alone, which may hide behaviour visible in other metrics.
### Models vs Dimensions:

### kNN vs Dimensionality

Optimal **k** follows a *U-shaped relationship* with dimensionality.

- Starts high at **D = 2** (mean *k* ≈ 17)
- Drops to a minimum around **D = 4–6** (mean *k* ≈ 7)
- Rises steadily to **k ≈ 23** at **D = 21**

The **generalization gap** remains nearly flat throughout (**≈ 0.01–0.018** across all conditions), indicating that kNN compensates for higher dimensionality by selecting larger neighborhood sizes. The model effectively adapts its capacity without accumulating additional overfitting.

#### Interpretation

- **Low dimension (D = 2):**  
  The problem is trivially simple. kNN requires many neighbors to avoid fitting noise along an almost trivial boundary.

- **Moderate dimension (D = 3–6):**  
  Added informative dimensions enrich structure, allowing smaller neighborhoods to capture decision boundaries efficiently.

- **High dimension (D ≥ 7):**  
  The **curse of dimensionality** dominates. Distances become less informative, forcing *k* to increase to maintain stable estimation.

---

#### Decision Trees vs Dimensionality

Optimal **max_depth** evolves as follows:

- ≈ **5** at **D = 2**
- Peaks at **≈ 11–12** around **D = 9–11**
- Plateaus at **≈ 8–9** for **D ≥ 12**

Unlike kNN, the **generalization gap increases substantially**:

- **0.012** at D = 2  
- **0.099** at D = 10  
- (~8× increase)

Even after depth stabilizes, the gap remains high, indicating that tuning `max_depth` alone cannot sufficiently control overfitting in higher dimensions.

#### Implication

Additional regularization is required, such as:

- `ccp_alpha` pruning  
- `min_samples_leaf`  
- `min_samples_split`

---

#### SVM vs Dimensionality (RBF Kernel)

Joint tuning of **C** and **gamma** reveals three clear regimes:

###### Regime 1 — Low Dimension (D = 2–7)
- Winning configs:
  - `C = 100, γ = 1.0`
  - `C = 10, γ = 1.0`
- **High C + High γ**
- Tight, highly local decision boundary.

###### Regime 2 — Transition Zone (D = 8–15)
- Winning config:
  - `C = 10, γ = 0.1`
- At **D = 10**, all 30 seeds agree.
- **Medium C + Medium γ**
- Represents a clean structural regime shift.

###### Regime 3 — High Dimension (D = 16–21)
- Winning config:
  - `C = 100, γ = 0.01`
- **High C + Very Low γ**
- Wide influence radius compensates for sparsity in high-dimensional space.

#### Generalization Behavior

The generalization gap increases monotonically:

- ≈ **0.0003** at D = 2  
- ≈ **0.043** at D = 21  

However, it remains substantially lower than Decision Trees, indicating stronger robustness to dimensionality when **C–γ joint tuning** is performed.

---

##### Model Comparison

| Model            | Gap at D = 2 | Gap at D = 21 | Adapts Successfully? |
|------------------|-------------|--------------|----------------------|
| kNN              | 0.004       | 0.013        | ✅ Yes — gap stays flat |
| Decision Tree    | 0.012       | 0.078        | ⚠️ Partially — gap grows 8× |
| SVM (RBF)        | 0.0003      | 0.043        | ✅ Partially — gap grows but remains low |

---

#### Key Takeaway

- **kNN** adapts by increasing neighborhood size.
- **Decision Trees** struggle without additional regularization.
- **RBF SVM** shows the strongest robustness through coordinated hyperparameter adaptation.

## Conclusion:

As noise increases, averaging out more points becomes crucial; models need stronger regularisation: kNN raises k to average more neighbours, SVM lowers C to smooth the boundary, and decision trees reduce max_depth to avoid fitting noise. When dimensionality increases, data becomes sparse, so models require more flexibility: k grows, trees deepen, and SVM often increases the value of C. These trends are helpful but not universal, as results are affected by other dataset aspects



## Future Work

Most of the fluctuations that we found in the visualisations were primarily due to the randomness of the synthetic data; that’s why we think that using real-world datasets could produce smoother trends and might show details we couldn’t see on synthetic data.
Explore how other less famous hyperparameters become influential in extreme cases, for example, the distance parameter in kNNq can become more influential in the case of higher dimensions.
Discover and experiment with other ways to assess the performance of the models, other than accuracy.


## References

1. Braga‑Neto, U. (2024). *Nonparametric Classification*. In Fundamentals of Pattern Recognition and Machine Learning (pp. 89-108). Cham: Springer International Publishing. [Springer PDF](https://link.springer.com/content/pdf/10.1007/978-3-031-60950-3_5.pdf)

2. Cornell University. (2018). *Lecture 2: K‑Nearest Neighbors*. CS4780/5780 — Machine Learning. [Lecture notes](https://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/lecturenote02_kNN.html)

3. Fletcher, T. (2009). *Support Vector Machines Explained*. Tutorial paper, 1118, 1-19. [PDF](https://www.csd.uwo.ca/~xling/cs860/papers/SVM_Explained.pdf)



> The research poster for this project can be found in the [BeyondAI Proceedings 2025](https://thinkingbeyond.education/beyondai_proceedings_2025/).

