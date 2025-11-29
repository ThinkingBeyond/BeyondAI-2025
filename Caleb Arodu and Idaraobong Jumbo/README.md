![BeyondAI Banner for Research Projects](../BeyondAI_Banner_Research_Projects_2025.png)

# The Impact of Class Imbalance on Non-linear Classifiers: Evaluating Performance and Mitigation Strategies

## Research Question

How does class imbalance affect non-linear classifiers, which evaluation metrics best capture this impact, and which mitigation strategies most effectively improve their performance?


## Motivation

Class imbalance is a persistent challenge in many machine-learning tasks where one class appears far less frequently than the other. This imbalance can mislead models into focusing heavily on the majority class, causing them to miss rare but important minority cases. As a result, models may report high accuracy while performing poorly on the class that matters most, revealing the limitations of traditional evaluation approaches.

Understanding which metrics truly reflect model performance under imbalance is essential. Metrics such as precision, recall, and F1-score provide insight into minority-class detection and help expose weaknesses that accuracy hides. Analyzing these metrics helps clarify how non-linear classifiers behave when trained on skewed data.

There is also a need to determine which mitigation techniques effectively address this issue. Methods such as random undersampling, SMOTE oversampling, and cost-sensitive learning offer different trade-offs in performance. Comparing these approaches across non-linear models allows us to identify strategies that improve recall without greatly increasing false positives, helping build more reliable and balanced classifiers.

## Methodology

### Dataset

For this project, we used the [Credit Card Fraud Detection dataset from Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) . The dataset contains transactions made by European credit cardholders over two days in September 2013. It is highly imbalanced, with only 492 fraudulent transactions out of 284,807 total transactions (approximately 0.172% of all transactions).

All input features are numerical and result from a PCA transformation due to confidentiality reasons. The dataset includes 28 principal components, labeled V1, V2, …, V28, as well as 'Time' and 'Amount', which were not transformed. The 'Time' feature represents the seconds elapsed between each transaction and the first transaction in the dataset, while 'Amount' is the transaction value. The response variable is 'Class', where 1 indicates a fraudulent transaction and 0 indicates a legitimate one.

### Data Preprocessing

Before training the models, the following preprocessing steps were applied to prepare the dataset:
- The 'Amount' feature was scaled using a Standard Scaler to ensure zero mean and unit variance.
- The 'Time' feature was dropped, as it was not relevant to the task at hand.
- Any missing values were identified and handled appropriately.
- The dataset was split into training and testing sets using a standard train-test split. A copy of the original test set was preserved to evaluate all models consistently across different mitigation strategies.
- No additional feature selection was performed; all remaining features were used in model training.

### Models Trained

We trained three non-linear classifiers on the dataset: Support Vector Machines (SVM), Decision Trees, and Random Forests. All models were used with their default hyperparameters.

To address class imbalance, we evaluated each model across multiple dataset scenarios:

- Original imbalanced dataset
- Random undersampled dataset
- SMOTE oversampled dataset
- Cost-weighted original dataset (class_weight='balanced' or 'balanced_subsample' for Random Forest)
- Cost-weighted random undersampled dataset
- Cost-weighted SMOTE oversampled dataset

For each scenario, we measured model performance using accuracy, precision, recall, and F1-score to capture both overall performance and minority-class detection effectiveness.

### Evaluation Metrics

To assess model performance, we used accuracy, precision, recall, and F1-score. Each metric provides different insight into how the models handle imbalanced data:

1. Accuracy – measures overall correctness, but can be misleading in imbalanced datasets because predicting the majority class (non-fraud) correctly dominates the score.
    - In fraud detection: high accuracy might just mean the model is correctly predicting legitimate transactions while missing most frauds.

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

2. Precision – the proportion of predicted frauds that are actually frauds, showing how well the model avoids false alarms.
    - In fraud detection: high precision means that when the model flags a transaction as fraud, it is likely correct, reducing unnecessary alerts.

$$\text{Precision} = \frac{TP}{TP + FP}$$

3. Recall – the proportion of actual frauds correctly identified, highlighting the model’s ability to detect minority-class instances.
    - In fraud detection: high recall means the model catches most fraudulent transactions, minimizing losses from undetected fraud.

$$\text{Recall} = \frac{TP}{TP + FN}$$

4. F1-score – the harmonic mean of precision and recall, providing a balanced metric when both false positives and false negatives matter.
    - In fraud detection: F1 balances catching as many frauds as possible (recall) while avoiding too many false alarms (precision).

$$F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

TP = True Positives (frauds correctly detected)

TN = True Negatives (legitimate transactions correctly identified)

FP = False Positives (legitimate transactions flagged as fraud)

FN = False Negatives (frauds missed by the model)

These metrics together allow us to evaluate both overall performance and the ability to detect rare fraud cases, which is crucial in highly imbalanced datasets

## Mitigation Strategies

To address the severe class imbalance in the dataset, we applied three different strategies and evaluated how each affected model performance.

### Random Undersampling (RUS)

Random undersampling reduces the size of the majority class by randomly removing samples until both classes are closer in proportion.

Advantage: Simplifies the dataset and speeds up training.

Limitation: Discards useful data, which may cause loss of important decision-boundary information.

Why it matters: With fewer legitimate transactions, models become more sensitive to the fraud class, often increasing recall but reducing precision.

### SMOTE (Synthetic Minority Oversampling Technique)

SMOTE generates synthetic minority-class samples by interpolating between existing minority samples and their nearest neighbors.

Formula:

$$x_{\text{new}} = x_i + \lambda (x_{\text{nn}} - x_i), \quad \lambda \in [0, 1]$$

Where:

$x_{\text{new}}$ — the newly generated synthetic sample

$x_i$ — the original minority class sample

$x_{\text{nn}}$ — the selected nearest neighbor of $x_i$

$\lambda$ — a random value drawn uniformly from 0 to 1

The formula generates a new sample along the line between $x_i$ and $x_{\text{nn}}$.

Advantage: Adds new minority samples without duplication, improving model exposure to fraud cases.

Limitation: Can introduce noise and create borderline samples that increase false positives.

Why it matters: Helps the model detect more frauds (higher recall) but may increase the number of incorrect fraud predictions (lower precision).

### Cost-Weighted Learning (Class Weighting)

We applied class weighting using class_weight='balanced' in SVM and Decision Trees, and class_weight='balanced_subsample' in Random Forest.
The minority class is assigned a higher penalty during training, forcing the model to prioritize detecting fraud.

Advantage: No change to dataset size; works directly inside the algorithm.

Limitation: Can make decision boundaries more aggressive, increasing false positives.

Why it matters: Helps the model be more sensitive to fraud, improving recall while balancing precision.

We also used ensembles of cost-weighing with the undersampled and oversampling using SMOTE

## Results

### Precision Across All Strategies
| Classifier        | Imbalanced | Undersampled | SMOTE | Cost-Weighting | US + CW | SMOTE + CW |
| ----------------- | ---------- | ------------ | ----- | -------------- | ------- | ---------- |
| **SVM**           | 0.909      | 0.083        | 0.090 | 0.281          | 0.083   | 0.090      |
| **Decision Tree** | 0.744      | 0.017        | 0.369 | 0.720          | 0.018   | 0.369      |
| **Random Forest** | 0.922      | 0.068        | 0.831 | 0.920          | 0.056   | 0.841      |

- The imbalanced dataset had the best precision across all classifiers due to the model's overwhelming bias toward the Majority Class.
- It was also observed that Random Forest had the best precision across all strategies.

### Recall Across All Strategies
| Classifier        | Imbalanced | Undersampled | SMOTE | Cost-Weighting | US + CW | SMOTE + CW |
| ----------------- | ---------- | ------------ | ----- | -------------- | ------- | ---------- |
| **SVM**           | 0.631      | 0.884        | 0.832 | 0.706          | 0.884   | 0.832      |
| **Decision Tree** | 0.706      | 0.916        | 0.726 | 0.621          | 0.895   | 0.739      |
| **Random Forest** | 0.737      | 0.895        | 0.779 | 0.716          | 0.895   | 0.779      |

-  The imbalanced dataset had the worst recall across all classifiers.
-  Random undersampling had the best recall at the expense of unacceptable precision which made it not proper for real world use.
-  Random Forest still had the best recall across all strategies it is an ensemble of many independent Decision Trees which provides superior structural resilience against the extreme bias caused by data imbalance.

### F1-Score Across All Strategies
| Classifier        | Imbalanced | Undersampled | SMOTE | Cost-Weighting | US + CW | SMOTE + CW |
| ----------------- | ---------- | ------------ | ----- | -------------- | ------- | ---------- |
| **SVM**           | 0.745      | 0.151        | 0.163 | 0.402          | 0.151   | 0.163      |
| **Decision Tree** | 0.717      | 0.032        | 0.500 | 0.666          | 0.035   | 0.507      |
| **Random Forest** | 0.819      | 0.116        | 0.809 | 0.804          | 0.106   | 0.808      |

- It was seen that the imbalanced dataset had the best F1-Score but isn't fit for use because of the unacceptable recall.
- The imbalanced dataset + Cost-Weighing had the second best F1-Score.

## Conclusion

Class imbalance significantly affects classifier behavior as trained on imbalanced data achieve high precision but likely miss many minority-class instances. 

A high F1-Score demonstrates that your model has found the optimal balance as it detects a large percentage of the minority class (high Recall) while ensuring that those detections are reliable enough to be useful in a real-world application (high Precision). It is the single best metric for summarizing the performance of a binary classifier on a highly skewed dataset. In our case, achieving a very high F1-Score(> 0.9) was not achieved due to tradeoffs.

We also found out that you can't increase the recall of a model without decreasing the precision which confirms the presicion-recall tradeoff which in turn led to us not having one strategy that had very high recall and precision as shown in the F1-Score table.

It was seen that while tree-based models handle SMOTE and cost-weighting well, SVM shows a sharp drop in precision with oversampling or undersampling.

Cost-Weighing on the imbalanced dataset and Cost-Weighing with SMOTE had their tradeoffs, as the former had better precision and the latter had better recall. Therefore for minority class detection, SMOTE + Cost-Weighing is advised.

Among SVM, Decision Tree, and Random Forest, Random Forest is the most robust and best overall classifier for imbalanced dataset as it achieves the best balance, delivering the highest overall F1-Score across all strategies.

## Limitations

- Dataset Constraints: Limited to a specific dataset, may not generalize to other domains or fraud types.

- Synthetic Oversampling Issues: SMOTE may create unrealistic minority samples, increasing false positives.

- Model Selection: Only SVM, Decision Tree, and Random Forest were tested; results may differ for other models.

- Evaluation & Deployment: Focused mainly on F1-score, precision, and recall; limited feature engineering and no real-world testing, so performance may vary in dynamic environments.

## Future Work

- Advanced Oversampling: Explore techniques like Borderline-SMOTE, ADASYN, or GAN-based synthetic data generation to improve minority-class detection.

- Hybrid Sampling & Ensemble Methods: Combine undersampling, oversampling, and cost-weighting in smarter ways, or use ensembles on multiple balanced datasets.

- Model Expansion: Test gradient boosting models (XGBoost, LightGBM) or neural networks with class-weighted loss functions for better imbalance handling.

- Feature Engineering: Investigate feature selection, dimensionality reduction, and feature importance to boost minority-class prediction.

- Real-World Validation: Apply models to live or evolving datasets, monitor performance over time, and consider metrics beyond F1-score for robust evaluation.
  

## References

1. Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: synthetic minority over-sampling technique. Journal of artificial intelligence research, 16, 321-357. [JAIR](https://www.jair.org/index.php/jair/article/view/10302/24590)
2. Abd Elrahman, S. M., & Abraham, A. (2013). A review of class imbalance problem. Journal of Network and Innovative Computing, 1, 9-9. [Download](https://cspub-jnic.org/index.php/jnic/article/download/42/33)
3. Picek, S., Heuser, A., Jovic, A., Bhasin, S., & Regazzoni, F. (2019). The curse of class imbalance and conflicting metrics with machine learning for side-channel evaluations. IACR Transactions on Cryptographic Hardware and Embedded Systems, 209-237. [IACR](https://moving-the-social.ub.rub.de/index.php/TCHES/article/view/7339)


---

> The research poster for this project can be found in the [BeyondAI Proceedings 2025](https://thinkingbeyond.education/beyondai_proceedings_2025/).

