![BeyondAI Banner for Research Projects](../BeyondAI_Banner_Research_Projects_2025.png)

# The Impact of Class Imbalance on Non-linear Classifiers: Evaluating Performance and Mitigation Strategies

***Provide a description of your project including*** 

1. motivating your research question
2. stating your research question
3. explaining your method and implementation
4. Briefly mention and discuss your results
5. Draw your conclusions
6. State what future investigations could be conducted
7. State your references 

### Further Guidance: Formating
- Structure this readme using subsections
- Your job is to 
    - keep it clear
    - provide sufficient detail, so what you did is understandable to the reader. This way other researchers and future cohorts of BeyondAI will be able to build on your research
    - List all your references at the end
- utilise markdown like *italics*, **bold**, numbered and unnumbered lists to make your document easier to read
- if you refer to links use the respective markdown for links, e.g. `[ThinkingBeyond](https://thinkingbeyond.education/)`
- If you have graphs and pictures you want to embed in your file use `![name](your_graphic.png)`
- If you want to present your results in a table use
    | Header 1            | Header 2  |
    |---------------------|-----------|
    | Lorem Ipsum         | 12345     |

**Tip:** Use tools to create markdown tables. For example, Obsidian has a table plugin, that makes creating tables much easier than doing it by hand.

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

## Your next subsection

Continue working through the points listed above with the help of sensibly named subsections. 

If you want to see some good examples of README files check out:
- [Example 1](https://github.com/ThinkingBeyond/BeyondAI-2024/blob/main/warenya-loulia/README.md)
- [Example 2](https://github.com/ThinkingBeyond/BeyondAI-2024/blob/main/shaana-karuna/README.md)

[ ... ]

## Future Work

State and explain what follow-up research could be conducted based on your work.

## References

List all your references here. Remember to put links into markdown. For example:

1.  Einstein, A. (1905). *On the Electrodynamics of Moving Bodies*. Annalen der Physik, 17, 891-921. [Internet Archive](https://archive.org/details/einstein-1905-relativity)

**Tip**: *If you have you references in BibTex, Google Scholar or Zotero*
1. Create/copy a list into ChatGPT
2. Ask it to turn it into an unsorted list in markdown

---

> The research poster for this project can be found in the [BeyondAI Proceedings 2025](https://thinkingbeyond.education/beyondai_proceedings_2025/).

