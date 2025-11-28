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

