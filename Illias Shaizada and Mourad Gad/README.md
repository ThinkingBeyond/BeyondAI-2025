![BeyondAI Banner for Research Projects](../BeyondAI_Banner_Research_Projects_2025.png)

# PCA & Elastic Net VS Double Descent
<!--
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
-->
![Double Descent Animation](double_descent_animation.gif)

## Overview

This project examines the impact of Principal Component Analysis (PCA) and Elastic Net regularization on the double descent phenomenon in high-dimensional polynomial regression. Double descent appears when model complexity approaches the interpolation threshold, causing test error to spike before decreasing again in the overparameterized regime.

We evaluate how PCA (with different variance-retention thresholds) and Elastic Net (with varying L1 ratios) reshape or suppress this behavior in polynomial regression trained on a controlled synthetic dataset.

- ### Double Descent
  A phenomenon in machine learning where test error first decreases, then increases near the interpolation threshold, and finally decreases again as model complexity continues to grow, showing a non-monotonic relationship between complexity and generalization.
  
- ### PCA (principal component analysis)
  A dimensionality reduction technique that transforms correlated features into a smaller set of uncorrelated variables (principal components) while retaining most of the original data’s variance. It helps simplify models and reduce noise.
  
- ### Elastic Net
  A regularization method that penalizes model combining both L1 (Lasso) and L2 (Ridge) penalties. It performs better on datasets with correlated features and it is considered more robust than using either methods alone.
  
## Research Question

### How do PCA and Elastic Net regularization affect the emergence, severity, and shape of the double descent curve in high-dimensional polynomial regression models?
We study whether these techniques suppress the interpolation spike, smooth the curve, or fundamentally alter model behavior.

## Motivation

Understanding the double descent phenomenon is crucial for developing robust machine learning models, particularly in high-dimensional settings where overparameterization is common, making it a compelling research topic. By investigating the effects of PCA and Elastic Net regularization, we aim to uncover strategies that mitigate excessive test error near the interpolation threshold. This knowledge can help practitioners design models that are both expressive and generalizable, especially when working with noisy or highly correlated features.

## Methodology
### Dataset
- a synthetic dataset generated
    - 100 samples
    - train/test split 80/20
    - input feature is $X \sim U [-1,\,1]$
      
$$
y = \sin(2\pi X) + \epsilon,\quad \epsilon \sim \mathcal{N}(0,\,1)
$$

### Model
- a polynomial regression model with degrees 1 -> 129 , fitted using Ridge regression with minimal regularization.
### Techniques
- PCA (Principal Component Analysis)
    - applied to the polynomial feature matrix to reduce the number of features based on the retained variance
    - Variance: 95% -> 100% 
- Elastic Net
    - applied to polynomial regression to regularize the model and reduce the noise
    - l1_ratio: 0.2-0.9
- Evaluation Metric: MSE ( mean square error )

 $$
\mathrm{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$
### Program Language : PYTHON => 3.12.12
### Libraries 
- NumPy => 2.0.2
- Scikit-Learn => 3.10.0
- Matplotlib => 1.6.1
<!--
## Your next subsection

Continue working through the points listed above with the help of sensibly named subsections. 

If you want to see some good examples of README files check out:
- [Example 1](https://github.com/ThinkingBeyond/BeyondAI-2024/blob/main/warenya-loulia/README.md)
- [Example 2](https://github.com/ThinkingBeyond/BeyondAI-2024/blob/main/shaana-karuna/README.md)

[ ... ]
-->
## Future Work
### Future investigations could expand in several directions:

1- Real-world datasets – Apply the PCA and Elastic Net analysis to real datasets with higher dimensionality or more complex feature correlations to validate findings beyond synthetic data.

2- Alternative dimensionality reduction techniques – Compare PCA with other methods to see how nonlinear feature reduction affects double descent.

3- Automated variance retention – Instead of using fixed thresholds (e.g., 95%), implement adaptive selection of PCA components based on cross-validation performance.

## References

- Nakkiran et al. (2021). Deep Double Descent: [Where Bigger Models and More Data Hurt](https://iopscience.iop.org/article/10.1088/1742-5468/ac3a74/meta).

- Gedon, D. et al. (2024). [The Effect of PCA on the Double Descent Risk Curve](https://proceedings.mlr.press/v235/gedon24a.html).

- Zou, H., & Hastie, T. (2005). [Regularization and Variable Selection via the Elastic Net](https://academic.oup.com/jrsssb/article/67/2/301/7109482).
<!--

List all your references here. Remember to put links into markdown. For example:

1.  Einstein, A. (1905). *On the Electrodynamics of Moving Bodies*. Annalen der Physik, 17, 891-921. [Internet Archive](https://archive.org/details/einstein-1905-relativity)

**Tip**: *If you have you references in BibTex, Google Scholar or Zotero*
1. Create/copy a list into ChatGPT
2. Ask it to turn it into an unsorted list in markdown

---
-->
> The research poster for this project can be found in the [BeyondAI Proceedings 2025](https://thinkingbeyond.education/beyondai_proceedings_2025/).

