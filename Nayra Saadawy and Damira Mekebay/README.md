![BeyondAI Banner for Research Projects](../BeyondAI_Banner_Research_Projects_2025.png)

# Non-linear Classifiers: How Dataset Conditions Such As Noise And Dimensionality Affect Regularization

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

In non-linear, nonparametric models how does the optimal level of regularization change when dataset conditions, such as dimensions and noise, are changed? Regularization occurs by tuning the model’s hyperparameters. The optimal settings of these hyperparameters are influenced by the dataset and its properties, for example: 
- noise     - dimensionality    - distribution      - number of classes

We focused on investigating the first two options: noise and dimensionality. We did so across 3 nonparametric nonlinear Machine Learning models:
- kNN    - SVM    - Decision Trees

We focused on the most influential hyperparameter for each of these models (k - kNN, C - SVM, max_depth - Decision Trees), and observed how the optimal settings of these hyperparameters change when the noise changes in the make_moons synthetic dataset, and the dimensions in the make_classifications synthetic dataset.

## Motivation

Nonparametric, nonlinear models have more freedom when drawing decision boundaries due to their minimal, intuitive underlying assumptions. However, this makes them prone to overfitting, especially as dataset conditions such as noise and dimensionality vary.

Theoretical guidelines are set to bound these modes’ behaviour; however, practically, their behaviour highly depends on specific dataset factors. That’s why our motivation was to explore the approximate trends in the behaviours of these models by changing these two dataset factors one at a time.

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

