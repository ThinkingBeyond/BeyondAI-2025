![BeyondAI Banner for Research Projects](../BeyondAI_Banner_Research_Projects_2025.png)

Evaluating Deep Learning Models for Pneumonia & Tuberculosis Classification
Across High and Low Resource Chest X-Ray Datasets

***Provide a description of your project including*** 

## Motivation
While most research focuses on large, well-curated datasets from high-income countries.The motivation arises from challenge of accurately and unexplored diagnosing Pneumonia and Tuberculosis (TB) using chest X-rays, particularly in lower Middle Income Country (LMICs) where radiology expert, high quality equipment, image quality, and standardized datasets are limited. This study systematically evaluates four deep learning models across six diverse chest X-ray datasets to understand how disease type, dataset size, balance, and income level affect model performance identifying what actually works in resource-constrained contexts.

## Research Question 
“Evaluating Deep Learning Models for Pneumonia & Tuberculosis Classification Across High and Low Resource Chest X-Ray Dataset”

Our research performs deep learning classification of chest X-rays into Healthy, Pneumonia, and Tuberculosis using four deep-learning architectures: a Baseline CNN, MobileNetV2, EfficientNet-B0, and ResNet-50. To examine real-world generalizability, we analyze performance across datasets originating from High-Income Countries (HICs), Low- and Middle-Income Countries (LMICs). This comparison highlights both the strengths and vulnerabilities of each model when applied to diverse imaging environments, offering insights for equitable and reliable global lung-disease screening.

## Method and Implementation
We evaluated four deep learning models (Baseline CNN, MobileNetV2, EfficientNet80, and ResNet50) classifying six datasets chest X-Ray into Healthy, Pneumonia and Tuberculosis categories. These were compiled for a total of 23,480 images and group according to their country’s income. Three datasets from High Income Country (HICs) which dataset 1, dataset 2, dataset 5) and another three datasets from Lower Middle Income Country (LMICs) dataset 3, dataset 4, dataset 6. All of these datasets underwent preprocessing as resizing to 224x224 pixels and standardized data to balance the class weighting during the training.

Each dataset was split into training (~70%), validation (~15%), and test sets (~15%). All models were train validation test, adam optimizer, categorical cross-entropy loss, early stopping, and learning rate reduction on plateau, models are under identical conditions. Model performance was measured and evaluated using weighted F1-score, per-class F1, confusion matrices, and training time.

## Result and Discussion


## Conclusion 
Overall, the findings of the model performance for chest X-ray classification are more highly influenced by dataset characteristics and disease type than the level income source country. MobileNetV2 appears as the strong model overall, achieving high and stable performance in both HICs and several LMICs datasets, specifically consistently having the highest F1-scores on HICs datasets. However, baseline CNN also has performed well on LMICs datasets, with ResNet50 remaining competitive for TB classification. This shows that simple architecture can outperform deeper models when trained with varied image qualities and giving the importance of selecting models based on dataset-specific factors such as size, balance, and quality rather than relying solely on model complexity or conventional expectations. 

## Future Work
Future work should focus on reducing the performance gap between HICs and LMICs datasets.
Incorporating tailored model selection and transfer learning strategies can optimize diagnostic performance, adding more diverse datasets (especially in LMICs regions), offering practical guidance for deploying AI effectively across diverse healthcare resource contexts and generalizability of models.


## References
List all your references here. Remember to put links into markdown. For example:

1.  Einstein, A. (1905). *On the Electrodynamics of Moving Bodies*. Annalen der Physik, 17, 891-921. [Internet Archive](https://archive.org/details/einstein-1905-relativity)

**Tip**: *If you have you references in BibTex, Google Scholar or Zotero*
1. Create/copy a list into ChatGPT
2. Ask it to turn it into an unsorted list in markdown

---

> The research poster for this project can be found in the [BeyondAI Proceedings 2025]([https://thinkingbeyond.education/beyondai_proceedings_2025/](https://www.canva.com/design/DAG56NGHhz8/gdlWMbL8WpTcnPAaTZ2ijA/edit)).

