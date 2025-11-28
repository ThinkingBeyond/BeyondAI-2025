![BeyondAI Banner for Research Projects](../BeyondAI_Banner_Research_Projects_2025.png)

Evaluating Deep Learning Models for Pneumonia & Tuberculosis Classification
Across High and Low Resource Chest X-Ray Datasets

## Motivation
While most research focuses on large, well-curated datasets from high-income countries.The motivation arises from challenge of accurately and unexplored diagnosing Pneumonia and Tuberculosis (TB) using chest X-rays, particularly in lower Middle Income Country (LMICs) where radiology expert, high quality equipment, image quality, and standardized datasets are limited. This study systematically evaluates four deep learning models across six diverse chest X-ray datasets to understand how disease type, dataset size, balance, and income level affect model performance identifying what actually works in resource-constrained contexts.

## Research Question 
“Evaluating Deep Learning Models for Pneumonia & Tuberculosis Classification Across High and Low Resource Chest X-Ray Dataset”

Our research performs deep learning classification of chest X-rays into Healthy, Pneumonia, and Tuberculosis using four deep-learning architectures: a Baseline CNN, MobileNetV2, EfficientNet-B0, and ResNet-50. To examine real-world generalizability, we analyze performance across datasets originating from High-Income Countries (HICs), Low- and Middle-Income Countries (LMICs). This comparison highlights both the strengths and vulnerabilities of each model when applied to diverse imaging environments, offering insights for equitable and reliable global lung-disease screening.

## Method and Implementation
We evaluated four deep learning models (Baseline CNN, MobileNetV2, EfficientNet80, and ResNet50) classifying six datasets chest X-Ray into Healthy, Pneumonia and Tuberculosis categories. These were compiled for a total of 23,480 images and group according to their country’s income. Three datasets from High Income Country (HICs) which dataset 1, dataset 2, dataset 5) and another three datasets from Lower Middle Income Country (LMICs) dataset 3, dataset 4, dataset 6. All of these datasets underwent preprocessing as resizing to 224x224 pixels and standardized data to balance the class weighting during the training.

Each dataset was split into training (~70%), validation (~15%), and test sets (~15%). All models were train validation test, adam optimizer, categorical cross-entropy loss, early stopping, and learning rate reduction on plateau, models are under identical conditions. Model performance was measured and evaluated using weighted F1-score, per-class F1, confusion matrices, and training time.

## Result and Discussion
F1 Scores Model For Each Datasets

(images)

The results showed model performance varied depending on datasets origin (HICs and LMICs) and disease type (Pneumonia and TB). Across all HIC datasets (Dataset 1, 2, and 5) achieve consistently high F1 scores, followed by Baseline CNN and ResNet50. EfficientNetB0 showed weaker performance indicating sensitivity to dataset and training.
In LMICs datasets, similar to HIC datasets, MobileNetV2 again achieved the strongest result in dataset 3 and 6, followed by Baseline CNN and ResNet50. In contrast, EfficientNetB0 struggles significantly. Unexpectedly, baseline CNN outperformed deep learning model in dataset 4.

1. HICs VS LMICs Comparison

(image)

The box and whiskers graph shows that the model performed differed significantly. Across both income levels MobileNetV2 (0.88-0.90) consistently achieves the highest F1 scores. Followed by Baseline CNN and ResNet50. However, EfficientNetBO performance dropped on LMICs and underperformed in HICs datasets. Overall this indicates that datasets characteristic (image quality) impact performance more than model complexity.

2. Disease Difficulty (TB vs Pneumonia)

(image)

The box and whisker graph shows that Tuberculosis (TB) achieves higher F1 scores in both HICs and LMICs datasets. While in Pneumonia varies more especially in LMIC datasets where image conditions affect model accuracy. LMIC Pneumonia datasets show significant drop giving that pneumonia is more sensitive to inconsistent image or labelling issues.

3. Training Time vs F1 Score

(image)

The graph plot shows no linear relationship between training time and F1 score, except in LMIC EfficientNetBO shows linear negative correlation. Models like MobileNetV2 achieve a high F1 score above 0.80 with a relatively short amount of time with less than 100 minutes. However, longer training times (some dataset in ResNet50) do not always mean better performance. This indicates that more complex models don't guarantee a better performance and accuracy

4. Best Model per Dataset

(image)

The bar graph shows that MobileNetV2 dominates as the top performer in most of the HICs and LMICs datasets in all of the pneumonia and some in TB datasets. In addition, Baseline CNN surprisingly outperformed more complex architecture in dataset 4 classifying TB in LMICs, this highlights that simpler models can be more strong when data is limited or inconsistent.

## Conclusion 
Overall, the findings of the model performance for chest X-ray classification are more highly influenced by dataset characteristics and disease type than the level income source country. MobileNetV2 appears as the strong model overall, achieving high and stable performance in both HICs and several LMICs datasets, specifically consistently having the highest F1-scores on HICs datasets. However, baseline CNN also has performed well on LMICs datasets, with ResNet50 remaining competitive for TB classification. This shows that simple architecture can outperform deeper models when trained with varied image qualities and giving the importance of selecting models based on dataset-specific factors such as size, balance, and quality rather than relying solely on model complexity or conventional expectations. 

## Future Work
Future work should focus on reducing the performance gap between HICs and LMICs datasets.
Incorporating tailored model selection and transfer learning strategies can optimize diagnostic performance, adding more diverse datasets (especially in LMICs regions), offering practical guidance for deploying AI effectively across diverse healthcare resource contexts and generalizability of models.


## References
1. Zhang K, Kermany D, Goldbaum M. Labeled Optical Coherence Tomography (OCT) and Chest X Ray Images for Classification [dataset]. Version 2. Mendeley Data; 2018. DOI: 10.17632/rscbjbr9sj.2.
2. Rahman T, Khandakar A, Kadir MA, Islam KR, Islam KF, Mahbub ZB, Ayari MA, Chowdhury MEH. Reliable Tuberculosis Detection using Chest X ray with Deep Learning, Segmentation and Visualization. IEEE Access. 2020;8:191586 191601. doi:10.1109/ACCESS.2020.3031384.
3. Musa A, Adamu MI, Kakudi HA, Lawal Y. Nigeria Chest X ray Dataset. Kaggle; 2024. doi:10.34740/KAGGLE/DSV/9370352.
4. Kermany D, Zhang K, Goldbaum M. Labeled Optical Coherence Tomography (OCT) and Chest X Ray Images for Classification. Mendeley Data. 2018;2. doi:10.17632/rscbjbr9sj.2.
5. Hira MIK, Bithee MMA, Ahmed S, Akter L, Anonna MJM. A Primary Chest X ray Dataset of Normal and Pneumonia Cases from Epic Chittagong, Bangladesh. Mendeley Data. 2025;2. doi:10.17632/wndbd5r26y.2.
6. Kiran S, Saira, Jabeen DI. Dataset of Tuberculosis Chest X-rays Images. Mendeley Data. 2024;v2. doi: 10.17632/8j2g3csprk.2.


> The research poster for this project can be found in the [BeyondAI Proceedings 2025]([https://thinkingbeyond.education/beyondai_proceedings_2025/](https://www.canva.com/design/DAG56NGHhz8/gdlWMbL8WpTcnPAaTZ2ijA/edit)).

