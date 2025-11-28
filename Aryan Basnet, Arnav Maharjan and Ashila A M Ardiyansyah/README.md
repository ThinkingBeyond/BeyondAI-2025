 ![BeyondAI Banner for Research Projects](../BeyondAI_Banner_Research_Projects_2025.png)

# Evaluating Deep Learning Models for Pneumonia & Tuberculosis Classification Across High and Low Resource Chest X-Ray Datasets

## Motivation
While most research focuses on large, well-curated datasets from high-income countries. Our motivation arises from challenge of accurately and unexplored diagnosing Pneumonia and Tuberculosis (TB) using chest X-rays, particularly in lower Middle Income Country (LMICs) where radiology expert, high quality equipment, image quality, and standardized datasets are limited. This study systematically evaluates four deep learning models across six diverse chest X-ray datasets to understand how disease type, dataset size, balance, and income level affect model performance identifying what actually works in resource-constrained contexts.

## Research Question 
### “Evaluating Deep Learning Models for Pneumonia & Tuberculosis Classification Across High and Low Resource Chest X-Ray Dataset” <br>
Our research performs deep learning classification of chest X-rays into Healthy, Pneumonia, and Tuberculosis using four deep-learning architectures: a Baseline CNN, MobileNetV2, EfficientNet-B0, and ResNet-50. To examine real-world generalizability, we analyze performance across datasets originating from High-Income Countries (HICs), Lower Middle Income Countries (LMICs). This comparison highlights both the strengths and vulnerabilities of each model when applied to diverse imaging environments, offering insights for equitable and reliable global lung-disease screening.

## Method and Implementation
We evaluated four deep learning models (Baseline CNN, MobileNetV2, EfficientNet-B0, and ResNet-50) classifying six datasets chest X-Ray into Healthy, Pneumonia and Tuberculosis categories. These were compiled for a total of 23,480 images and group according to their country level of incomes. Three datasets from High Income Country (HICs) which dataset 1, dataset 2, dataset 5 and another three datasets from Lower Middle Income Country (LMICs) dataset 3, dataset 4, dataset 6. All of these datasets underwent preprocessing as resizing to 224x224 pixels and standardized data to balance the class weighting during the training.

Each dataset was split into training (~70%), validation (~15%), and test sets (~15%). All models were trained with data augmentation, class weighting to handle imbalance, Adam optimizer, categorical cross-entropy loss, early stopping, and learning rate reduction on plateau. Models are under identical conditions. Model performance was measured and evaluated using Weighted F1-score, per-class F1-score, confusion matrices, and training time.

## Result and Discussion
F1 Scores Model For Each Datasets
Dataset 1 (Pneumonia HIC)|Dataset 2 (TB HIC)
:-------------------------:|:-------------------------:
![Dataset 1](https://github.com/user-attachments/assets/0766d26c-a5c3-4fb8-9114-17fba505a2b0)  |  ![Dataset 2](https://github.com/user-attachments/assets/6147bfbb-2f99-45a5-867f-afb392dd87eb)

Dataset 3 (Nigeria LMIC)|Dataset 4 (TB LMIC)
:-------------------------:|:-------------------------:
![Dataset 3](https://github.com/user-attachments/assets/fc4fc5fd-0c48-4cfa-8bfc-49b52087721a)  |  ![Dataset 4](https://github.com/user-attachments/assets/d3eec016-2e69-4752-a1d8-6118dc565cf8)

Dataset 5 (Pneumonia HIC)|Dataset 6 (Bangladesh LMIC)
:-------------------------:|:-------------------------:
![Dataset 5](https://github.com/user-attachments/assets/508ad5bf-59e6-481b-a232-c34bd99bfbef)  |  ![Dataset 6](https://github.com/user-attachments/assets/4fd18d01-6519-49bf-844b-ffddd7bf58e1)

The results showed model performance varied depending on datasets origin (HICs and LMICs) and disease type (Pneumonia and TB). Across all HICs datasets (Dataset 1, 2, and 5) achieve consistently high F1-scores, followed by Baseline CNN and ResNet-50. EfficientNet-B0 showed weaker performance indicating sensitivity to dataset and training.
In LMICs datasets, similar to HICs datasets, MobileNetV2 again achieved the strongest result in dataset 3 and 6, followed by Baseline CNN and ResNet-50. In contrast, EfficientNet-B0 struggles significantly. Unexpectedly, Baseline CNN outperformed deep learning model in dataset 4.

## 1. <br>
HICs VS LMICs Comparison|  
:-------------------------:|
![Comparison](https://github.com/user-attachments/assets/d2ac1d17-795f-43b6-bb50-477647cbbc30) | <br>
The box and whiskers diagram shows that the model performed different significantly. Across both income levels MobileNetV2 (0.88-0.90) consistently achieves the highest F1-scores. Followed by Baseline CNN and ResNet-50. However, EfficientNet-B0 performance dropped on LMICs and underperformed in HICs datasets. Overall this indicates that datasets characteristic (image quality) impact performance more than model complexity.

## 2. <br>
Disease Difficulty (TB vs Pneumonia)|  
:-------------------------:|
![Disease Difficulty](https://github.com/user-attachments/assets/8fcf3dc8-a6ee-4a33-b74c-b57e8215ca0c) | <br>
The box and whisker diagram shows that Tuberculosis (TB) achieves higher F1-scores in both HICs and LMICs datasets. While in Pneumonia varies more especially in LMICs datasets where image conditions affect model accuracy. LMICs Pneumonia datasets show significant drop indicating that Pneumonia is more sensitive to inconsistent image or labelling issues.

## 3. <br>
Training Time vs F1 Score|  
:-------------------------:|
![Training Time VS F1 Score](https://github.com/user-attachments/assets/cd312eaf-62ee-4540-96a5-61d475a08a8b)| <br>
The plot diagram shows no linear relationship between training time and F1-score, except in LMIC EfficientNet-B0 shows linear negative correlation. Models like MobileNetV2 achieve a high F1-score above 0.80 with a relatively short amount of time with less than 100 minutes. However, longer training times (in some dataset like ResNet-50) do not always mean better performance. This indicates that more complex models don't guarantee a better performance and accuracy.

## 4. <br>
Best Model per Dataset|  
:-------------------------:|
![Top Performer](https://github.com/user-attachments/assets/232151f0-f893-447e-b7cb-74fad5bc6c47)| <br>
The bar graph shows that MobileNetV2 dominates as the top performer in most of the HICs and LMICs datasets in all of the Pneumonia and some in TB datasets. In addition, Baseline CNN surprisingly outperformed more complex architecture in dataset 4 classifying TB in LMICs, this highlights that simpler models can be more stronger when data is limited or inconsistent.

## Conclusion 
Overall, the findings of the model performance for chest X-ray classification are more highly influenced by dataset characteristics and disease type than the level income source country. MobileNetV2 appears as the strong model overall, achieving high and stable performance in both HICs and several LMICs datasets, specifically consistently having the highest F1-scores on HICs datasets. However, baseline CNN also has performed well on LMICs datasets, with ResNet-50 remaining competitive for TB classification. This shows that simple architecture can outperform deeper models when trained with varied image qualities and giving the importance of selecting models based on dataset-specific factors such as size, balance, and quality rather than relying solely on model complexity or conventional expectations. 

## Future Work
We think future work should focus on reducing the performance gap between HICs and LMICs datasets. Incorporating tailored model selection and transfer learning strategies can optimize diagnostic performance, adding more diverse datasets (especially in LMICs regions), offering practical guidance for deploying AI effectively across diverse healthcare resource contexts and generalizability of models.

## How to Install and Run the Project **(WARNING)**
**Disclaimer :** Running this hour and hour of work, highly recommended to run it online in Google Colab or the final file.

## Credits
**Student Researchers:** Arnav Maharjan, Ashila Atha Makkah Ardiyansyah, Aryan Basnet <br> 
**Mentors:** Dr. Devendra Singh Dhami

## References
1. Zhang K, Kermany D, Goldbaum M. Labeled Optical Coherence Tomography (OCT) and Chest X Ray Images for Classification [dataset]. Version 2. Mendeley Data; 2018. DOI: 10.17632/rscbjbr9sj.2. [Mendelay Data](https://data.mendeley.com/datasets/rscbjbr9sj/2)
2. Rahman T, Khandakar A, Kadir MA, Islam KR, Islam KF, Mahbub ZB, Ayari MA, Chowdhury MEH. Reliable Tuberculosis Detection using Chest X ray with Deep Learning, Segmentation and Visualization. IEEE Access. 2020;8:191586 191601. doi:10.1109/ACCESS.2020.3031384. [IEE Access](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9224622)
3. Musa A, Adamu MI, Kakudi HA, Lawal Y. Nigeria Chest X ray Dataset. Kaggle; 2024. doi:10.34740/KAGGLE/DSV/9370352. [Kaggle](https://www.kaggle.com/datasets/aminumusa/nigeria-chest-x-ray-dataset)
4. Hira MIK, Bithee MMA, Ahmed S, Akter L, Anonna MJM. A Primary Chest X ray Dataset of Normal and Pneumonia Cases from Epic Chittagong, Bangladesh. Mendeley Data. 2025;2. doi:10.17632/wndbd5r26y.2. [Mendelay Data](https://data.mendeley.com/datasets/wndbd5r26y/2)
6. Kiran S, Saira, Jabeen DI. Dataset of Tuberculosis Chest X-rays Images. Mendeley Data. 2024;v2. doi: 10.17632/8j2g3csprk.2. [Mendelay Data](https://data.mendeley.com/datasets/8j2g3csprk/2)


> The research poster for this project can be found in the [BeyondAI Proceedings 2025](https://www.canva.com/design/DAG56NGHhz8/gdlWMbL8WpTcnPAaTZ2ijA/edit).

