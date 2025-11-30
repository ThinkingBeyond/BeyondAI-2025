 ![BeyondAI Banner for Research Projects](../BeyondAI_Banner_Research_Projects_2025.png)

# Evaluating Deep Learning Models for Pneumonia & Tuberculosis Classification Across High and Low Resource Chest X-Ray Datasets

## Motivation
While most existing research relies on large, well-curated chest X-ray datasets from high-income countries, such settings do not reflect the realities of diagnosing Pneumonia and Tuberculosis (TB) in lower-middle-income countries (LMICs), where radiology expertise, imaging quality, and standardized datasets are often limited. Motivated by these gaps, this study systematically evaluates four deep learning models across six diverse chest X-ray datasets to examine how factors such as disease type, dataset size, class balance, and country income level influence model performance, ultimately identifying what truly works in resource-constrained environments.

## Research Question 
Our research is guided by the central question: How do different deep learning models (Baseline CNN, ResNet50, EfficientNet-B0, and MobileNetV2) perform in classifying Pneumonia and Tuberculosis when evaluated on chest X-ray datasets from both High-Income Countries (HICs) and Low- and Middle-Income Countries (LMICs), and which model is ultimately best suited for resource-constrained LMICs settings? <br>
Building on this question, we conduct deep learning classification of chest X-rays into Healthy, Pneumonia, and Tuberculosis using the four selected architectures. To assess real-world generalizability, these models are systematically tested across datasets originating from both HIC and LMIC environments. By comparing their performance under varying imaging conditions, resource levels, and dataset characteristics, this study reveals the strengths and weaknesses of each model and offers practical insights into designing equitable and reliable global lung-disease screening systems.

## Literature Review
Deep learning has become central to automated chest X-ray analysis, particularly for pneumonia and tuberculosis detection. Prior studies demonstrate that convolutional neural networks (CNNs) consistently outperform traditional diagnostic algorithms by learning discriminative radiological patterns directly from imaging data (Journal of Big Data 2022). Research also confirms the strong benefit of using pretrained architectures such as ResNet and EfficientNet, whose transfer-learned representations significantly improve pneumonia and TB classification accuracy. This aligns with broader findings that deep learning models can reach expert-level performance when trained on large, high-quality datasets.

However, multiple studies emphasize that performance varies substantially across datasets from different geographical or clinical contexts. A multi-country evaluation showed that domain shift differences in imaging equipment, disease severity, or population characteristics can cause large performance drops when models trained on high-income country (HIC) datasets are applied to low- and middle-income country (LMIC) settings (Springer BMC 2022). As models often generalize poorly outside the environment they were trained on. The literature highlights this as a critical barrier to deploying robust AI for TB and pneumonia diagnosis in global health contexts.

Lightweight architectures such as MobileNetV2 and EfficientNetB0 are repeatedly shown to perform well on smaller or lower-quality datasets while requiring far fewer computational resources (Computers in Biology and Medicine 2021). This makes them more convinient for LMIC screening programs where GPU infrastructure is limited. Studies evaluating TB detection pipelines found that these compact models can match or exceed heavier models like ResNet50 when proper augmentation and balancing strategies are used, reinforcing your poster’s results where MobileNetV2 performed competitively across both HIC and LMIC datasets.

Recent work also underscores that transfer learning alone is not always sufficient to guarantee cross-regional robustness. A 2024 Nature study found that although pretrained models significantly boost baseline accuracy, their performance still depends strongly on dataset-specific factors such as disease presentation patterns, imaging quality, and labeling depth. 

## Method and Implementation
We evaluated four deep learning models (Baseline CNN, MobileNetV2, EfficientNet-B0, and ResNet-50) classifying six datasets chest X-Ray into Healthy, Pneumonia and Tuberculosis categories. These were compiled for a total of 23,480 images and group according to their country level of incomes. Three datasets from High Income Countries (HICs) which dataset 1, dataset 2, dataset 5 and another three datasets from Lower Middle Income Countries Country (LMICs) dataset 3, dataset 4, dataset 6. All of these datasets underwent preprocessing as resizing to 224x224 pixels and standardized data to balance the class weighting during the training.

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
**Disclaimer**: Not recommended to run this, because running this code locally may require several hours due to the computational demands of training deep learning models. It is highly recommended to execute it in Google Colab or directly run the final file (AA_master_analysis_all_datasets), preprocessed notebook for a smoother and faster experience.

## Acknowledgement <br>
We would like to express our gratitude to BeyondAI for providing and making this AI program available to a selection of people. We also want to thank Dr. Filip Bar and the entire volunteers that contributed and their dedication on making this program possible. Especially Dr. Filip Bar on teaching throughout the course stage and guidance at the research development stage. His insight gives us a fundamental stepping stone and vital role in the direction of our study.
Finally, we would like to thank Dr. Davendra Singh Dhami, our mentor for his valuable advice, constructive feedback, and support throughout the research process. His mentorship was crucial to direct and complete our research successfully.

## Credits
**Student Researchers:** Arnav Maharjan (Main Contributor) & Ashila Atha Makkah Ardiyansyah<br> 
**Mentor:** Dr. Devendra Singh Dhami

## Notes
One of the inside of the chest X-ray result in the checkpoints, Baseline CNN is incomplete. This only gives a purpose to sort the epochs result so it won't crash or runtime error repeatedly.

## References
Datasets References<br>
1. Kermany D, Zhang K, Goldbaum M. Chest X-Ray Images (Pneumonia) Dataset. Mendeley Data; 2018. DOI: 10.17632/rscbjbr9sj.2. [Dataset 1](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
2. Saira Kiran, Ishrat Jabeen. Dataset of Tuberculosis Chest X-rays Images. Mendeley Data; 2024. DOI: 10.17632/8j2g3csprk.2. [Dataset 2](https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset)
3. Nigerian Chest X-ray Dataset (Aminu Musa, et al.). Nigeria Chest X-Ray Dataset. Kaggle Dataset. Accessed 2025. [Dataset 3](https://www.kaggle.com/datasets/aminumusa/nigeria-chest-x-ray-dataset)
4. Tuberculosis Chest X-ray Images (Local Pakistan Hospital). Mendeley Data; 2024. DOI: 10.17632/8j2g3csprk.2. [Dataset 4](https://data.mendeley.com/datasets/8j2g3csprk/2)
5. Rahman T, Khandakar A, Kadir MA, Islam KR, Islam KF, Mahbub ZB, Ayari MA, Chowdhury MEH. Reliable Tuberculosis Detection using Chest X ray with Deep Learning, Segmentation and Visualization. IEEE Access. 2020;8:191586 191601. doi:10.1109/ACCESS.2020.3031384. [Dataset 5](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9224622)
6. Hira MIK, Bithee MMA, Ahmed S, Akter L, Anonna MJM. A Primary Chest X ray Dataset of Normal and Pneumonia Cases from Epic Chittagong, Bangladesh. Mendeley Data. 2025;2. doi:10.17632/wndbd5r26y.2. [Dataset 6](https://data.mendeley.com/datasets/wndbd5r26y/2)
7. RSUA Chest X-Ray Dataset. Airlangga University Hospital, Indonesia; 2023. PubMed-accessible via Mendeley Data — Dataset DOI: 10.17632/2jg8vfdmpm.1. [Dataset 0](https://data.mendeley.com/datasets/2jg8vfdmpm/1)

Literature Review References <br>
1. Abdulkarem M, Geman O, Al-Hadhrami T, et al. Deep learning for multi-class chest disease classification using chest X-ray images. Journal of Big Data. 2022. PubMed Central (PMC9090861).[Journal of Big Data](https://pmc.ncbi.nlm.nih.gov/articles/PMC9090861/)
2. Ozturk T, Talo M, Yildirim EA, Baloglu UB, Yildirim O, Acharya UR. An explainable deep learning approach for detecting COVID-19 and pneumonia from chest X-rays. Computers in Biology and Medicine. 2021. PubMed Central (PMC8117675).[Computers in Biology and Medicine](https://pmc.ncbi.nlm.nih.gov/articles/PMC8117675/)
3.  Schaaf C, Maduke T, Breuninger T, et al. Performance variation of deep learning–based chest X-ray classifiers across global clinical settings: a multi-country evaluation. BMC Medical Imaging. 2022. [BMC Medical Imaging](https://link.springer.com/article/10.1186/s12880-022-00793-7)
4. hang Y, Li H, Xu C, et al. Generalization limits of deep learning for global chest X-ray diagnosis across heterogeneous imaging domains. Scientific Reports (Nature). 2024.[Scientific Reports (Nature)](https://www.nature.com/articles/s41598-024-65703-z)


> The research poster for this project can be found in the [BeyondAI Proceedings 2025](https://www.canva.com/design/DAG56NGHhz8/gdlWMbL8WpTcnPAaTZ2ijA/edit).

