![BeyondAI Banner for Research Projects](../BeyondAI_Banner_Research_Projects_2025.png)

# Deep Learning Models for Diabetic Retinopathy

### Student Researchers
Renesh Thaipulley and Sahana Dillibabu

### Mentor
Dr. Devendra Singh Dhami

## Description 

Diabetic retinopathy is an eye disease that can lead to vision loss and blindness in people with diabetes. It is the leading cause of vision impairment in working-age adults worldwide, but early detection and treatment can prevent up to 95% of cases. Thus, it is no surprise that researchers, including those in previous BeyondAI cohorts, have turned to machine learning to facilitate early diagnosis.
2024's project by Nafiul Haque and Dr. Devendra Singh Dhami [Machiene Learning in Early Detection of Diabetic Retinopathy](https://github.com/ThinkingBeyond/BeyondAI-2024/tree/main/nafiul) focused on comparing the performance of CNNs (convolutional neural networks) and ViTs (vision transformers) on a diabetic retinopathy classification task. They found that each model had its own strengths - ViTs were better at generalization, while CNNs excelled at feature extraction. Building on their work, we studied 3 cutting-edge models to evaluate differences in *speed*, *accuracy*, and *confidence*. These models are: 

**MetaCLIP** - MetaCLIP is an improved way of training image–text models like CLIP. CLIP models learn by matching an image with the text that describes it, but large web datasets usually have noisy or inaccurate captions that hurt performance. MetaCLIP fixes this by using a better filtering and matching process to create image-text pairings that are more accurate. By improving the quality of the training data, the model learns stronger and more reliable connections, improving its overall accuracy even on very large datasets.

**ResNet** - ResNet is a type of CNN designed to make very deep models easier to train. Normally, when you keep adding more layers to a neural network, the gradient can become extremely small or unstable during backpropagation. This is called the vanishing/exploding gradient problem, and it causes deeper models to perform worse even though they should be more powerful. ResNet solves this by adding residual blocks with skip connections. These connections allow the input of a block to “skip” over a few layers and be added directly to the output. This helps gradients flow through the network more easily, letting the model perform well even when it has dozens or hundreds of layers.

## Research Methodology

To conduct our research, we each tested a model on an EYEPACS dataset sourced from Hugging Face. The dataset contained ~35000 retinal fundus images and comprised 5 classes: (0)no_diabetic_retinopathy, (1)mild_retinopathy, (2)moderate_retinopathy, (3)severe_retinopathy, and (4)proliferative_retinopathy. Each model/network was trained on a training set (70% of the dataset) and testing on a validation set (30% of the dataset). The results were then measured through the calculation of Precision, Accuracy, and F1 scores. A comparison of the scores was then carried out to test the properties of speed, accuracy, and confidence for each label.

## Research Question

How do cutting-edge networks and models (ResNet, EfficientNet, and MetaCLIP) compare to the vanilla CNN and ViT? Additionally, what properties differentiate performance?

## Motivation

There is a real need for accurate, efficient, and effective diagnostic tools for diabetic retinopathy. However, limited clinical resources and the difficulty of identifying subtle retinal features makes screening at a large scale difficult. Each machine learning architecture has its strengths and weaknesses. We wanted to investigate whether new architectures could offer meaningful improvements in performance over traditional CNNs and vision transformers. By doing this, we hope to gain insights into which modern models are best for screening, and implement what we have learned into real-world situations through hybrid models and other solutions.

## Results

### MetaCLIP
### <ins>Zero Shot</ins>
- Time: 29276.318501535003 seconds or about 8 hours
- Count: (in order) \[24636, 9946, 3, 0, 523]
<img width="295" height="227.5" alt="image" src="https://github.com/user-attachments/assets/007516e4-0583-4094-98a3-4b2aa7e7a08f" />

  
**Classification Report**

| Class                                | Precision | Recall | F1-Score | Support |
|--------------------------------------|-----------|--------|----------|---------|
| no_diabetic_retinopathy              | 0.75      | 0.72   | 0.73     | 25,802  |
| mild_retinopathy                     | 0.00      | 0.00   | 0.00     | 0       |
| moderate_retinopathy                 | 0.00      | 0.00   | 0.00     | 0       |
| severe_retinopathy                   | 0.00      | 0.00   | 0.00     | 0       |
| proliferative_retinopathy            | 0.00      | 0.00   | 0.00     | 0       |

**Averages**

| Average Type | Precision | Recall | F1-Score | Support |
|--------------|-----------|--------|----------|---------|
| Micro Avg    | 0.53      | 0.72   | 0.61     | 25,802  |
| Macro Avg    | 0.15      | 0.14   | 0.15     | 25,802  |
| Weighted Avg | 0.75      | 0.72   | 0.73     | 25,802  |

### <ins>Fine Tuning</ins>
| Epoch | Training Loss | Validation Loss | Accuracy |
|-------|--------------|-----------------|----------|
| 1 | 0.8264 | 0.7128 | 76.83% |
| 2 | 0.6736 | 0.6245 | 79.28% |
| 3 | 0.6321 | 0.5959 | 80.41% |
| 4 | 0.5818 | 0.5755 | 81.14% |

Final test accuracy: **81.1%** | Test loss: **0.575** | Evaluation speed: **50.8 samples/second**
### Resnet
- Time: 121.9s per epoch (dependent on GPU and runtime)
- Count: (in order, total) \[25802, 2438, 5288, 872, 708]

**Classification Report**

| Class                                | Precision | Recall | F1-Score | 
|--------------------------------------|-----------|--------|----------|
| no_diabetic_retinopathy              | 0.79      | 0.97   | 0.87     | 
| mild_retinopathy                     | 0.00      | 0.00   | 0.00     | 
| moderate_retinopathy                 | 0.49      | 0.21   | 0.30     | 
| severe_retinopathy                   | 0.31      | 0.54   | 0.40     | 
| proliferative_retinopathy            | 0.80      | 0.02   | 0.04     |


**Averages**

| Average Type | Precision | Recall | F1-Score |
|--------------|-----------|--------|----------|
| Micro Avg    | 0.75      | 0.75   | 0.32     |
| Macro Avg    | 0.48      | 0.35   | 0.75     |
| Weighted Avg | 0.67      | 0.75   | 0.71     |


## Conclusion and Discussion

### MetaCLIP
The zero shot model ran and tested the different different labels. From the graph it can be seen that the accuracy of the model wasn't very good. For `no_diabetic_retinopathy` the model could somwhat accurately predict most of them with the precision score being around 0.75. However, for the other labels, it wasn't really the same. This can mostly be caused by having the normalization of the images to not be proper, however, it could also be the case where the model decided that the highest chance of getting the image right was to guess that there was no diabetic retinopathy. Addtionally, the model took about eight hours to fully run through all images in the dataset. Now, this can be fixed with a stronger GPU and CPU along with more efficient code but the amount of time taken was about one second per image.

With fine tuning the model, the class imbalance was fixed using `RandomOverSampler` to even out the imbalances. For example, if the moderate class had 5000 images while severe had 2500 images, all the rows that was connected with the severe class would be duplicated to have 5000 images, matching the moderate class. In the code, the image distribution changed from `{0: 25802, 2: 5288, 1: 2438, 3: 872, 4: 708}` to `{0: 25802, 1: 25802, 2: 25802, 4: 25802, 3: 25802}`. This fixed the class imbalance issue. Additionally, changing the properties of the images with size and rotation allowed allowed for the model to be better at detecting variabilities. The images were rotated randomly and the sharpness was adjusted in this case. Four epochs were implemented using the HuggingFace Trainer. All of these changed led to about a 50% increase in the accuracy for better detection of diabetic retinopathy.     

### ResNet
The ResNet model performed strongly in identifying no diabetic retinopathy, with a high recall(0.97) and F1 score (0.87). However, the model struggles with minority classes, especially mild retinopathy. For moderate and severe retinopathy, that have low recall and moderate F1 scores, the performance is inconsistent, while proliferative retinopathy has a high precision but a low recall. When we account for class imbalance, the weighted F1 score of 0.71 shows decent performance, but the macro average recall of 0.35 shows that the performance across classes is uneven. Improving the quality of our dataset by finding more images from other classes and using more aggressive data augmentation methods could improve the model's clinical usefulness. 

## Future Work
Future iterations of this project could cover: 
- finding and testing more properties
- development of a hybrid model with best properties of its components
- testing more, newer networks to see which one will work the best
- optimizing a model to work in low-resource settings, or with poor-quality images

## References

1.  He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. In 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (pp. 770–778). 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR). IEEE. [IEEE Xplore](https://doi.org/10.1109/cvpr.2016.90)
2. Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. [Internet Archive](https://doi.org/10.48550/ARXIV.1905.11946)
3. Chuang, Y.-S., Li, Y., Wang, D., Yeh, C.-F., Lyu, K., Raghavendra, R., Glass, J., Huang, L., Weston, J., Zettlemoyer, L., Chen, X., Liu, Z., Xie, S., Yih, W., Li, S.-W., & Xu, H. (2025). Meta CLIP 2: A Worldwide Scaling Recipe (Version 3). [Internet Archive](https://doi.org/10.48550/ARXIV.2507.22062)
4. Gulshan, V., Peng, L., Coram, M., Stumpe, M. C., Wu, D., Narayanaswamy, A., ... & Webster, D. R. (2016). Development and validation of a deep learning algorithm for detection of diabetic retinopathy in retinal fundus photographs. Jama, 316(22), 2402-2410. [Jama Network](https://jamanetwork.com/journals/jama/fullarticle/2588763)
5. Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). Imagenet classification with deep convolutional neural networks. In Advances in neural information processing systems (pp. 1097-1105). [NIPS](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)

---

> The research poster for this project can be found in the [BeyondAI Proceedings 2025](https://thinkingbeyond.education/beyondai_proceedings_2025/).

