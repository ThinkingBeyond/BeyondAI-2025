![BeyondAI Banner for Research Projects](../BeyondAI_Banner_Research_Projects_2025.png)

# Models and Networks in Early Detection of Diabetic Retinopathy

***Description*** 

Building off of what [Nafiul Haque and Dr. Devendra Singh Dhami](https://github.com/ThinkingBeyond/BeyondAI-2024/tree/main/nafiul), we looked into more networks and models to evaluate the differences in *speed*, *accuracy*, and *confidence*. The neural networks chosen are **ResNet** (a deep neural network designed to tackle the vanishing gradients problem) and **EfficientNet** (a group of CNNs to provide high performance with low resources), and the model chosen is **MetaCLIP** (a vision model developed off of OpenAI's CLIP model with additional features). We used the papers from *Mingxing Tan and Quoc V. Le* from Google Research; *Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun*; and *Hu Xu, Saining Xie, Xiaoqing Ellen Tan, Po-Yao Huang, Russell Howes, Vasu Sharma, Shang-Wen Li, Gargi Ghosh, Luke Zettlemoyer, and Christoph Feichtenhofer*. All the papers have been referenced below. We have used additional papers and websites to develop our research and those too will be refered below.

## Research Methodology

To conduct our research, we each tested a model on a dataset from HuggingFace. The dataset contained ~35k images of eyes labeled no_diabetic_retinopathy, mild_diabetic_retinopathy, moderate_diabetic_retinopathy, severe_diabetic_retinopathy, and proliferative_diabetic_retinopathy. Each label was overall equal in size and each model/network searched through and provided a guess which was then graphed based on Precision, Accuracy, and F1 scores. These scores were then compared to test the properties of speed, accuracy, and confidence for each label.

## Research Question

How can some other networks and models (such as ResNet, EfficientNet, and MetaCLIP) compare to ViTs (Vision Transformers) and CNNs? Additionally, what makes the properties of these better or worse than others?

## Motivation

We wanted to do more testing on machine learning in the medical field to see its effectiveness. Hence, we chose well known and strong image classifiers to test. We thought of combining the knowledge we have gained from each of our respective models and testing them to see how and why they work as well as they do compared to the others. This can help in developing a hybrid machine learning model to provide accurate results, combining properties from the models tested.

## Conclusion


## Future Work

In future iterations of this project,
- finding and testing more properties 
- a hybrid model could be developed with the best properties from each model and network tested
- testing other networks to see which one will work the best

## References

1.  He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. In 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (pp. 770–778). 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR). IEEE. [IEEE Xplore](https://doi.org/10.1109/cvpr.2016.90)
2. Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. [Internet Archive](https://doi.org/10.48550/ARXIV.1905.11946)
3. Chuang, Y.-S., Li, Y., Wang, D., Yeh, C.-F., Lyu, K., Raghavendra, R., Glass, J., Huang, L., Weston, J., Zettlemoyer, L., Chen, X., Liu, Z., Xie, S., Yih, W., Li, S.-W., & Xu, H. (2025). Meta CLIP 2: A Worldwide Scaling Recipe (Version 3). [Internet Archive](https://doi.org/10.48550/ARXIV.2507.22062)
4. Gulshan, V., Peng, L., Coram, M., Stumpe, M. C., Wu, D., Narayanaswamy, A., ... & Webster, D. R. (2016). Development and validation of a deep learning algorithm for detection of diabetic retinopathy in retinal fundus photographs. Jama, 316(22), 2402-2410. [Jama Network](https://jamanetwork.com/journals/jama/fullarticle/2588763)
5. Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). Imagenet classification with deep convolutional neural networks. In Advances in neural information processing systems (pp. 1097-1105). [NIPS](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)

---

> The research poster for this project can be found in the [BeyondAI Proceedings 2025](https://thinkingbeyond.education/beyondai_proceedings_2025/).

