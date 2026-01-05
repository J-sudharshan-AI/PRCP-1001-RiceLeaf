PROJECT OVERVIEW:-

Rice is one of the most important staple crops worldwide, especially in Asia.
 
 Rice plants are highly susceptible to various leaf diseases that can significantly reduce crop yield if not detected early. This project aims to build an automated rice leaf disease classification system using deep learning and computer vision techniques.

The system classifies rice leaf images into three major disease categories using Convolutional Neural Networks (CNNs) and Transfer Learning.

OJECTIVE:-

Analyze a small image dataset of rice leaf diseases
Build and compare multiple deep learning models
Study the impact of data augmentation and regularization
Select the best-performing model for production use
Provide insights into overfitting and underfitting behavior.


DATASET DESCRIPTION:-

Total Images: 119
Classes: 3
Bacterial Leaf Blight
Brown Spot
Leaf Smut
Image format: JPG


MODEL IMPLEMENTATION:-

The following four models were developed and evaluated:
* CNN without Data Augmentation
* CNN with Data Augmentation
* CNN with Regularization (Dropout + Batch Normalization)
* MobileNet (Transfer Learning)

BEST MODEL SELECTION:-

MobileNet with Transfer Learning was selected as the final model because:
* It achieved the highest validation accuracy
* It recorded the lowest validation loss
* It generalized well despite the small dataset size
* It leveraged pretrained ImageNet features effectively.

TECHNIQUE USED:-

* Data Augmentation to increase data diversity
* Regularization analysis to diagnose underfitting
* Transfer Learning to improve performance on limited data
* Validation accuracy and loss for fair model comparison.

TECHNOLOGIES USED:-

* Python
* TensorFlow / Keras
* NumPy
* Matplotlib
* Streamlit (for deployment)
* VS code

FUTURE SCOPE:-

* Increase dataset size with real-world field images
* Add more rice disease categories
* Fine-tune pretrained layers in MobileNet
* Deploy as a mobile or web-based application
* Integrate with precision agriculture systems (IoT, drones)


CONCLUSION:-

This project demonstrates the effectiveness of deep learning in agricultural disease detection. Comparative analysis of multiple models highlights the importance of data augmentation, proper regularization, and transfer learning when working with small datasets. The final MobileNet-based model shows strong potential for real-world deployment in precision agriculture.




Author:-  J.Sudharshan
