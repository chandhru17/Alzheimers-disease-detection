# Alzheimer's Disease Detection Using CNN

## 1. Introduction

Alzheimer's disease (AD) is a progressive neurodegenerative disorder characterized by cognitive decline and structural brain changes. It begins with mild memory loss and can lead to the inability to carry on a conversation and respond to the environment. Alzheimer's disease affects parts of the brain that control thought, memory, and language, significantly impacting daily activities. The disease disrupts the brain's natural asymmetry, causing faster atrophy in the left hemisphere and widespread loss of grey matter, especially in memory-critical areas like the hippocampus. This imbalance contributes to the progressive cognitive decline experienced by Alzheimer's patients.

This project uses Convolutional Neural Networks (CNNs), a prevalent computer vision technique for image classification, to identify differences in MRI scans of brains from individuals with no Alzheimer's and those with varying stages of early Alzheimer's Disease (Very Mildly Demented, Mildly Demented, and Moderately Demented).

## 2. Problem Statement

Identifying Mild Cognitive Impairment at the early stages is crucial for the detection and diagnosis of Alzheimer’s Disease (AD). This project aims to accurately detect mild dementia based on MRI images of the cross-section of the brain. The future scope of this project includes choosing the best CNN model architecture that produces the most accurate results and building an intuitive interface for early detection of cognitive impairments using computer vision and tests.

## 3. Project Overview

This research aims to develop a deep learning model utilizing the ResNet architecture within Convolutional Neural Networks (CNNs) for accurate classification of Alzheimer's disease (AD) from brain MRI images. The objective is to differentiate between healthy individuals and those affected by AD by leveraging the distinctive structural patterns captured by the uniquely optimized architecture. This study addresses the critical need for early and precise diagnosis of AD, facilitating timely interventions and improving patient care.

## 4. Objectives

1. To design and implement a CNN architecture for early Alzheimer's detection and classification from MRI images.
2. To curate and preprocess a comprehensive dataset of MRI scans for training and validation.
3. To evaluate the CNN model's accuracy, sensitivity, and specificity through rigorous testing and validation procedures.
4. To incorporate validated cognitive assessment tools and the most accurate model to analyze user responses and predict an individual’s risk of developing AD based on MRIs and clinical test scores.
5. To ensure data security and compliance with privacy regulations in handling patient information.

## 5. Models

### Custom ResNet64

The proposed system architecture is an extension of the ResNet (Residual Neural Network) architecture, known for its effectiveness in deep learning tasks, particularly image classification. ResNet trains very deep neural networks by utilizing residual blocks, enabling the learning of residual functions with reference to the layer inputs.

The extended architecture includes additional layers and features to enhance its capabilities. It begins with an initial convolutional layer followed by batch normalization and rectified linear unit (ReLU) activation to extract features from input images. A max-pooling layer reduces spatial dimensions while retaining important features.

The core of the extension lies in the series of residual blocks employed thereafter. Each residual block consists of two convolutional layers with batch normalization and ReLU activation after each convolution. The residual blocks allow the network to learn residual mappings, facilitating the training of deeper networks and mitigating the vanishing gradient problem.

The extended architecture includes multiple stacks of residual blocks, with the number of filters increasing at deeper layers, allowing the network to capture increasingly complex features. Some residual blocks utilize strided convolutions to downsample spatial dimensions, effectively increasing the receptive field of the network and reducing computational complexity.

Dropout is applied before the final classification layer to regularize the network and prevent overfitting. A global average pooling layer aggregates spatial information from previous layers, followed by a dense layer with softmax activation to produce class probabilities.

**Accuracy:** 93%

### CNN – SVM

The second proposed system architecture combines a Convolutional Neural Network (CNN) and a Support Vector Machine (SVM) classifier for image classification tasks. It begins with data preprocessing to standardize input images, resizing them to a standard size of (190, 200) pixels and converting them to float32 format.

The CNN model, serving as the feature extractor, consists of several convolutional layers with ReLU activation functions, max-pooling layers to downsample feature maps, and batch normalization layers to stabilize and accelerate training. The CNN culminates with a GlobalAveragePooling2D layer, reducing feature maps to a single vector compatible with the SVM classifier. Dropout is incorporated to prevent overfitting.

The standardized features extracted by the CNN model are fed into the SVM classifier. These features are further standardized using a StandardScaler to ensure zero mean and unit variance, enhancing the SVM's performance. The SVM classifier, initialized with a radial basis function (RBF) kernel, classifies feature vectors into different classes.

During testing, the trained CNN model extracts features from validation images, scaled using the same StandardScaler. The scaled features are inputted into the trained SVM classifier to predict labels for validation images. The accuracy of the SVM classifier is calculated by comparing predicted labels with true labels from the validation dataset.

**Accuracy:** 70%

## 6. Conclusion

This project demonstrates the potential of deep learning models, specifically CNN architectures, in the early detection of Alzheimer's disease from MRI images. The custom ResNet64 model achieved an accuracy of 93%, while the CNN-SVM model achieved 70%. Future work will focus on further improving model accuracy and developing an intuitive interface for early cognitive impairment detection.
