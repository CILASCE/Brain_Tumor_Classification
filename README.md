# A Deep Learning Project for Brain Tumors Classification using MRI images

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-ee4c2c.svg)](https://pytorch.org/)
[![Gradio](https://img.shields.io/badge/UI-Gradio-orange.svg)](https://gradio.app/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 Summary
Accurate and timely diagnosis of brain tumors is a critical factor in determining effective treatment strategies and improving patient survival rates. Although Magnetic Resonance Imaging (MRI) serves as the primary diagnostic tool, manual interpretation by radiologists is a time-consuming process susceptible to human error.

This project presents a high-performance **Convolutional Neural Network (CNN)** architecture designed for the automatic classification of brain MRI scans into four categories: **Glioma, Meningioma, Pituitary Tumor, and Healthy tissue**. By integrating stochastic data augmentation and automated hyperparameters optimization, the system achieves state-of-the-art diagnostic accuracy while maintaining computational efficiency.

---

## 🔬 System Design & Implementation

### Advanced Image Preprocessing
To ensure high-fidelity feature extraction and reduce computational overhead, the data pipeline includes the following stages:
* **Standardization**: All images were resized to 224 x 224 pixels, to standardize the inputs and reduce computational overhead
* **Grayscaling**: Images are converted to grayscale to focus the model exclusively on tissue texture.
* **Histogram Equalization**: Histogram equalization is applied to enhance global image contrast.
* **Statistical Normalization**: Pixel intensities are normalized to a zero mean and unit standard deviation.

### Stochastic Data Augmentation
To enhance the network's generalization capabilities and prevent the memorization of specific image artifacts, the training process incorporates dynamic, on-the-fly stochastic transformations:
* **Geometric Transformations**: Random horizontal flipping (p=0.5) and rotations (±45º).
* **Photometric Adjustments**: Dynamic changes to brightness and contrast by a factor of up to 0.3.
* **Spatial Scaling**: Random zooming between 80% and 120%.

### Tailored CNN Architecture
The custom-designed architecture comprises a high-depth feature extraction module and a robust classification head:
* **Feature Extractor**: Four sequential convolutional blocks using 3 x 3 kernels, with filter sizes scaling systematically (32 to 256).
* **Regularization**: Batch Normalization (BN) is integrated into every block to stabilize learning and improve convergence.
* **Pooling**: Each block concludes with a 2 x 2 Max Pooling operation to reduce spatial dimensions.
* **Classification Head**: Four fully connected (Dense) layers activated by ReLU.
* **Output Layer**: A 4-neuron Softmax layer provides the final probability distribution.

---

## 🧪 Automated Hyperparameters Optimization
To reach the absolute optimal configuration, the model's architecture and training settings were tuned using the **Tree-structured Parzen Estimator (TPE)** algorithm via the **Optuna** framework.

| Hyperparameter | Optimized Value |
| :--- | :--- |
| **Learning Rate** | 0.0001 |
| **Batch Size** | 16 |
| **Dense Layer 1** | 3,694 Neurons |
| **Dense Layer 2** | 1,093 Neurons |
| **Dense Layer 3** | 562 Neurons |
| **Dense Layer 4** | 374 Neurons |

---

## 📊 Performance Analysis
The model was evaluated on an unseen test set of 703 MRI scans, demonstrating exceptional diagnostic reliability.

| Metric | Achievement |
| :--- | :--- |
| **Accuracy** | **0.988** |
| **Test Loss** | **0.053** |
| **Precision** | **0.986** |
| **Recall** | **0.987** |
| **F1-Score** | **0.987** |
| **Specificity** | **0.996** |

---

## 🚀 Clinical Deployment
To demonstrate practical utility, the optimized CNN is deployed via an interactive web application utilizing the **Gradio** framework.
* **User Interface**: Provides a streamlined platform for uploading raw MRI scans and receiving rapid classifications.
* **Confidence Reporting**: The interface displays the model's prediction confidence, serving as a reliable proof-of-concept for computer-aided diagnosis in medical settings.
