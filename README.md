
# Advanced Vision AI: Fast-Tracking Image Classification with Transfer Learning

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)](https://www.tensorflow.org/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)

A comprehensive project demonstrating transfer learning techniques for image classification using pre-trained convolutional neural networks (CNNs) on multiple datasets. This repository contains both educational examples and practical assignments for mastering transfer learning in computer vision.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Notebooks](#notebooks)
- [Key Features](#key-features)
- [Requirements](#requirements)
- [Results Summary](#results-summary)
- [Project Structure](#project-structure)

  
## 🎯 Project Overview

This project explores the power of transfer learning in computer vision by leveraging pre-trained models trained on ImageNet to solve image classification tasks. We compare three state-of-the-art architectures:

- **ResNet50**: Deep residual network with skip connections for better gradient flow
- **VGG16**: Classic deep CNN with small convolution filters
- **MobileNetV2**: Efficient architecture optimized for mobile and edge devices

The project demonstrates how transfer learning can significantly reduce training time and improve performance compared to training from scratch, especially on datasets with limited samples.

## 📚 Notebooks

### 1. Fast-Tracking Image Classification with Transfer Learning
**File:** `Fast-Tracking Image Classification with Transfer Learning.ipynb`

This notebook provides a comprehensive introduction to transfer learning using the CIFAR-100 dataset:

- **Dataset**: CIFAR-100 (100 classes, 60,000 32×32 color images)
- **Focus**: Understanding transfer learning fundamentals
- **Content**:
  - Data loading and preprocessing for each model
  - Model adaptation with custom classification heads
  - Fine-tuning techniques
  - Performance comparison across architectures
  - Training history visualization

### 2. Assignment: Transfer Learning on Oxford Flowers 102
**File:** `Assignment_Oxford_Flowers_Transfer_Learning.ipynb`

A memory-optimized assignment notebook designed for Google Colab, focusing on real-world application:

- **Dataset**: Oxford Flowers 102 (102 flower categories, higher resolution images)
- **Optimizations**: Memory-efficient training for Colab environments
- **Content**:
  - Complete transfer learning pipeline
  - Model comparison and analysis
  - Assignment questions for learning assessment
  - Optional advanced tasks (fine-tuning, data augmentation)
  - Performance visualization and error analysis

## ✨ Key Features

- **Multiple Architectures**: Compare ResNet50, VGG16, and MobileNetV2 performance
- **Two Datasets**: CIFAR-100 and Oxford Flowers 102 for diverse evaluation
- **Transfer Learning Pipeline**: Complete workflow from data preprocessing to evaluation
- **Memory Optimization**: Colab-compatible with reduced batch sizes and epochs
- **Educational Content**: Detailed explanations and assignment questions
- **Visualization**: Training history, performance comparison, and prediction analysis
- **Best Practices**: Proper preprocessing, callbacks, and model saving

## 🔧 Requirements

- Python 3.8+
- TensorFlow 2.13+
- TensorFlow Datasets
- NumPy
- Matplotlib
- Seaborn
- Pandas
- Jupyter Notebook

## 📊 Results Summary

### Oxford Flowers 102 Dataset Results

| Model      | Test Accuracy | Top-5 Accuracy | Parameters |
|------------|---------------|----------------|------------|
| ResNet50   | 81.27%       | 93.51%        | 23.8M     |
| MobileNetV2| 74.19%       | 89.22%        | 2.4M      |
| VGG16      | 51.67%       | 76.47%        | 14.8M     |

**Key Findings:**
- **ResNet50** achieved the best performance with 81.27% test accuracy
- Transfer learning enabled rapid convergence (5 epochs vs. training from scratch)
- All models significantly outperformed CIFAR-100 results due to higher resolution images
- Pre-trained ImageNet features proved highly relevant for flower classification

### CIFAR-100 Dataset Results

The CIFAR-100 notebook demonstrates transfer learning fundamentals with generally lower accuracies due to smaller 32×32 images, but shows the same performance ranking: ResNet50 > MobileNetV2 > VGG16.

## 📁 Project Structure

```
advanced-vision-ai-transfer-learning/
│
├── Fast-Tracking Image Classification with Transfer Learning.ipynb
├── Assignment_Oxford_Flowers_Transfer_Learning.ipynb
├── README.md
│
└── models/ (created during training)
    ├── best_resnet50_flowers.h5
    ├── best_vgg16_flowers.h5
    └── best_mobilenetv2_flowers.h5
```

## 🙏 Acknowledgments

- **TensorFlow Team** for the excellent deep learning framework
- **Oxford Visual Geometry Group** for the Flowers 102 dataset
- **University of Toronto** for the CIFAR datasets
- **Google Colab** for providing free GPU resources

## 📞 Contact
**Email** rohanpaswan001782gmail.com
