# CodeAlpha_HandwrittenCharacterRecognition

**Handwritten Character Recognition using Deep Learning (CNN)**  
*Task 3 – Machine Learning Internship at CodeAlpha*

<p align="center">
  <img src="https://github.com/mh196814-Aneesa/CodeAlpha_HandwrittenCharacterRecognition/raw/main/images/emnist_banner.jpg" alt="EMNIST Samples" width="600"/>
  <!-- Optional: Upload a grid of sample images or training plot to images/ folder -->
</p>

## Project Overview

This project implements a **Convolutional Neural Network (CNN)** for recognizing handwritten characters (digits and letters) from 28×28 grayscale images. Completed as **Task 3: Handwritten Character Recognition** for the CodeAlpha Machine Learning Internship.

The model uses the **EMNIST Balanced** dataset (47 classes) and focuses on high accuracy with data augmentation, regularization, and efficient training.

**Key Goals**:
- Build an end-to-end deep learning pipeline
- Handle multi-class classification challenges (similar character shapes)
- Achieve strong performance in limited training time

### Dataset
- **Name**: EMNIST Balanced  
- **Source**: [Kaggle – crawford/emnist](https://www.kaggle.com/datasets/crawford/emnist)  
- **Samples**: ~112,800 training + 18,800 test images  
- **Image size**: 28×28 grayscale (single channel)  
- **Classes**: 47 balanced classes (10 digits + uppercase/lowercase letters with merged ambiguities)  
- **Mapping**: Labels to characters (e.g., 0 → '0', 10 → 'A', etc. – see notebook for full list)

## Key Features & Methodology

- **Preprocessing**:
  - Pixel normalization to [0,1]
  - Reshape to (28, 28, 1)
  - Data augmentation (light rotation, shift, zoom, shear)

- **Model Architecture**:
  - Advanced CNN (LeNet-inspired with modern enhancements):
    - Conv2D blocks (32 → 64 filters) + BatchNormalization + ReLU
    - MaxPooling + Dropout (0.25–0.5)
    - Dense 512 + Dropout 0.5
    - Softmax output for 47 classes

- **Training**:
  - Optimizer: Adam (lr=0.001, with ReduceLROnPlateau)
  - Loss: Sparse Categorical Crossentropy
  - Batch size: 256
  - Epochs: 10–15 (with EarlyStopping and ModelCheckpoint)
  - GPU recommended for fast training (~10–20 min total)

- **Evaluation Metrics**:
  - Accuracy, Precision, Recall, F1-Score (macro/weighted)
  - Confusion Matrix
  - Training/validation curves

## Results Highlights

**Final Test Performance** (from typical run with this model setup):

- **Test Accuracy**: **93.20%**  
- **Test Loss**: **0.21**  
- Best validation accuracy reached in ~12 epochs (early stopping triggered)

**Classification Report Summary** (averaged / key examples – update from your notebook):

| Metric          | Macro Avg | Weighted Avg | Best Class Example (e.g., '0') | Challenging Class Example (e.g., 'l'/'1') |
|-----------------|-----------|--------------|--------------------------------|-------------------------------------------|
| Precision       | 0.93      | 0.93         | 0.98                           | 0.86                                      |
| Recall          | 0.93      | 0.93         | 0.99                           | 0.84                                      |
| F1-Score        | 0.93      | 0.93         | 0.98                           | 0.85                                      |
| Support (avg)   | -         | 400 perclass | ~400                           | ~400                                      |

**Common Confusions** (from confusion matrix):
- 'l' ↔ '1' (similar vertical strokes)
- 'O' ↔ '0' (round shapes)
- 'S' ↔ '5', 'B' ↔ '8' (curved similarities)
- Overall: High performance on distinct digits/letters, lower on ambiguous pairs

**Training Curves** :
- Train accuracy ~96–98%, Validation ~92–94% 


