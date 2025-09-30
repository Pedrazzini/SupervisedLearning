# Food Image Classification Project
A comprehensive machine learning project for classifying food images using both deep learning (CNN) and traditional computer vision approaches (SIFT + Bag-of-Words + SVM).

## Overview
This project implements and compares two approaches for multi-class food image classification on a dataset with 251 classes:

- Deep Learning: Convolutional Neural Network (CNN)
- Classical Computer Vision: SIFT features with Bag-of-Words representation and Support Vector Machine classifier

## Dataset

Total Classes: 251 food categories
- Training Set: Images organized by class with labels from [this dataset](https://github.com/karansikka1/iFood_2019?tab=readme-ov-file) (train_labels.csv)
- Validation Set: 20% split from training data
- Test Set: Separate test images with labels from val_labels.csv
Class Distribution: Between 100-600 images per class (with augmentation applied to balance)

## Project Structure

├── Preprocessing_supervised_def.ipynb    # Data preprocessing and augmentation

├── processing-def.ipynb                  # CNN training and evaluation

├── sift-bow-svm-final2.ipynb            # SIFT+BoW+SVM implementation

└── grid-search-def.ipynb                # Hyperparameter tuning for SVM
 

## Methodology

### 1. Data Preprocessing
Image Organization (Preprocessing_supervised_def.ipynb)

Automatic folder structure creation for each class
Image distribution: 80% training, 20% validation

#### Data Augmentation
Applied to classes with insufficient samples (<300 images):

- Rotation: Random angles (0-180°)
- Flipping: Horizontal, vertical, and both
- Brightness adjustment: Both increase and decrease

- Special augmentation for class 162 (<100 images):

2 different rotations
3 types of flipping
Brightness modifications
Target: ~270 images per class minimum


### 2. Deep Learning Approach (CNN)
Architecture (processing-def.ipynb)

Input (128×128×3)

├── Conv2D(8 filters, 5×5) + LeakyReLU + MaxPool

├── Conv2D(12 filters, 4×4) + LeakyReLU + MaxPool

├── Conv2D(16 filters, 3×3) + LeakyReLU + MaxPool

├── Conv2D(30 filters, 2×2) + LeakyReLU + MaxPool

├── Flatten

├── FC(1470 → 550) + LeakyReLU + Dropout(0.5)

└── FC(550 → 251)

## Training Configuration

- Loss Function: CrossEntropyLoss
- Optimizer: SGD with momentum (0.9)
- Learning Rate: 0.01 with Cosine Annealing
- Scheduler: CosineAnnealingLR (min_lr=1e-8)
- Epochs: 15
- Batch Size: 64
- Image Size: 128×128
- Normalization: Computed dataset-specific mean and std

### Features

- Leaky ReLU activation for better gradient flow
- Dropout regularization (50%)
- Best model selection based on validation loss
- Training/validation loss visualization

### 3. Classical Computer Vision Approach
Pipeline (sift-bow-svm-final2.ipynb)

#### 1. Feature Extraction: SIFT (Scale-Invariant Feature Transform)

- Images resized to 224×224
- Converted to grayscale
- SIFT descriptors extracted from each image


#### 2. Feature Representation: Bag-of-Words (BoW)

- K-means clustering on SIFT descriptors
- Vocabulary size: G = 380 words
- Histogram representation for each image


#### 3. Classification: Support Vector Machine (SVM)

- Kernel: RBF
- Regularization parameter: C = 30
- Training on 30% of dataset (computational efficiency)



#### 4. Hyperparameter Tuning (grid-search-def.ipynb)

- 5-fold cross-validation
Parameters tested:

- C: [20, 25, 30, 35, 40, 45]
- Kernel: ['rbf', 'poly']


- Training subset: 10% of data for efficiency


## Evaluation Metrics

- Overall accuracy on test set
- Per-class accuracy analysis
- Confusion matrix for specific class pairs
- Training/validation loss curves

# Usage
### 1. Preprocessing
```
Run preprocessing notebook
jupyter notebook Preprocessing_supervised_def.ipynb
```

### 2. CNN Training
```
Train CNN model
jupyter notebook processing-def.ipynb
```
### 3. SIFT+BoW+SVM
```
Run traditional CV approach
jupyter notebook sift-bow-svm-final2.ipynb
```
### 4. Hyperparameter Tuning
```
Optimize SVM parameters
jupyter notebook grid-search-def.ipynb
```

## Authors

- [Ernesto Pedrazzini](https://github.com/Pedrazzini)
- [Viola Cavedoni](https://github.com/violacave)