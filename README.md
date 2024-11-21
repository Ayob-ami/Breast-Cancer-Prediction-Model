# Breast-Cancer-Prediction-Model

This project is a machine learning-based web application for predicting breast cancer diagnosis using cell nucleus features derived from the **Breast Cancer Wisconsin Dataset**. The app uses a **Support Vector Machine (SVM)** model for classification and is built with Flask for deployment.

---

## Table of Contents
- [Overview](#overview)
- [Model Training and Comparison](#model-training-and-comparison)
  - [Grid Search for Hyperparameter Tuning](#grid-search-for-hyperparameter-tuning)
  - [Model 1: Decision Tree](#model-1-decision-tree)
  - [Model 2: Random Forest](#model-2-random-forest)
  - [Model 3: Support Vector Machine (SVM)](#model-3-support-vector-machine-svm)
- [Web Application Deployment](#web-application-deployment)
- [Usage](#usage)
- [Acknowledgments](#acknowledgments)
- [License](#license)

---

## Overview

The Breast Cancer Prediction App accepts 30 diagnostic features as input and predicts whether a tumor is **Malignant** (cancerous) or **Benign** (non-cancerous). The prediction is accompanied by a brief explanation. This tool is designed to assist in research and should not be used as a substitute for professional medical advice.

---

## Model Training and Comparison

### Dataset
The **Breast Cancer Wisconsin Dataset** was used for training and evaluation. It contains:
- **569 samples**: Diagnoses of breast cancer tumors.
- **30 numerical features**: Derived from diagnostic imaging, such as mean radius, worst area, etc.
- **Target labels**:
  - `0`: Benign
  - `1`: Malignant

### Grid Search for Hyperparameter Tuning
Grid Search was applied to each model to tune the hyperparameters and improve their performance. **Cross-validation** was employed to ensure robust evaluation.

1. **Support Vector Machine (SVM)**:
   - **Best Parameters**: 
     - `C=1`, `coef0=0.5`, `degree=2`, `gamma='scale'`, `kernel='poly'`
   - **Cross-validation score**: ~97%
   - **Test Set Accuracy**: ~98%
   - **Test Set Precision**: ~98%
   - **Confusion Matrix**: 
     ```
     [[ 61   2]
      [  0 108]]
     ```
   - **Classification Report**:
     ```
                 precision    recall  f1-score   support
            0       1.00      0.97      0.98        63
            1       0.98      1.00      0.99       108
      accuracy                           0.99       171
     macro avg       0.99      0.98      0.99       171
     weighted avg       0.99      0.99      0.99       171
     ```

2. **Decision Tree**:
   - **Best Parameters**: 
     - `max_depth=8`, `max_leaf_nodes=9`, `min_samples_leaf=3`
   - **Cross-validation score**: ~93%
   - **Test Set Accuracy**: ~97%
   - **Test Set Precision**: ~98%
   - **Confusion Matrix**:
     ```
     [[ 61   2]
      [  3 105]]
     ```
   - **Classification Report**:
     ```
                 precision    recall  f1-score   support
            0       0.95      0.97      0.96        63
            1       0.98      0.97      0.98       108
      accuracy                           0.97       171
     macro avg       0.97      0.97      0.97       171
     weighted avg       0.97      0.97      0.97       171
     ```

3. **Random Forest**:
   - **Best Parameters**: 
     - `n_estimators=50`, `max_depth=20`, `max_samples=10`, `min_samples_leaf=1`
   - **Cross-validation score**: ~95%
   - **Test Set Accuracy**: ~94%
   - **Test Set Precision**: ~93%
   - **Confusion Matrix**:
     ```
     [[ 55   8]
      [  2 106]]
     ```
   - **Classification Report**:
     ```
                 precision    recall  f1-score   support
            0       0.96      0.87      0.92        63
            1       0.93      0.98      0.95       108
      accuracy                           0.94       171
     macro avg       0.95      0.93      0.94       171
     weighted avg       0.94      0.94      0.94       171
     ```

---

### Model 1: Decision Tree
- **Description**: A decision tree was implemented and tuned using Grid Search.
- **Performance**:
  - Cross-validation Accuracy: ~93%
  - Test Set Accuracy: ~97%
- **Issues**: Despite hyperparameter tuning with Grid Search, it performed worse than the SVM model in terms of overall accuracy and precision.

---

### Model 2: Random Forest
- **Description**: An ensemble method using multiple decision trees was used.
- **Performance**:
  - Cross-validation Accuracy: ~95%
  - Test Set Accuracy: ~94%
- **Strengths**: Random Forest performed better than the decision tree, providing good generalization.
- **Issues**: The performance was still lower than the **SVM model**, especially in precision.

---

### Model 3: Support Vector Machine (SVM)
- **Description**: SVM with a polynomial kernel was implemented and fine-tuned using Grid Search.
- **Performance**:
  - Cross-validation Accuracy: ~97%
  - Test Set Accuracy: ~98%
  - Test Set Precision: ~98%
- **Strengths**: The **SVM model** outperformed all other models, providing excellent accuracy and precision on both cross-validation and test sets.
- **Final Decision**: The **SVM model** was selected for deployment.

---

## Web Application Deployment

### Backend
The backend is implemented using **Flask**, which provides:
- A web interface for entering input features.
- An API endpoint (`/api/predict`) for programmatic access to the model.

### Deployment
The app can be deployed locally or on a hosting platform like Render or Heroku. It loads the pre-trained SVM model from a pickle file (`SVM_model.pkl`).

---

## Usage

### Features
- Accepts 30 numerical inputs (cell nucleus characteristics).
- Outputs:
  - **Malignant**: Indicates a cancerous tumor.
  - **Benign**: Indicates a non-cancerous tumor.
- Includes a brief explanation of the prediction.

---

### Acknowledgments
- **Dataset**: The Breast Cancer Wisconsin Dataset provided by the UCI Machine Learning Repository.
- **Libraries Used**:
   - Flask for backend development
   - scikit-learn for model training and prediction
   - numpy for numerical computations

### License

MIT License
