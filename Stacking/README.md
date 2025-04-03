# Stacking Model Documentation

## Overview
This notebook implements a Stacking ensemble classifier for cancer classification, combining the strengths of multiple base models through a meta-learning approach. The implementation explores different combinations of base estimators and meta-classifiers to achieve optimal performance.

## Features
- Implementation of Stacking ensemble learning architecture
- Combination of diverse base models (SVM, KNN, Decision Tree/Logistic Regression)
- Meta-level learning with Logistic Regression and Decision Tree
- Cross-validation during the stacking process
- Comprehensive model evaluation

## Step-by-Step Workflow

### 1. Setup and Data Preparation
- Imports necessary libraries (pandas, numpy, scikit-learn, matplotlib, seaborn)
- Loads the pre-cleaned cancer dataset
- Separates features (X) and target variable (y)
- Splits data into training and testing sets with stratification

### 2. Base Estimator Configuration
- Configures optimized base models from previous experiments:
  - Support Vector Machine (SVM) with RBF kernel
  - K-Nearest Neighbors (KNN) with Manhattan distance and k=2
  - Decision Tree classifier (initial configuration)
- Combines these models into an array of base estimators

### 3. First Stacking Ensemble
- Creates a Stacking classifier with Logistic Regression as meta-classifier
- Configures cross-validation (k=3) for the stacking process
- Sets parallel processing for efficiency
- Trains the stacked model on the training data
- Makes predictions on the test set
- Evaluates model performance using custom metrics function

### 4. Alternative Stacking Configuration
- Modifies the base estimator set:
  - Replaces Decision Tree with Logistic Regression in the base estimators
  - Changes meta-classifier to Decision Tree
- Trains this alternative stacked model
- Evaluates and compares performance with the first configuration

### 5. Model Evaluation
- Analyzes accuracy, precision, recall, and F1-score for both configurations
- Identifies the best-performing stacking architecture

## Model Parameters

### Base Estimators
- **SVM**: 
  - kernel: RBF
  - C: 1.0
  - gamma: scale
- **KNN**:
  - n_neighbors: 2
  - metric: manhattan
  - weights: uniform
- **Decision Tree/Logistic Regression**: Default parameters

### Meta-Classifier
- **Logistic Regression**:
  - C: 1.0
  - solver: liblinear
  - penalty: l2
  - max_iter: 100
- **Decision Tree**: Default parameters

### Stacking Parameters
- **cv**: 3 (number of cross-validation folds)
- **n_jobs**: -1 (utilize all available CPU cores)

## Performance Metrics
The model is evaluated using:
- Accuracy
- Precision
- Recall
- F1-score

## Results
The notebook provides:
- Performance comparison between different stacking configurations
- Analysis of how meta-learner choice affects ensemble performance
- Insights into the effectiveness of combining different algorithm types

## Usage Instructions
1. Ensure the cleaned dataset is available in the specified path
2. Run all cells sequentially
3. Review the performance metrics for both stacking configurations
4. Select the best-performing configuration for deployment

## Key Insights
- Stacking combines the strengths of multiple algorithms, often outperforming individual models
- The choice of meta-learner significantly impacts overall performance
- Cross-validation during stacking helps prevent overfitting
- The model leverages optimized base classifiers identified in previous experiments
- Models with different learning approaches complement each other when combined
