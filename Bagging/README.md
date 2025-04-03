# Bagging Model Documentation

## Overview
This notebook implements a Bagging (Bootstrap Aggregating) classifier for cancer classification, focusing on ensemble learning through multiple base estimators trained on different bootstrap samples. The implementation explores the effect of ensemble size and base estimator parameters on model performance.

## Features
- Implementation of Bagging algorithm with Decision Tree base estimators
- Analysis of ensemble size impact on model performance
- Hyperparameter optimization through grid search
- Class imbalance handling with SMOTE
- Comprehensive model evaluation

## Step-by-Step Workflow

### 1. Setup and Data Preparation
- Imports necessary libraries (pandas, numpy, scikit-learn, matplotlib)
- Loads the pre-cleaned cancer dataset
- Separates features (X) and target variable (y)
- Sets random seed for reproducibility
- Splits data into training and testing sets with stratification

### 2. Initial Model Building
- Creates a Bagging classifier with Decision Tree base estimators
- Configures Decision Trees with entropy criterion, max depth of 5, and min samples leaf of 1
- Sets the number of estimators to 40 with bootstrap sampling
- Trains the model on the training data
- Creates a custom evaluation function to calculate metrics

### 3. Performance Analysis
- Implements a function to analyze model performance with varying ensemble sizes
- Tests ensemble sizes from 1 to 69 estimators
- Averages results over 20 iterations for robustness
- Visualizes training and testing accuracy trends as the ensemble size increases

### 4. Hyperparameter Tuning
- Performs grid search across multiple parameters:
  - Number of estimators (1 to 79, odd values)
  - Decision Tree max depth (1 to 39, odd values)
- Uses 5-fold cross-validation with accuracy scoring
- Identifies optimal hyperparameters for the ensemble

### 5. Model Optimization
- Creates an optimized model with the best parameters from grid search
- Evaluates the optimized model on test data

### 6. Class Imbalance Handling
- Applies SMOTE to address class imbalance in the training data
- Configures an optimized Bagging classifier with the best parameters
- Retrains the model on the balanced data
- Performs final evaluation on both training and test sets

## Model Parameters
- **n_estimators**: 21 (optimized through grid search)
- **base_estimator**: Decision Tree with:
  - criterion: entropy
  - max_depth: 5
  - min_samples_leaf: 1
- **bootstrap**: True (sampling with replacement)

## Performance Metrics
The model is evaluated using:
- Accuracy
- Precision
- Recall
- F-score (with beta=5 to emphasize recall for cancer detection)

## Results
The notebook provides:
- Analysis of how ensemble size affects model performance
- Grid search results identifying optimal hyperparameters
- Comparison of model performance before and after SMOTE
- Final evaluation metrics on both training and test sets

## Usage Instructions
1. Ensure the cleaned dataset is available in the specified path
2. Run all cells sequentially
3. Examine the performance trend plot to understand ensemble size effects
4. Review grid search results to understand hyperparameter selection
5. Analyze the final model performance metrics

## Key Insights
- Bagging reduces variance in Decision Tree models, improving generalization
- Ensemble size significantly affects model performance up to a certain point
- The optimized model achieves high accuracy while maintaining good balance between precision and recall
- Class balancing with SMOTE improves detection of malignant cases
