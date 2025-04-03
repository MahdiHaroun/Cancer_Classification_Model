# Random Forest (RF) Model Documentation

## Overview
This notebook implements a Random Forest classifier for cancer classification, leveraging the power of ensemble learning through multiple decision trees. The implementation focuses on optimizing forest parameters and feature selection to achieve high prediction accuracy.

## Features
- Implementation of Random Forest algorithm with optimized parameters
- Automatic feature importance assessment
- Hyperparameter optimization through grid search
- Class imbalance handling with SMOTE
- Comprehensive model evaluation

## Step-by-Step Workflow

### 1. Setup and Data Preparation
- Imports necessary libraries (pandas, numpy, scikit-learn, matplotlib)
- Loads the pre-cleaned cancer dataset
- Separates features (X) and target variable (y)
- Splits data into training and testing sets with stratification

### 2. Initial Model Configuration
- Sets the number of estimators (trees) to 20
- Calculates optimal max_features parameter based on dataset dimensions
- Implements square root heuristic for feature selection at each split
- Creates a custom evaluation function to calculate metrics

### 3. Initial Model Building
- Creates a Random Forest classifier with the specified parameters
- Trains the model on the training data
- Makes predictions on the test set
- Evaluates initial model performance

### 4. Hyperparameter Tuning
- Performs grid search across multiple parameters:
  - Number of estimators (odd values from 1 to 39)
  - Maximum depth of trees (odd values from 1 to 19)
  - Feature selection methods ("auto", "sqrt", "log2")
- Uses cross-validation with accuracy scoring
- Identifies optimal hyperparameters for the ensemble

### 5. Model Optimization
- Retrieves best hyperparameters from grid search
- Creates an optimized model with these parameters
- Evaluates the optimized model on test data

### 6. Class Imbalance Handling
- Applies SMOTE to address class imbalance in the training data
- Configures optimized Random Forest with best parameters
- Retrains the model on the balanced data
- Performs final evaluation on both training and test sets

## Model Parameters
- **n_estimators**: Number of trees in the forest (optimized through grid search)
- **max_features**: Number of features to consider at each split (optimized through grid search)
- **max_depth**: Maximum depth of the trees (optimized through grid search)
- **random_state**: 123 (for reproducibility)

## Performance Metrics
The model is evaluated using:
- Accuracy
- Precision
- Recall
- F-score (with beta=5 to emphasize recall for cancer detection)

## Results
The notebook provides:
- Performance comparison before and after hyperparameter tuning
- Grid search results identifying optimal parameters
- Comparison of model performance before and after SMOTE
- Final evaluation metrics on both training and test sets

## Usage Instructions
1. Ensure the cleaned dataset is available in the specified path
2. Run all cells sequentially
3. Review the performance of the initial model
4. Examine grid search results to understand optimal parameters
5. Analyze the final model performance metrics

## Key Insights
- Random Forest provides robust performance for cancer classification
- The ensemble approach reduces overfitting compared to single decision trees
- Feature selection methodology significantly impacts model performance
- Class balancing with SMOTE improves detection sensitivity for malignant cases
