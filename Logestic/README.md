# Logistic Regression Model Documentation 📈

## Overview 🔍
This notebook implements a Logistic Regression classifier for cancer classification, focusing on different regularization techniques (L1 and L2) and hyperparameter optimization. The model achieves a balance between simplicity and performance.

## Features ⭐
- Implementation of L1 (Lasso) and L2 (Ridge) regularization techniques
- Pipeline architecture for streamlined model development
- Hyperparameter tuning through GridSearchCV
- Visualization of regularization strength effects
- Class imbalance handling with SMOTE
- Comprehensive model evaluation

## Step-by-Step Workflow 📋

### 1. Setup and Data Preparation 📥
- Imports necessary libraries (pandas, numpy, scikit-learn, matplotlib, seaborn)
- Sets random seed for reproducibility
- Loads the pre-cleaned cancer dataset
- Separates features (X) and target variable (y)
- Splits data into training and testing sets with stratification
- Confirms dataset dimensions

### 2. L2 Regularization Implementation 🔢
- Configures initial Logistic Regression with L2 penalty and 'lbfgs' solver
- Sets regularization strength (C=1.0) and maximum iterations
- Trains the model on the training data
- Creates a custom evaluation function to calculate metrics
- Evaluates model performance on test data

### 3. L1 Regularization Implementation 🔢
- Reconfigures Logistic Regression with L1 penalty and 'liblinear' solver
- Maintains similar regularization strength and iterations
- Trains and evaluates the model
- Compares performance against L2 regularization

### 4. Hyperparameter Tuning ⚙️
- Implements a scikit-learn pipeline for model development
- Performs grid search across multiple parameters:
  - Regularization strength (C values: 0.001 to 10)
  - Penalty types (L1, L2)
  - Maximum iterations (100 to 1000)
- Uses 3-fold cross-validation with accuracy scoring
- Identifies optimal hyperparameters
- Evaluates the best model on test data

### 5. Regularization Analysis 📊
- Tests a range of C values (0.001 to 100) to visualize regularization effects
- Calculates F1 score, precision, and recall for each value
- Plots performance metrics against regularization strength
- Identifies optimal regularization strength

### 6. Model Evaluation 📏
- Generates confusion matrix to visualize classification results
- Analyzes true positives, false positives, true negatives, and false negatives
- Evaluates model on both training and test sets

### 7. Class Imbalance Handling ⚖️
- Applies SMOTE to address class imbalance in the training data
- Visualizes the balanced class distribution
- Retrains the model on the balanced data
- Performs final evaluation with emphasis on malignant case detection

## Model Parameters ⚙️
- **penalty**: L1 or L2 (determined through grid search)
- **C**: Regularization strength (optimized through analysis)
- **solver**: liblinear (compatible with both L1 and L2)
- **max_iter**: Model convergence parameter (200 iterations)

## Performance Metrics 📏
The model is evaluated using:
- Accuracy
- Precision
- Recall
- F-score (with beta=5 to emphasize recall for cancer detection)
- Confusion Matrix

## Results 🏆
The notebook provides:
- Comparison of L1 vs. L2 regularization performance
- Analysis of regularization strength impact on model metrics
- Visualization of model performance across different configurations
- Final performance metrics on both training and test sets

## Usage Instructions 📝
1. Ensure the cleaned dataset is available in the specified path
2. Run all cells sequentially
3. Examine the regularization effects plot to understand parameter impact
4. Review the grid search results to understand hyperparameter selection
5. Analyze the final model performance metrics

## Key Insights 💡
- Logistic Regression provides a good baseline model for cancer classification
- Regularization strength significantly impacts model performance
- Pipeline architecture streamlines the modeling process
- The model achieves improved sensitivity to malignant cases after SMOTE application
