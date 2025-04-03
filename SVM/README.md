# Support Vector Machine (SVM) Model Documentation

## Overview
This notebook implements a Support Vector Machine classifier for cancer classification, focusing on different kernel types and hyperparameter tuning. The implementation explores the impact of regularization and kernel functions to achieve optimal decision boundaries for classification.

## Features
- Implementation of SVM with various kernel functions
- Exploration of regularization strength (C parameter)
- Hyperparameter optimization through grid search
- Class imbalance handling with SMOTE
- Comprehensive model evaluation

## Step-by-Step Workflow

### 1. Setup and Data Preparation
- Imports necessary libraries (pandas, numpy, scikit-learn, matplotlib, seaborn)
- Sets random seed for reproducibility
- Loads the pre-cleaned cancer dataset
- Separates features (X) and target variable (y)
- Splits data into training and testing sets with stratification

### 2. Initial Model Building
- Creates a basic SVM classifier with default parameters
- Trains the model on the training data
- Creates a custom evaluation function to calculate metrics
- Evaluates initial model performance

### 3. Model Parameter Exploration
- Examines default parameters of SVM implementation
- Tests SVM with RBF kernel and higher C value (C=10)
- Experiments with sigmoid kernel and higher regularization (C=20)
- Compares performance across different configurations

### 4. Hyperparameter Tuning
- Performs grid search across multiple parameters:
  - Regularization strength (C values: 1, 10, 100)
  - Kernel types (polynomial, RBF, sigmoid)
- Uses 5-fold cross-validation with F1 score optimization
- Identifies optimal hyperparameters (kernel type and C value)

### 5. Model Optimization
- Creates an optimized SVM with best parameters from grid search
- Evaluates the optimized model on test data
- Analyzes performance improvement over baseline model

### 6. Class Imbalance Handling
- Applies SMOTE to address class imbalance in the training data
- Retrains the optimized model on the balanced data
- Performs final evaluation on both training and test sets
- Uses beta=5 for F-score to emphasize recall for cancer detection

## Model Parameters
- **C**: Regularization parameter (controls trade-off between smooth decision boundary and classifying training points correctly)
- **kernel**: Function type that transforms input space (options: rbf, sigmoid, poly)
- **gamma**: Kernel coefficient (default: 'scale')

## Performance Metrics
The model is evaluated using:
- Accuracy
- Precision
- Recall
- F-score (with beta=5 to emphasize recall for cancer detection)

## Results
The notebook provides:
- Performance comparison across different kernel types
- Analysis of regularization strength impact
- Grid search results identifying optimal parameters
- Final evaluation metrics on both training and test sets after SMOTE application

## Usage Instructions
1. Ensure the cleaned dataset is available in the specified path
2. Run all cells sequentially
3. Review the performance of different SVM configurations
4. Examine grid search results to understand optimal parameters
5. Analyze the final model performance metrics after class balancing

## Key Insights
- SVM performance varies significantly with kernel selection and regularization strength
- RBF kernel typically outperforms other kernels for this cancer classification task
- Proper regularization prevents overfitting while maintaining good classification performance
- Class balancing with SMOTE improves sensitivity to malignant cases
- The optimized SVM model achieves high accuracy and good balance between precision and recall
