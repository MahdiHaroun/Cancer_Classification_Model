# Decision Tree (DT) Model Documentation

## Overview
This notebook implements a Decision Tree classifier for cancer classification, with a focus on hyperparameter tuning and visualization of the decision-making process. The model is optimized to improve prediction accuracy and interpretability.

## Features
- Implementation of Decision Tree algorithm with different splitting criteria
- Tree visualization for model interpretability
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
- Creates a basic Decision Tree classifier with default parameters
- Trains the model on the training data
- Creates a custom evaluation function to calculate metrics
- Evaluates initial model performance

### 3. Decision Tree Visualization
- Implements a function to visualize the decision tree structure
- Displays the tree with feature names for interpretability
- Analyzes the decision paths and node criteria

### 4. Custom Model Experimentation
- Tests different combinations of hyperparameters:
  - Different splitting criteria (gini, entropy)
  - Various maximum depths (10, 15)
  - Different minimum samples per leaf (3, 5)
- Evaluates each model's performance

### 5. Hyperparameter Tuning
- Performs grid search across multiple parameters:
  - Criterion (gini, entropy)
  - Max depth (5, 10, 15, 20)
  - Min samples per leaf (1, 2, 5)
- Uses 5-fold cross-validation with F1 score optimization
- Identifies optimal hyperparameters

### 6. Model Optimization
- Creates an optimized model with the best parameters from grid search
- Evaluates the optimized model on test data

### 7. Class Imbalance Handling
- Applies SMOTE to address class imbalance in the training data
- Retrains the optimized model on the balanced data
- Performs final evaluation on both training and test sets

## Model Parameters
- **Criterion**: Entropy (optimized through grid search)
- **Max Depth**: 5 (optimized to prevent overfitting)
- **Min Samples Leaf**: 1 (optimized through grid search)

## Performance Metrics
The model is evaluated using:
- Accuracy
- Precision
- Recall
- F-score (with beta=5 to emphasize recall for cancer detection)

## Results
The notebook provides:
- Comparison of model performance with different hyperparameter settings
- Visualizations of the decision tree structure
- Performance before and after handling class imbalance
- Training vs. test set performance analysis

## Usage Instructions
1. Ensure the cleaned dataset is available in the specified path
2. Run all cells sequentially
3. Examine the decision tree visualizations to understand the model's decision process
4. Review the grid search results to understand hyperparameter selection
5. Analyze the final model performance metrics

## Key Insights
- Decision Trees provide an interpretable model for cancer classification
- Optimal tree depth prevents overfitting while maintaining good predictive performance
- Entropy criterion works better than Gini for this classification task
- Balancing the classes with SMOTE improves model sensitivity to malignant cases
