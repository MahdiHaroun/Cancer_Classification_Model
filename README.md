# Cancer Classification Model 🚀

## Project Overview 🔬

This repository contains machine learning models designed for cancer classification using various algorithms including Logistic Regression, K-Nearest Neighbors (KNN), Support Vector Machines (SVM), Random Forest, AdaBoost, Bagging, XGBoost, and Stacking. The models analyze various features from a cancer dataset to classify samples as malignant or benign.

## Setup and Installation ⚙️

### Prerequisites
- Python 3.8+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- imbalanced-learn (for SMOTE)

### Installation
```bash
git clone https://github.com/[username]/Cancer_Classification_Model.git
cd Cancer_Classification_Model
pip install -r requirements.txt
```

## Data Description 📊

The dataset contains features extracted from breast cancer cell nuclei, including:
- Radius
- Texture
- Perimeter
- Area
- Smoothness
- And other characteristics

Each sample is classified as malignant (1) or benign (0).

## Directory Structure 📁

```
Cancer_Classification_Model/
├── EDA.ipynb                  # Exploratory Data Analysis
├── README.md                  # Project documentation
├── Logistic/                  # Logistic Regression implementation
│   └── Model.ipynb
├── KNN/                       # K-Nearest Neighbors implementation  
│   └── Model.ipynb
├── SVM/                       # Support Vector Machine implementation
│   └── Model.ipynb
├── RF/                        # Random Forest implementation
│   └── Model.ipynb
├── ADA/                       # AdaBoost implementation
│   └── Model.ipynb
├── Bagging/                   # Bagging implementation
│   └── Model.ipynb
├── XGBoost/                   # XGBoost implementation
│   └── Model.ipynb
└── Stacking/                  # Stacking implementation
    └── Model.ipynb
```

## Models Implemented 🤖

### Logistic Regression 📈
- Implementation of both L1 and L2 regularization
- Hyperparameter tuning via GridSearchCV
- Pipeline integration for workflow management

### K-Nearest Neighbors 🧮
- Different distance metrics (Euclidean, Manhattan, Minkowski)
- Various weighting schemes (uniform, distance)
- Hyperparameter optimization for k value

### Support Vector Machine 🔍
- Linear and non-linear kernels
- Hyperparameter tuning for C and gamma parameters
- Class imbalance handling

### Random Forest 🌲
- Ensemble of decision trees
- Feature importance analysis
- Bootstrapping and random feature selection
- Hyperparameter optimization for number of trees and tree depth

### AdaBoost 🔄
- Adaptive boosting algorithm
- Sequential learning of weak classifiers
- Weight adjustment for misclassified samples
- Hyperparameter tuning for learning rate and number of estimators

### Bagging 📦
- Bootstrap aggregating technique
- Parallel ensemble of base classifiers
- Reduction of variance in the model
- Customizable base estimator selection

### XGBoost 🚀
- Gradient boosting implementation with regularization
- Early stopping to prevent overfitting
- Efficient handling of sparse data
- Advanced hyperparameter tuning

### Stacking 🏗️
- Meta-ensemble learning approach
- Multiple base models with a meta-classifier
- Cross-validation for meta-model training
- Diverse base model selection for robust predictions

## Data Preprocessing 🧹

- Feature standardization/scaling
- Outlier detection and handling
- Checking for missing values
- Correlation analysis
- SMOTE for handling class imbalance

## Evaluation Metrics 📏

The models are evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

## Usage Instructions 📋

### Exploratory Data Analysis
```python
# Run the EDA notebook
jupyter notebook EDA.ipynb
```

### Model Training and Evaluation
```python
# Run the respective model notebook
jupyter notebook [Model_Type]/Model.ipynb  # where Model_Type is one of: Logistic, KNN, SVM, RF, ADA, Bagging, XGBoost, Stacking
```

## Results and Comparison 🏆

Each model's performance is evaluated on both training and testing datasets to compare accuracy, precision, recall, and F1-scores. The evaluation helps determine the best model for cancer classification based on these metrics.

Key performance highlights:

- **KNN** 🥇 achieved the highest performance with 98% accuracy and 100% precision without balancing the dataset, and maintained 98% accuracy with 98% precision after balancing using SMOTE
- **AdaBoost** 🥈 performed strongly with 97% accuracy and 97% precision
- **XGBoost** 🥉 delivered robust results with 97% accuracy and 95% precision
- Other models showed varying performance levels across metrics, with ensemble methods generally outperforming individual classifiers

## Best Practices Used ✅

- Cross-validation for robust model evaluation
- Pipeline architecture to prevent data leakage
- Hyperparameter optimization using GridSearchCV
- Proper train-test splitting
- Handling class imbalance with SMOTE

## Future Improvements 🔮

- Deep learning approaches

## Source 📑
[Cancer Dataset](https://www.kaggle.com/datasets/erdemtaha/cancer-data)


