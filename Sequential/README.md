# Neural Network Models for Cancer Classification 🧠

## Overview 🔍
This notebook implements various neural network architectures using Keras for breast cancer classification. Deep learning models are built with increasing complexity to classify tumors as malignant or benign based on multiple features extracted from diagnostic images.

## Features ⭐
- Multiple sequential neural network architectures
- Implementation using Keras with TensorFlow backend
- Advanced techniques including:
  - Dropout regularization
  - LeakyReLU activation
  - Early stopping
- Model performance comparison
- Comprehensive visualization of training metrics

## Data 📊
The models use the same cleaned_data.csv dataset as other models in the repository, containing standardized features extracted from breast cancer cell images, with the `diagnosis` column as the target variable (1 for malignant, 0 for benign).

## Model Architectures 🏗️

### Basic Model (model)
- Single hidden layer with 16 neurons and ReLU activation
- Output layer with sigmoid activation for binary classification
- Trained with SGD optimizer (learning_rate=0.003)

### Two-Layer Model (model_2)
- Two hidden layers with 16 neurons each and ReLU activation
- Output layer with sigmoid activation
- Improved architecture for learning more complex patterns

### Advanced Model (model_3)
- Multiple implementations with:
  - Dense layers (64→32→1 neurons)
  - LeakyReLU activation (alpha=0.01)
  - Dropout layers (0.2) for regularization
  - Adam optimizer (learning_rate=0.001)
  - Early stopping based on validation loss
  - Batch processing with configurable batch sizes

## Implementation Steps 📋

### 1. Data Preparation
- Load the cancer dataset
- Split features (x) and target (y)
- Create training and testing sets (75%/25% split)

### 2. Model Building & Training
- Define sequential models with different architectures
- Configure loss function (binary_crossentropy) and metrics (accuracy)
- Train models with validation monitoring
- Implement early stopping to prevent overfitting

### 3. Evaluation & Visualization
- Calculate accuracy and ROC-AUC scores
- Plot training and validation metrics:
  - Loss curves to monitor convergence
  - Accuracy curves to track performance
  - Side-by-side comparisons for model assessment

## Performance Metrics 📏
Models are evaluated using:
- Accuracy score
- ROC-AUC score
- Training vs. validation loss comparisons
- Training vs. validation accuracy comparisons

## Results 🏆
The notebook showcases neural network performance for cancer classification, with the advanced model achieving high accuracy (~0.98) and ROC-AUC scores (~0.99), demonstrating the power of neural networks for this classification task.

## Requirements 📦
- Python 3.x
- TensorFlow/Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

## Usage 💻
1. Ensure dependencies are installed
2. Run cells sequentially
3. Experiment with different model architectures by adjusting:
   - Number of layers and neurons
   - Activation functions
   - Dropout rates
   - Batch sizes
   - Learning rates

