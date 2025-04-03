# Exploratory Data Analysis (EDA) Documentation 📊

## Overview 🔍
This notebook performs exploratory data analysis on the cancer dataset to understand the data structure, clean the data, and prepare it for machine learning model training.

## Features ⭐
- Data loading and initial examination
- Data cleaning and preprocessing
- Statistical analysis
- Feature distribution analysis
- Correlation analysis
- Outlier detection and handling
- Visualization of data patterns

## Step-by-Step Workflow 📋

### 1. Data Loading and Initial Exploration 📥
- Imports necessary Python libraries (pandas, numpy, matplotlib, seaborn)
- Loads the cancer dataset from 'data.csv'
- Displays the first few rows for data familiarization
- Lists all columns in the dataset

### 2. Data Cleaning 🧹
- Removes unnecessary columns ('Unnamed: 32', 'id')
- Checks for missing values
- Identifies and counts duplicate entries

### 3. Statistical Analysis 📈
- Generates descriptive statistics of the dataset
- Converts diagnosis labels from categorical (M/B) to numerical (1/0)
- Examines class distribution of the target variable

### 4. Data Visualization 📊
- Creates bar plots showing class distribution
- Generates histograms for feature distributions
- Creates pair plots to show relationships between features
- Builds correlation matrix heatmap to identify highly correlated features

### 5. Data Preprocessing ⚙️
- Standardizes features using StandardScaler
- Identifies outliers using the IQR method
- Creates a cleaned dataset with outliers removed

### 6. Feature Analysis 🔍
- Calculates correlations between features and the target variable
- Identifies the most important features for prediction

### 7. Data Export 📤
- Saves the cleaned and preprocessed dataset to "cleaned_data.csv"

## Usage Instructions 📝

1. Ensure you have the required Python libraries installed
2. Place your 'data.csv' file in the same directory as this notebook
3. Execute the notebook cells in sequence
4. Analyze the generated visualizations and statistics
5. Use the cleaned dataset for model training

## Key Insights 💡

- The correlation matrix reveals which features are highly related
- Feature distributions show the spread and central tendency of measurements
- Outlier analysis helps identify abnormal data points that may affect model performance
- The class distribution analysis shows whether the dataset is balanced or imbalanced

## Output 📦
The notebook outputs a cleaned dataset ("cleaned_data.csv") that can be used directly for model training.
