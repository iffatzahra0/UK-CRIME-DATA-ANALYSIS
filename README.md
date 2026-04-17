# UK Crime Data Analysis

## Overview
This project performs multi-class classification on UK crime data using machine learning techniques. The goal is to predict crime types based on spatial and temporal features. The workflow includes data preprocessing, feature engineering, class balancing, model training, and performance evaluation.

## Dataset
- Source: Kaggle (UK Crime Data – Oct–Dec 2022)
- Records used: ~300,000
- Final features: 13
- Target classes: 11 crime categories (top crimes + "other-crime")

## Workflow

### 1. Data Preprocessing
- Removed non-predictive columns (IDs, source files)
- Dropped high-cardinality text features (e.g., street names)
- Consolidated rare crime types into `other-crime`
- Handled missing values:
  - Numerical → median
  - Categorical → mode

### 2. Feature Engineering
- Extracted:
  - Month number
  - Quarter
- Created location-based features:
  - Latitude/Longitude binning
  - `location_zone`

### 3. Encoding & Scaling
- Label encoding for categorical features and target
- Standardization using `StandardScaler`

### 4. Train-Test Split
- 80% training, 20% testing
- Stratified sampling to preserve class distribution

### 5. Handling Class Imbalance
- Applied SMOTE on training data to balance classes

### 6. Dimensionality Reduction
- PCA (2 components) for visualization and analysis

## Models Used

### Supervised Learning
- K-Nearest Neighbors (KNN)
- Logistic Regression
- Logistic Regression (L1 Regularization)
- Logistic Regression (L2 Regularization)

### Unsupervised Learning
- K-Means Clustering (for pattern analysis)

## Hyperparameters

- **KNN**
  - n_neighbors = 7
  - weights = distance
  - metric = manhattan

- **Logistic Regression**
  - C = 10
  - max_iter = 2000
  - solver = saga

- **Regularized Models**
  - L1 (Lasso), L2 (Ridge)
  - C = 1.0

## Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- ROC-AUC Curve
- Precision-Recall Curve

## Results

| Model                         | Accuracy | F1-Score |
|------------------------------|----------|----------|
| K-Nearest Neighbors          | 91.58%   | 0.9140   |
| Logistic Regression          | 89.55%   | 0.8918   |
| Logistic Regression (L2)     | 89.44%   | 0.8908   |
| Logistic Regression (L1)     | 89.57%   | 0.8920   |

**Best Model:** K-Nearest Neighbors

## Visual Outputs
The project generates:
- Class distribution plots
- Scaling comparison
- SMOTE effect visualization
- PCA projection
- K-Means clustering plots
- Performance comparison charts
- Confusion matrices
- ROC curves
- Precision-recall curves
- Bias-variance analysis
# view zip file for all outputs
## Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Matplotlib, Seaborn

# made using google collab
