# 🏦 Bank Customer Churn Prediction

This project builds a supervised machine learning pipeline to predict bank customer churn using classical machine learning models and best practices in preprocessing, model tuning, and evaluation.

---

## 📁 Dataset

- `Bank Customer Churn Prediction.csv`
- Target variable: `churn`
- Categorical features: `country`, `gender`
- Numerical features: `credit_score`, `age`, `balance`, `estimated_salary`, `tenure`

---

## ✅ Objectives

1. **Preprocess the data**  
2. **Build and evaluate predictive models**  
3. **Draw conclusions from model results**  

---

## ⚙️ Preprocessing Steps

- Removed `customer_id`
- Checked for null values
- One-hot encoded categorical variables (`country`, `gender`)
- Dropped redundant dummy columns to avoid multicollinearity
- Applied `log1p` transformation to skewed `balance` column
- Scaled numerical features using `StandardScaler`

---

## 📊 Exploratory Data Analysis 

- Class distribution of `churn`: ~80% no churn, ~20% churn
- Histograms of numerical features grouped by `churn`
- Correlation heatmap of numeric features

---

## 🧠 Models Evaluated

- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest
- k-Nearest Neighbors (k-NN)

Each model was evaluated with:
- **StratifiedKFold Cross-Validation** (`cv=5`)
- **GridSearchCV** for hyperparameter tuning
- **Balanced class weights** to handle class imbalance

---

## 🔍 Model Tuning (Hyperparameters)

Example tuning setup for Random Forest:
```python
{
  'n_estimators': [1, 5, 10],
  'max_depth': [None, 10, 20, 30],
  'min_samples_split': [2, 5, 10],
  'min_samples_leaf': [1, 2, 4],
  'bootstrap': [True, False]
}
