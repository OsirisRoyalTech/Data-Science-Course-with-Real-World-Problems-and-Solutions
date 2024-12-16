# Module 3: Classification - Predicting Customer Churn
# Problem 3: Predicting Customer Churn (Classification Problem)
# Dataset: Telco customer churn data (e.g., Kaggle’s "Telco Customer Churn")
"""
Objective: Build a classification model to predict whether a customer will churn
based on their usage patterns, account information, etc.
"""
"""
Tasks:
•	Preprocess data: Handle categorical features, missing values, and feature scaling.
•	Build and evaluate classification models (Logistic Regression, Random Forest, SVM).
•	Use metrics like accuracy, precision, recall, and ROC-AUC.
"""

# Source Code:
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

# Load dataset
df = pd.read_csv('telco_churn.csv')

# Data preprocessing: Encode categorical variables
df = pd.get_dummies(df, drop_first=True)

# Handle missing values
df.fillna(df.mean(), inplace=True)

# Feature selection (target: Churn)
X = df.drop('Churn', axis=1)
y = df['Churn']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model 1: Logistic Regression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_pred_log_reg = log_reg.predict(X_test)

# Model 2: Random Forest
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Evaluate models: Accuracy, Precision, Recall
print(f"Logistic Regression Accuracy: {accuracy_score(y_test, y_pred_log_reg)}")
print(f"Random Forest Accuracy: {accuracy_score(y_test, y_pred_rf)}")

# ROC-AUC for Random Forest
roc_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
print(f"Random Forest ROC-AUC: {roc_auc}")

"""
Outcome: Students will gain experience in building classification models and 
evaluating their performance using various metrics.
"""