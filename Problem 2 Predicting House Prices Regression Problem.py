# Module 2: Predictive Modeling - Regression Analysis
# Problem 2: Predicting House Prices (Regression Problem)
# Dataset: Kaggle’s “House Prices: Advanced Regression Techniques”
# Objective: Build a machine learning model to predict house prices based on features like size, location, number of rooms, etc.

"""
Tasks:
•	Preprocess data: Handle missing values, encode categorical features, and normalize numerical features.
•	Build and evaluate multiple regression models.
•	Tune the model using cross-validation and hyperparameter optimization.
"""

# Source Code:
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv('house_prices.csv')

# Data preprocessing: Fill missing values
df.fillna(df.mean(), inplace=True)

# Encode categorical features
df = pd.get_dummies(df, drop_first=True)

# Feature selection (excluding target variable)
X = df.drop('SalePrice', axis=1)
y = df['SalePrice']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model: Linear Regression
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Predictions
y_pred = model.predict(X_test_scaled)

# Model evaluation: RMSE
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"Root Mean Squared Error (RMSE): {rmse}")

"""
Outcome: Students will understand how to build a regression model, 
evaluate it using metrics like RMSE, and perform data preprocessing.
"""