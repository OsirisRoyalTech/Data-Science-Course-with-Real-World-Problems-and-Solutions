# Module 1: Real-World Data Wrangling and Exploratory Data Analysis (EDA)

#Problem 1: Sales Data Analysis for a Retail Store
#Dataset: Retail sales data (e.g., Kaggle's “Retail Data Analysis”)
#Objective: Analyze sales data and extract meaningful insights to help business owners understand trends, customer preferences, and sales performance.

"""
Tasks:
•	Load and clean the data (handle missing values, outliers, and duplicate entries).
•	Perform data aggregation and grouping to identify sales patterns.
•	Visualize trends using charts like bar plots, line charts, and heatmaps.
"""

#Source Code:
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('retail_sales.csv')

# Data cleaning: Handling missing values
df.fillna(df.mean(), inplace=True)

# EDA: Group sales by store
store_sales = df.groupby('Store')['Sales'].sum().reset_index()

# Visualization: Bar chart of total sales by store
plt.figure(figsize=(10,6))
sns.barplot(x='Store', y='Sales', data=store_sales)
plt.title('Total Sales by Store')
plt.show()

# Additional EDA
df['Date'] = pd.to_datetime(df['Date'])
monthly_sales = df.groupby(df['Date'].dt.to_period('M'))['Sales'].sum().reset_index()

# Visualization: Line plot for monthly sales
plt.figure(figsize=(10,6))
sns.lineplot(x='Date', y='Sales', data=monthly_sales)
plt.title('Monthly Sales Trends')
plt.xticks(rotation=45)
plt.show()

"""
Outcome: This project helps students understand how to clean data, aggregate it, 
and visualize trends for better decision-making in business.
"""