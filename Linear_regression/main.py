"""
The Single and Multiple Linear Regression Project

Author:
Date: 2024-05-19

Description:
This project is about performing Single and Multiple Linear Regression on the Startups dataset.
The dataset contains the following columns:
- R&D Spend: Research and Development Spend
- Administration: Administration Spend
- Marketing Spend: Marketing Spend
- State: State of the startup
- Profit: Profit of the startup

The project involves the following steps:
1. Read the dataset
2. Display the shape of the data
3. Display the column names
4. Display the first 5 rows of the data
5. Display the last 5 rows of the data
6. Display the information of the data
7. Display the summary statistics of the data
8. Display the number of unique values in each column of the dataset
9. Display Null values (There is not any null-values in the dataset)
10. Perform Single Linear Regression for each independent variable
11. Perform Multiple Linear Regression using the following combinations of independent variables:
    a. R&D Spend and Administration
    b. Administration and Marketing Spend
    c. R&D Spend, Administration, and Marketing Spend
12. Plot the actual vs predicted values for each multiple linear regression model
"""

# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# Read the Startups data
data = pd.read_csv("50_Startups-without State.csv")

# Display the shape of the data
print(data.shape)

# Display the Column names
print(list(data.columns))

# Display the first 5 rows of the data
print(data.head())

# Display the last 5 rows of the data
print(data.tail())

# Display the information of the data
print(data.info())

# Display the summary statistics of the data
print(data.describe())

# Display the Number of unique values in each column of the dataset
print(data.nunique())

# Display Null values (There is not any null-values in the dataset)
print(data.isnull().sum())

# Single Linear Regression
# Define the independent variables and the target variable
X = data[["R&D Spend", "Administration", "Marketing Spend"]]
y = data["Profit"]

# Initialize the linear regression model
single_model = LinearRegression()

# Perform linear regression for each independent variable
results = {}
for column in X.columns:
    X_single = X[[column]]
    single_model.fit(X_single, y)
    single_y_pred = single_model.predict(X_single)
    results[column] = {
        "model": single_model,
        "y_pred": single_y_pred
    }

    # Plot the results
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=X_single[column], y=y, label="Actual Profit")
    sns.lineplot(x=X_single[column], y=single_y_pred, color="red", label="Predicted Profit")
    plt.title(f"Linear Regression: {column} vs Profit")
    plt.xlabel(column)
    plt.ylabel("Profit")
    plt.legend()
    plt.savefig(f"Linear_Regression_{column}_vs_Profit.png")
    # plt.show()


# Multi Linear Regression
# Function to perform and plot multiple linear regression
def multiple_linear_regression(X, y, title):
    multi_model = LinearRegression()
    multi_model.fit(X, y)
    multi_y_pred = multi_model.predict(X)

    # Create a plot for the actual vs predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(y, multi_y_pred, edgecolor='k', facecolor='c', alpha=0.7, label='Predicted vs Actual')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label='Perfect Prediction')
    plt.xlabel('Actual Profit')
    plt.ylabel('Predicted Profit')
    plt.title(title)
    plt.legend()
    plt.savefig(f"{title}.png")
    # plt.show()


# 1. Use first 2 columns (R&D Spend, Administration) to predict Profit
X1 = data[["R&D Spend", "Administration"]]
multiple_linear_regression(X1, y, 'Multiple_Linear_Regression_R&D_Spend_and_Administration_vs_Profit')

# 2. Use second and third columns (Administration, Marketing Spend) to predict Profit
X2 = data[["Administration", "Marketing Spend"]]
multiple_linear_regression(X2, y, 'Multiple_Linear_Regression_Administration_and_Marketing_Spend_vs_Profit')

# 3. Use all three columns (R&D Spend, Administration, Marketing Spend) to predict Profit
X3 = data[["R&D Spend", "Administration", "Marketing Spend"]]
multiple_linear_regression(X3, y, 'Multiple_Linear_Regression_R&D_Spend_Administration_and_Marketing_Spend_vs_Profit')