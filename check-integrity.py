import pandas as pd

# Load the datasets
red_wine = pd.read_csv('Data/winequality-red.csv', delimiter=";")
white_wine = pd.read_csv('Data/winequality-white.csv', delimiter=";")

# Print column names to check for 'quality'
print("Red Wine Columns:", red_wine.columns)
print("White Wine Columns:", white_wine.columns)


#Author: Morteza Farrokhnejad