import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

plt.style.use('ggplot')

# Load the dataset
dengue_df = pd.read_csv('dengue.csv')

# Check for any missing values
print(dengue_df.isnull().sum())

# Pair plot for initial visualization
sns.set_style("whitegrid")
sns.pairplot(dengue_df, hue='Outcome', height=2.5, markers=["o", "s"])
plt.suptitle("Dengue Pair Plot", y=1.02)  # Title with spacing adjustment
plt.show()

# Basic summary statistics
print("Statistical Summary:")
print(dengue_df.describe())

# First few rows of the data
print("\nFirst 5 Rows of Data:")
print(dengue_df.head())

# Check the column names to ensure 'Outcome' is correctly loaded
print("\nColumns in the dataset:")
print(dengue_df.columns)

# Count the number of observations per class
print("\nClass Distribution:")
print(dengue_df['Outcome'].value_counts())

# Drop non-numeric columns for correlation matrix calculation
numeric_df = dengue_df.select_dtypes(include=[np.number])  # Select only numeric columns

# Correlation Matrix Heatmap
plt.figure(figsize=(10, 8))
correlation_matrix = numeric_df.corr()  # Compute correlation matrix for numerical columns
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1, square=True)
plt.title("Correlation Matrix of Dengue Dataset Features")
plt.show()

