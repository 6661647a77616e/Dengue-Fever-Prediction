import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

plt.style.use('ggplot')

# Load the dataset
dengue_df = pd.read_csv('dengue.csv')

# Check for any missing values
print(dengue_df.isnull().sum())

# label encoding
label_encoder = LabelEncoder()
dengue_df['areatype'] = label_encoder.fit_transform(dengue_df['AreaType'])
dengue_df['housetype'] = label_encoder.fit_transform(dengue_df['HouseType'])

dengue_numeric_only = dengue_df.select_dtypes(include=['number'])

# Basic summary statistics
print("Statistical Summary:")
print(dengue_df.describe())

# First few rows
print("\nFirst 5 Rows of Data:")
print(dengue_df.head())

# Check the column names to ensure 'Class' is correctly loaded
print("\nColumns in the dataset:")
print(dengue_df.columns)

# Count the number of observations per class
print("\nClass Distribution:")
print(dengue_df['Outcome'].value_counts())

# Ensure 'Outcome' is treated as a categorical variable
dengue_df['Outcome'] = dengue_df['Outcome'].astype('category')

# Filter numeric columns and include 'Outcome' for hue
dengue_numeric_only = dengue_df.select_dtypes(include=['number']).copy()
dengue_numeric_only['Outcome'] = dengue_df['Outcome']

# Drop any NaN values if present
dengue_numeric_only = dengue_numeric_only.dropna()

# Plot pair plot with specified hue
sns.set_style("whitegrid")
pair_plot = sns.pairplot(dengue_numeric_only, hue='Outcome', height=2.5, markers=["o", "s"],
                         palette="bright", plot_kws={'s': 60, 'alpha': 0.7})

# Add title and display plot
plt.suptitle("Dengue Pair Plot", y=1.02)
plt.savefig("dengue_pairplot.png", dpi=300, bbox_inches='tight')
plt.show()

# Create box plots for each feature against the target 'Outcome'
plt.figure(figsize=(15, 10))
for i, column in enumerate(dengue_df.select_dtypes(include=['number']).columns, 1):
    plt.subplot(3, 3, i)  # Adjust subplot layout based on the number of features
    sns.boxplot(data=dengue_df, x='Outcome', y=column, palette='bright')
    plt.title(f'Boxplot of {column} by Outcome')
    plt.xlabel('Outcome')
    plt.ylabel(column)

plt.tight_layout()
plt.savefig("dengue_boxplots.png", dpi=300, bbox_inches='tight')
plt.show()



