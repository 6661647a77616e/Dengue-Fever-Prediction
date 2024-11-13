
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder

plt.style.use('ggplot')

# Load the dataset
dengue_df = pd.read_csv('dengue.csv')

# Check for any missing values
print(dengue_df.isnull().sum())

dengue_numeric_only = dengue_df.select_dtypes(include=['number'])

# Dataset shape
print("Row and Column")
print(dengue_df.shape)

# Basic summary statistics 
print("\nStatistical Summary:")
print(dengue_df.describe(include='number'))

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

# label encoding
label_encoder = LabelEncoder()
dengue_df['AreaType'] = label_encoder.fit_transform(dengue_df['AreaType'])
dengue_df['HouseType'] = label_encoder.fit_transform(dengue_df['HouseType'])

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

# Separate features and target variable
X = dengue_df.drop(['Outcome', 'Gender','District','Area'], axis=1)  
y = dengue_df['Outcome']               

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize and train the classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Generate and display the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(conf_matrix, display_labels=model.classes_).plot(cmap="plasma")
plt.title("Confusion Matrix")
plt.show()

# Print the classification report for additional metrics
print("Classification Report:")
print(classification_report(y_test, y_pred))
