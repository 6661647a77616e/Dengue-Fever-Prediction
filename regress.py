import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
import pickle

# Set plot style
plt.style.use('ggplot')

# Load the dataset
dengue_df = pd.read_csv('Project Machine Learning/dengue.csv')

# Perform EDA

# Check for missing values
print("Missing values in each column:")
print(dengue_df.isnull().sum())

# Describe the dataset's statistics
print("\nStatistical Summary:")
print(dengue_df.describe())

# Describe non-numeric columns
print("\nStatistical Summary of non-numeric columns:")
print(dengue_df.describe(include=['object']))

# Display first few rows
print("\nFirst 5 Rows of Data:")
print(dengue_df.head())

# Count the number of observations per class
print("\nClass Distribution (Outcome):")
print(dengue_df['Outcome'].value_counts())

# Visualize the correlation matrix
dengue_numeric_only = dengue_df.select_dtypes(include=['number'])
correlation_matrix = dengue_numeric_only.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Correlation Matrix of Dengue Dataset Features")
plt.show()

# Visualize the distribution of features
dengue_numeric_only.hist(bins=20, figsize=(12, 10), color='skyblue', edgecolor='black')
plt.suptitle("Distribution of Numerical Features in Dengue Dataset", y=1.02)
plt.tight_layout()
plt.show()

# Boxplot for each numerical feature by Outcome
plt.figure(figsize=(15, 10))
for i, column in enumerate(dengue_df.select_dtypes(include=['number']).columns, 1):
    plt.subplot(3, 3, i)  # Adjust subplot layout based on the number of features
    sns.boxplot(data=dengue_df, x='Outcome', y=column, palette='bright')
    plt.title(f'Boxplot of {column} by Outcome')
    plt.xlabel('Outcome')
    plt.ylabel(column)
plt.tight_layout()
plt.show()

# Feature Engineering

# Label encode categorical features (e.g., AreaType and HouseType)
label_encoder = LabelEncoder()
dengue_df['AreaType'] = label_encoder.fit_transform(dengue_df['AreaType'])
dengue_df['HouseType'] = label_encoder.fit_transform(dengue_df['HouseType'])

# OneHot Encoding for 'AreaType' and 'HouseType'
encoder = OneHotEncoder(drop='first', sparse_output=False)
encoded_columns = encoder.fit_transform(dengue_df[['AreaType', 'HouseType']])
encoded_df = pd.DataFrame(encoded_columns, columns=encoder.get_feature_names_out(['AreaType', 'HouseType']))

# Generate interaction terms for numerical variables (Age, NS1, IgG, IgM)
numeric_df = dengue_df[['Age', 'NS1', 'IgG', 'IgM']]
poly = PolynomialFeatures(degree=2, include_bias=False)
interaction_terms = poly.fit_transform(numeric_df)
interaction_df = pd.DataFrame(interaction_terms, columns=poly.get_feature_names_out(numeric_df.columns))

# Combine everything into the final DataFrame
final_df = pd.concat([dengue_df.drop(['AreaType', 'HouseType'], axis=1), encoded_df, interaction_df], axis=1)

# Drop original numeric columns if they are now redundant (Age, NS1, IgG, IgM)
final_df = final_df.loc[:, ~final_df.columns.duplicated()]

# Split the dataset into features and target variable
X = final_df.drop(['Outcome', 'Gender', 'District', 'Area'], axis=1)
y = dengue_df['Outcome']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the logistic regression model
log_reg_model = LogisticRegression(max_iter=1000)
log_reg_model.fit(X_train, y_train)

# Make predictions
y_pred = log_reg_model.predict(X_test)

# Model evaluation

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(conf_matrix, display_labels=[0, 1]).plot(cmap="plasma")
plt.title("Confusion Matrix for Logistic Regression")
plt.show()

# Classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

with open('regress_dengue.pkl', 'wb') as file:
    pickle.dump(log_reg_model, file)
