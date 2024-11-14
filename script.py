
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,PolynomialFeatures,OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder

plt.style.use('ggplot')

# Load the dataset
dengue_df = pd.read_csv('Project Machine Learning\dengue.csv')

# Check for any missing values
print(dengue_df.isnull().sum())

dengue_numeric_only = dengue_df.select_dtypes(include=['number'])

# Dataset shape
print("Row and Column")
print(dengue_df.shape)

# Basic summary statistics 
print("\nStatistical Summary:")
print(dengue_df.describe(include='number'))

# Basic summary statistics 
print("\nStatistical Summary of non numbers:")
print(dengue_df.describe(exclude='number'))

# First few rows
print("\nFirst 5 Rows of Data:")
print(dengue_df.head())

# Check the column names to ensure 'Class' is correctly loaded
print("\nColumns in the dataset:")
print(dengue_df.columns)

# Count the number of observations per class
print("\nClass Distribution:")
print(dengue_df['Outcome'].value_counts())

# Calculate the correlation matrix
correlation_matrix = dengue_numeric_only.corr()

# Set up the matplotlib figure
plt.figure(figsize=(12, 8))

# Draw the heatmap with a color palette and annotations
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Correlation Matrix of Dengue Dataset Features")
plt.savefig("dengue_correlation_matrix.png", dpi=300, bbox_inches='tight')
plt.show()

# Ensure 'Outcome' is treated as a categorical variable
dengue_df['Outcome'] = dengue_df['Outcome'].astype('int')

# Filter numeric columns and include 'Outcome' for hue
dengue_numeric_only = dengue_df.select_dtypes(include=['number']).copy()
dengue_numeric_only['Outcome'] = dengue_df['Outcome']

# Drop any NaN values if present
dengue_numeric_only = dengue_numeric_only.dropna()

# label encoding
label_encoder = LabelEncoder()
dengue_df['AreaType'] = label_encoder.fit_transform(dengue_df['AreaType'])
dengue_df['HouseType'] = label_encoder.fit_transform(dengue_df['HouseType'])

# Assuming 'df' is your original dataframe with the necessary columns
encoder = OneHotEncoder(drop='first', sparse_output=False)
poly = PolynomialFeatures(degree=2, include_bias=False)

# Step 1: Encode categorical variables (e.g., AreaType and HouseType)
encoded_columns = encoder.fit_transform(dengue_df[['AreaType', 'HouseType']])
encoded_df = pd.DataFrame(encoded_columns, columns=encoder.get_feature_names_out(['AreaType', 'HouseType']))

# Step 2: Generate interaction terms for numeric variables (e.g., Age, NS1, IgG, IgM)
numeric_df = dengue_df[['Age', 'NS1', 'IgG', 'IgM']]  # Adjust based on your dataset
interaction_terms = poly.fit_transform(numeric_df)

# Convert interaction terms to a DataFrame with proper column names
interaction_df = pd.DataFrame(interaction_terms, columns=poly.get_feature_names_out(numeric_df.columns))

# Step 3: Combine everything into the final DataFrame
# Drop the original categorical columns to avoid duplication
final_df = pd.concat([dengue_df.drop(['AreaType', 'HouseType'], axis=1), encoded_df, interaction_df], axis=1)

# Step 4: Remove redundant columns (e.g., 'Age', 'NS1', etc., if they are now duplicated)
final_df = final_df.loc[:, ~final_df.columns.duplicated()]

# Adjust pandas display options for better readability
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', None)  # Don't truncate rows

print("\nData with Interaction Terms column:")
print(final_df.columns)

# Print the final DataFrame to see the cleaned-up output
print("\nData with Interaction Terms:")
print(final_df.head())

# Plot pair plot with specified hue
sns.set_style("whitegrid")
pair_plot = sns.pairplot(dengue_numeric_only, hue='Outcome', height=2.5, markers=["o", "s"],
                         palette="bright", plot_kws={'s': 60, 'alpha': 0.7})

# Add title and display plot
plt.suptitle("Dengue Pair Plot", y=1.02)
plt.savefig("dengue_pairplot.png", dpi=300, bbox_inches='tight')
plt.show()

# Plot histograms for each numerical feature
dengue_numeric_only.hist(bins=20, figsize=(12, 10), color='skyblue', edgecolor='black')
plt.suptitle("Distribution of Numerical Features in Dengue Dataset", y=1.02)
plt.tight_layout()
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
X = dengue_df.drop(['Outcome','Gender','District','Area'], axis=1)
y = dengue_df['Outcome']               

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize the neural network model
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
print("Training started..., this can take a while:")
model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.2, verbose=1)

# Evaluate the model
y_pred = (model.predict(X_test) > 0.5).astype("int32")  # Convert probabilities to binary outcomes

# Generate and display the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(conf_matrix, display_labels=[0, 1]).plot(cmap="plasma")
plt.title("Confusion Matrix")
plt.show()

# Print the classification report for additional metrics
print("Classification Report:")
print(classification_report(y_test, y_pred))
