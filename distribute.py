import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
dengue_df = pd.read_csv('Project Machine Learning/dataset.csv')

# Select only the numerical columns
numeric_df = dengue_df.select_dtypes(include=[np.number])

# Plot histograms for each numerical feature
numeric_df.hist(bins=20, figsize=(12, 10), color='skyblue', edgecolor='black')
plt.suptitle("Distribution of Numerical Features in Dengue Dataset", y=1.02)
plt.tight_layout()
plt.show()
