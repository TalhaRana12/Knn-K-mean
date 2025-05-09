# -*- coding: utf-8 -*-
"""AI.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1R9a6kZD82zP5kfOZyvUXxh_md4NGWmgB

# Task
Tell me about this dataset.

Here is all the data you need:
"RT_IOT2022_transformed.csv"

## Data loading

### Subtask:
Load the dataset "RT_IOT2022_transformed.csv" into a pandas DataFrame.

**Reasoning**:
Load the dataset into a pandas DataFrame and display its first few rows and shape.
"""

import pandas as pd

try:
    df = pd.read_csv('RT_IOT2022_transformed.csv')
    display(df.head())
    print(df.shape)
except FileNotFoundError:
    print("Error: 'RT_IOT2022_transformed.csv' not found.")
    df = None
except Exception as e:
    print(f"An error occurred: {e}")
    df = None

"""## Data preparation

### Subtask:
Prepare the data for clustering by handling missing values and scaling numerical features.

**Reasoning**:
Handle missing values and scale numerical features in the dataframe.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler

# 1. Handle Missing Values
# Check for missing values
missing_values = df.isnull().sum()
print("Missing values per column:\n", missing_values)

# Impute missing values with the mean for numerical features
numerical_cols = df.select_dtypes(include=['number']).columns
for col in numerical_cols:
    if df[col].isnull().any():
        df[col].fillna(df[col].mean(), inplace=True)

# 2. Scale Numerical Features
# Separate numerical features
numerical_features = df.select_dtypes(include=['number']).columns
df_numerical = df[numerical_features]

# Scale numerical features using StandardScaler
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_numerical), columns=numerical_features)

# Concatenate scaled numerical features with other columns
df_scaled = pd.concat([df_scaled, df.select_dtypes(exclude=['number'])], axis=1)

display(df_scaled.head())

"""**Reasoning**:
Fix the chained assignment warning and complete the data preparation.


"""

import pandas as pd
from sklearn.preprocessing import StandardScaler

# 1. Handle Missing Values
# Check for missing values
missing_values = df.isnull().sum()
print("Missing values per column:\n", missing_values)

# Impute missing values with the mean for numerical features
numerical_cols = df.select_dtypes(include=['number']).columns
for col in numerical_cols:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].mean())

# 2. Scale Numerical Features
# Separate numerical features
numerical_features = df.select_dtypes(include=['number']).columns
df_numerical = df[numerical_features]

# Scale numerical features using StandardScaler
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_numerical), columns=numerical_features)

# Concatenate scaled numerical features with other columns
non_numerical_cols = df.select_dtypes(exclude=['number']).columns
df_scaled = pd.concat([df_scaled, df[non_numerical_cols]], axis=1)

display(df_scaled.head())

"""## Data clustering

### Subtask:
Perform K-means clustering on the scaled data. Experiment with different numbers of clusters (k) to find an optimal solution, and store the results.

**Reasoning**:
Perform K-means clustering on the scaled data with different k values and store the results in new columns.
"""

from sklearn.cluster import KMeans

# Select numerical features for clustering
numerical_features = df_scaled.select_dtypes(include=['number']).columns

# KMeans clustering with different k values
k_values = [5, 10, 20]
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
    kmeans.fit(df_scaled[numerical_features])
    cluster_labels = kmeans.labels_
    df_scaled[f'cluster_{k}'] = cluster_labels

display(df_scaled.head())

"""## Data visualization

### Subtask:
Visualize the clusters obtained from the K-means algorithm using scatter plots.

**Reasoning**:
Create scatter plots to visualize the clusters for different k values. Use two principal components for the x and y axes and color the points by cluster assignment.
"""

import matplotlib.pyplot as plt

# Select two principal components or relevant numerical features
x_axis = 'fwd_pkts_tot'
y_axis = 'bwd_pkts_tot'

# Create scatter plots for each k value
k_values = [5, 10, 20]
fig, axes = plt.subplots(1, 3, figsize=(18, 6))  # Adjust figure size

for i, k in enumerate(k_values):
    ax = axes[i]
    ax.scatter(df_scaled[x_axis], df_scaled[y_axis], c=df_scaled[f'cluster_{k}'], cmap='viridis')
    ax.set_title(f'K-means Clustering (k={k})')
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)

plt.tight_layout()
plt.show()

"""## Data analysis

### Subtask:
Analyze the characteristics of each cluster identified in the previous step.

**Reasoning**:
Group the data by cluster labels and calculate descriptive statistics for each cluster.
"""

# Group data by cluster labels and calculate descriptive statistics
k_values = [5, 10, 20]
for k in k_values:
    cluster_col = f'cluster_{k}'
    grouped = df_scaled.groupby(cluster_col)
    print(f"\nDescriptive statistics for k = {k}:\n")
    # Select numerical features for description
    numerical_features = df_scaled.select_dtypes(include=['number']).columns
    numerical_features = numerical_features.drop(cluster_col)
    numerical_features = numerical_features.drop('Unnamed: 0')
    description = grouped[numerical_features].describe()
    print(description)

"""**Reasoning**:
The previous code block printed the descriptive statistics for each cluster.  Now, I will create a summary table for each k value, presenting the mean values of several important features for each cluster.


"""

# Create summary tables for each k value
k_values = [5, 10, 20]
important_features = ['fwd_pkts_tot', 'bwd_pkts_tot', 'flow_duration', 'fwd_header_size_tot', 'bwd_header_size_tot', 'flow_iat.avg']

for k in k_values:
    cluster_col = f'cluster_{k}'
    grouped = df_scaled.groupby(cluster_col)
    summary = grouped[important_features].mean()
    print(f"\nSummary table for k = {k}:\n")
    display(summary)

"""## Summary:

### Q&A

No questions were explicitly asked in the provided text.  However, the overall goal of the analysis was to explore the dataset "RT_IOT2022_transformed.csv", prepare it for clustering, perform K-means clustering with different numbers of clusters, visualize the results, and analyze the characteristics of each cluster.


### Data Analysis Key Findings

* **Missing Value Imputation:** Missing numerical values were imputed using the mean of each respective column.
* **Feature Scaling:** Numerical features were scaled using `StandardScaler` to ensure features with larger values did not unduly influence the clustering process.
* **K-means Clustering:** K-means clustering was performed with k values of 5, 10, and 20.  The `n_init` parameter was set to 10, meaning the algorithm ran 10 times with different centroid seeds, selecting the best result.  Three new cluster assignment columns (`cluster_5`, `cluster_10`, `cluster_20`) were added to the dataframe.
* **Cluster Visualization:** Scatter plots were created to visualize the clusters based on 'fwd_pkts_tot' and 'bwd_pkts_tot', color-coded by cluster assignment for k=5, k=10, and k=20.
* **Cluster Characteristic Analysis:** Descriptive statistics (count, mean, std, min, 25%, 50%, 75%, max) were calculated for numerical features within each cluster for each k value.  Summary tables showing the mean values of specific features ('fwd_pkts_tot', 'bwd_pkts_tot', 'flow_duration', 'fwd_header_size_tot', 'bwd_header_size_tot', and 'flow_iat.avg') were also generated for each k value to highlight key differences between clusters.


### Insights or Next Steps

* **Optimal k Value:**  Compare the descriptive statistics and summary tables across different k values (5, 10, and 20) to determine which k value produces the most distinct and meaningful clusters.  Consider using metrics like silhouette score or Davies-Bouldin index to objectively evaluate the clustering quality for each k.
* **Feature Importance:** Investigate the relative importance of different features in defining the clusters. Techniques like feature importance scores from tree-based models or permutation importance could be used to identify the most influential features.  This can help better interpret the clusters and understand the underlying patterns within the data.

"""