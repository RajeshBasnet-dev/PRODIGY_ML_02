# K-Means Clustering for Customer Segmentation
# Task-02: Group customers by income and spending behavior
# Dataset: Mall_Customers.csv from Kaggle

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import os

# Load dataset if available
dataset_file = 'Mall_Customers.csv'
if not os.path.exists(dataset_file):
    print(f"Error: '{dataset_file}' not found.")
    print("Download it from: https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python")
    exit()

# Read the data
print("Loading dataset...")
data = pd.read_csv(dataset_file)
print("Sample data:")
print(data.head())

# Select features for clustering
features = data[['Annual Income (k$)', 'Spending Score (1-100)']].values
print("\nUsing 'Annual Income' and 'Spending Score' for clustering.")

# Find best number of clusters using Elbow Method
wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(features)
    wcss.append(kmeans.inertia_)

# Plot Elbow Curve
plt.figure(figsize=(8, 5))
sns.set(style="whitegrid")
plt.plot(range(1, 11), wcss, marker='o', linestyle='--', color='blue')
plt.title('Elbow Method: Best Number of Clusters', fontsize=14, fontweight='bold')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('WCSS')
plt.grid(True)
plt.show()

# Apply K-means with chosen number of clusters
optimal_k = 5
print(f"\nClustering into {optimal_k} groups...")
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
data['Cluster'] = kmeans.fit_predict(features)
centroids = kmeans.cluster_centers_

# Visualize the clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x=features[:, 0], y=features[:, 1], hue=data['Cluster'], palette='Set2', s=100, alpha=0.7)
plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='black', marker='X', label='Centers')
plt.title('Customer Groups by Income and Spending', fontsize=14, fontweight='bold')
plt.xlabel('Annual Income (in $1000)')
plt.ylabel('Spending Score')
plt.legend(title='Group')
plt.show()

# Show summary stats for each group
print("\nCustomer Group Summary:")
group_counts = data['Cluster'].value_counts().sort_index()
group_income = data.groupby('Cluster')['Annual Income (k$)'].mean()
group_spending = data.groupby('Cluster')['Spending Score (1-100)'].mean()

for i in range(optimal_k):
    print(f"Group {i}:")
    print(f"  - Customers: {group_counts[i]}")
    print(f"  - Avg. Income: ${group_income[i]:.2f}k")
    print(f"  - Avg. Spending: {group_spending[i]:.2f}")

# Bar chart: Number of customers in each group
plt.figure(figsize=(8, 4))
sns.barplot(x=group_counts.index, y=group_counts.values, palette='Set2')
plt.title('Number of Customers per Group', fontsize=14, fontweight='bold')
plt.xlabel('Group')
plt.ylabel('Customer Count')
plt.show()

# Bar chart: Avg income per group
plt.figure(figsize=(8, 4))
sns.barplot(x=group_income.index, y=group_income.values, palette='Set3')
plt.title('Average Income by Group', fontsize=14, fontweight='bold')
plt.xlabel('Group')
plt.ylabel('Avg. Income ($1000)')
plt.show()

# Bar chart: Avg spending score per group
plt.figure(figsize=(8, 4))
sns.barplot(x=group_spending.index, y=group_spending.values, palette='coolwarm')
plt.title('Average Spending Score by Group', fontsize=14, fontweight='bold')
plt.xlabel('Group')
plt.ylabel('Spending Score')
plt.show()
