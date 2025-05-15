# Customer Segmentation using K-Means Clustering
# Groups customers based on Annual Income and Spending Score

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Load the dataset (ensure 'Mall_Customers.csv' is in the same folder)
data = pd.read_csv('Mall_Customers.csv')

# View first few rows
print("First 5 rows of dataset:")
print(data.head())

# Plot Annual Income vs Spending Score
plt.figure(figsize=(8, 5))
sns.scatterplot(data=data, x='Annual Income (k$)', y='Spending Score (1-100)')
plt.title('Customer Distribution: Income vs Spending Score')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score')
plt.show()

# Select features for clustering
features = data[['Annual Income (k$)', 'Spending Score (1-100)']].values

# Use Elbow Method to find optimal number of clusters
wcss = []
for k in range(1, 11):
 kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(features)

wcss.append(kmeans.inertia_)

# Plot WCSS to find the elbow
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method for Optimal Clusters')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('WCSS')
plt.grid(True)
plt.show()

# Choose optimal clusters (e.g., 5) and fit KMeans
optimal_k = 5
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
data['Cluster'] = kmeans.fit_predict(features)

# Plot clusters with centroids
plt.figure(figsize=(8, 5))
sns.scatterplot(x=features[:, 0], y=features[:, 1], hue=data['Cluster'], palette='Set1')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
s=200, c='black', marker='X', label='Centroids')
plt.title('Customer Segments by Income and Spending Score')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score')
plt.legend(title='Cluster')
plt.show()

# Print summary of clusters
for i in range(optimal_k):
 cluster = data[data['Cluster'] == i]
print(f"Cluster {i}: {len(cluster)} customers")
print(f"  Average Income: {cluster['Annual Income (k$)'].mean():.2f}k")
print(f"  Average Spending Score: {cluster['Spending Score (1-100)'].mean():.2f}\n")
