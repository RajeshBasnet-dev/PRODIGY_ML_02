# Customer Segmentation using K-Means Clustering
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Load the dataset
data = pd.read_csv('Mall_Customers.csv')

# Select features for clustering
features = data[['Annual Income (k$)', 'Spending Score (1-100)']].values

# Use Elbow Method to find optimal k
wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(features)
    wcss.append(kmeans.inertia_)

# Elbow Plot with style
plt.figure(figsize=(8, 5))
sns.set(style="whitegrid")
plt.plot(range(1, 11), wcss, 'o--', color='mediumvioletred', markerfacecolor='gold', linewidth=2, markersize=8)
plt.title('Optimal Clusters Using Elbow Method', fontsize=14, fontweight='bold')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('WCSS (Inertia)')
plt.xticks(range(1, 11))
plt.grid(True)
plt.show()

# Apply KMeans with k=5
optimal_k = 5
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
data['Cluster'] = kmeans.fit_predict(features)
centroids = kmeans.cluster_centers_

# Plot Clusters with custom colors and centroids
plt.figure(figsize=(10, 6))
palette = sns.color_palette("Set2", optimal_k)
sns.scatterplot(x=features[:, 0], y=features[:, 1], hue=data['Cluster'], palette=palette, s=100, alpha=0.7)

# Plot centroids
plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='black', marker='X', label='Centroids')

# Annotate cluster numbers
for i, (x, y) in enumerate(centroids):
    plt.text(x, y+1, f'Cluster {i}', color='black', fontsize=10, ha='center')

plt.title('Customer Segments by Income and Spending Score', fontsize=14, fontweight='bold')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score')
plt.legend(title='Cluster', loc='upper right')
plt.show()
