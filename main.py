# Step 1: Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Step 2: Load dataset (make sure you downloaded 'Mall_Customers.csv' from Kaggle)
data = pd.read_csv('Mall_Customers.csv')

# Step 3: Data exploration 
plt.figure(figsize=(8,5))
sns.scatterplot(data=data, x='Annual Income (k$)', y='Spending Score (1-100)')
plt.title('Customer Distribution by Income and Spending Score')
plt.show()

# # Step 4: Prepare data for clustering
# X = data[['Annual Income (k$)', 'Spending Score (1-100)']].values

# # Step 5: Use Elbow method to find optimal number of clusters
# wcss = []
# for k in range(1, 11):
#     kmeans = KMeans(n_clusters=k, random_state=42)
#     kmeans.fit(X)
#     wcss.append(kmeans.inertia_)

# plt.figure(figsize=(8,5))
# plt.plot(range(1, 11), wcss, marker='o')
# plt.title('Elbow Method to find optimal k')
# plt.xlabel('Number of clusters')
# plt.ylabel('WCSS')
# plt.show()

# # Step 6: Apply K-means clustering with k=5 (based on elbow plot)
# k = 5
# kmeans = KMeans(n_clusters=k, random_state=42)
# clusters = kmeans.fit_predict(X)

# # Step 7: Add cluster labels to dataframe
# data['Cluster'] = clusters

# # Step 8: Visualize clusters
# plt.figure(figsize=(8,5))
# sns.scatterplot(x=X[:,0], y=X[:,1], hue=clusters, palette='Set1')
# plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=200, c='black', marker='X')  # centroids
# plt.title('Customer Segments based on Income and Spending Score')
# plt.xlabel('Annual Income (k$)')
# plt.ylabel('Spending Score (1-100)')
# plt.show()
