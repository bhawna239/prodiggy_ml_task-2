import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1. Load the dataset
# Simulated sample data
data = {
    'CustomerID': range(1, 11),
    'Annual Income (k$)': [15, 16, 17, 18, 19, 60, 62, 63, 64, 65],
    'Spending Score (1-100)': [39, 81, 6, 77, 40, 50, 42, 55, 47, 49],
    'Purchase Frequency': [5, 8, 3, 6, 7, 20, 19, 18, 21, 22]
}

df = pd.DataFrame(data)

# 2. Select features for clustering
X = df[['Annual Income (k$)', 'Spending Score (1-100)', 'Purchase Frequency']]

# 3. Feature scaling (important for K-means)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Determine the optimal number of clusters using the Elbow Method
inertia = []
K = range(1, 10)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plot Elbow Curve
plt.figure(figsize=(8, 4))
plt.plot(K, inertia, 'bo-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method For Optimal k')
plt.show()

# 5. Fit KMeans with optimal number of clusters (let's say k=3 for this example)
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# 6. Visualize clusters (using first 2 principal components for simplicity)
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
components = pca.fit_transform(X_scaled)
df['PCA1'] = components[:, 0]
df['PCA2'] = components[:, 1]

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Cluster', palette='viridis', s=100)
plt.title('Customer Segments by K-means Clustering')
plt.show()

# 7. Optional: Examine cluster profiles
cluster_summary = df.groupby('Cluster')[['Annual Income (k$)', 'Spending Score (1-100)', 'Purchase Frequency']].mean()
print("Cluster Profiles:\n", cluster_summary)
df = pd.read_csv('customer_purchase_data.csv')
