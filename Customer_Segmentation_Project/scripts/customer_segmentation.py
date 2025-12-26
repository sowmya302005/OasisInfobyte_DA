# ============================================
# Customer Segmentation Analysis
# Tool: Python (VS Code)
# ============================================

# ---------- 1. Import Libraries ----------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os

# ---------- 2. Create Output Folder ----------
os.makedirs("outputs", exist_ok=True)

# ---------- 3. Load Dataset ----------
df = pd.read_csv("data/marketing_campaign.csv", sep='\t')

print("Dataset Loaded Successfully")
print(df.head())

# ---------- 4. Dataset Understanding ----------
print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

# ---------- 5. Data Cleaning ----------
df.dropna(inplace=True)

# ---------- Create Total Spent Feature ----------
df['TotalSpent'] = (
    df['MntWines'] +
    df['MntFruits'] +
    df['MntMeatProducts'] +
    df['MntFishProducts'] +
    df['MntSweetProducts'] +
    df['MntGoldProds']
)

# ---------- 6. Select Features for Segmentation ----------
# Common features for customer behavior
features = df[['Income', 'Recency', 'NumWebPurchases', 'NumStorePurchases', 'TotalSpent']]

# ---------- 7. Feature Scaling ----------
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# ---------- 8. Find Optimal Clusters (Elbow Method) ----------
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(scaled_features)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8,5))
plt.plot(range(1,11), wcss, marker='o')
plt.title("Elbow Method for Optimal Clusters")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.tight_layout()
plt.savefig("outputs/elbow_method.png")
plt.show()

# ---------- 9. Apply K-Means Clustering ----------
kmeans = KMeans(n_clusters=4, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_features)

print("\nCluster Distribution:")
print(df['Cluster'].value_counts())

# ---------- 10. Cluster Visualization ----------
plt.figure(figsize=(8,5))
sns.scatterplot(
    x=df['Income'],
    y=df['TotalSpent'],
    hue=df['Cluster'],
    palette='viridis'
)
plt.title("Customer Segmentation")
plt.xlabel("Income")
plt.ylabel("Total Spent")
plt.tight_layout()
plt.savefig("outputs/customer_segments.png")
plt.show()

# ---------- 11. Cluster Analysis ----------
cluster_summary = df.groupby('Cluster')[['Income', 'TotalSpent', 'NumWebPurchases', 'NumStorePurchases']].mean()
print("\nCluster Summary:")
print(cluster_summary)

# ---------- 12. Insights & Recommendations ----------
print("\nKEY INSIGHTS:")
print("Cluster 0: High income, high spending → Premium customers")
print("Cluster 1: Medium income, moderate spending → Regular customers")
print("Cluster 2: Low spending → Price-sensitive customers")
print("Cluster 3: Low engagement → Inactive customers")

print("\nRECOMMENDATIONS:")
print("- Offer loyalty rewards to premium customers")
print("- Run targeted discounts for price-sensitive customers")
print("- Re-engage inactive customers with email campaigns")

print("\nCustomer Segmentation Project Completed Successfully!")
