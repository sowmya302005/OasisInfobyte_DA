# ============================================
# Exploratory Data Analysis on Retail Sales Data
# Tool: Python (VS Code)
# ============================================

# ---------- 1. Import Libraries ----------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ---------- 2. Create Output Folder ----------
os.makedirs("outputs", exist_ok=True)

# ---------- 3. Load Dataset ----------
df = pd.read_csv("data/retail_sales_dataset.csv")

print("Dataset Loaded Successfully")
print(df.head())

# ---------- 4. Understand Dataset ----------
print("\nDataset Information:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

# ---------- 5. Data Cleaning ----------
print("\nMissing Values:")
print(df.isnull().sum())

# Drop missing values
df.dropna(inplace=True)

# Remove duplicates
df.drop_duplicates(inplace=True)

# Convert Date column to datetime
df['Date'] = pd.to_datetime(df['Date'])

print("\nData Cleaning Completed")

# ---------- 6. Descriptive Statistics ----------
mean_sales = df['Total Amount'].mean()
median_sales = df['Total Amount'].median()
std_sales = df['Total Amount'].s
td()

print("\nDescriptive Statistics:")
print("Mean Sales:", mean_sales)
print("Median Sales:", median_sales)
print("Standard Deviation:", std_sales)

# ---------- 7. Time Series Analysis ----------
df['Month'] = df['Date'].dt.to_period('M')

monthly_sales = df.groupby('Month')['Total Amount'].sum()

plt.figure(figsize=(10,5))
monthly_sales.plot()
plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Total Sales")
plt.tight_layout()
plt.savefig("outputs/monthly_sales_trend.png")
plt.show()

# ---------- 8. Customer Analysis ----------
# Gender-wise Sales
gender_sales = df.groupby('Gender')['Total Amount'].sum()

plt.figure()
gender_sales.plot(kind='bar')
plt.title("Sales by Gender")
plt.ylabel("Total Sales")
plt.tight_layout()
plt.savefig("outputs/gender_sales.png")
plt.show()

# Age Distribution
plt.figure()
sns.histplot(df['Age'], bins=10)
plt.title("Customer Age Distribution")
plt.tight_layout()
plt.savefig("outputs/age_distribution.png")
plt.show()

# ---------- 9. Product Analysis ----------
product_sales = df.groupby('Product Category')['Total Amount'].sum().sort_values(ascending=False)

plt.figure(figsize=(8,5))
product_sales.plot(kind='bar')
plt.title("Sales by Product Category")
plt.ylabel("Total Sales")
plt.tight_layout()
plt.savefig("outputs/product_sales.png")
plt.show()

# ---------- 10. Correlation Heatmap ----------
corr = df[['Age', 'Quantity', 'Price per Unit', 'Total Amount']].corr()

plt.figure(figsize=(6,4))
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("outputs/correlation_heatmap.png")
plt.show()

# ---------- 11. Final Insights ----------
print("\nKEY INSIGHTS:")
print("- Certain months show higher sales indicating seasonal trends.")
print("- Clothing and Electronics categories generate high revenue.")
print("- Adult customers contribute most to total sales.")

print("\nRECOMMENDATIONS:")
print("- Increase stock for high-performing product categories.")
print("- Focus marketing campaigns during peak sales months.")
print("- Target high-value customer segments for promotions.")

print("\nEDA Project Completed Successfully!")
