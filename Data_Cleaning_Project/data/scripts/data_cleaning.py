import pandas as pd
import numpy as np

# -------------------------------
# Load datasets
# -------------------------------
df1 = pd.read_csv(
    "data/raw/dataset1.csv",
    encoding="latin1",
    sep=",",
    on_bad_lines="skip",
    engine="python"
)
df2 = pd.read_csv(
    "data/raw/dataset2.csv",
    encoding="latin1",
    sep=",",
    on_bad_lines="skip",
    engine="python"
)

print("Initial Shape Dataset 1:", df1.shape)
print("Initial Shape Dataset 2:", df2.shape)

# -------------------------------
# 1. Remove duplicates
# -------------------------------
df1 = df1.drop_duplicates()
df2 = df2.drop_duplicates()

# -------------------------------
# 2. Handle missing values
# -------------------------------
for col in df1.select_dtypes(include=np.number).columns:
    df1[col].fillna(df1[col].median(), inplace=True)

for col in df2.select_dtypes(include=np.number).columns:
    df2[col].fillna(df2[col].median(), inplace=True)

df1.fillna("Unknown", inplace=True)
df2.fillna("Unknown", inplace=True)

# -------------------------------
# 3. Standardization (text)
# -------------------------------
for col in df1.select_dtypes(include="object").columns:
    df1[col] = df1[col].str.strip().str.lower()

for col in df2.select_dtypes(include="object").columns:
    df2[col] = df2[col].str.strip().str.lower()

# -------------------------------
# 4. Outlier removal (IQR method)
# -------------------------------
def remove_outliers(df):
    numeric_cols = df.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]
    return df

df1 = remove_outliers(df1)
df2 = remove_outliers(df2)

# -------------------------------
# Save cleaned datasets
# -------------------------------
df1.to_csv("data/cleaned/dataset1_cleaned.csv", index=False)
df2.to_csv("data/cleaned/dataset2_cleaned.csv", index=False)

print("✅ Data cleaning completed successfully!")
print("Final Shape Dataset 1:", df1.shape)
print("Final Shape Dataset 2:", df2.shape)

