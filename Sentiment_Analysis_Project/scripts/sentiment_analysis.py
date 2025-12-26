# scripts/sentiment_analysis.py

import pandas as pd
import nltk
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# ----------------------------
# NLTK setup
# ----------------------------
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

# ----------------------------
# Load Dataset
# ----------------------------
df = pd.read_csv("data/sentiment_dataset.csv")  # change name if needed

print("Initial Shape:", df.shape)
print("Columns:", df.columns.tolist())

# ----------------------------
# Use correct columns directly
# ----------------------------
df = df[['clean_text', 'category']]
df.columns = ['text', 'sentiment']

# ----------------------------
# Drop missing values ONLY
# ----------------------------
df.dropna(inplace=True)

# ----------------------------
# Minimal text cleaning (DO NOT over-clean)
# ----------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z ]", "", text)
    return text.strip()

df['text'] = df['text'].apply(clean_text)

# Remove empty rows AFTER cleaning
df = df[df['text'].str.len() > 2]

print("After Cleaning Shape:", df.shape)

# ----------------------------
# Split data
# ----------------------------
X = df['text']
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# TF-IDF Vectorization
# ----------------------------
vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words='english'
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# ----------------------------
# Train Model
# ----------------------------
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# ----------------------------
# Evaluate Model
# ----------------------------
y_pred = model.predict(X_test_tfidf)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
