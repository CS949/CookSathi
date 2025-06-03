import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load dataset
df = pd.read_csv("recipes.csv")

# Normalize column names
df.columns = df.columns.str.lower()

# Use the correct column
ingredient_column = 'cleaned-ingredients'

# Check if it exists
if ingredient_column not in df.columns:
    raise Exception(f"Column '{ingredient_column}' not found. Available columns: {df.columns.tolist()}")

# Preprocess ingredients
df[ingredient_column] = df[ingredient_column].astype(str).str.lower().str.replace('[^a-zA-Z, ]', '', regex=True)

# Vectorize ingredients
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(df[ingredient_column])

# Save model and data
with open("model.pkl", "wb") as f:
    pickle.dump((df, tfidf, tfidf_matrix), f)

print("Model trained and saved as model.pkl")
