# preprocessing/clean_dataset.py
import pandas as pd
import re
import os
import sys
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Setup paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import config

# Download necessary NLTK data (runs only if not already downloaded)
print(" Downloading NLTK resources (if missing)...")
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Initialize Lemmatizer and Stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    # 1. Remove HTML tags
    text = BeautifulSoup(text, "lxml").get_text()
    
    # 2. Lowercase
    text = text.lower()
    
    # 3. Remove Special Characters & Numbers (Keep only alphabets)
    text = re.sub(r'[^a-z\s]', ' ', text)
    
    # 4. Tokenization
    tokens = word_tokenize(text)
    
    # 5. Stopword Removal & 6. Lemmatization
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    
    # Rejoin tokens to form the cleaned string
    return " ".join(cleaned_tokens)

def process_dataset():
    input_path = os.path.join(config.DATA_DIR, "unified_programming_problems.csv")
    
    # Output path setup
    clean_dir = os.path.join(config.DATA_DIR, "clean")
    os.makedirs(clean_dir, exist_ok=True)
    output_path = os.path.join(clean_dir, "cleaned_programming_problems.csv")
    
    if not os.path.exists(input_path):
        print(f"Error: Master dataset not found at {input_path}")
        return

    print(f" Loading Master Dataset from {input_path}...")
    df = pd.read_csv(input_path)
    
    initial_count = len(df)
    
    print(" Removing duplicates and handling missing values...")
    df.drop_duplicates(subset=['problem_id', 'platform'], inplace=True)
    df.fillna("", inplace=True)
    
    print(" Starting NLP Text Cleaning on Problem Descriptions (This may take a few minutes)...")
    # Apply NLP cleaning on the 'description' column
    df['clean_description'] = df['description'].apply(clean_text)
    
    # (Optional) Clean tags column too if needed
    df['clean_tags'] = df['tags'].apply(lambda x: re.sub(r'[^a-z\s]', ' ', str(x).lower()) if x else "")

    # Save the cleaned dataset
    df.to_csv(output_path, index=False)
    
    print("\n" + "="*50)
    print(" SUCCESS: PHASE 2 DATA CLEANING COMPLETE! 🎉")
    print("="*50)
    print(f" Original Problems  : {initial_count}")
    print(f"Cleaned Problems   : {len(df)}")
    print(f" Cleaned File saved : {output_path}")
    print("="*50)

if __name__ == "__main__":
    process_dataset()