# preprocessing/clean_dataset.py
import pandas as pd
import re
import os
import sys
import time
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Setup paths (so we can import config)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import config

# --- SMART NLTK LOADER ---
def ensure_nltk_downloads():
    print(" Checking NLTK resources...")
    packages = {
        'tokenizers/punkt': 'punkt', 
        'corpora/stopwords': 'stopwords', 
        'corpora/wordnet': 'wordnet'
    }
    for path, name in packages.items():
        try:
            nltk.data.find(path)
        except LookupError:
            print(f"    Downloading missing resource: {name}")
            nltk.download(name, quiet=True)

ensure_nltk_downloads()

lemmatizer = WordNetLemmatizer()

# --- CUSTOM STOPWORDS ---
core_stopwords = set(stopwords.words('english'))
important_programming_words = {"if", "else", "only", "not", "every", "each"}
custom_stopwords = core_stopwords - important_programming_words


def clean_text(text):
    if pd.isna(text) or not str(text).strip():
        return ""
    
    text = str(text)
    
    # 1. Remove HTML tags (Using built-in html.parser)
    text = BeautifulSoup(text, "html.parser").get_text(separator=" ")
    
    # 2. Lowercase
    text = text.lower()
    
    # 3 & 4. Keep numbers & alphabets (Constraints like 10^5, O(n) are saved!)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    
    # 5. Tokenization
    tokens = word_tokenize(text)
    
    # 6 & 7. Stopwords & basic Lemmatization
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in custom_stopwords]
    
    #  8. Remove extra spaces and strip
    final_text = " ".join(cleaned_tokens)
    final_text = re.sub(r'\s+', ' ', final_text).strip()
    
    return final_text


def count_valid_tags(tags):
    if pd.isna(tags) or str(tags).strip() == "":
        return 0
    valid_tags = [t for t in str(tags).split(',') if t.strip()]
    return len(valid_tags)


def process_dataset():
    start_time = time.time()  # Track execution time
    print("Starting Phase 2: Data Cleaning (Version 1 Final)...")
    
    # Input File Handling
    input_path = os.path.join(config.DATA_DIR, "unified_programming_problems.csv")
    if not os.path.exists(input_path):
        print(f" Not found at primary path, checking alternate...")
        input_path = os.path.join(config.DATA_DIR, "merged", "unified_programming_problems.csv")
        
    clean_dir = os.path.join(config.DATA_DIR, "clean")
    os.makedirs(clean_dir, exist_ok=True)
    output_path = os.path.join(clean_dir, "cleaned_programming_problems.csv")
    
    if not os.path.exists(input_path):
        print(f"Error: Master dataset not found at {input_path}.")
        return

    print(f"📂 Master Dataset loaded from: {input_path}")
    df = pd.read_csv(input_path)
    
    #  Required Columns Validation
    required_columns = ['problem_id', 'platform', 'description', 'tags']
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        print(f" ERROR: Missing required columns: {missing_cols}. Stopping pipeline.")
        sys.exit(1)
        
    initial_count = len(df)
    
    # Track missing values before filling
    missing_descriptions = df['description'].isna().sum()
    missing_tags = df['tags'].isna().sum()
    
    print("🧹 Removing duplicates...")
    df.drop_duplicates(subset=['problem_id', 'platform'], inplace=True)
    duplicates_removed = initial_count - len(df)
    
    print("🛠️ Handling missing values safely...")
    df.fillna("", inplace=True)
    
    print("Applying NLP Text Cleaning (Yeh thoda time lega, patience rakhna!)...")
    df['clean_description'] = df['description'].apply(clean_text)
    
    print("Engineering new columns...")
    df['description_length'] = df['description'].apply(lambda x: len(str(x)) if pd.notna(x) and str(x).strip() else 0)
    df['word_count'] = df['clean_description'].apply(lambda x: len(str(x).split()) if pd.notna(x) else 0)
    df['has_description'] = df['description_length'] > 0
    df['num_tags'] = df['tags'].apply(count_valid_tags)

    # Sort before saving
    print(" Sorting dataset by platform and problem_id...")
    df.sort_values(by=['platform', 'problem_id'], inplace=True)

    print(f" Saving cleaned dataset to CSV (UTF-8)...")
    df.to_csv(output_path, index=False, encoding='utf-8')
    
    end_time = time.time()
    exec_time = round(end_time - start_time, 2)
    
    # Data Quality Metrics
    avg_desc_len = round(df['description_length'].mean(), 2)
    avg_word_count = round(df['word_count'].mean(), 2)
    
    print("\n" + "="*50)
    print("SUCCESS: PHASE 2 DATA CLEANING COMPLETE! ")
    print("="*50)
    print(" DATA QUALITY REPORT:")
    print(f"   - Total Rows              : {len(df)}")
    print(f"   - Duplicate Rows Removed  : {duplicates_removed}")
    print(f"   - Missing Descriptions    : {missing_descriptions}")
    print(f"   - Missing Tags            : {missing_tags}")
    print(f"   - Avg Description Length  : {avg_desc_len} chars")
    print(f"   - Avg Word Count          : {avg_word_count} words")
    print("-" * 50)
    print(f" Total Execution Time     : {exec_time} seconds")
    print(f" Cleaned File saved at    : {output_path}")
    print("="*50)
    print(" Ready for Phase 3: tokenizer.py")

if __name__ == "__main__":
    process_dataset()