# preprocessing/tokenizer.py
import pandas as pd
import os
import sys
import pickle
import json
from tensorflow.keras.preprocessing.text import Tokenizer

# Setup paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import config

# --- Hyperparameters ---
MAX_VOCAB_SIZE = 10000
OOV_TOKEN = "<OOV>"

def run_tokenizer():
    print("Starting Phase 4: Tokenizer Generation (Version 1 Final)...")
    
    # Paths setup
    input_path = os.path.join(config.DATA_DIR, "clean", "cleaned_programming_problems.csv")
    artifacts_dir = os.path.join(ROOT_DIR, "artifacts")
    processed_dir = os.path.join(config.DATA_DIR, "processed")
    
    os.makedirs(artifacts_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    
    tokenizer_path = os.path.join(artifacts_dir, "tokenizer.pkl")
    vocab_path = os.path.join(artifacts_dir, "vocabulary.json")
    config_path = os.path.join(artifacts_dir, "tokenizer_config.json")
    filtered_data_path = os.path.join(processed_dir, "tokenizer_dataset.csv")
    
    # ✓ Loading Dataset
    print("✓ Loading Dataset...")
    if not os.path.exists(input_path):
        print(f"ERROR: Cleaned dataset not found at {input_path}")
        sys.exit(1)
        
    df = pd.read_csv(input_path)
    
    # ✓ Validating Dataset
    print("✓ Validating Dataset Columns...")
    required_columns = ['clean_description', 'has_description']
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        print(f"ERROR: Missing required columns: {missing_cols}. Stopping pipeline.")
        sys.exit(1)
        
    # ✓ Filtering Valid Descriptions
    print("✓ Filtering Valid Descriptions...")
    valid_df = df[df['has_description'] == True].copy()
    total_samples = len(valid_df)
    
    # 2. EMPTY DATASET VALIDATION
    if total_samples == 0:
        print(" ERROR: 0 valid descriptions found after filtering. Cannot train Tokenizer.")
        sys.exit(1)
        
    # ✓ Saving Filtered Dataset
    print("✓ Saving Filtered Dataset (tokenizer_dataset.csv)...")
    valid_df.to_csv(filtered_data_path, index=False, encoding='utf-8')
        
    texts = valid_df['clean_description'].astype(str).tolist()
    
    # ✓ Building Vocabulary
    print("✓ Building Vocabulary...")
    tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token=OOV_TOKEN)
    tokenizer.fit_on_texts(texts)
    
    # ✓ Generating Sequences
    print("✓ Generating Sequences (calculating statistics)...")
    sequences = tokenizer.texts_to_sequences(texts)
    
    # Calculate Stats for the report
    seq_lengths = [len(seq) for seq in sequences]
    max_seq_len = int(max(seq_lengths))
    avg_seq_len = float(sum(seq_lengths) / len(seq_lengths))
    actual_vocab_size = len(tokenizer.word_index)
    
    # Tokenizer Vocabulary Size logic (+1 for padding index)
    tokenizer_vocab_size = min(actual_vocab_size + 1, MAX_VOCAB_SIZE)
    
    # ✓ Saving Tokenizer
    print("✓ Saving Tokenizer...")
    with open(tokenizer_path, 'wb') as f:
        pickle.dump(tokenizer, f)
        
    # ✓ Saving Vocabulary
    print("✓ Saving Vocabulary...")
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump(tokenizer.word_index, f, ensure_ascii=False, indent=4)
        
    # ✓ Saving Configuration
    print("✓ Saving Configuration...")
    tokenizer_config = {
        "MAX_VOCAB_SIZE": MAX_VOCAB_SIZE,
        "OOV_TOKEN": OOV_TOKEN,
        "Total Training Samples": total_samples,
        "Total Unique Words": actual_vocab_size,
        "Tokenizer Vocabulary Size": tokenizer_vocab_size,
        "Maximum Sequence Length": max_seq_len,
        "Average Sequence Length": round(avg_seq_len, 2)
    }
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(tokenizer_config, f, ensure_ascii=False, indent=4)
        
    # ✓ Tokenizer Completed
    print("✓ Tokenizer Completed!\n")
    
    # 3. VOCABULARY REPORT
    print("="*55)
    print("SUCCESS: PHASE 4 TOKENIZER FINALIZED! ")
    print("="*55)
    print(" VOCABULARY REPORT:")
    print(f"   - Total Training Samples   : {total_samples}")
    print(f"   - Total Unique Words Found : {actual_vocab_size}")
    print(f"   - Tokenizer Vocab Size     : {tokenizer_vocab_size} (Includes Padding)")
    print(f"   - OOV Token                : '{OOV_TOKEN}'")
    print(f"   - Maximum Sequence Length  : {max_seq_len} tokens")
    print(f"   - Average Sequence Length  : {avg_seq_len:.2f} tokens")
    print("="*55)
    print(f" Filtered Dataset saved   : {filtered_data_path}")
    print(f" Config saved at          : {config_path}")
    print(" Ready for Phase 5: Sequence Generator (sequence_generator.py)!")

if __name__ == "__main__":
    run_tokenizer()