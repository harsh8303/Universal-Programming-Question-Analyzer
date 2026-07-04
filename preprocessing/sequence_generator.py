# preprocessing/sequence_generator.py
import pandas as pd
import numpy as np
import os
import sys
import pickle
import json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

# Setup paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import config

def generate_sequences():
    print(" Starting Phase 5: Sequence Generation & Distribution Analysis (Version 1 Final)...")
    
    # Paths & Loading Files
    processed_dir = os.path.join(config.DATA_DIR, "processed")
    artifacts_dir = os.path.join(ROOT_DIR, "artifacts")
    
    dataset_path = os.path.join(processed_dir, "tokenizer_dataset.csv")
    tokenizer_path = os.path.join(artifacts_dir, "tokenizer.pkl")
    config_path = os.path.join(artifacts_dir, "sequence_config.json")
    
    if not os.path.exists(dataset_path) or not os.path.exists(tokenizer_path):
        print(" ERROR: Required files not found. Run Phase 4 first.")
        sys.exit(1)
        
    print(" Loading Filtered Dataset & Tokenizer...")
    df = pd.read_csv(dataset_path)
    
    # 1. Required Columns Validation
    required_columns = ['clean_description', 'difficulty', 'problem_id', 'platform', 'title', 'tags']
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        print(f" ERROR: Missing required columns: {missing_cols}. Stopping pipeline.")
        sys.exit(1)
        
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
        
    print("✓ Converting descriptions to sequences...")
    texts = df['clean_description'].astype(str).tolist()
    sequences = tokenizer.texts_to_sequences(texts)
    
    # 2. Sequence Length Validation (Prevent crash)
    if len(sequences) == 0:
        print("ERROR: No sequences generated. Dataset might be empty. Stopping pipeline.")
        sys.exit(1)
        
    # Sequence Length Distribution Analysis
    lengths = np.array([len(seq) for seq in sequences])
    
    print("\n SEQUENCE LENGTH DISTRIBUTION:")
    print(f"   - Min Length       : {lengths.min()}")
    print(f"   - Max Length       : {lengths.max()}")
    print(f"   - Mean Length      : {lengths.mean():.2f}")
    print(f"   - 50th Percentile  : {np.percentile(lengths, 50)}")
    print(f"   - 80th Percentile  : {np.percentile(lengths, 80)}")
    print(f"   - 90th Percentile  : {np.percentile(lengths, 90)}")
    print(f"   - 95th Percentile  : {np.percentile(lengths, 95)}")
    print(f"   - 99th Percentile  : {np.percentile(lengths, 99)}")
    
    # Best MAX_SEQUENCE_LENGTH Select (Dynamic 95th Percentile)
    optimal_length = int(np.percentile(lengths, 95))
    MAX_SEQ_LENGTH = optimal_length
    
    print(f"\n DATA-DRIVEN DECISION:")
    print(f"   Setting MAX_SEQUENCE_LENGTH to {MAX_SEQ_LENGTH} (Covers 95% of the dataset)")
    
    print(f"\n Applying Padding & Truncation (padding='post', truncating='post')...")
    X_padded = pad_sequences(sequences, maxlen=MAX_SEQ_LENGTH, padding='post', truncating='post')
    
    print(" Encoding Difficulty Labels...")
    df['difficulty'] = df['difficulty'].fillna('Unknown').astype(str)
    encoder = LabelEncoder()
    y_difficulty = encoder.fit_transform(df['difficulty'])
    
    # 4. Print Class Distribution
    print("\n DIFFICULTY CLASS DISTRIBUTION:")
    class_counts = df['difficulty'].value_counts()
    for cls, count in class_counts.items():
        print(f"   - {cls:<10} : {count}")
        
    encoder_path = os.path.join(artifacts_dir, "difficulty_encoder.pkl")
    with open(encoder_path, 'wb') as f:
        pickle.dump(encoder, f)
        
    print("\n✓ Saving Sequences and Labels (.npy formats)...")
    np.save(os.path.join(processed_dir, "X_sequences.npy"), X_padded)
    np.save(os.path.join(processed_dir, "y_difficulty.npy"), y_difficulty)
    
    print("✓ Generating Metadata File...")
    df[required_columns].to_csv(os.path.join(processed_dir, "processed_metadata.csv"), index=False)
    
    # 3. Enhanced Sequence Config
    print("✓ Saving Configuration...")
    seq_config = {
        "Total Samples": len(sequences),
        "Vocabulary Size": len(tokenizer.word_index) + 1,
        "Padding": "post",
        "Truncation": "post",
        "Maximum Sequence Length": MAX_SEQ_LENGTH,
        "Difficulty Classes": encoder.classes_.tolist()
    }
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(seq_config, f, indent=4)
        
    print("\n" + "="*55)
    print(" SUCCESS: PHASE 5 SEQUENCE GENERATOR FINALIZED! 🎉")
    print("="*55)
    print(f" X (Inputs) Shape     : {X_padded.shape}")
    print(f" y (Difficulty) Shape : {y_difficulty.shape}")
    print("="*55)
    print(" Ready for Phase 6: Train / Validation / Test Split (train_test_split.py)!")

if __name__ == "__main__":
    generate_sequences()