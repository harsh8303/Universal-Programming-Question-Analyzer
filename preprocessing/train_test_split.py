# preprocessing/train_test_split.py
import numpy as np
import pandas as pd
import os
import sys
import json
import pickle
from collections import Counter
from sklearn.model_selection import train_test_split

# Setup paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import config

def split_dataset():
    print(" Starting Phase 6: Train / Validation / Test Stratified Split (Version 1 Final)...")
    
    processed_dir = os.path.join(config.DATA_DIR, "processed")
    artifacts_dir = os.path.join(ROOT_DIR, "artifacts")
    
    X_path = os.path.join(processed_dir, "X_sequences.npy")
    y_path = os.path.join(processed_dir, "y_difficulty.npy")
    metadata_path = os.path.join(processed_dir, "processed_metadata.csv")
    encoder_path = os.path.join(artifacts_dir, "difficulty_encoder.pkl")
    config_path = os.path.join(artifacts_dir, "split_config.json")
    
    if not os.path.exists(X_path) or not os.path.exists(y_path) or not os.path.exists(metadata_path):
        print(" ERROR: Required data files not found! Run Phase 5 first.")
        sys.exit(1)
        
    print("✓ Loading Numpy Arrays & Metadata...")
    X = np.load(X_path)
    y = np.load(y_path)
    df_meta = pd.read_csv(metadata_path)
    
    # 1. Validate Input Shapes
    if len(X) != len(y) or len(X) != len(df_meta):
        print(" ERROR: X, y, and metadata have different number of samples.")
        sys.exit(1)
        
    # 2. Check Empty Dataset
    if len(X) == 0:
        print(" ERROR: Dataset is empty.")
        sys.exit(1)
        
    print(f"   - Total Samples Input: {X.shape[0]}")
    
    # Load LabelEncoder to print actual class names (Easy, Medium, Hard)
    try:
        with open(encoder_path, 'rb') as f:
            encoder = pickle.load(f)
        class_mapping = {i: cls for i, cls in enumerate(encoder.classes_)}
    except FileNotFoundError:
        print(" WARNING: Encoder not found. Using numeric class labels instead.")
        class_mapping = {i: str(i) for i in np.unique(y)}
    
    # Split 1: Train (80%) and Temp (20%)
    print("✓ Performing First Split (80% Train, 20% Temp)...")
    X_train, X_temp, y_train, y_temp, meta_train, meta_temp = train_test_split(
        X, y, df_meta, test_size=0.20, random_state=42, stratify=y
    )
    
    # Split 2: Validation (10%) and Test (10%) from Temp
    print("✓ Performing Second Split (Splitting Temp into 50% Val, 50% Test)...")
    X_val, X_test, y_val, y_test, meta_val, meta_test = train_test_split(
        X_temp, y_temp, meta_temp, test_size=0.50, random_state=42, stratify=y_temp
    )
    
    # Validation to prevent -O flag issues
    if X_train.shape[0] + X_val.shape[0] + X_test.shape[0] != X.shape[0]:
        print(" ERROR: Data leakage or loss during split!")
        sys.exit(1)
        
    print("✓ Saving Arrays and Metadata for Model Training...")
    
    # Save Numpy Arrays
    np.save(os.path.join(processed_dir, "X_train.npy"), X_train)
    np.save(os.path.join(processed_dir, "X_val.npy"), X_val)
    np.save(os.path.join(processed_dir, "X_test.npy"), X_test)
    np.save(os.path.join(processed_dir, "y_train.npy"), y_train)
    np.save(os.path.join(processed_dir, "y_val.npy"), y_val)
    np.save(os.path.join(processed_dir, "y_test.npy"), y_test)
    
    # Save Metadata Splits (Resetting index for clean sequential rows)
    meta_train.reset_index(drop=True).to_csv(os.path.join(processed_dir, "metadata_train.csv"), index=False)
    meta_val.reset_index(drop=True).to_csv(os.path.join(processed_dir, "metadata_val.csv"), index=False)
    meta_test.reset_index(drop=True).to_csv(os.path.join(processed_dir, "metadata_test.csv"), index=False)
    
    # 3. Save split_config.json
    print("✓ Saving Configuration...")
    split_config = {
        "Train Samples": int(X_train.shape[0]),
        "Validation Samples": int(X_val.shape[0]),
        "Test Samples": int(X_test.shape[0]),
        "Random State": 42,
        "Stratified": True,
        "Train %": 80,
        "Validation %": 10,
        "Test %": 10
    }
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(split_config, f, indent=4)
        
    # Helper function for printing distribution with sorted output
    def print_dist(y_split, split_name):
        counts = Counter(y_split)
        print(f"\n{split_name}")
        for label_idx in sorted(counts.keys()):
            print(f"   {class_mapping[label_idx]:<10} : {counts[label_idx]}")
            
    # 4. Print Class Distribution
    print_dist(y_train, "TRAIN")
    print_dist(y_val, "VALIDATION")
    print_dist(y_test, "TEST")
    
    print("\n" + "="*55)
    print("SUCCESS: PHASE 6 DATA SPLIT FINALIZED! 🎉")
    print("="*55)
    print("📊 STRATIFIED SPLIT SUMMARY:")
    print(f"   - Train Set      : {X_train.shape[0]} samples (80%)")
    print(f"   - Validation Set : {X_val.shape[0]} samples (10%)")
    print(f"   - Test Set       : {X_test.shape[0]} samples (10%)")
    print(f"📁 Config saved at  : {config_path}")
    print("="*55)
    print(" NLP Preprocessing Fully Completed (70% Project Complete)!")
    print(" Next Stop -> Phase 7 & 8: Deep Learning (models/bilstm_attention.py)!")

if __name__ == "__main__":
    split_dataset()