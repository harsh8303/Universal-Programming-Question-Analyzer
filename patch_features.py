# patch_features.py
import pandas as pd
import os
import sys

# Setup paths properly
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import config

def apply_feature_fusion():
    print(" Initiating Feature Fusion (Preserving Lineage)...")
    
    # CORRECTED PATH: Looking into the 'clean' folder!
    file_path = os.path.join(config.DATA_DIR, "clean", "cleaned_programming_problems.csv")
    
    if not os.path.exists(file_path):
        print(f" ERROR: File not found at {file_path}")
        sys.exit(1)
        
    df = pd.read_csv(file_path)
    
    df['title'] = df['title'].fillna('')
    df['tags'] = df['tags'].fillna('')
    df['clean_description'] = df['clean_description'].fillna('')
    
    # New column creation (Title + Tags + Description)
    df['model_text'] = df['title'] + " " + df['tags'] + " " + df['clean_description']
    
    df.to_csv(file_path, index=False)
    print(f" SUCCESS: 'model_text' column generated successfully in {file_path}!")

if __name__ == "__main__":
    apply_feature_fusion()