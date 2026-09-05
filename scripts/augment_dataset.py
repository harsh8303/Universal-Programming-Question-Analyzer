# scripts/augment_dataset.py
import pandas as pd
import os
import random
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import config

def augment_text(text):
    """
    Smart NLP Augmentation: Replaces words with programming synonyms 
    randomly to create a fresh variation of the question.
    """
    if pd.isna(text) or not isinstance(text, str):
        return str(text)
        
    replacements = {
        "array": "list", "list": "array",
        "integer": "number", "number": "value",
        "string": "text",
        "calculate": "compute", "compute": "calculate",
        "return": "output", "print": "display",
        "find": "determine", "determine": "find",
        "maximum": "largest", "largest": "maximum",
        "minimum": "smallest", "smallest": "minimum",
        "given": "provided",
        "function": "method",
        "element": "item",
        "sum": "total", "total": "sum"
    }
    
    words = text.split()
    new_words = []
    
    for word in words:
        clean_word = word.lower().strip(".,!?:;()")
        # 30% chance to replace the word if synonym exists (Subtle changes, high realism)
        if clean_word in replacements and random.random() < 0.30:
            new_word = replacements[clean_word]
            if word[0].isupper():
                new_word = new_word.capitalize()
            new_words.append(new_word)
        else:
            new_words.append(word)
            
    return " ".join(new_words)

def generate_augmented_dataset():
    print("Starting Data Augmentation to reach 100,000+ problems...")
    
    # Input folder: data/clean
    input_path = os.path.join(config.DATA_DIR, "clean", "cleaned_programming_problems.csv")
    
    # Output Folder: data/clean (Tere requirement ke hisab se)
    output_dir = os.path.join(config.DATA_DIR, "clean")
    os.makedirs(output_dir, exist_ok=True)
    # New File Name (Do not overwrite the original clean file)
    output_path = os.path.join(output_dir, "augmented_100k_problems.csv")
    
    if not os.path.exists(input_path):
        print(f" Error: Could not find clean data at {input_path}")
        print("Please run 'python preprocessing/clean_dataset.py' first.")
        return
        
    print(f" Loading Clean Master Dataset...")
    df = pd.read_csv(input_path)
    current_size = len(df)
    target_size = 100000
    
    print(f" Current Size: {current_size} problems")
    print(f" Target Size: {target_size} problems")
    
    if current_size >= target_size:
        print(" Dataset is already 100k+! Skipping augmentation.")
        return
        
    needed = target_size - current_size
    print(f" Generating {needed} augmented problems to prevent overfitting...")
    
    # Randomly sample 'needed' rows from our dataset
    df_sample = df.sample(n=needed, replace=True, random_state=42).copy()
    
    print(" Applying NLP Transformations (This will take 1-2 minutes)...")
    df_sample['clean_description'] = df_sample['clean_description'].apply(augment_text)
    
    # Update IDs and Tags so we know these are augmented
    df_sample['problem_id'] = df_sample['problem_id'].astype(str) + "_AUG"
    df_sample['platform'] = df_sample['platform'].astype(str) + "_Augmented"
    
    # Merge original and augmented
    print(" Merging Original and Augmented Data...")
    final_df = pd.concat([df, df_sample], ignore_index=True)
    
    # Final Shuffle
    print(" Final Shuffling for ML Training...")
    final_df = final_df.sample(frac=1, random_state=99).reset_index(drop=True)
    
    # Save the ultimate dataset
    final_df.to_csv(output_path, index=False)
    
    print("\n" + "="*50)
    print(" THE 100k MASTER DATASET IS READY! 👑")
    print("="*50)
    print(f" Final Size: {len(final_df)} problems!")
    print(f"Saved at: {output_path}")
    print("="*50)

if __name__ == "__main__":
    generate_augmented_dataset()