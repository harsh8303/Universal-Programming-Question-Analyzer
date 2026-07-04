# scripts/merge_dataset.py
import pandas as pd
import os
import glob
import sys

# Setup paths to access config
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import config

def merge_all_platforms():
    print("🚀 Starting Phase 3: Data Integration & Merging...")
    
    # data/raw folder se saari CSV files dhundhna
    raw_files = glob.glob(os.path.join(config.RAW_DATA_DIR, "*.csv"))
    
    if not raw_files:
        print("❌ Error: No raw CSV files found in data/raw/")
        return

    print(f"📂 Found {len(raw_files)} files to check.\n")
    
    all_data = []
    for file in raw_files:
        filename = os.path.basename(file)
        
        # 💡 FIX 1: Ignore raw Kaggle files and empty GFG file
        if "raw" in filename.lower() or filename == "gfg.csv":
            print(f"⏭️ Skipping irrelevant file: {filename}")
            continue
            
        print(f"⏳ Reading: {filename}...")
        try:
            df = pd.read_csv(file)
            all_data.append(df)
        except Exception as e:
            print(f"⚠️ Could not read {filename}: {e}")
    
    if not all_data:
        print("❌ No valid data found to merge.")
        return

    # 1. Concatenate all datasets
    print("\n🔄 Merging datasets into Universal Schema...")
    master_df = pd.concat(all_data, ignore_index=True)
    initial_count = len(master_df)
    
    # 2. Clean up duplicates
    print("🧹 Cleaning up duplicates...")
    master_df.drop_duplicates(subset=['problem_id', 'platform'], inplace=True)
    final_count = len(master_df)

    # 3. 💡 FIX 2: Smart handling of missing values (Numbers vs Text)
    print("🛠️ Handling missing values safely...")
    for col in master_df.columns:
        if pd.api.types.is_numeric_dtype(master_df[col]):
            master_df[col] = master_df[col].fillna(0)  # Numbers ki jagah 0
        else:
            master_df[col] = master_df[col].fillna("") # Text ki jagah empty string
    
    # 4. Save the Final Master Dataset
    output_path = os.path.join(config.DATA_DIR, "unified_programming_problems.csv")
    master_df.to_csv(output_path, index=False)
    
    # 5. Print Analytics
    print("\n" + "="*50)
    print("✅ SUCCESS: UNIVERSAL MASTER DATASET GENERATED! 🎉")
    print("="*50)
    print(f"📊 Total problems merged   : {initial_count}")
    print(f"📉 Problems after cleaning : {final_count}")
    print(f"💾 Master File saved at    : {output_path}")
    print("-" * 50)
    
    # Platform-wise breakdown
    print("📈 Platform-wise Breakdown:")
    breakdown = master_df['platform'].value_counts()
    for platform, count in breakdown.items():
        print(f"   - {platform}: {count} problems")
    print("="*50)

if __name__ == "__main__":
    merge_all_platforms()