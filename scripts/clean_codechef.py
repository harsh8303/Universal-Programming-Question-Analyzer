# scripts/clean_codechef.py
import pandas as pd
import os
import sys

# Setup paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import config

def clean_codechef_data():
    raw_path = os.path.join(config.RAW_DATA_DIR, "codechef_raw.csv")
    output_path = os.path.join(config.RAW_DATA_DIR, "codechef.csv")

    if not os.path.exists(raw_path):
        print(f"❌ Error: {raw_path} not found. Pehle file ko raw folder mein daalo.")
        return

    print("⏳ Reading raw CodeChef dataset...")
    df_raw = pd.read_csv(raw_path)
    
    # 💡 Bulletproof trick: Sab columns ke naam lower case aur bina extra space ke kar do
    df_raw.columns = df_raw.columns.str.strip().str.lower()
    
    # NaN values ko empty string se replace kar do
    df_raw.fillna("", inplace=True)

    df_clean = pd.DataFrame()

    # Difficulty extract karne ka logic (tags ya level se)
    def extract_difficulty(row):
        # Pehle 'level' column check karo
        lvl = str(row.get('level', '')).lower().strip()
        if lvl in ['school', 'beginner', 'easy']: return "Easy"
        elif lvl in ['medium']: return "Medium"
        elif lvl in ['hard', 'challenge']: return "Hard"
        
        # Agar 'level' khali hai, toh 'tags' ke andar dhoondho (jaise screenshot mein hai)
        tags = str(row.get('tags', '')).lower()
        if 'easy' in tags or 'beginner' in tags or 'school' in tags: return "Easy"
        elif 'medium' in tags: return "Medium"
        elif 'hard' in tags or 'challenge' in tags: return "Hard"
        
        return "Unknown"

    print("🔄 Mapping to Universal Schema...")
    
    # Hamari 19-column mapping (using safe .get() to avoid KeyError)
    df_clean["problem_id"] = df_raw.get("qcode", pd.Series([""] * len(df_raw))).astype(str)
    df_clean["platform"] = "CodeChef"
    df_clean["title"] = df_raw.get("title", "")
    df_clean["title_slug"] = df_clean["problem_id"].str.lower()
    df_clean["difficulty"] = df_raw.apply(extract_difficulty, axis=1)
    df_clean["description"] = df_raw.get("statement", "")
    df_clean["examples"] = ""
    df_clean["constraints"] = ""
    df_clean["tags"] = df_raw.get("tags", "")
    df_clean["companies"] = ""
    df_clean["hints"] = ""
    df_clean["acceptance_rate"] = ""
    df_clean["likes"] = 0
    df_clean["dislikes"] = 0
    df_clean["premium"] = False
    df_clean["similar_questions"] = ""
    
    # Agar editorial column mein link hai, toh True, warna False
    df_clean["editorial_available"] = df_raw.get("editorial", "") != ""
    
    df_clean["video_solution_available"] = False
    
    # Agar link explicitly column mein hai, toh woh lo, warna generate karo
    if "link" in df_raw.columns:
        df_clean["source_url"] = df_raw["link"]
    else:
        df_clean["source_url"] = "https://www.codechef.com/problems/" + df_clean["problem_id"]

    # Final columns order guarantee (19 Columns)
    columns_order = [
        "problem_id", "platform", "title", "title_slug", "difficulty", 
        "description", "examples", "constraints", "tags", "companies", 
        "hints", "acceptance_rate", "likes", "dislikes", "premium", 
        "similar_questions", "editorial_available", "video_solution_available", "source_url"
    ]
    
    for col in columns_order:
        if col not in df_clean.columns:
            df_clean[col] = ""
            
    df_clean = df_clean[columns_order]

    # Clean empty rows (agar koi kachra row aa gayi ho)
    df_clean = df_clean[df_clean["problem_id"] != ""]

    df_clean.to_csv(output_path, index=False)
    print(f"\n✅ Success! CodeChef data converted to Universal Schema.")
    print(f"📊 Total Problems Cleaned: {len(df_clean)}")
    print(f"💾 Saved to: {output_path}")

if __name__ == "__main__":
    clean_codechef_data()