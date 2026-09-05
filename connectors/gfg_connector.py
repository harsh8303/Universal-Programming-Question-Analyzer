# connectors/gfg_connector.py
import os
import sys
import requests
import pandas as pd
import io
import gc

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from base_connector import BaseConnector
import config

class GFGConnector(BaseConnector):
    def __init__(self):
        super().__init__(platform_name="geeksforgeeks")
        
        # List of known public GFG datasets on Hugging Face (Fail-Safe Mechanism)
        self.gfg_datasets = [
            "Saurabh3949/Geeksforgeeks-DSA-Datset",
            "karthikbhandary2/gfg-dsa-dataset",
            "Amod/GeeksForGeeks-Dataset",
            "shashank-m/GeeksForGeeks-DSA-Dataset"
        ]

    def fetch_problems(self):
        print("\n [STEP 1] Fetching GeeksForGeeks (GFG) Dataset via Parquet API...")
        
        parquet_files = []
        # Fallback mechanism: Try datasets until one works
        for repo in self.gfg_datasets:
            print(f" Trying GFG source: {repo}...")
            api_url = f"https://datasets-server.huggingface.co/parquet?dataset={repo}"
            try:
                res = requests.get(api_url, timeout=10)
                if res.status_code == 200:
                    parquet_files = res.json().get("parquet_files", [])
                    if parquet_files:
                        print(f" Successfully connected to {repo}!")
                        break
            except Exception as e:
                pass
                
        if not parquet_files:
            print(" All public HF GFG APIs failed. Servers might be down.")
            return []

        all_urls = [p['url'] for p in parquet_files if p.get('split') == 'train']
        
        cleaned_list = []
        seen_descriptions = set()
        class_counts = {"Basic": 0, "Easy": 0, "Medium": 0, "Hard": 0}

        for url in all_urls:
            try:
                print(f"   -> Downloading chunk: {url.split('/')[-1]}...")
                response = requests.get(url, timeout=40)
                response.raise_for_status()
                df_chunk = pd.read_parquet(io.BytesIO(response.content))
                
                # Dynamic Column Mapping (different repos use different column names)
                col_title = next((c for c in df_chunk.columns if c.lower() in ['title', 'name', 'problem_name', 'problem']), None)
                col_desc = next((c for c in df_chunk.columns if c.lower() in ['description', 'problem_description', 'text', 'question']), None)
                col_diff = next((c for c in df_chunk.columns if c.lower() in ['difficulty', 'level', 'difficulty_level']), None)
                
                if not col_desc or not col_diff:
                    print(" Missing required columns in this chunk. Skipping.")
                    continue

                for idx, row in df_chunk.iterrows():
                    # 1. Clean Description
                    description = str(row.get(col_desc, '')).strip()
                    if not description or len(description.split()) < 10:
                        continue
                        
                    # 2. Smart Deduplication
                    desc_key = " ".join(description.split()).lower()
                    if desc_key in seen_descriptions:
                        continue
                    seen_descriptions.add(desc_key)

                    # 3. Ground-Truth Difficulty Extraction
                    raw_diff = str(row.get(col_diff, '')).strip().title()
                    
                    if "Basic" in raw_diff or "School" in raw_diff: diff = "Basic"
                    elif "Easy" in raw_diff: diff = "Easy"
                    elif "Medium" in raw_diff: diff = "Medium"
                    elif "Hard" in raw_diff: diff = "Hard"
                    else: continue # Skip if difficulty is not explicitly labeled
                    
                    class_counts[diff] += 1
                    title = str(row.get(col_title, f'GFG Problem {idx}'))
                    
                    cleaned_list.append({
                        "problem_id": f"GFG_{idx}",
                        "platform": "GeeksForGeeks",
                        "title": title,
                        "title_slug": title.lower().replace(" ", "-"),
                        "difficulty": diff,
                        "description": description,
                        "examples": "", "constraints": "", "tags": "dsa, interview",
                        "companies": "", "hints": "", "acceptance_rate": "",
                        "likes": 0, "dislikes": 0, "premium": False,
                        "similar_questions": "", "editorial_available": False,
                        "video_solution_available": False, "source_url": ""
                    })
                
                del df_chunk
                gc.collect()
                
            except Exception as e:
                print(f" Failed chunk: {e}")
                
        print("\n GEEKSFORGEEKS CLASS BALANCE REPORT:")
        for diff, count in class_counts.items():
            print(f"   - {diff}: {count} problems")
            
        print(f"\n Total Valid GFG Problems Extracted: {len(cleaned_list)}")
        return cleaned_list

    def save_to_csv(self, data):
        if not data:
            print(" No data to save.")
            return
            
        # Optional: Map 'Basic' to 'Easy' so it aligns with your 3-class system
        print(" Mapping 'Basic' GFG problems to 'Easy' for model compatibility...")
        for row in data:
            if row['difficulty'] == 'Basic':
                row['difficulty'] = 'Easy'
                
        df = pd.DataFrame(data)
        output_path = os.path.join(config.RAW_DATA_DIR, "gfg.csv")
        
        # Will overwrite the empty gfg.csv you had previously
        df.to_csv(output_path, index=False)
        print(f" Successfully saved premium GFG data to {output_path}")

if __name__ == "__main__":
    connector = GFGConnector()
    data = connector.fetch_problems()
    connector.save_to_csv(data)