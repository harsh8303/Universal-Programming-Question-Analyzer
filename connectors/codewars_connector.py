# connectors/codewars_connector.py
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

class CodeWarsConnector(BaseConnector):
    def __init__(self):
        super().__init__(platform_name="codewars")

    def fetch_problems(self):
        print("\n [STEP 1] Fetching CodeWars Dataset via Parquet API (Public Version)...")
        #  FIXED: Using a guaranteed public CodeWars dataset
        api_url = "https://datasets-server.huggingface.co/parquet?dataset=NyanDoggo/codewars_dataset"
        try:
            res = requests.get(api_url, timeout=15)
            res.raise_for_status()
            parquet_files = res.json().get("parquet_files", [])
        except Exception as e:
            print(f" API Error: {e}")
            return []

        all_urls = [p['url'] for p in parquet_files if p.get('split') == 'train']
        
        cleaned_list = []
        seen_descriptions = set()
        
        # Track class balance on the fly
        class_counts = {"Easy": 0, "Medium": 0, "Hard": 0}

        for url in all_urls:
            try:
                print(f"   -> Downloading chunk: {url.split('/')[-1]}...")
                response = requests.get(url, timeout=40)
                response.raise_for_status()
                df_chunk = pd.read_parquet(io.BytesIO(response.content))
                
                for idx, row in df_chunk.iterrows():
                    # 1. Clean and Filter Description
                    description = str(row.get('description', '')).strip()
                    if not description or len(description.split()) < 15:
                        continue
                        
                    # 2. Smart Deduplication
                    desc_key = " ".join(description.split()).lower()
                    if desc_key in seen_descriptions:
                        continue
                    seen_descriptions.add(desc_key)

                    # 3. CodeWars Rank to Difficulty Mapping
                    # Rank is stored as "8 kyu", "7 kyu", etc.
                    rank_str = str(row.get('rank_name', '8 kyu'))
                    try:
                        # Extract just the integer (e.g., from "8 kyu" -> 8)
                        rank = int(rank_str.split()[0])
                    except:
                        rank = 8
                    
                    if rank >= 7: 
                        diff = 'Easy'
                    elif 5 <= rank <= 6: 
                        diff = 'Medium'
                    else: 
                        diff = 'Hard'
                    
                    class_counts[diff] += 1
                    
                    # Handle tags properly if they are in a list/array
                    tags_raw = row.get('tags', '')
                    if isinstance(tags_raw, (list, tuple, type(pd.Series()))):
                        tags_raw = ", ".join(str(x) for x in tags_raw if x)
                        
                    title = str(row.get('name', f'CodeWars Problem {idx}'))
                    
                    cleaned_list.append({
                        "problem_id": f"CW_{row.get('id', idx)}",
                        "platform": "CodeWars",
                        "title": title,
                        "title_slug": title.lower().replace(" ", "-"),
                        "difficulty": diff,
                        "description": description,
                        "examples": "", 
                        "constraints": "", 
                        "tags": str(tags_raw),
                        "companies": "", 
                        "hints": "", 
                        "acceptance_rate": "",
                        "likes": 0, 
                        "dislikes": 0, 
                        "premium": False,
                        "similar_questions": "", 
                        "editorial_available": False,
                        "video_solution_available": False, 
                        "source_url": str(row.get('url', ''))
                    })
                
                # 4. RAM Cleanup
                del df_chunk
                gc.collect()
                
            except Exception as e:
                print(f" Failed chunk: {e}")
                
        print("\n CODEWARS CLASS BALANCE REPORT:")
        for diff, count in class_counts.items():
            print(f"   - {diff}: {count} problems")
            
        print(f"\n Total Valid CodeWars Problems Extracted: {len(cleaned_list)}")
        return cleaned_list

    def save_to_csv(self, data):
        if not data:
            print(" No data to save.")
            return
        df = pd.DataFrame(data)
        output_path = os.path.join(config.RAW_DATA_DIR, "codewars.csv")
        df.to_csv(output_path, index=False)
        print(f" Successfully saved to {output_path}")

if __name__ == "__main__":
    connector = CodeWarsConnector()
    data = connector.fetch_problems()
    connector.save_to_csv(data)