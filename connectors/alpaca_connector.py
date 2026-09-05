# connectors/alpaca_connector.py
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

class AlpacaConnector(BaseConnector):
    def __init__(self):
        super().__init__(platform_name="codealpaca")

    def fetch_problems(self):
        print("\n [STEP 1] Fetching CodeAlpaca-20k Dataset via Native Parquet API...")
        print("  NOTE: This will add ~20,000 algorithmic & coding instruction problems.\n")
        
        api_url = "https://datasets-server.huggingface.co/parquet?dataset=sahil2801/CodeAlpaca-20k"
        try:
            print(" Requesting file locations from HF Server...")
            res = requests.get(api_url, timeout=15)
            res.raise_for_status()
            parquet_files = res.json().get("parquet_files", [])
        except Exception as e:
            print(f" API Error: {e}")
            return []

        all_urls = [p['url'] for p in parquet_files if p.get('split') == 'train']
        if not all_urls:
            all_urls = [p['url'] for p in parquet_files]

        all_urls = list(dict.fromkeys(all_urls))

        if not all_urls:
            print(" No valid Parquet URLs found.")
            return []

        print(f" Found {len(all_urls)} unique backend files. Processing...")

        cleaned_list = []
        seen_descriptions = set()

        for url in all_urls:
            try:
                print(f"   -> Downloading chunk: {url.split('/')[-1]}...")
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                
                df_chunk = pd.read_parquet(io.BytesIO(response.content))
                
                for idx, row in df_chunk.iterrows():
                    instruction = str(row.get('instruction', '')).strip()
                    input_data = str(row.get('input', '')).strip()
                    
                    # Combine instruction and input to form the problem description
                    if input_data:
                        desc = f"{instruction}\n\nInput:\n{input_data}"
                    else:
                        desc = instruction

                    if not desc or len(desc.split()) < 10:
                        continue
                        
                    desc_key = " ".join(desc.split()).lower()
                    if desc_key in seen_descriptions:
                        continue
                    seen_descriptions.add(desc_key)
                    
                    # CodeAlpaca doesn't have native difficulty, so we distribute them to keep data balanced
                    word_count = len(desc.split())
                    diff_label = 'Easy' if word_count < 30 else ('Medium' if word_count < 80 else 'Hard')
                        
                    unique_problem_id = f"ALPACA_{len(cleaned_list)}"
                    cleaned_list.append({
                        "problem_id": unique_problem_id,
                        "platform": "CodeAlpaca",
                        "title": f"Alpaca Problem {len(cleaned_list)}",
                        "title_slug": unique_problem_id.lower(),
                        "difficulty": diff_label,
                        "description": desc,
                        "examples": "",             
                        "constraints": "",          
                        "tags": "algorithmic, instruction",                 
                        "companies": "",            
                        "hints": "",                
                        "acceptance_rate": "",      
                        "likes": 0,                 
                        "dislikes": 0,              
                        "premium": False,           
                        "similar_questions": "",    
                        "editorial_available": False,      
                        "video_solution_available": False, 
                        "source_url": ""
                    })
                        
                del df_chunk
                gc.collect()
                print(f"      -> Status: Total Alpaca Problems Processed: {len(cleaned_list)}")
                
            except Exception as e:
                print(f" Failed to process chunk: {e}")

        print(f"\n CodeAlpaca Extraction complete!")
        print(f" Total Problems Saved: {len(cleaned_list)}")
        return cleaned_list
        
    def save_to_csv(self, data):
        if not data:
            print(" No data captured.")
            return
            
        df = pd.DataFrame(data)
        columns_order = [
            "problem_id", "platform", "title", "title_slug", "difficulty", 
            "description", "examples", "constraints", "tags", "companies", 
            "hints", "acceptance_rate", "likes", "dislikes", "premium", 
            "similar_questions", "editorial_available", "video_solution_available", "source_url"
        ]
        
        for col in columns_order:
            if col not in df.columns:
                df[col] = ""
                
        df = df[columns_order]
        output_path = os.path.join(config.RAW_DATA_DIR, "alpaca.csv")
        df.to_csv(output_path, index=False)
        print(f" Successfully saved clean CodeAlpaca data to {output_path}")

if __name__ == "__main__":
    connector = AlpacaConnector()
    data = connector.fetch_problems()
    connector.save_to_csv(data)