# connectors/deepmind_connector.py
import os
import sys
import time
import requests
import pandas as pd

# Setup paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from base_connector import BaseConnector
import config

class DeepMindConnector(BaseConnector):
    def __init__(self):
        super().__init__(platform_name="deepmind")

    def fetch_problems(self):
        print("\n🚀 [STEP 1] Using FAST API Mode to fetch DeepMind's dataset...")
        
        cleaned_list = []
        easy_count, medium_count, hard_count = 0, 0, 0
        limit_per_class = 20000  
        seen_descriptions = set()

        offset = 0
        batch_size = 100 # Ek baar mein 100 questions layega
        
        while True:
            # Hugging Face Datasets Server API (No heavy downloads!)
            url = f"https://datasets-server.huggingface.co/rows?dataset=deepmind/code_contests&config=default&split=train&offset={offset}&length={batch_size}"
            
            try:
                response = requests.get(url)
                if response.status_code != 200:
                    print(f"\n❌ API Error ya End of Dataset aa gaya. Status Code: {response.status_code}")
                    break
                    
                data = response.json()
                rows = data.get('rows', [])
                
                if not rows:
                    print("\n✅ Saare questions khatam ho gaye!")
                    break
                    
                # Process API Data
                for item_wrapper in rows:
                    item = item_wrapper.get('row', {})
                    
                    desc = item.get('description', '')
                    source = item.get('source', 'unknown')
                    name = item.get('name', f'problem_{offset}')
                    
                    if not desc or len(desc.split()) < 20:
                        continue
                        
                    desc_hash = hash(desc)
                    if desc_hash in seen_descriptions:
                        continue
                    seen_descriptions.add(desc_hash)

                    diff_label = None
                    raw_diff = item.get('difficulty', None)
                    
                    try:
                        diff_int = int(raw_diff) if raw_diff is not None else -1
                        if diff_int > 0:
                            if diff_int <= 1299 and easy_count < limit_per_class:
                                diff_label = "Easy"
                                easy_count += 1
                            elif 1300 <= diff_int <= 1899 and medium_count < limit_per_class:
                                diff_label = "Medium"
                                medium_count += 1
                            elif diff_int > 1899 and hard_count < limit_per_class:
                                diff_label = "Hard"
                                hard_count += 1
                    except (ValueError, TypeError):
                        if isinstance(raw_diff, str):
                            val = raw_diff.upper()
                            if 'EASY' in val and easy_count < limit_per_class: 
                                diff_label = 'Easy'
                                easy_count += 1
                            elif 'MEDIUM' in val and medium_count < limit_per_class: 
                                diff_label = 'Medium'
                                medium_count += 1
                            elif 'HARD' in val and hard_count < limit_per_class: 
                                diff_label = 'Hard'
                                hard_count += 1
                    
                    if diff_label:
                        problem_id = f"DM_{offset + item_wrapper.get('row_idx', 0)}"
                        cleaned_list.append({
                            "problem_id": problem_id,
                            "platform": f"DeepMind_{source}",
                            "title": name,
                            "title_slug": problem_id.lower(),
                            "difficulty": diff_label,
                            "description": desc,
                            "examples": "",             
                            "constraints": "",          
                            "tags": "",                 
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
                
                # Check target limit
                if easy_count >= limit_per_class and medium_count >= limit_per_class and hard_count >= limit_per_class:
                    print("\n🎯 Target reached for all classes! Stopping extraction.", flush=True)
                    break
                
                print(f"⚡ Fetched {offset + batch_size} questions via API... | Saved -> Easy: {easy_count}, Medium: {medium_count}, Hard: {hard_count}", flush=True)
                
                offset += batch_size
                time.sleep(0.2) # API ko overload na karne ke liye chhota pause
                
            except Exception as e:
                print(f"⚠️ API Fetch Error: {e}")
                break
                
        print(f"\n✅ Extraction complete! Easy: {easy_count}, Medium: {medium_count}, Hard: {hard_count}")
        return cleaned_list

    def save_to_csv(self, data):
        if not data:
            print("❌ No DeepMind data captured to save.")
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
        output_path = os.path.join(config.RAW_DATA_DIR, "deepmind.csv")
        df.to_csv(output_path, index=False)
        print(f"💾 Successfully saved balanced raw DeepMind data to {output_path}")

if __name__ == "__main__":
    dm = DeepMindConnector()
    data = dm.fetch_problems()
    dm.save_to_csv(data)