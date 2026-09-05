# connectors/taco_connector.py
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

class CodeforcesConnector(BaseConnector):
    def __init__(self):
        super().__init__(platform_name="codeforces_openr1")

    def fetch_problems(self):
        print("\n [STEP 1] Fetching Open-R1 CodeForces Dataset via Native Parquet API...")
        print("  NOTE: Using native parquet endpoint with precise rating-based difficulty mapping.\n")
        
        api_url = "https://datasets-server.huggingface.co/parquet?dataset=open-r1/codeforces"
        try:
            print(" Requesting file locations from HF Server...")
            res = requests.get(api_url, timeout=15)
            res.raise_for_status()
            parquet_files = res.json().get("parquet_files", [])
        except Exception as e:
            print(f" API Error: {e}")
            return []

        if not parquet_files:
            print(" No valid Parquet URLs found.")
            return []

        #  THE FIX: Sirf ek config uthayenge taaki duplicate files download hi na hon!
        first_config = parquet_files[0].get('config')
        all_urls = [p['url'] for p in parquet_files if p.get('config') == first_config and p.get('split') == 'train']
        all_urls = list(dict.fromkeys(all_urls)) # Final safety net

        print(f" Found {len(all_urls)} unique backend files for config '{first_config}'. Processing...")

        cleaned_list = []
        seen_descriptions = set()
        
        limit_per_class = 5000  
        easy_c, med_c, hard_c = 0, 0, 0

        for url in all_urls:
            if easy_c >= limit_per_class and med_c >= limit_per_class and hard_c >= limit_per_class:
                print("🎯 Target per class reached!")
                break
                
            try:
                print(f"   -> Downloading chunk: {url.split('/')[-1]}...")
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                
                df_chunk = pd.read_parquet(io.BytesIO(response.content))
                
                for idx, row in df_chunk.iterrows():
                    if easy_c >= limit_per_class and med_c >= limit_per_class and hard_c >= limit_per_class:
                        break
                        
                    desc = str(row.get('description', ''))
                    title = str(row.get('title', ''))
                    raw_id = str(row.get('id', f"{len(cleaned_list)}"))
                    rating = row.get('rating', None)

                    if not desc or len(desc.split()) < 15:
                        continue
                        
                    desc_key = " ".join(desc.split()).lower()
                    if desc_key in seen_descriptions:
                        continue
                    seen_descriptions.add(desc_key)
                    
                    if rating is None or pd.isna(rating):
                        continue
                        
                    try:
                        rating = int(rating)
                    except:
                        continue

                    diff_label = None
                    if rating < 1400 and easy_c < limit_per_class:
                        diff_label = 'Easy'
                        easy_c += 1
                    elif 1400 <= rating <= 2000 and med_c < limit_per_class:
                        diff_label = 'Medium'
                        med_c += 1
                    elif rating > 2000 and hard_c < limit_per_class:
                        diff_label = 'Hard'
                        hard_c += 1
                        
                    if diff_label:
                        unique_problem_id = f"CF_{raw_id}_{len(cleaned_list)}"
                        cleaned_list.append({
                            "problem_id": unique_problem_id,
                            "platform": "CodeForces_OpenR1",
                            "title": title if title else f"CodeForces Problem {raw_id}",
                            "title_slug": unique_problem_id.lower(),
                            "difficulty": diff_label,
                            "description": desc,
                            "examples": "",             
                            "constraints": "",          
                            "tags": str(row.get('tags', '')),                 
                            "companies": "",            
                            "hints": "",                
                            "acceptance_rate": "",      
                            "likes": 0,                 
                            "dislikes": 0,              
                            "premium": False,           
                            "similar_questions": "",    
                            "editorial_available": bool(row.get('editorial')),      
                            "video_solution_available": False, 
                            "source_url": ""
                        })
                        
                del df_chunk
                gc.collect()
                print(f"      -> Status: Easy: {easy_c} | Med: {med_c} | Hard: {hard_c}")
                
            except Exception as e:
                print(f"Failed to process chunk: {e}")

        print(f"\n🎉 CodeForces Extraction complete! Easy: {easy_c}, Medium: {med_c}, Hard: {hard_c}")
        print(f"📊 Total Problems Saved: {len(cleaned_list)}")
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
        output_path = os.path.join(config.RAW_DATA_DIR, "taco.csv")
        df.to_csv(output_path, index=False)
        print(f" Successfully saved clean CodeForces data to {output_path}")

if __name__ == "__main__":
    connector = CodeforcesConnector()
    data = connector.fetch_problems()
    connector.save_to_csv(data)