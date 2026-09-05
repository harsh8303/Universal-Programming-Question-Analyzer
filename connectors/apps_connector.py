# connectors/apps_connector.py
import os
import sys
import time
import requests
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from base_connector import BaseConnector
import config

class AppsConnector(BaseConnector):
    def __init__(self):
        super().__init__(platform_name="apps")

    def fetch_problems(self):
        print("\n🚀 [STEP 1] Fetching APPS Dataset with Safe Batching & Medium Priority...")
        
        output_path = os.path.join(config.RAW_DATA_DIR, "apps.csv")
        seen_descriptions = set()
        
        # Load existing data if file already exists so we don't lose anything
        all_cleaned_list = []
        if os.path.exists(output_path):
            try:
                existing_df = pd.read_csv(output_path)
                if not existing_df.empty and 'description' in existing_df.columns:
                    all_cleaned_list = existing_df.to_dict('records')
                    for item in all_cleaned_list:
                        desc = item.get('description', '')
                        if desc:
                            seen_descriptions.add(" ".join(desc.split()).lower())
                    print(f"📂 Loaded {len(all_cleaned_list)} existing problems from apps.csv (Appending mode active).")
            except Exception as e:
                print(f"⚠️ Could not load existing CSV: {e}")

        configs_mapping = {
            'introductory': 'Easy',
            'interview': 'Medium',
            'competition': 'Hard'
        }
        
        limit_per_class = 3000  # Safe target per class to prevent server blocks
        
        for cfg_name, diff_label in configs_mapping.items():
            print(f"\n📂 Fetching config: '{cfg_name}' as '{diff_label}'...")
            offset = 0
            batch_size = 50  # Smaller batches to avoid 502 server timeouts
            count_for_cfg = sum(1 for x in all_cleaned_list if x['difficulty'] == diff_label)
            
            if count_for_cfg >= limit_per_class:
                print(f"   -> Already have {count_for_cfg} {diff_label} problems. Skipping.")
                continue

            consecutive_failures = 0
            
            while count_for_cfg < limit_per_class:
                url = f"https://datasets-server.huggingface.co/rows?dataset=codeparrot/apps&config={cfg_name}&split=train&offset={offset}&length={batch_size}"
                
                success = False
                data = None
                
                for attempt in range(4):
                    try:
                        response = requests.get(url, timeout=20)
                        if response.status_code == 200:
                            data = response.json()
                            success = True
                            consecutive_failures = 0
                            break
                        elif response.status_code in [502, 504, 429]:
                            time.sleep(3 * (attempt + 1))
                        else:
                            break
                    except requests.RequestException:
                        time.sleep(3)
                
                if not success or not data:
                    consecutive_failures += 1
                    print(f"⚠️ Server resting at offset {offset} for {cfg_name}. Retrying shift...")
                    offset += batch_size
                    time.sleep(2)
                    if consecutive_failures > 5:
                        print(f"⏩ Server too busy for {cfg_name}. Moving ahead.")
                        break
                    continue
                    
                rows = data.get('rows', [])
                if not rows:
                    print(f"✅ Reached natural end of {cfg_name} dataset.")
                    break
                    
                added_in_batch = 0
                for item_wrapper in rows:
                    item = item_wrapper.get('row', {})
                    desc = item.get('question', '')
                    raw_id = item.get('problem_id', str(offset))
                    
                    if not desc or len(desc.split()) < 15:
                        continue
                        
                    desc_key = " ".join(desc.split()).lower()
                    if desc_key in seen_descriptions:
                        continue
                    seen_descriptions.add(desc_key)
                    
                    unique_problem_id = f"APPS_{cfg_name}_{raw_id}_{len(all_cleaned_list)}"
                        
                    all_cleaned_list.append({
                        "problem_id": unique_problem_id,
                        "platform": f"APPS_{cfg_name}",
                        "title": f"APPS Problem {raw_id}",
                        "title_slug": unique_problem_id.lower(),
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
                    
                    count_for_cfg += 1
                    added_in_batch += 1
                    if count_for_cfg >= limit_per_class:
                        break
                
                print(f"   -> Offset {offset} processed | Total {diff_label}: {count_for_cfg}", flush=True)
                
                if count_for_cfg >= limit_per_class:
                    break
                
                if len(rows) < batch_size:
                    break
                    
                offset += batch_size
                time.sleep(0.8) # Polite delay to keep connection alive
                
        # Final stats
        easy_c = sum(1 for x in all_cleaned_list if x['difficulty'] == 'Easy')
        med_c = sum(1 for x in all_cleaned_list if x['difficulty'] == 'Medium')
        hard_c = sum(1 for x in all_cleaned_list if x['difficulty'] == 'Hard')
        
        print(f"\n🎉 APPS Extraction complete! Easy: {easy_c}, Medium: {med_c}, Hard: {hard_c}")
        return all_cleaned_list

    def save_to_csv(self, data):
        if not data:
            print("❌ No APPS data captured.")
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
        output_path = os.path.join(config.RAW_DATA_DIR, "apps.csv")
        df.to_csv(output_path, index=False)
        print(f"💾 Successfully saved total {len(df)} rows to {output_path}")

if __name__ == "__main__":
    app_connector = AppsConnector()
    data = app_connector.fetch_problems()
    app_connector.save_to_csv(data)