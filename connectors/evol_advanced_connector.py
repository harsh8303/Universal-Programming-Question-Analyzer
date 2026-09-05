# connectors/evol_advanced_connector.py
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

class EvolAdvancedConnector(BaseConnector):
    def __init__(self):
        super().__init__(platform_name="evol_advanced")

    def calculate_complexity(self, text):
        text_lower = text.lower()
        score = len(text.split())  # Base score is length
        
        # Add massive bonus points for complex constraints and advanced DSA topics
        if "o(" in text_lower or "time complexity" in text_lower: score += 50
        if "space complexity" in text_lower: score += 50
        if "constraint" in text_lower: score += 30
        
        hard_words = ['dp', 'dynamic programming', 'dijkstra', 'graph', 'tree', 'trie', 'backtracking']
        for hw in hard_words:
            if hw in text_lower: score += 40
            
        medium_words = ['matrix', 'sort', 'binary search', 'linked list', 'hash', 'stack', 'queue']
        for mw in medium_words:
            if mw in text_lower: score += 20
            
        return score

    def fetch_problems(self):
        print("\n [ADVANCED PIPELINE] Fetching 30,000 algorithmic problems from Evol-Instruct...")
        api_url = "https://datasets-server.huggingface.co/parquet?dataset=nickrosh/Evol-Instruct-Code-80k-v1"
        try:
            res = requests.get(api_url, timeout=15)
            res.raise_for_status()
            parquet_files = res.json().get("parquet_files", [])
        except Exception as e:
            print(f" API Error: {e}")
            return []

        all_urls = [p['url'] for p in parquet_files if p.get('split') == 'train']
        
        raw_problems = []
        seen_descriptions = set()
        
        print(" Harvesting raw data...")
        for url in all_urls:
            if len(raw_problems) >= 30000:
                break
                
            try:
                response = requests.get(url, timeout=40)
                df_chunk = pd.read_parquet(io.BytesIO(response.content))
                
                for idx, row in df_chunk.iterrows():
                    if len(raw_problems) >= 30000:
                        break
                        
                    instruction = str(row.get('instruction', '')).strip()
                    if not instruction or len(instruction.split()) < 15:
                        continue
                        
                    desc_key = " ".join(instruction.split()).lower()
                    if desc_key in seen_descriptions:
                        continue
                    seen_descriptions.add(desc_key)
                    
                    # Store raw data with calculated complexity score
                    complexity = self.calculate_complexity(instruction)
                    
                    raw_problems.append({
                        "description": instruction,
                        "score": complexity
                    })
                    
                del df_chunk
                gc.collect()
            except Exception as e:
                print(f" Failed chunk: {e}")

        #  THE PERCENTILE BUCKETING (Ensures 100% Perfect Class Balance)
        print("\n Sorting by algorithmic complexity and balancing classes...")
        raw_problems.sort(key=lambda x: x['score']) # Sort from lowest score to highest
        
        total_extracted = len(raw_problems)
        chunk_size = total_extracted // 3
        
        cleaned_list = []
        for i, item in enumerate(raw_problems):
            if i < chunk_size:
                diff = "Easy"
            elif i < chunk_size * 2:
                diff = "Medium"
            else:
                diff = "Hard"
                
            unique_id = f"EVOL_ADV_{i}"
            cleaned_list.append({
                "problem_id": unique_id,
                "platform": "Evol_Advanced",
                "title": f"Advanced Algorithmic Problem {i}",
                "title_slug": unique_id.lower(),
                "difficulty": diff,
                "description": item['description'],
                "examples": "", "constraints": "", "tags": f"{diff.lower()}_logic",
                "companies": "", "hints": "", "acceptance_rate": "",
                "likes": 0, "dislikes": 0, "premium": False,
                "similar_questions": "", "editorial_available": False,
                "video_solution_available": False, "source_url": ""
            })

        print("\n BALANCED DATASET DISTRIBUTION:")
        df_check = pd.DataFrame(cleaned_list)
        print(df_check['difficulty'].value_counts())
            
        print(f"\n Total Balanced Problems Extracted: {len(cleaned_list)}")
        return cleaned_list

    def save_to_csv(self, data):
        if not data: return
        df = pd.DataFrame(data)
        output_path = os.path.join(config.RAW_DATA_DIR, "evol_advanced.csv")
        df.to_csv(output_path, index=False)
        print(f" DATA SECURED! Saved to {output_path}")

if __name__ == "__main__":
    connector = EvolAdvancedConnector()
    data = connector.fetch_problems()
    connector.save_to_csv(data)