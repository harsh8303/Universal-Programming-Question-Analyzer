# connectors/codechef_connector.py
import os
import sys
import time
import requests
import pandas as pd

# Path setup to ensure cross-module imports work perfectly
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from base_connector import BaseConnector
import config

class CodeChefConnector(BaseConnector):
    def __init__(self):
        super().__init__(platform_name="codechef")
        # 🔥 The new API endpoint you found!
        self.api_url = "https://www.codechef.com/api/practice/catalog"

    def map_difficulty(self, rating):
        """
        Maps CodeChef numerical rating to Universal difficulty bands.
        Rating < 1400 -> Easy
        1400 - 2000 -> Medium
        > 2000 -> Hard
        """
        try:
            val = float(rating)
            if val < 1400:
                return "Easy"
            elif 1400 <= val <= 2000:
                return "Medium"
            elif val > 2000:
                return "Hard"
        except (ValueError, TypeError):
            return "Unknown"
        return "Unknown"

    def fetch_problems(self):
        self.logger.info("Starting CodeChef extraction using the New Catalog API...")
        cleaned_list = []
        page = 0  # The new API usually starts pagination from 0
        limit = 50 
        
        session = requests.Session()
        
        # Fresh browser-like headers
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
            "Referer": "https://www.codechef.com/practice",
            "Origin": "https://www.codechef.com"
        }
        
        # Pre-flight request to establish connection
        session.get("https://www.codechef.com/practice", headers=headers, timeout=10)
        
        while True:
            self.logger.info(f"Fetching CodeChef Page {page + 1}...")
            
            # Simplified parameters for the new catalog API
            params = {
                "page": page,
                "limit": limit
            }
            
            try:
                response = session.get(self.api_url, params=params, headers=headers, timeout=15)
                
                if response.status_code != 200:
                    self.logger.error(f"Server rejected request. Status Code: {response.status_code}")
                    break
                    
                data = response.json()
                
            except Exception as e:
                self.logger.error(f"Request/Parsing failed at page {page}: {e}")
                break
            
            # The new catalog API structure usually wraps data inside payload -> data
            payload = data.get("payload", data)
            problem_list = payload.get("data", []) if isinstance(payload, dict) else []
                
            if not problem_list or len(problem_list) == 0:
                self.logger.info("No more problems found. Pagination complete.")
                break
                
            for p in problem_list:
                problem_code = p.get("code", "")
                title = p.get("name", "")
                
                if not problem_code or not title:
                    continue
                    
                accuracy = p.get("accuracy", "")
                if accuracy:
                    try:
                        accuracy = f"{round(float(accuracy), 2)}%"
                    except ValueError:
                        accuracy = str(accuracy)
                        
                cleaned_list.append({
                    "problem_id": problem_code,
                    "platform": "CodeChef",
                    "title": title,
                    "title_slug": problem_code.lower(),
                    "difficulty": self.map_difficulty(p.get("rating")),
                    "description": "", "examples": "", "constraints": "", 
                    "tags": "", "companies": "", "hints": "",                
                    "acceptance_rate": accuracy, "likes": 0, "dislikes": 0,              
                    "premium": False, "similar_questions": "",    
                    "editorial_available": p.get("has_solution", False),
                    "video_solution_available": p.get("has_video_editorial", False),
                    "source_url": f"https://www.codechef.com/problems/{problem_code}"
                })
                
            page += 1
            time.sleep(1.5) 
            
        self.logger.info(f"Successfully processed {len(cleaned_list)} CodeChef problems.")
        return cleaned_list

    def save_to_csv(self, data):
        if not data:
            self.logger.warning("No CodeChef data captured to save.")
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
        
        output_path = os.path.join(config.RAW_DATA_DIR, "codechef.csv")
        df.to_csv(output_path, index=False)
        self.logger.info(f"Successfully saved raw CodeChef data to {output_path}")

if __name__ == "__main__":
    cc = CodeChefConnector()
    data = cc.fetch_problems()
    cc.save_to_csv(data)