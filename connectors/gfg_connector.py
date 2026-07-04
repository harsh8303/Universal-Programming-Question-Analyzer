# connectors/gfg_connector.py
import os
import sys
import time
import json
import pandas as pd
import requests
from bs4 import BeautifulSoup

# Path setup for internal module access
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from base_connector import BaseConnector
import config

class GFGConnector(BaseConnector):
    def __init__(self):
        super().__init__(platform_name="geeksforgeeks")
        # Direct website URL instead of the blocked API
        self.base_url = "https://practice.geeksforgeeks.org/explore"

    def map_difficulty(self, val):
        """Standardizes GFG difficulty integers to Universal schema bands."""
        try:
            val = int(val)
            if val <= 1: return "Easy"
            elif val == 2: return "Medium"
            elif val >= 3: return "Hard"
        except:
            return "Unknown"
        return "Unknown"

    def fetch_problems(self):
        self.logger.info("Starting GFG extraction using Next.js HTML Parsing (Bypassing API)...")
        cleaned_list = []
        page = 1
        
        session = requests.Session()
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1"
        }
        
        # Testing limit - remove or increase for full dataset
        testing_limit_pages = 3 
        
        while page <= testing_limit_pages:
            self.logger.info(f"Scraping HTML for GFG Page {page}...")
            url = f"{self.base_url}?page={page}&sortBy=submissions"
            
            try:
                response = session.get(url, headers=headers, timeout=15)
                if response.status_code != 200:
                    self.logger.error(f"Failed to load page. Status Code: {response.status_code}")
                    break
                
                # Parsing HTML to find the hidden Next.js JSON data
                soup = BeautifulSoup(response.text, 'html.parser')
                next_data_script = soup.find('script', id='__NEXT_DATA__')
                
                if not next_data_script:
                    self.logger.error("Could not find __NEXT_DATA__ script tag. GFG might have changed their UI structure.")
                    break
                    
                json_data = json.loads(next_data_script.string)
                
                # Navigating the deeply nested JSON structure of Next.js
                # Note: The exact path might vary slightly based on GFG's current build, 
                # but it generally resides inside pageProps -> initialData
                try:
                    # Generic traversal to find the 'results' or 'problems' array
                    page_props = json_data.get('props', {}).get('pageProps', {})
                    
                    # Depending on the route, it might be nested differently. 
                    # We look for standard keys GFG uses for the problem list.
                    problems_list = []
                    
                    # Deep search for the problem array in the JSON state
                    def extract_problems(obj):
                        if isinstance(obj, dict):
                            if 'results' in obj and isinstance(obj['results'], list) and len(obj['results']) > 0 and 'problem_name' in obj['results'][0]:
                                return obj['results']
                            for key, value in obj.items():
                                res = extract_problems(value)
                                if res: return res
                        elif isinstance(obj, list):
                            for item in obj:
                                res = extract_problems(item)
                                if res: return res
                        return None

                    problems_list = extract_problems(page_props)
                    
                    if not problems_list:
                        self.logger.warning(f"No problems found in JSON on page {page}. Ending extraction.")
                        break

                    for p in problems_list:
                        title_slug = p.get("problem_url", "").strip("/")
                        cleaned_list.append({
                            "problem_id": str(p.get("id", "")),
                            "platform": "GeeksforGeeks",
                            "title": p.get("problem_name", ""),
                            "title_slug": title_slug,
                            "difficulty": self.map_difficulty(p.get("difficulty")),
                            "description": "", "examples": "", "constraints": "", 
                            "tags": ",".join([t.get("name", "") for t in p.get("topic_tags", [])]) if isinstance(p.get("topic_tags"), list) else "",
                            "companies": ",".join([c.get("name", "") for c in p.get("company_tags", [])]) if isinstance(p.get("company_tags"), list) else "",
                            "hints": "", "acceptance_rate": str(p.get("accuracy", "")), 
                            "likes": 0, "dislikes": 0, "premium": p.get("is_premium", False),
                            "similar_questions": "", "editorial_available": p.get("has_editorial", False),
                            "video_solution_available": p.get("has_video", False),
                            "source_url": f"https://practice.geeksforgeeks.org/problems/{title_slug}/1"
                        })
                        
                except Exception as e:
                    self.logger.error(f"JSON traversal error: {e}")
                    break
                    
                page += 1
                time.sleep(3) # Be gentle with HTML scraping
                
            except Exception as e:
                self.logger.error(f"Request error: {e}")
                break
            
        self.logger.info(f"Successfully processed {len(cleaned_list)} GFG problems via Next.js HTML Scraping.")
        return cleaned_list

    def save_to_csv(self, data):
        if not data: return
        df = pd.DataFrame(data)
        
        # Enforcing schema order per specification
        columns_order = [
            "problem_id", "platform", "title", "title_slug", "difficulty", 
            "description", "examples", "constraints", "tags", "companies", 
            "hints", "acceptance_rate", "likes", "dislikes", "premium", 
            "similar_questions", "editorial_available", "video_solution_available", "source_url"
        ]
        
        for col in columns_order:
            if col not in df.columns: df[col] = ""
                
        df = df[columns_order]
        df.to_csv(os.path.join(config.RAW_DATA_DIR, "gfg.csv"), index=False)
        self.logger.info("Saved raw GFG data to gfg.csv.")

if __name__ == "__main__":
    gfg = GFGConnector()
    gfg.save_to_csv(gfg.fetch_problems())