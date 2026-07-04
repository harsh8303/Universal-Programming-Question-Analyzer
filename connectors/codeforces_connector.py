# connectors/codeforces_connector.py
import os
import sys
import pandas as pd

# Setup paths to ensure cross-module imports work perfectly
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from base_connector import BaseConnector
import config

class CodeforcesConnector(BaseConnector):
    def __init__(self):
        super().__init__(platform_name="codeforces")
        # Official Codeforces API endpoint
        self.api_url = "https://codeforces.com/api/problemset.problems"

    def map_difficulty(self, rating):
        """
        Maps Codeforces numerical rating to Universal difficulty bands.
        Rating < 1200 → Easy
        Rating 1200–1900 → Medium
        Rating > 1900 → Hard
        """
        if rating is None:
            return "Unknown"
            
        if rating < 1200:
            return "Easy"
        elif 1200 <= rating <= 1900:
            return "Medium"
        else:
            return "Hard"

    def fetch_problems(self):
        self.logger.info("Fetching master problem list from Codeforces API...")
        response = self.make_request(self.api_url)
        
        if not response:
            self.logger.error("Failed to fetch Codeforces data.")
            return []
            
        data = response.json()
        if data.get("status") != "OK":
            self.logger.error(f"API Error: {data.get('comment', 'Unknown error')}")
            return []
            
        problems = data["result"]["problems"]
        self.logger.info(f"Total problems found on Codeforces: {len(problems)}")
        self.logger.info("Starting Universal Schema mapping for Codeforces...")
        
        cleaned_list = []
        
        for p in problems:
            contest_id = p.get("contestId")
            index = p.get("index")
            
            # Skip if critical identifiers are missing
            if contest_id is None or index is None:
                continue
                
            unique_id = f"{contest_id}{index}"
            cf_rating = p.get("rating")
            
            # --- UNIVERSAL SCHEMA MAPPING (Strictly per Specification V1) ---
            cleaned_list.append({
                "problem_id": unique_id,
                "platform": "Codeforces",
                "title": p.get("name", ""),
                "title_slug": unique_id.lower(),
                "difficulty": self.map_difficulty(cf_rating),
                "description": "",          # Not Available (left empty)
                "examples": "",             # Not Available (left empty)
                "constraints": "",          # Not Available (left empty)
                "tags": ",".join(p.get("tags", [])),
                "companies": "",            # Not Available (left empty)
                "hints": "",                # Not Available (left empty)
                "acceptance_rate": "",      # Not Available (left empty)
                "likes": 0,                 # Default Value
                "dislikes": 0,              # Default Value
                "premium": False,           # Default Value
                "similar_questions": "",    # Not Available (left empty)
                "editorial_available": False,      # Default Value
                "video_solution_available": False, # Default Value
                "source_url": f"https://codeforces.com/problemset/problem/{contest_id}/{index}"
            })
            
        return cleaned_list

    def save_to_csv(self, data):
        if not data:
            self.logger.warning("No Codeforces data captured to save.")
            return
            
        df = pd.DataFrame(data)
        
        # Enforcing the Universal Schema order strictly (19 Columns)
        columns_order = [
            "problem_id", "platform", "title", "title_slug", "difficulty", 
            "description", "examples", "constraints", "tags", "companies", 
            "hints", "acceptance_rate", "likes", "dislikes", "premium", 
            "similar_questions", "editorial_available", "video_solution_available", "source_url"
        ]
        
        # Failsafe: Ensure all columns exist before saving
        for col in columns_order:
            if col not in df.columns:
                df[col] = ""
                
        df = df[columns_order]
        
        output_path = os.path.join(config.RAW_DATA_DIR, "codeforces.csv")
        df.to_csv(output_path, index=False)
        self.logger.info(f"Successfully saved raw Codeforces data to {output_path}")

if __name__ == "__main__":
    cf = CodeforcesConnector()
    data = cf.fetch_problems()
    cf.save_to_csv(data)