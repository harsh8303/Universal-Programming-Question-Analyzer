# connectors/leetcode_connector.py
import os
import sys
import time
import json
import pandas as pd
from bs4 import BeautifulSoup

# Setup paths to ensure cross-module imports work perfectly
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from base_connector import BaseConnector
import config

class LeetCodeConnector(BaseConnector):
    def __init__(self):
        super().__init__(platform_name="leetcode")
        self.graphql_url = "https://leetcode.com/graphql"
        
    def fetch_all_problems_list(self):
        self.logger.info("Fetching master problem list from LeetCode API...")
        all_problems_url = "https://leetcode.com/api/problems/all/"
        response = self.make_request(all_problems_url)
        
        if not response:
            self.logger.error("Failed to fetch LeetCode master problem list.")
            return []
            
        data = response.json()
        return data.get("stat_status_pairs", [])

    def fetch_problem_details(self, title_slug):
        # Updated GraphQL query to fetch editorial and video solution data
        query = """
        query questionData($titleSlug: String!) {
            question(titleSlug: $titleSlug) {
                questionId
                title
                titleSlug
                content
                difficulty
                topicTags { name }
                hints
                stats
                likes
                dislikes
                isPaidOnly
                similarQuestions
                hasSolution
                hasVideoSolution
            }
        }
        """
        variables = {"titleSlug": title_slug}
        
        response = self.make_request(
            url=self.graphql_url,
            method="POST",
            json_data={"query": query, "variables": variables},
            headers={"Content-Type": "application/json"}
        )
        
        if response:
            res_json = response.json()
            if "data" in res_json and res_json["data"]["question"]:
                return res_json["data"]["question"]
        return None

    def fetch_problems(self):
        raw_pairs = self.fetch_all_problems_list()
        if not raw_pairs:
            return []

        total_problems = len(raw_pairs)
        self.logger.info(f"Total problems found on LeetCode: {total_problems}")
        
        cleaned_list = []
        
        # Yahan humne limit hata kar 'total_problems' daal diya hai
        self.logger.info(f"Starting detailed extraction for ALL {total_problems} problems with Universal Schema...")
        
        # Loop ab poore 'raw_pairs' par chalega bina kisi [:limit] ke
        for index, pair in enumerate(raw_pairs):
            stat = pair.get("stat", {})
            title_slug = stat.get("question__title_slug")
            
            if not title_slug:
                continue
                
            self.logger.info(f"[{index + 1}/{total_problems}] Fetching details for slug: {title_slug}")
            details = self.fetch_problem_details(title_slug)
            
            if details:
                # 1. Tags extraction
                tags = ",".join([t["name"] for t in details.get("topicTags", [])])
                
                # 2. Hints processing
                hints = " | ".join(details.get("hints", []))
                
                # 3. Stats JSON processing
                stats_str = details.get("stats", "{}")
                try:
                    stats_dict = json.loads(stats_str)
                    acceptance_rate = stats_dict.get("acRate", "")
                except:
                    acceptance_rate = ""

                # 4. Similar Questions parsing
                similar_q_str = details.get("similarQuestions", "[]")
                try:
                    similar_qs = json.loads(similar_q_str)
                    similar_questions = ",".join([q.get("titleSlug", "") for q in similar_qs])
                except:
                    similar_questions = ""

                # 5. HTML Cleaning & Separation
                raw_html = details.get("content") or ""
                clean_text = BeautifulSoup(raw_html, "html.parser").get_text(separator=" ")
                clean_text = " ".join(clean_text.split())
                
                desc_part = clean_text
                examples_part = ""
                constraints_part = ""
                
                if "Example 1:" in clean_text:
                    parts = clean_text.split("Example 1:")
                    desc_part = parts[0].strip()
                    rest_text = "Example 1:" + parts[1]
                    
                    if "Constraints:" in rest_text:
                        sub_parts = rest_text.split("Constraints:")
                        examples_part = sub_parts[0].strip()
                        constraints_part = sub_parts[1].strip()
                    else:
                        examples_part = rest_text.strip()
                elif "Constraints:" in clean_text:
                    parts = clean_text.split("Constraints:")
                    desc_part = parts[0].strip()
                    constraints_part = parts[1].strip()

                # 6. UNIVERSAL SCHEMA MAPPING
                cleaned_list.append({
                    "problem_id": details.get("questionId", ""),
                    "platform": "LeetCode",
                    "title": details.get("title", ""),
                    "title_slug": details.get("titleSlug", title_slug),
                    "difficulty": details.get("difficulty", ""),
                    "description": desc_part,
                    "examples": examples_part,
                    "constraints": constraints_part,
                    "tags": tags,
                    "companies": "", # Placeholder to maintain universal schema
                    "hints": hints,
                    "acceptance_rate": acceptance_rate,
                    "likes": details.get("likes", 0),
                    "dislikes": details.get("dislikes", 0),
                    "premium": details.get("isPaidOnly", False),
                    "similar_questions": similar_questions,
                    "editorial_available": details.get("hasSolution", False),
                    "video_solution_available": details.get("hasVideoSolution", False),
                    "source_url": f"https://leetcode.com/problems/{title_slug}/"
                })
                
            time.sleep(1.5) 
            
        return cleaned_list
    def save_to_csv(self, data):
        if not data:
            self.logger.warning("No LeetCode data captured to save.")
            return
            
        df = pd.DataFrame(data)
        
        # Enforcing the Universal Schema order strictly
        columns_order = [
            "problem_id", "platform", "title", "title_slug", "difficulty", 
            "description", "examples", "constraints", "tags", "companies", 
            "hints", "acceptance_rate", "likes", "dislikes", "premium", 
            "similar_questions", "editorial_available", "video_solution_available", "source_url"
        ]
        df = df[columns_order]
        
        output_path = os.path.join(config.RAW_DATA_DIR, "leetcode.csv")
        df.to_csv(output_path, index=False)
        self.logger.info(f"Successfully saved raw LeetCode data to {output_path}")

if __name__ == "__main__":
    lc = LeetCodeConnector()
    data = lc.fetch_problems()
    lc.save_to_csv(data)

        