# connectors/base_connector.py
import os
import sys
import time
import requests
from abc import ABC, abstractmethod

# Root directory ko path mein sabse pehle add karo
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.logger import setup_logger
import config

class BaseConnector(ABC):
    def __init__(self, platform_name):
        self.platform_name = platform_name
        self.logger = setup_logger(f"{platform_name}_connector")
        
    def make_request(self, url, method="GET", params=None, json_data=None, headers=None):
        retries = config.MAX_RETRIES
        backoff = config.BACKOFF_FACTOR
        
        for attempt in range(retries):
            try:
                if method.upper() == "GET":
                    response = requests.get(url, params=params, headers=headers, timeout=config.REQUEST_TIMEOUT)
                elif method.upper() == "POST":
                    response = requests.post(url, json=json_data, headers=headers, timeout=config.REQUEST_TIMEOUT)
                
                response.raise_for_status()
                return response
                
            except requests.exceptions.RequestException as e:
                self.logger.warning(f"Attempt {attempt + 1} failed for URL: {url}. Error: {e}")
                if attempt < retries - 1:
                    sleep_time = backoff ** attempt
                    self.logger.info(f"Sleeping for {sleep_time} seconds before retrying...")
                    time.sleep(sleep_time)
                else:
                    self.logger.error(f"All retries failed for URL: {url}")
                    return None

    @abstractmethod
    def fetch_problems(self):
        pass

    @abstractmethod
    def save_to_csv(self, data):
        pass