# config.py
import os

# Base Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
CLEAN_DATA_DIR = os.path.join(DATA_DIR, "clean")
MERGED_DATA_DIR = os.path.join(DATA_DIR, "merged")

# Automatically create directories if they don't exist
for folder in [RAW_DATA_DIR, CLEAN_DATA_DIR, MERGED_DATA_DIR]:
    os.makedirs(folder, exist_ok=True)

# API Configurations
REQUEST_TIMEOUT = 15
BACKOFF_FACTOR = 2
MAX_RETRIES = 3