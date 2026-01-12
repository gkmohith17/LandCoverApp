"""
Helper Functions
================

Common utility functions used throughout the application.
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path

def create_folder(path):
    """Create a folder if it doesn't exist"""
    Path(path).mkdir(parents=True, exist_ok=True)

def save_config(config, filepath):
    """Save configuration to JSON file"""
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=4)

def load_config(filepath):
    """Load configuration from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def get_file_size_mb(filepath):
    """Get file size in megabytes"""
    size_bytes = os.path.getsize(filepath)
    return round(size_bytes / (1024 * 1024), 2)

def print_dataset_info(data):
    """Print dataset information"""
    print("\n" + "="*50)
    print("📊 DATASET INFORMATION")
    print("="*50)
    print(f"Total samples: {len(data):,}")
    if 'label' in data.columns:
        print(f"\nClass distribution:")
        print(data['label'].value_counts())
    print("="*50 + "\n")
