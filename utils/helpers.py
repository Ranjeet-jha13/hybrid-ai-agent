"""
Helper functions for the project
"""

import os
import yaml
import torch
import numpy as np
from datetime import datetime


def load_config(config_path="configs/config.yaml"):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_directories(config):
    """Create necessary directories if they don't exist"""
    dirs = [
        config['paths']['data'],
        config['paths']['logs'],
        config['paths']['models'],
        config['paths']['manuals']
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    
    print("✓ Directories setup complete")


def get_device():
    """Get available device (CPU/GPU)"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


def set_seed(seed=42):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    print(f"✓ Random seed set to {seed}")


def get_timestamp():
    """Get current timestamp string"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


class Logger:
    """Simple logger for training metrics"""
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, f"training_{get_timestamp()}.log")
        
    def log(self, message):
        """Log message to file and console"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        
        with open(self.log_file, 'a') as f:
            f.write(log_message + '\n')


if __name__ == "__main__":
    # Test utilities
    print("Testing utilities...\n")
    
    config = load_config()
    print(f"✓ Config loaded: {config['project']['name']}")
    
    setup_directories(config)
    
    device = get_device()
    
    set_seed(42)
    
    logger = Logger()
    logger.log("Test log message")
    
    print("\n✓ All utilities working!")