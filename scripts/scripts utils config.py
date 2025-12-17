"""
scripts/utils/config.py
Configuration loading and validation utilities.
"""

import yaml
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def load_config(config_path):
    """
    Load YAML configuration file.
    
    Args:
        config_path (str): Path to YAML config file
    
    Returns:
        dict: Configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Loaded configuration from {config_path}")
    return config


def load_fred_series(config_path):
    """
    Load FRED series configuration.
    
    Args:
        config_path (str): Path to YAML config
    
    Returns:
        dict: FRED series mapping
    """
    config = load_config(config_path)
    return config.get('fred_series', {})


def load_json_config(config_path):
    """
    Load JSON configuration file.
    
    Args:
        config_path (str): Path to JSON config
    
    Returns:
        dict: Configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    logger.info(f"Loaded JSON configuration from {config_path}")
    return config


def validate_config(config):
    """
    Validate configuration dictionary for required fields.
    
    Args:
        config (dict): Configuration dictionary
    
    Returns:
        bool: True if valid, raises exception otherwise
    """
    required_keys = [
        'data_paths',
        'date_range',
        'data_split',
        'feature_engineering',
        'target',
    ]
    
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")
    
    return True
