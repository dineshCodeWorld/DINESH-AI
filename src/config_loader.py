"""
Configuration loader for Dinesh AI
Loads settings from config.yaml in project root
All modules should import configuration from this file
"""

import yaml
from pathlib import Path
import logging
import os
import sys

# Get project root (parent of src directory)
PROJECT_ROOT = Path(__file__).parent.parent

# Determine which config to use
def get_config_file():
    """Determine which config file to use based on environment or command line"""
    # Check command line argument
    if '--config' in sys.argv:
        idx = sys.argv.index('--config')
        if idx + 1 < len(sys.argv):
            config_name = sys.argv[idx + 1]
            return PROJECT_ROOT / config_name
    
    # Check environment variable
    env_config = os.getenv('DINESH_AI_CONFIG')
    if env_config:
        return PROJECT_ROOT / env_config
    
    # Default to production config
    return PROJECT_ROOT / "config.yaml"

# Load configuration from appropriate config file
CONFIG_FILE = get_config_file()

def load_config():
    """Load configuration from config.yaml"""
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(f"Config file not found: {CONFIG_FILE}")
    
    with open(CONFIG_FILE, 'r') as f:
        config = yaml.safe_load(f)
    
    if config is None:
        raise ValueError("Config file is empty")
    
    return config

# Load configuration once at module import
try:
    CONFIG = load_config()
except Exception as e:
    print(f"Warning: Could not load config.yaml: {e}")
    CONFIG = {}

# ==================== PROJECT PATHS ====================
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
CACHE_DIR = PROJECT_ROOT / "cache"

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, LOGS_DIR, CACHE_DIR]:
    directory.mkdir(exist_ok=True)

# ==================== DATA CONFIGURATION ====================
DATA_SOURCES = CONFIG.get('data_sources', {})

# ==================== MODEL CONFIGURATION ====================
model_cfg = CONFIG.get('model', {})
MODEL_CONFIG = {
    "model_type": model_cfg.get("model_type", "custom_gpt"),
    "vocab_size": model_cfg.get("vocab_size", 50000),
    "d_model": model_cfg.get("d_model", 768),
    "num_layers": model_cfg.get("num_layers", 12),
    "num_heads": model_cfg.get("num_heads", 12),
    "d_ff": model_cfg.get("d_ff", 3072),
    "max_length": model_cfg.get("max_length", 512),
    "dropout": model_cfg.get("dropout", 0.1),
}

# ==================== TRAINING CONFIGURATION ====================
training_cfg = CONFIG.get('training', {})
TRAINING_CONFIG = {
    "batch_size": training_cfg.get("batch_size", 16),
    "learning_rate": training_cfg.get("learning_rate", 2e-5),
    "epochs": training_cfg.get("epochs", 3),
    "device": training_cfg.get("device", "cpu"),
    "save_interval": training_cfg.get("save_interval", 500),
    "eval_interval": training_cfg.get("eval_interval", 1000),
    "warmup_steps": training_cfg.get("warmup_steps", 500),
    "weight_decay": training_cfg.get("weight_decay", 0.01),
    "optimizer": training_cfg.get("optimizer", "adamw"),
    "scheduler": training_cfg.get("scheduler", "cosine"),
    "gradient_clip": training_cfg.get("gradient_clip", 1.0),
}

# ==================== DEPLOYMENT CONFIGURATION ====================
deployment_cfg = CONFIG.get('deployment', {})
DEPLOYMENT_CONFIG = {
    "push_to_hub": deployment_cfg.get("push_to_hub", True),
    "hf_repo_name": deployment_cfg.get("hf_repo_name", "dinesh-ai-custom-gpt"),
    "hf_private": deployment_cfg.get("hf_private", False),
    "version_format": deployment_cfg.get("version_format", "v{number}-{date}"),
}

# ==================== CONTINUOUS LEARNING CONFIGURATION ====================
continuous_cfg = CONFIG.get('continuous', {})
CONTINUOUS_CONFIG = {
    "enabled": continuous_cfg.get("enabled", True),
    "collection_interval_hours": continuous_cfg.get("collection_interval_hours", 24),
    "fine_tune_interval_hours": continuous_cfg.get("fine_tune_interval_hours", 72),
    "weekly_versioning": continuous_cfg.get("weekly_versioning", True),
    "version_release_day": continuous_cfg.get("version_release_day", "sunday"),
    "version_release_time": continuous_cfg.get("version_release_time", "00:00"),
    "auto_deploy": continuous_cfg.get("auto_deploy", True),
}

# ==================== LOGGING CONFIGURATION ====================
logging_cfg = CONFIG.get('logging', {})
LOGGING_CONFIG = {
    "level": logging_cfg.get("level", "INFO"),
    "format": logging_cfg.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
}

# ==================== SYSTEM CONFIGURATION ====================
system_cfg = CONFIG.get('system', {})
SYSTEM_CONFIG = {
    "num_workers": system_cfg.get("num_workers", 4),
    "seed": system_cfg.get("seed", 42),
    "deterministic": system_cfg.get("deterministic", True),
}

# ==================== APP CONFIGURATION ====================
app_cfg = CONFIG.get('app', {})
APP_CONFIG = {
    "page_title": app_cfg.get("page_title", "Dinesh AI"),
    "page_icon": app_cfg.get("page_icon", "✨"),
    "layout": app_cfg.get("layout", "wide"),
    "model_cache_ttl": app_cfg.get("model_cache_ttl", 60),
    "dark_theme": app_cfg.get("dark_theme", {}),
    "light_theme": app_cfg.get("light_theme", {}),
    "default_temperature": app_cfg.get("default_temperature", 0.8),
    "default_top_k": app_cfg.get("default_top_k", 50),
    "default_max_length": app_cfg.get("default_max_length", 150),
    "example_prompts": app_cfg.get("example_prompts", []),
}
