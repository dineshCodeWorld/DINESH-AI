from huggingface_hub import HfApi
from datetime import datetime
from pathlib import Path
import os
import re

def get_next_version_number(api, repo_id, token):
    """Get next version number by checking existing versions in HF repo"""
    try:
        files = api.list_repo_files(repo_id=repo_id, token=token)
        version_files = [f for f in files if f.startswith('versions/weekly/') and 'v' in f]
        
        if not version_files:
            return 1
        
        # Extract version numbers from filenames like dinesh_ai_model_v1_2026-02-28_143052.pth
        version_numbers = []
        for f in version_files:
            match = re.search(r'_v(\d+)_', f)
            if match:
                version_numbers.append(int(match.group(1)))
        
        return max(version_numbers) + 1 if version_numbers else 1
    except:
        return 1

def find_latest_model():
    """Find the most recently created model directory"""
    models_dir = Path("models")
    if not models_dir.exists():
        raise FileNotFoundError("Models directory not found")
    
    model_dirs = [d for d in models_dir.iterdir() 
                  if d.is_dir() and d.name.startswith("model_")]
    
    if not model_dirs:
        raise FileNotFoundError("No trained model found in models/")
    
    # Get most recent by modification time
    latest = max(model_dirs, key=lambda x: x.stat().st_mtime)
    model_file = latest / "model.pt"
    
    if not model_file.exists():
        raise FileNotFoundError(f"model.pt not found in {latest}")
    
    return model_file

def create_weekly_version():
    token = os.environ.get('HF_TOKEN')
    repo_id = os.environ.get('HF_REPO')
    
    if not token or not repo_id:
        print("❌ HF_TOKEN or HF_REPO not set")
        return False
    
    try:
        # Find latest trained model
        model_path = find_latest_model()
        print(f"📦 Found model: {model_path}")
        
        api = HfApi()
        
        # Get next version number
        version_num = get_next_version_number(api, repo_id, token)
        timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
        version = f"v{version_num}_{timestamp}"
        
        print(f"📦 Creating version: {version}")
        
        api.upload_file(
            path_or_fileobj=str(model_path),
            path_in_repo=f"versions/weekly/dinesh_ai_model_{version}.pth",
            repo_id=repo_id,
            token=token,
            commit_message=f"Version {version}"
        )
        
        print(f"✅ Weekly version created: v{version}")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    create_weekly_version()
