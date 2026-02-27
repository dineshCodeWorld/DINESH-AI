from huggingface_hub import HfApi
from datetime import datetime
from pathlib import Path
import os

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
        version = datetime.now().strftime('%Y-%m-%d')
        
        print(f"📦 Creating weekly version: v{version}")
        
        api.upload_file(
            path_or_fileobj=str(model_path),
            path_in_repo=f"versions/weekly/dinesh_ai_model_v{version}.pth",
            repo_id=repo_id,
            token=token,
            commit_message=f"Weekly version {version}"
        )
        
        print(f"✅ Weekly version created: v{version}")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    create_weekly_version()
