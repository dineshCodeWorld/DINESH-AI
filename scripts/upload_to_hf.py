from huggingface_hub import HfApi, create_repo, login
from datetime import datetime
from pathlib import Path
import os, sys

def find_latest_model():
    """Find the most recently created model directory"""
    models_dir = Path("models")
    if not models_dir.exists():
        raise FileNotFoundError("Models directory not found")
    
    model_dirs = [d for d in models_dir.iterdir() 
                  if d.is_dir() and d.name.startswith("model_")]
    
    if not model_dirs:
        raise FileNotFoundError("No trained model found in models/")
    
    latest = max(model_dirs, key=lambda x: x.stat().st_mtime)
    return latest

def upload_model(repo_id=None, token=None):
    if token is None:
        token = os.environ.get('HF_TOKEN')
    if token is None:
        print("❌ HF_TOKEN not found. Set: export HF_TOKEN=hf_your_token")
        return False
    if repo_id is None:
        repo_id = os.environ.get('HF_REPO', 'yourusername/dinesh-ai')
    
    print(f"📤 Uploading to {repo_id}...")
    login(token=token)
    api = HfApi()
    
    try:
        create_repo(repo_id, token=token, exist_ok=True, repo_type="model")
    except: pass
    
    try:
        # Find latest model directory
        model_dir = find_latest_model()
        print(f"Found model directory: {model_dir}")
        
        model_file = model_dir / "model.pt"
        tokenizer_file = model_dir / "tokenizer.json"
        config_file = model_dir / "model_config.json"
        
        if not model_file.exists():
            print(f"❌ Model file not found: {model_file}")
            return False
        
        # Upload model
        api.upload_file(
            path_or_fileobj=str(model_file),
            path_in_repo="dinesh_ai_model.pth",
            repo_id=repo_id,
            token=token,
            commit_message=f"Update {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        )
        print("✅ Model uploaded")
        
        # Upload tokenizer
        if tokenizer_file.exists():
            api.upload_file(
                path_or_fileobj=str(tokenizer_file),
                path_in_repo="tokenizer.json",
                repo_id=repo_id,
                token=token
            )
            print("✅ Tokenizer uploaded")
        
        # Upload config
        if config_file.exists():
            api.upload_file(
                path_or_fileobj=str(config_file),
                path_in_repo="model_config.json",
                repo_id=repo_id,
                token=token
            )
            print("✅ Config uploaded")
        
        # Create version backup
        version = datetime.now().strftime('%Y-%m-%d')
        api.upload_file(
            path_or_fileobj=str(model_file),
            path_in_repo=f"versions/dinesh_ai_model_v{version}.pth",
            repo_id=repo_id,
            token=token,
            commit_message=f"Version {version}"
        )
        print(f"✅ Version backup: v{version}")
        print(f"🔗 https://huggingface.co/{repo_id}")
        return True
        
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return False

if __name__ == "__main__":
    repo_id = sys.argv[1] if len(sys.argv) > 1 else None
    upload_model(repo_id=repo_id)
