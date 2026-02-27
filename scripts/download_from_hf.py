from huggingface_hub import hf_hub_download, list_repo_files
import os, sys

def download_model(repo_id=None, version=None):
    if repo_id is None:
        repo_id = os.environ.get('HF_REPO', 'yourusername/dinesh-ai')
    
    print(f"📥 Downloading from {repo_id}...")
    os.makedirs("models", exist_ok=True)
    
    try:
        filename = f"versions/dinesh_ai_model_v{version}.pth" if version else "dinesh_ai_model.pth"
        print(f"Downloading {filename}...")
        
        model_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir="models",
            local_dir_use_symlinks=False
        )
        print(f"✅ Model: {model_path}")
        
        tokenizer_path = hf_hub_download(
            repo_id=repo_id,
            filename="tokenizer.json",
            local_dir="models",
            local_dir_use_symlinks=False
        )
        print(f"✅ Tokenizer: {tokenizer_path}")
        print("🎉 Download complete!")
        return True
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False

def list_versions(repo_id=None):
    if repo_id is None:
        repo_id = os.environ.get('HF_REPO', 'yourusername/dinesh-ai')
    
    print(f"📋 Versions in {repo_id}:\n")
    try:
        files = list_repo_files(repo_id)
        versions = [f.replace('versions/dinesh_ai_model_v', '').replace('.pth', '') 
                   for f in files if f.startswith('versions/')]
        for v in sorted(versions):
            print(f"  - {v}")
        print(f"\nTotal: {len(versions)} versions")
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "list":
            repo_id = sys.argv[2] if len(sys.argv) > 2 else None
            list_versions(repo_id)
        else:
            repo_id = sys.argv[1]
            version = sys.argv[2] if len(sys.argv) > 2 else None
            download_model(repo_id, version)
    else:
        download_model()
