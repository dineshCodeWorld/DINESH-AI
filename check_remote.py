import os
from huggingface_hub import HfApi, hf_hub_download
import json

HF_TOKEN = input("Enter HF_TOKEN: ").strip()
HF_REPO = input("Enter HF_REPO: ").strip()

api = HfApi()

print(f"\n🔍 Analyzing {HF_REPO}\n")

# Check all files
files = api.list_repo_files(repo_id=HF_REPO, token=HF_TOKEN)

print("📁 Files in repo:")
for f in sorted(files):
    info = api.repo_info(repo_id=HF_REPO, token=HF_TOKEN, files_metadata=True)
    for file in info.siblings:
        if file.rfilename == f:
            print(f"  {f}: {file.size/1024/1024:.2f} MB")
            break

# Download and check config
try:
    config_path = hf_hub_download(repo_id=HF_REPO, filename="model_config.json", token=HF_TOKEN)
    with open(config_path) as f:
        config = json.load(f)
    
    print(f"\n🤖 Model Config:")
    print(f"  Vocab: {config.get('vocab_size', 'N/A')}")
    print(f"  Layers: {config.get('num_layers', 'N/A')}")
    print(f"  Dimension: {config.get('d_model', 'N/A')}")
    print(f"  Parameters: {config.get('parameter_count', 'N/A')}")
    print(f"  Training samples: {config.get('training_samples', 'N/A')}")
    print(f"  Final loss: {config.get('final_loss', 'N/A')}")
except Exception as e:
    print(f"\n❌ Config error: {e}")

# Check data files
data_files = [f for f in files if 'data' in f.lower()]
print(f"\n📊 Data-related files: {len(data_files)}")
for f in data_files:
    print(f"  {f}")
