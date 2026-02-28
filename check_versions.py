import os
from huggingface_hub import HfApi, list_repo_commits

# Get credentials
HF_TOKEN = os.getenv("HF_TOKEN")
HF_REPO = os.getenv("HF_REPO")

if not HF_TOKEN:
    HF_TOKEN = input("Enter your HF_TOKEN: ").strip()
if not HF_REPO:
    HF_REPO = input("Enter your HF_REPO (username/repo-name): ").strip()

api = HfApi()

print(f"🔍 Checking repository: {HF_REPO}\n")

# List all commits (versions)
try:
    commits = list(list_repo_commits(repo_id=HF_REPO, token=HF_TOKEN))
    
    print(f"📦 Total versions found: {len(commits)}\n")
    
    for i, commit in enumerate(commits, 1):
        print(f"Version {i}:")
        print(f"  Commit ID: {commit.commit_id[:8]}")
        print(f"  Date: {commit.created_at}")
        print(f"  Message: {commit.title}")
        print()
        
except Exception as e:
    print(f"❌ Error: {e}")
