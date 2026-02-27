# 🆓 Free GPU Deployment & Storage

## Solution: Kaggle GPU + Hugging Face Storage

**Training**: Kaggle (30 hours/week FREE GPU P100/T4)
**Storage**: Hugging Face Hub (UNLIMITED free storage)
**Cost**: $0/month

## Part 1: Hugging Face Setup (Model Storage)

### Create Account & Repo

1. Go to https://huggingface.co/join
2. Create account (free)
3. Settings → Access Tokens → New Token
4. Copy token: `hf_xxxxxxxxxxxxx`
5. Create repo: https://huggingface.co/new
   - Name: `dinesh-ai`
   - Type: Model
   - License: MIT
   - Public (for unlimited storage)

### Install & Login

```bash
pip install huggingface_hub

huggingface-cli login
# Paste your token
```

### Test Upload

```bash
python scripts/upload_to_hf.py yourusername/dinesh-ai
```

## Part 2: Kaggle GPU Setup (Training)

### Create Kaggle Account

1. Go to https://www.kaggle.com
2. Sign up (free)
3. Verify phone number (REQUIRED for GPU access)
4. Settings → Account → Phone Verification

### Create Notebook

1. Kaggle → Code → New Notebook
2. Settings → Accelerator → **GPU P100** (or T4)
3. Settings → Internet → **ON** (required)
4. Settings → Persistence → **Files only**

### Copy Training Code

Open `kaggle_notebook.py` in this project and copy all code to Kaggle notebook.

Update these values:
```python
HF_TOKEN = "hf_your_actual_token"
HF_REPO = "yourusername/dinesh-ai"
```

### Run Training

1. Click "Run All" in Kaggle
2. Training takes 10-15 minutes with GPU
3. Model auto-uploads to Hugging Face
4. Check your HF repo for uploaded model

### Schedule Daily Training

1. Notebook → Settings → Schedule
2. Run frequency: **Daily** (or custom)
3. GPU: **ON**
4. Save schedule

**Result**: Trains automatically every day with free GPU!

## Part 3: Download Models

### From Hugging Face

```bash
# Download latest
python scripts/download_from_hf.py yourusername/dinesh-ai

# Download specific version
python scripts/download_from_hf.py yourusername/dinesh-ai 2024-02-26

# List all versions
python scripts/download_from_hf.py list yourusername/dinesh-ai
```

### Use in Streamlit

```bash
# Download model first
python scripts/download_from_hf.py yourusername/dinesh-ai

# Run app
streamlit run app.py
```

## Storage Comparison

| Service | Free Storage | GPU | Best For |
|---------|--------------|-----|----------|
| **Hugging Face** | Unlimited | ❌ | Primary storage |
| **Kaggle** | 100GB | ✅ 30h/week | Training |
| **GitHub LFS** | 2GB | ❌ | Code backup |
| **Google Drive** | 15GB | ❌ | Personal backup |

## Kaggle GPU Limits

- **30 hours/week** GPU time (resets Monday)
- **30 hours/week** TPU time
- **12 hours** max session
- **100GB** dataset storage
- **20GB** notebook output

## Complete Workflow

```
1. Train on Kaggle GPU (10-15 min)
2. Auto-upload to Hugging Face
3. Create versioned backup
4. Schedule runs daily
5. Download anytime from HF
6. Use in Streamlit locally
```

## Troubleshooting

**Kaggle GPU quota exceeded**: Wait for Monday reset
**HF upload fails**: Check token has write permission
**Model not found**: Verify repo name is correct
**Kaggle session timeout**: Enable persistence in settings

## Quick Start Commands

```bash
# 1. Setup HF
huggingface-cli login

# 2. Upload model
python scripts/upload_to_hf.py yourusername/dinesh-ai

# 3. Download model
python scripts/download_from_hf.py yourusername/dinesh-ai

# 4. Run locally
streamlit run app.py
```

See `SETUP_FREE_GPU.md` for detailed step-by-step instructions.
