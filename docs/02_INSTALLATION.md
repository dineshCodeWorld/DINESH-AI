# 📦 Installation Guide

Complete installation instructions for Dinesh AI.

## 🎯 System Requirements

### Minimum (Local Testing)
- **OS**: Windows 10/11, Linux, macOS
- **Python**: 3.8 or higher
- **RAM**: 2GB
- **Disk**: 5GB free space
- **Internet**: Required for data collection

### Recommended (Production)
- **OS**: Windows 10/11, Linux, macOS
- **Python**: 3.8 or higher
- **RAM**: 8GB (16GB preferred)
- **Disk**: 20GB free space
- **GPU**: NVIDIA GPU with CUDA (optional, 30-50× faster)
- **Internet**: Required for data collection

## 🚀 Installation Steps

### 1. Install Python

**Check if Python is installed:**
```bash
python --version
```

**If not installed:**
- **Windows**: Download from [python.org](https://www.python.org/downloads/)
- **Linux**: `sudo apt install python3 python3-pip`
- **macOS**: `brew install python3`

### 2. Clone or Download Project

**Option A: Git Clone**
```bash
git clone https://github.com/yourusername/Dinesh-AI.git
cd Dinesh-AI
```

**Option B: Download ZIP**
1. Download project ZIP
2. Extract to desired location
3. Open terminal in project folder

### 3. Create Virtual Environment (Recommended)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

**What gets installed:**
- `torch` - PyTorch deep learning framework
- `transformers` - Hugging Face transformers
- `tokenizers` - Fast BPE tokenization
- `streamlit` - Web interface
- `requests` - HTTP requests
- `beautifulsoup4` - HTML parsing
- `pyyaml` - YAML configuration
- `tqdm` - Progress bars
- `numpy` - Numerical computing

**Installation time:** 2-5 minutes

### 5. Verify Installation

```bash
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import streamlit; print('Streamlit:', streamlit.__version__)"
```

**Expected output:**
```
PyTorch: 2.x.x
Streamlit: 1.x.x
```

## 🎮 GPU Setup (Optional)

### Check GPU Availability

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

### Install CUDA (if needed)

**Windows/Linux:**
1. Download CUDA Toolkit from [NVIDIA](https://developer.nvidia.com/cuda-downloads)
2. Install following instructions
3. Reinstall PyTorch with CUDA:
```bash
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

**Verify GPU:**
```bash
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

## 📁 Project Structure

After installation, your project should look like:

```
Dinesh-AI/
├── src/                      # Source code
│   ├── core/                 # Model architecture
│   │   ├── custom_model.py
│   │   └── model_trainer.py
│   ├── data/                 # Data handling
│   │   ├── data_collector.py
│   │   └── data_preprocessor.py
│   ├── continuous/           # 24/7 training
│   │   └── continuous_trainer.py
│   ├── deployment/           # Deployment
│   │   └── model_deployer.py
│   └── config_loader.py      # Config management
├── scripts/                  # Training scripts
│   ├── train.py
│   └── continuous_train.py
├── docs/                     # Documentation
├── data/                     # Data storage (created on first run)
├── models/                   # Trained models (created on first run)
├── logs/                     # Log files (created on first run)
├── config.yaml              # Production config
├── config.local.yaml        # Local testing config
├── requirements.txt         # Dependencies
├── app.py                   # Web interface
└── README.md               # Project overview
```

## 🔧 Configuration

### Create Required Directories

These are created automatically on first run, but you can create manually:

```bash
mkdir data models logs
mkdir data/raw data/processed
```

### Verify Configuration Files

**Check config files exist:**
```bash
ls config.yaml config.local.yaml
```

Both files should be present.

## ✅ Verify Installation

### Run Quick Test

```bash
python scripts/train.py --local
```

**This will:**
1. Collect 35 items from Wikipedia, ArXiv, Gutenberg
2. Train a small model
3. Save to `models/` directory

**Expected time:** 10-15 minutes

### Start Web Interface

```bash
streamlit run app.py
```

**Open:** http://localhost:8501

If you see the web interface, installation is successful!

## 🐛 Troubleshooting

### "Python not found"
- Install Python 3.8+
- Add Python to PATH
- Restart terminal

### "pip not found"
```bash
python -m ensurepip --upgrade
```

### "No module named 'torch'"
```bash
pip install -r requirements.txt
```

### "CUDA out of memory"
- Reduce batch_size in config
- Use CPU instead: `device: cpu`
- Close other applications

### "Permission denied"
- Run as administrator (Windows)
- Use `sudo` (Linux/macOS)
- Check file permissions

### "Connection timeout"
- Check internet connection
- Try again later
- Use VPN if blocked

## 🔄 Updating

### Update Dependencies
```bash
pip install --upgrade -r requirements.txt
```

### Update Project
```bash
git pull origin main
```

## 🗑️ Uninstallation

### Remove Virtual Environment
```bash
deactivate
rm -rf venv  # Linux/macOS
rmdir /s venv  # Windows
```

### Remove Project
```bash
cd ..
rm -rf Dinesh-AI  # Linux/macOS
rmdir /s Dinesh-AI  # Windows
```

## 📊 Disk Space Usage

| Component | Size |
|-----------|------|
| Dependencies | ~2GB |
| Local model | ~5MB |
| Production model | ~380MB |
| Training data | ~500MB |
| Logs | ~10MB |
| **Total (Local)** | **~2.5GB** |
| **Total (Production)** | **~3GB** |

## 💡 Tips

1. **Use virtual environment** to avoid conflicts
2. **Enable GPU** for 30-50× faster training
3. **Start with local config** to verify setup
4. **Monitor disk space** during training
5. **Keep dependencies updated** for bug fixes

## 🎓 Next Steps

After successful installation:

1. **[Quick Start Guide](01_QUICK_START.md)** - Train your first model
2. **[Configuration Guide](03_CONFIGURATION.md)** - Customize settings
3. **[Training Pipeline](05_TRAINING_PIPELINE.md)** - Understand training

---

**Installation complete!** → [Quick Start Guide](01_QUICK_START.md)

*Last Updated: February 26, 2026*
