# 🐛 Troubleshooting Guide

Solutions to common issues in Dinesh AI.

## 🎯 Quick Fixes

| Issue | Quick Fix |
|-------|-----------|
| Model not loading | Check model_config.json has num_heads and d_ff |
| Out of memory | Reduce batch_size in config |
| Streamlit errors | Press F5 to refresh browser |
| No module errors | Run `pip install -r requirements.txt` |
| Training fails | Check logs in `logs/` directory |
| JSON serialization error | Fixed in latest version |

## 📋 Installation Issues

### "Python not found"

**Problem:** Python not installed or not in PATH

**Solution:**
```bash
# Check Python
python --version

# If not found, install:
# Windows: Download from python.org
# Linux: sudo apt install python3
# macOS: brew install python3
```

### "pip not found"

**Problem:** pip not installed

**Solution:**
```bash
python -m ensurepip --upgrade
```

### "No module named 'torch'"

**Problem:** Dependencies not installed

**Solution:**
```bash
pip install -r requirements.txt
```

### "Permission denied"

**Problem:** Insufficient permissions

**Solution:**
```bash
# Windows: Run as Administrator
# Linux/macOS: Use sudo
sudo pip install -r requirements.txt
```

## 🚂 Training Issues

### "Out of memory"

**Problem:** Not enough RAM

**Solution:**
```yaml
# Edit config file
training:
  batch_size: 2  # Reduce from 4 or 16
  
model:
  d_model: 128   # Reduce if still failing
```

### "CUDA out of memory"

**Problem:** GPU memory insufficient

**Solution:**
```yaml
# Option 1: Use CPU
training:
  device: "cpu"

# Option 2: Reduce batch size
training:
  batch_size: 4  # Reduce further if needed
  
# Option 3: Reduce model size
model:
  d_model: 512   # Reduce from 768
```

### "d_model must be divisible by num_heads"

**Problem:** Invalid configuration

**Solution:**
```yaml
# Ensure d_model % num_heads == 0
model:
  d_model: 128   # Must be divisible by num_heads
  num_heads: 2   # 128 / 2 = 64 ✓
  
# Valid combinations:
# d_model: 128, num_heads: 1, 2, 4
# d_model: 256, num_heads: 1, 2, 4, 8
# d_model: 512, num_heads: 1, 2, 4, 8, 16
# d_model: 768, num_heads: 1, 2, 3, 4, 6, 8, 12
```

### "Training very slow"

**Problem:** Using CPU instead of GPU

**Solution:**
```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# If True, enable in config:
training:
  device: "cuda"
  
# If False, GPU not available - use CPU or install CUDA
```

### "Loss not decreasing"

**Problem:** Learning rate too high/low or bad data

**Solution:**
```yaml
# Try different learning rate
training:
  learning_rate: 0.0001  # Increase if too slow
  learning_rate: 0.00001 # Decrease if unstable
  
# Or increase epochs
training:
  epochs: 5  # More training
```

### "Training crashes midway"

**Problem:** Various causes

**Solution:**
1. Check logs: `logs/training_*.log`
2. Verify disk space: `df -h`
3. Check memory: `free -h`
4. Reduce batch_size
5. Enable checkpointing

## 🌐 Web Interface Issues

### "Streamlit not found"

**Problem:** Streamlit not installed

**Solution:**
```bash
pip install streamlit
```

### "Network issue" error in Streamlit

**Problem:** Frontend cache issue

**Solution:**
1. Press **F5** to refresh
2. If persists: **Ctrl+Shift+Delete** → Clear cache
3. Restart Streamlit:
```bash
# Stop: Ctrl+C
streamlit run app.py
```

### "Error loading model"

**Problem:** Model config incomplete

**Solution:**
```json
// Check models/model_*/model_config.json has:
{
  "vocab_size": 5000,
  "d_model": 128,
  "num_layers": 2,
  "num_heads": 2,      // Must be present
  "d_ff": 512,         // Must be present
  "max_seq_len": 128,
  "device": "cpu"
}
```

### "No trained model found"

**Problem:** No model trained yet

**Solution:**
```bash
python scripts/train.py --local
```

### "index out of range in self"

**Problem:** Model trained with insufficient data

**Solution:**
Train with more data:
```bash
# Use updated local config (35 items)
python scripts/train.py --local

# Or use production config
python scripts/train.py
```

## 📊 Data Collection Issues

### "Connection timeout"

**Problem:** Network issues or API limits

**Solution:**
1. Check internet connection
2. Wait and retry
3. Reduce collection limits:
```yaml
data_sources:
  wikipedia:
    limit: 10  # Reduce from 20
```

### "No data collected"

**Problem:** API issues or wrong configuration

**Solution:**
1. Check logs: `logs/data_collection_*.log`
2. Verify internet connection
3. Check data sources enabled:
```yaml
data_sources:
  wikipedia:
    enabled: true  # Must be true
```

### "Duplicate data"

**Problem:** Deduplication not working

**Solution:**
Check `data/seen_content_hashes.json` exists and is valid JSON

## 🔧 Configuration Issues

### "Invalid YAML"

**Problem:** Syntax error in config file

**Solution:**
```bash
# Validate YAML
python -c "import yaml; yaml.safe_load(open('config.yaml'))"

# Common issues:
# - Wrong indentation (use spaces, not tabs)
# - Missing colons
# - Unquoted special characters
```

### "Config file not found"

**Problem:** Wrong path or missing file

**Solution:**
```bash
# Check files exist
ls config.yaml config.local.yaml

# If missing, restore from backup or repository
```

### "Wrong config loaded"

**Problem:** Not using intended config

**Solution:**
```bash
# Explicitly specify config
python scripts/train.py --config config.local.yaml

# Or use --local flag
python scripts/train.py --local

# Check which config is loaded (shown in logs)
```

## 💾 File System Issues

### "Permission denied" on Windows

**Problem:** File in use or no permissions

**Solution:**
1. Close all applications using the file
2. Run as Administrator
3. Check file permissions

### "Disk space full"

**Problem:** Insufficient disk space

**Solution:**
```bash
# Check space
df -h  # Linux/macOS
dir    # Windows

# Clean up:
# - Delete old models in models/
# - Delete old logs in logs/
# - Delete training data in data/ (after training)
```

### "File not found"

**Problem:** Missing required files

**Solution:**
```bash
# Verify project structure
ls -la

# Recreate directories
mkdir -p data/raw data/processed models logs
```

## 🔄 Model Issues

### "Model generates gibberish"

**Problem:** Model too small or undertrained

**Solution:**
1. Train with production config:
```bash
python scripts/train.py
```

2. Or increase local config:
```yaml
model:
  d_model: 256  # Increase
  num_layers: 4  # Increase
  vocab_size: 10000  # Increase
```

### "Model repeats same text"

**Problem:** Temperature too low or model too small

**Solution:**
1. Increase temperature in web interface: 0.7 → 1.0
2. Train larger model
3. Train for more epochs

### "Model too slow"

**Problem:** Model too large or CPU-bound

**Solution:**
1. Use GPU:
```yaml
training:
  device: "cuda"
```

2. Or reduce model size:
```yaml
model:
  d_model: 256  # Reduce from 768
  num_layers: 6  # Reduce from 12
```

## 🐍 Python Issues

### "SyntaxError"

**Problem:** Python version too old

**Solution:**
```bash
# Check version
python --version

# Need Python 3.8+
# Upgrade if needed
```

### "ImportError"

**Problem:** Missing dependencies

**Solution:**
```bash
# Reinstall all dependencies
pip install --upgrade -r requirements.txt
```

### "ModuleNotFoundError"

**Problem:** Package not in path

**Solution:**
```bash
# Run from project root
cd Dinesh-AI
python scripts/train.py
```

## 📝 Logging Issues

### "Cannot write to log file"

**Problem:** Permission or disk space

**Solution:**
```bash
# Check permissions
chmod 755 logs/

# Check disk space
df -h
```

### "Log file too large"

**Problem:** Logs accumulating

**Solution:**
```bash
# Clean old logs
rm logs/*.log

# Or configure rotation in config.yaml:
logging:
  max_file_size: 10485760  # 10MB
  backup_count: 3
```

## 🔍 Debugging Tips

### Enable Debug Logging
```yaml
logging:
  level: "DEBUG"  # More detailed logs
```

### Check Logs
```bash
# View latest log
tail -f logs/training_*.log

# Search for errors
grep ERROR logs/*.log
```

### Verify Installation
```bash
# Check all imports
python -c "
import torch
import transformers
import streamlit
import yaml
print('All imports successful')
"
```

### Test Components
```bash
# Test data collection only
python -c "
from src.data.data_collector import DataCollector
dc = DataCollector()
print('Data collector works')
"
```

## 📞 Getting More Help

### Check Documentation
1. [Quick Start](01_QUICK_START.md)
2. [Configuration](03_CONFIGURATION.md)
3. [Architecture](04_ARCHITECTURE.md)

### Check Logs
```bash
# Training logs
cat logs/training_*.log

# Data collection logs
cat logs/data_collection_*.log

# Application logs
cat logs/*.log
```

### Verify Setup
```bash
# Run diagnostic
python -c "
import sys
print('Python:', sys.version)
import torch
print('PyTorch:', torch.__version__)
print('CUDA:', torch.cuda.is_available())
"
```

## ✅ Prevention Tips

1. **Always use virtual environment**
2. **Keep dependencies updated**
3. **Monitor disk space**
4. **Check logs regularly**
5. **Start with local config**
6. **Backup important models**
7. **Test after changes**

---

**Still having issues?** Check logs in `logs/` directory for detailed error messages.

*Last Updated: February 26, 2026*
