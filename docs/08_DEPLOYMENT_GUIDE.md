# 🚀 Deployment Guide - Local to Production

Complete end-to-end deployment guide for Dinesh AI.

## 📋 Deployment Overview

```
Local Development → Testing → Production → Continuous Learning
```

## 🎯 Phase 1: Local Development Setup

### Step 1: Environment Setup

```bash
# Clone/navigate to project
cd Dinesh-AI

# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Verify Installation

```bash
# Check Python version
python --version  # Should be 3.8+

# Verify packages
pip list | findstr torch
pip list | findstr streamlit
pip list | findstr tokenizers
```

### Step 3: Configuration Check

```bash
# Verify config exists
type config.yaml

# Check key settings
# - vocab_size: 50000
# - num_layers: 12
# - d_model: 768
```

## 🧪 Phase 2: Local Testing

### Step 1: Initial Training

```bash
# Start production training (2-3 hours)
python scripts/train.py

# Monitor progress
# [0%]   Starting...
# [10%]  Data collection (1,500 items)
# [15%]  Preprocessing
# [20%]  Tokenization (50K vocab)
# [95%]  Training (10 epochs)
# [100%] Complete
```

### Step 2: Verify Training Output

```bash
# Check model files created
dir models\
# Should see:
# - dinesh_ai_model.pth (380MB)
# - tokenizer.json (2MB)
# - model_info.json

# Check data collected
dir data\raw\
# Should see:
# - wikipedia_*.json
# - arxiv_*.json
# - gutenberg_*.json

# Check logs
dir logs\
# - training_*.log
# - data_collection.log
```

### Step 3: Test Web Interface

```bash
# Start Streamlit
streamlit run app.py

# Opens: http://localhost:8501

# Test prompts:
# - "What is artificial intelligence?"
# - "Explain machine learning"
# - "What is Python programming?"
```

### Step 4: Quality Validation

**Check responses for:**
- Coherent sentences
- Relevant content
- No repetition
- Proper grammar

**If quality is poor:**
- Check vocab_size in models/model_info.json (should be 50,000)
- Check training loss in logs (should be < 3.0)
- Verify 1,500 items collected

## 🏭 Phase 3: Production Deployment

### Option A: Local Production Server

**Best for**: Personal use, small teams, testing

**Requirements:**
- Dedicated computer/server
- 8GB+ RAM
- 20GB+ disk space
- Stable internet

**Setup:**

```bash
# 1. Complete Phase 1 & 2 first

# 2. Set up as Windows service (optional)
# Create run_server.bat:
@echo off
cd C:\Users\dinesh\OneDrive\vscodeprojects\Dinesh-AI
call venv\Scripts\activate
streamlit run app.py --server.port 8501

# 3. Configure firewall (if accessing from network)
# Allow port 8501 in Windows Firewall

# 4. Get local IP
ipconfig
# Note IPv4 Address (e.g., 192.168.1.100)

# 5. Access from other devices
# http://192.168.1.100:8501
```

### Option B: Cloud Deployment (AWS)

**Best for**: Public access, scalability, 24/7 availability

**Cost**: ~$10-30/month (t3.medium instance)

**Step 1: Launch EC2 Instance**

```bash
# AWS Console → EC2 → Launch Instance
# - AMI: Ubuntu 22.04 LTS
# - Instance type: t3.medium (2 vCPU, 4GB RAM)
# - Storage: 30GB
# - Security group: Allow ports 22 (SSH), 8501 (Streamlit)
```

**Step 2: Connect and Setup**

```bash
# SSH into instance
ssh -i your-key.pem ubuntu@your-instance-ip

# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3.10
sudo apt install python3.10 python3-pip -y

# Install git
sudo apt install git -y

# Clone project
git clone https://github.com/yourusername/Dinesh-AI.git
cd Dinesh-AI

# Install dependencies
pip3 install -r requirements.txt
```

**Step 3: Train Model on Cloud**

```bash
# Start training (2-3 hours)
python3 scripts/train.py

# Monitor with screen/tmux
screen -S training
python3 scripts/train.py
# Detach: Ctrl+A, D
# Reattach: screen -r training
```

**Step 4: Deploy Streamlit**

```bash
# Run in background
nohup streamlit run app.py --server.port 8501 --server.address 0.0.0.0 &

# Check running
ps aux | grep streamlit

# Access via browser
# http://your-instance-ip:8501
```

**Step 5: Set Up Domain (Optional)**

```bash
# 1. Register domain (e.g., dinesh-ai.com)
# 2. Point A record to EC2 IP
# 3. Install nginx as reverse proxy

sudo apt install nginx -y

# Configure nginx
sudo nano /etc/nginx/sites-available/dinesh-ai

# Add:
server {
    listen 80;
    server_name dinesh-ai.com;
    
    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}

# Enable site
sudo ln -s /etc/nginx/sites-available/dinesh-ai /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx

# Access: http://dinesh-ai.com
```

### Option C: Free Cloud (Google Colab)

**Best for**: Testing, demos, temporary use

**Limitations:**
- 12-hour session limit
- Must restart daily
- No persistent storage

**Setup:**

```python
# In Colab notebook:

# 1. Clone repo
!git clone https://github.com/yourusername/Dinesh-AI.git
%cd Dinesh-AI

# 2. Install dependencies
!pip install -r requirements.txt

# 3. Train model
!python scripts/train.py

# 4. Run Streamlit (with ngrok for public URL)
!pip install pyngrok
from pyngrok import ngrok

# Start Streamlit in background
!streamlit run app.py &

# Create public URL
public_url = ngrok.connect(8501)
print(f"Access at: {public_url}")
```

## 🔄 Phase 4: Continuous Learning Setup

### Step 1: Configure Continuous Training

```yaml
# config.yaml
CONTINUOUS_LEARNING:
  collection_interval_hours: 6    # Collect every 6 hours
  training_interval_hours: 24     # Train daily
  backup_interval_days: 7         # Backup weekly
  max_backups: 4                  # Keep 4 versions
```

### Step 2: Start Continuous Learning

**Local:**
```bash
# Run in background
start /B python scripts/continuous_train.py

# Or use Task Scheduler for auto-start
```

**Cloud (AWS):**
```bash
# Use screen/tmux
screen -S continuous
python3 scripts/continuous_train.py
# Detach: Ctrl+A, D

# Or create systemd service
sudo nano /etc/systemd/system/dinesh-ai.service

# Add:
[Unit]
Description=Dinesh AI Continuous Learning
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/Dinesh-AI
ExecStart=/usr/bin/python3 scripts/continuous_train.py
Restart=always

[Install]
WantedBy=multi-user.target

# Enable and start
sudo systemctl enable dinesh-ai
sudo systemctl start dinesh-ai
sudo systemctl status dinesh-ai
```

### Step 3: Monitor Continuous Learning

```bash
# Check logs
tail -f logs/continuous_learning.log

# Expected output:
# [10:00] Collected 45 new items
# [16:00] Collected 38 new items
# [22:00] Collected 52 new items
# [10:00] Fine-tuning on 135 new items...
# [10:28] Fine-tuning complete
```

## 📊 Phase 5: Monitoring & Maintenance

### Daily Checks

```bash
# 1. Check service status
# Local: Task Manager → Check python.exe running
# Cloud: systemctl status dinesh-ai

# 2. Check logs for errors
type logs\continuous_learning.log | findstr ERROR

# 3. Check disk space
dir models\  # Should have backups

# 4. Test web interface
# Visit http://localhost:8501 or your domain
```

### Weekly Tasks

```bash
# 1. Verify model backups created
dir models\dinesh_ai_model_v*.pth

# 2. Test model quality
streamlit run app.py
# Test with standard prompts

# 3. Check data collection stats
type logs\data_collection.log

# 4. Review training metrics
type logs\fine_tuning.log
```

### Monthly Tasks

```bash
# 1. Clean old logs
del logs\*_old.log

# 2. Verify backup integrity
# Test loading old versions

# 3. Update dependencies
pip install --upgrade -r requirements.txt

# 4. Review and adjust config
# Based on performance metrics
```

## 🚨 Troubleshooting Deployment

### Issue 1: Port 8501 already in use

```bash
# Find process using port
netstat -ano | findstr :8501

# Kill process
taskkill /PID <PID> /F

# Restart Streamlit
streamlit run app.py
```

### Issue 2: Model not loading in production

```bash
# Check file exists
dir models\dinesh_ai_model.pth

# Check file size (should be ~380MB)
# If 0 bytes, model corrupted - restore backup

# Verify config path
python -c "from src.config_loader import load_config; print(load_config())"
```

### Issue 3: Out of memory on cloud

```bash
# Check memory usage
free -h

# Reduce batch size in config.yaml
TRAINING:
  batch_size: 4  # Reduce from 8

# Or upgrade instance type
# t3.medium → t3.large (8GB RAM)
```

### Issue 4: Continuous learning stopped

```bash
# Check process
ps aux | grep continuous_train

# Check logs for errors
tail -100 logs/continuous_learning.log

# Restart
python3 scripts/continuous_train.py
```

## 📈 Scaling Production

### Horizontal Scaling (Multiple Instances)

```bash
# Use load balancer (nginx/AWS ALB)
# Deploy multiple Streamlit instances
# Each serves requests independently
```

### Vertical Scaling (Bigger Instance)

```bash
# Upgrade instance:
# t3.medium (4GB) → t3.large (8GB) → t3.xlarge (16GB)

# Increase model size:
# 100M params → 300M params → 1B params
```

### GPU Acceleration

```bash
# AWS: Use g4dn.xlarge instance (NVIDIA T4 GPU)
# Training time: 2-3 hours → 10-15 minutes

# Verify GPU available
python -c "import torch; print(torch.cuda.is_available())"
```

## 🎯 Deployment Checklist

**Before Production:**
- [ ] Training completed successfully
- [ ] Model quality validated
- [ ] Web interface tested
- [ ] Backups configured
- [ ] Logs reviewed
- [ ] Disk space sufficient (20GB+)

**Production Launch:**
- [ ] Server/instance running
- [ ] Firewall configured
- [ ] Domain pointed (if using)
- [ ] SSL certificate installed (if public)
- [ ] Monitoring set up
- [ ] Backup system active

**Post-Launch:**
- [ ] Continuous learning running
- [ ] Daily log checks scheduled
- [ ] Weekly quality tests
- [ ] Monthly maintenance planned
- [ ] Rollback procedure documented

## 📚 Related Documentation

- [Quick Start](01_QUICK_START.md) - Initial setup
- [Training Pipeline](05_TRAINING_PIPELINE.md) - Training process
- [Continuous Learning](09_CONTINUOUS_LEARNING.md) - 24/7 system
- [Model Safety](11_MODEL_SAFETY.md) - Backups
- [Troubleshooting](12_TROUBLESHOOTING.md) - Fix issues

---

**Ready to deploy?** Follow phases 1-5 in order for successful deployment!
