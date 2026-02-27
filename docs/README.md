# 📚 Dinesh AI Documentation

Complete documentation for the Dinesh AI project - a custom GPT-style language model trained from scratch.

## 📖 Documentation Index

### Getting Started
1. **[Quick Start Guide](01_QUICK_START.md)** - Get running in 10-15 minutes
2. **[Installation](02_INSTALLATION.md)** - Detailed setup instructions
3. **[Configuration Guide](03_CONFIGURATION.md)** - Understanding config files

### Core Concepts
4. **[Architecture Overview](04_ARCHITECTURE.md)** - How the model works
5. **[Training Pipeline](05_TRAINING_PIPELINE.md)** - Complete training workflow
6. **[Data Collection](06_DATA_COLLECTION.md)** - Multi-source data gathering

### Usage & Features
7. **[Web Interface](07_WEB_INTERFACE.md)** - Using the Streamlit UI
8. **[Deployment Guide](08_DEPLOYMENT_GUIDE.md)** - Local to production deployment
9. **[Continuous Learning](09_CONTINUOUS_LEARNING.md)** - 24/7 training system

### Advanced Topics
10. **[Fine-Tuning System](10_FINE_TUNING.md)** - Incremental learning explained
11. **[Model Safety](11_MODEL_SAFETY.md)** - Backups and recovery
12. **[Troubleshooting](12_TROUBLESHOOTING.md)** - Common issues and fixes
13. **[Free GPU Deployment](13_FREE_GPU_DEPLOYMENT.md)** - Kaggle GPU + HuggingFace storage

## 🚀 Quick Navigation

**Want to start immediately?** → [Quick Start Guide](01_QUICK_START.md)

**Understanding the system?** → [Architecture Overview](04_ARCHITECTURE.md)

**Having issues?** → [Troubleshooting](12_TROUBLESHOOTING.md)

**Deploying to production?** → [Deployment Guide](08_DEPLOYMENT_GUIDE.md)

## 📊 Project Overview

Dinesh AI is a custom GPT-style transformer language model built completely from scratch:

- **Custom Architecture**: Decoder-only transformer with multi-head attention
- **Training from Scratch**: No pretrained models used
- **Multi-Source Data**: Wikipedia, ArXiv, Project Gutenberg
- **Continuous Learning**: Trains daily, versions weekly
- **Free Infrastructure**: Runs on free-tier services
- **Incremental Learning**: Fine-tunes on new data without forgetting

## 🎯 Key Features

✅ **100M+ parameter model** (configurable)
✅ **50,000 word vocabulary** with BPE tokenization
✅ **Automatic deduplication** prevents duplicate data
✅ **Fine-tuning system** for continuous learning
✅ **Model backups** prevent knowledge loss
✅ **Web interface** for easy interaction
✅ **Two training modes** (local testing + production)

## 📁 Project Structure

```
Dinesh-AI/
├── src/                      # Source code
│   ├── core/                 # Model architecture
│   ├── data/                 # Data collection & preprocessing
│   ├── continuous/           # 24/7 training system
│   └── deployment/           # Model deployment
├── scripts/                  # Training scripts
├── docs/                     # Documentation (you are here)
├── config.yaml              # Production configuration
├── config.local.yaml        # Local testing configuration
└── app.py                   # Web interface

```

## 🔧 System Requirements

**Minimum (Local Testing):**
- Python 3.8+
- 2GB RAM
- 5GB disk space

**Recommended (Production):**
- Python 3.8+
- 8GB RAM (16GB preferred)
- 20GB disk space
- GPU (optional, 30-50× faster)

## 💡 Quick Commands

```bash
# Local testing (10-15 min)
python scripts/train.py --local

# Production training (1-2 hours)
python scripts/train.py

# Start web interface
streamlit run app.py

# Continuous 24/7 training
python scripts/continuous_train.py
```

## 📞 Getting Help

1. Check [Troubleshooting Guide](12_TROUBLESHOOTING.md)
2. Review relevant documentation section
3. Check logs in `logs/` directory
4. Verify configuration files

## 🎓 Learning Path

**Beginner:**
1. Quick Start Guide
2. Web Interface
3. Deployment Guide

**Intermediate:**
4. Architecture Overview
5. Training Pipeline
6. Configuration Guide

**Advanced:**
7. Fine-Tuning System
8. Continuous Learning
9. Model Safety

---

**Ready to start?** → [Quick Start Guide](01_QUICK_START.md)

*Last Updated: February 26, 2026*
