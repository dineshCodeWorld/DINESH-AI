# ⚡ Quick Start Guide

Get your custom GPT model running in 10-15 minutes!

## 🎯 Overview

This guide will help you:
1. Install dependencies (2 min)
2. Train a minimal functional model (10-15 min)
3. Test it in web interface (1 min)

## 📋 Prerequisites

- Python 3.8 or higher
- 2GB RAM minimum
- Internet connection
- Windows/Linux/macOS

## 🚀 Step-by-Step Guide

### Step 1: Install Dependencies (2 min)

```bash
cd Dinesh-AI
pip install -r requirements.txt
```

**What gets installed:**
- PyTorch (deep learning)
- Transformers (tokenization)
- Streamlit (web interface)
- Other utilities

### Step 2: Train Model (10-15 min)

**For quick testing:**
```bash
python scripts/train.py --local
```

**What happens:**
- ✅ Collects 35 items (20 Wikipedia + 10 ArXiv + 5 Gutenberg)
- ✅ Trains BPE tokenizer (5,000 vocabulary)
- ✅ Creates 2M parameter model (2 layers, 128 dimensions)
- ✅ Trains for 2 epochs
- ✅ Saves to `models/model_YYYYMMDD_HHMMSS/`

**Expected output:**
```
[STEP 1] Collecting data from sources...
✓ Data collection completed

[STEP 2] Preprocessing collected data...
✓ Data preprocessing completed: 35 samples

[STEP 3] Preparing training data...
✓ Training data prepared

[STEP 4] Training model...
Epoch 1/2: 100%|████████| Loss: 3.45
Epoch 2/2: 100%|████████| Loss: 2.87
✓ Model training completed

[STEP 5] Preparing for deployment...
✓ Deployment preparation completed

PIPELINE COMPLETED SUCCESSFULLY
Total time: 12.5 minutes
```

### Step 3: Start Web Interface (1 min)

```bash
streamlit run app.py
```

**Open browser:** http://localhost:8501

### Step 4: Chat with Your Model

1. Click **💬 Chat** tab
2. Type a question: "What is artificial intelligence?"
3. Click **🚀 Generate**
4. See the response!

## 📊 What You Get

### Model Files
```
models/model_20260226_143022/
├── model.pt              # Trained weights (~5MB)
├── tokenizer.json        # BPE tokenizer
├── model_config.json     # Architecture config
└── README.md            # Model card
```

### Model Specifications
- **Parameters**: ~2 million
- **Vocabulary**: 5,000 tokens
- **Layers**: 2 transformer blocks
- **Training Data**: 35 diverse items
- **Model Size**: ~5MB

## 🎮 Using the Web Interface

### Chat Tab
- Ask questions
- Adjust temperature (creativity)
- Control response length
- See real-time generation

### Model Info Tab
- View model statistics
- Architecture details
- Parameter count
- Training information

### Testing Tab
- Pre-built test prompts
- Custom prompt testing
- Experiment with settings

## ⚙️ Configuration Options

### Temperature (Creativity)
- **0.1-0.5**: Focused, deterministic
- **0.7**: Balanced (default)
- **1.0-2.0**: Creative, random

### Max Length
- **50**: Short responses
- **150**: Medium (default)
- **300**: Long responses

## 🔄 Next Steps

### For Better Quality
Train with production config (1-2 hours):
```bash
python scripts/train.py
```

**Production model:**
- 100M parameters (vs 2M)
- 50,000 vocabulary (vs 5,000)
- 1,600 training items (vs 35)
- 380MB model size (vs 5MB)

### Learn More
- **[Local vs Production](08_LOCAL_VS_PRODUCTION.md)** - Compare training modes
- **[Architecture](04_ARCHITECTURE.md)** - Understand how it works
- **[Configuration](03_CONFIGURATION.md)** - Customize settings

### Enable 24/7 Learning
```bash
python scripts/continuous_train.py
```

## ❓ Common Questions

**Q: Why is the model small?**
A: Local config is for quick testing. Use production config for full model.

**Q: Can I use GPU?**
A: Yes! It's 30-50× faster. Set `device: cuda` in config.yaml

**Q: How do I retrain?**
A: Just run `python scripts/train.py` again. It will fine-tune on new data.

**Q: Where are logs?**
A: Check `logs/` directory for detailed training logs.

**Q: Model not responding well?**
A: Local model is minimal. Train production model for better quality.

## 🐛 Troubleshooting

### "No module named 'torch'"
```bash
pip install -r requirements.txt
```

### "No trained model found"
```bash
python scripts/train.py --local
```

### "Out of memory"
Reduce batch_size in config.local.yaml:
```yaml
training:
  batch_size: 2  # Lower if needed
```

### Streamlit errors
Press F5 to refresh browser

## 💡 Tips

1. **Start with local config** to verify everything works
2. **Use production config** for actual use
3. **Enable GPU** if available (much faster)
4. **Monitor logs** to track progress
5. **Experiment with temperature** for different response styles

## 📈 Expected Times

| Config | Data Collection | Training | Total |
|--------|----------------|----------|-------|
| Local | 3-5 min | 5-10 min | 10-15 min |
| Production | 20-30 min | 1-2 hours | 2-3 hours |

## ✅ Success Checklist

- [ ] Dependencies installed
- [ ] Model trained successfully
- [ ] Web interface running
- [ ] Can generate responses
- [ ] Understand local vs production

---

**Congratulations! Your custom GPT model is running!** 🎉

**Next:** [Configuration Guide](03_CONFIGURATION.md) to customize your model

*Last Updated: February 26, 2026*
