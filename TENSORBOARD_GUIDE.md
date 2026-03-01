# 📊 TensorBoard Real-Time Visualization Guide

## What You'll See Now

TensorBoard will show **real-time data collection progress** during training!

## 🎯 Data Collection Metrics (NEW!)

### During Data Collection Phase (Step 1/5):

**DataCollection/** tab shows:
- `Wikipedia` - Articles collected (updates every 10 items)
- `ArXiv` - Papers collected (updates every 10 items)
- `Gutenberg` - Books collected (updates after completion)
- `Reddit` - Posts collected (updates after completion)
- `HackerNews` - Stories collected (updates after completion)
- `News` - Articles collected (updates after completion)
- `TotalCollected` - Running total across all sources
- `TotalTarget` - Your target collection goal
- `DuplicatesFiltered` - How many duplicates were removed

### Example View:
```
DataCollection/Wikipedia: 0 → 10 → 20 → ... → 500
DataCollection/ArXiv: 0 → 10 → 20 → ... → 300
DataCollection/TotalCollected: 0 → 800 → 1200 → 1700
DataCollection/TotalTarget: 1700 (constant line)
```

## 🔧 Pipeline Stages

**Pipeline/Stage** tab shows text updates:
- "Data Collection Started"
- "Data Preprocessing"
- "Training Data Preparation"

## 📈 Training Metrics (Existing)

### During Training Phase (Step 5/6):

**Loss/** tab:
- `train` - Training loss (updates every 10 steps)

**Learning_Rate** tab:
- Shows learning rate schedule (updates every 10 steps)

**Metrics/** tab:
- `perplexity` - Model perplexity (updates every 100 steps)
- `vocab_match` - Real word percentage (updates every 100 steps)

**Model/** tab:
- `Parameters` - Total model parameters
- `Layers` - Number of layers

**Tokenizer/** tab:
- `VocabSize` - Vocabulary size

**Data/** tab:
- `TotalSamples` - Training samples prepared

## 🚀 How to Use

### 1. Start Training:
```bash
python scripts/train.py
```

### 2. Open TensorBoard (in another terminal):
```bash
tensorboard --logdir=runs
```

### 3. Open Browser:
```
http://localhost:6006
```

### 4. Watch Real-Time:
- **First 10-20 minutes**: Watch `DataCollection/` tab to see data being collected
- **Next 2-3 hours**: Watch `Loss/train` and `Metrics/` tabs during training

## 📊 What Each Graph Means

### DataCollection/Wikipedia (0-500)
- **Going up**: Successfully collecting Wikipedia articles
- **Flat line**: Rate limit hit or network issue
- **Target**: Should reach your config limit (default: 500)

### DataCollection/TotalCollected (0-1700)
- **Steady increase**: Pipeline working correctly
- **Should match**: Sum of all individual sources
- **Final value**: Total unique items after deduplication

### Loss/train (starts high, goes down)
- **High at start (8-10)**: Model is learning
- **Decreasing**: Good! Model improving
- **Flat or increasing**: Potential issue (overfitting or bad data)
- **Target**: Should drop to 2-4 after training

### Metrics/perplexity (starts high, goes down)
- **Lower is better**: Measures prediction confidence
- **Good range**: 50-150 (your current model)
- **Excellent**: < 50 (ChatGPT-level)

### Metrics/vocab_match (0.0-1.0)
- **Higher is better**: Percentage of real English words
- **Your current**: 0.40-0.50 (40-50%)
- **Target**: 0.80+ (80%+ real words)
- **ChatGPT-level**: 0.95+ (95%+ real words)

## 🎨 TensorBoard Tips

### Smooth Curves:
- Use the "Smoothing" slider (left sidebar) to reduce noise
- Recommended: 0.6-0.8 for training metrics

### Compare Runs:
- Each training run creates a new folder in `runs/`
- TensorBoard shows all runs together
- Use checkboxes to compare different versions

### Refresh Data:
- TensorBoard auto-refreshes every 30 seconds
- Click "Reload" button (top-right) for immediate update

### Download Data:
- Click "Show data download links" (top-left)
- Export CSV for analysis in Excel/Python

## 🔍 Troubleshooting

### "No data found"
- **During data collection**: Normal! Wait 2-3 minutes for first update
- **During training**: Check if training started (look at console logs)

### Graphs not updating
- Check if `python scripts/train.py` is still running
- Refresh browser (Ctrl+R)
- Check `runs/` folder exists

### Multiple runs cluttering view
- Delete old runs: `rmdir /s runs` (Windows) or `rm -rf runs` (Linux)
- Or uncheck old runs in TensorBoard sidebar

## 📱 Mobile Access

Access TensorBoard from phone/tablet on same network:
```bash
# Find your computer's IP
ipconfig  # Windows
ifconfig  # Linux/Mac

# Start TensorBoard with host binding
tensorboard --logdir=runs --host=0.0.0.0

# Access from phone
http://YOUR_COMPUTER_IP:6006
```

## 🎯 What to Watch For

### ✅ Good Signs:
- DataCollection graphs steadily increasing
- Loss decreasing over time
- Perplexity dropping below 100
- Vocab match increasing above 0.5

### ⚠️ Warning Signs:
- DataCollection stuck at 0 for >5 minutes (network issue)
- Loss increasing (learning rate too high)
- Perplexity > 500 (model not learning)
- Vocab match stuck at 0.1-0.2 (bad tokenizer)

## 💡 Pro Tips

1. **Keep TensorBoard open** during entire training (2-3 hours)
2. **Take screenshots** of final metrics for comparison
3. **Compare versions** by keeping old runs in `runs/` folder
4. **Monitor early** - First 100 steps show if training will succeed
5. **Check data collection** - If collection fails, training will fail too

---

**Next Steps**: Run `python scripts/train.py` and watch the magic happen! 🚀
