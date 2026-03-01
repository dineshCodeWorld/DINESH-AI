# 🌐 Weights & Biases (W&B) - Live Cloud Monitoring

## 🎯 What You Get

✅ **Live training dashboard** - Watch training from anywhere (phone, tablet, laptop)  
✅ **GitHub Actions integration** - Monitor training during automated runs  
✅ **Compare commits** - See which code changes improved the model  
✅ **Select best model** - Pick which version to deploy  
✅ **Public shareable links** - Share progress with anyone  
✅ **100% FREE** for public projects  

## 🚀 5-Minute Setup

### Step 1: Create W&B Account
```
1. Go to: https://wandb.ai/signup
2. Sign up (free)
3. Verify email
```

### Step 2: Get API Key
```
1. Go to: https://wandb.ai/authorize
2. Copy your API key: wandb_xxxxxxxxxxxxxxxxxxxxx
```

### Step 3: Add to GitHub Secrets
```
1. Go to: https://github.com/YOUR_USERNAME/Dinesh-AI/settings/secrets/actions
2. Click "New repository secret"
3. Name: WANDB_API_KEY
4. Value: wandb_xxxxxxxxxxxxxxxxxxxxx (paste your key)
5. Click "Add secret"
```

### Step 4: Test Locally (Optional)
```bash
# Install wandb
pip install wandb

# Login
wandb login
# Paste your API key when prompted

# Run training
python scripts/train.py
```

### Step 5: Push to GitHub
```bash
git add .
git commit -m "Add W&B monitoring"
git push
```

**Done!** W&B will automatically track all GitHub Actions training runs.

## 📊 What Gets Tracked

### Data Collection Phase:
- `collection/wikipedia` - Wikipedia articles collected
- `collection/arxiv` - ArXiv papers collected
- `collection/total` - Total items collected
- `collection/duplicates` - Duplicates filtered

### Training Phase:
- `train/loss` - Training loss (every 10 steps)
- `train/lr` - Learning rate schedule
- `train/step` - Current training step
- `metrics/perplexity` - Model perplexity (every 100 steps)
- `metrics/vocab_match` - Real word percentage (every 100 steps)

### Pipeline Progress:
- `pipeline/progress` - Overall progress (0-100%)
- `pipeline/stage` - Current stage (collection, preprocessing, training)
- `pipeline/total_time` - Total time taken
- `data/samples` - Training samples prepared

## 🌐 Access Your Dashboard

### During Local Training:
```
1. Run: python scripts/train.py
2. Look for: ✓ W&B enabled: https://wandb.ai/YOUR_USERNAME/dinesh-ai/runs/xxxxx
3. Click the link or copy to browser
```

### During GitHub Actions:
```
1. Go to: https://github.com/YOUR_USERNAME/Dinesh-AI/actions
2. Click on running workflow
3. Expand "Run training script" step
4. Look for: ✓ W&B enabled: https://wandb.ai/...
5. Click the link
```

### Direct Access:
```
https://wandb.ai/YOUR_USERNAME/dinesh-ai
```

## 📱 Features You'll Love

### 1. Live Charts
- Real-time loss curve
- Perplexity dropping
- Vocab match increasing
- Data collection progress

### 2. Compare Runs
- Side-by-side comparison of different commits
- See which changes improved performance
- Filter by tags (github-actions vs local)

### 3. System Metrics
- CPU/GPU usage
- Memory consumption
- Training speed (steps/sec)

### 4. Logs
- All console output captured
- Search through logs
- Download logs as text

### 5. Model Selection
- Mark best runs with stars ⭐
- Add notes to runs
- Tag runs for deployment

## 🎨 Dashboard Views

### Overview Tab:
- All runs listed
- Quick metrics comparison
- Filter by date/tag/status

### Workspace Tab:
- Live charts during training
- Customizable layout
- Add/remove metrics

### System Tab:
- CPU/GPU/Memory graphs
- Training speed
- Resource utilization

### Logs Tab:
- Real-time console output
- Search functionality
- Download logs

## 🔍 Compare Different Commits

### Scenario: You changed vocab_size from 8K to 32K

**Before (Commit A):**
```
vocab_size: 8000
Final vocab_match: 0.45 (45%)
Final perplexity: 120
```

**After (Commit B):**
```
vocab_size: 32000
Final vocab_match: 0.62 (62%)
Final perplexity: 85
```

**In W&B:**
1. Select both runs (checkboxes)
2. Click "Compare"
3. See side-by-side charts
4. Conclusion: Commit B is better! ✅

## 🎯 Select Model for Deployment

### Method 1: Star Best Run
```
1. Open W&B dashboard
2. Find best run (lowest perplexity)
3. Click star icon ⭐
4. Add note: "Best model - deploy this"
```

### Method 2: Tag for Deployment
```
1. Click on run
2. Add tag: "deploy-candidate"
3. Team can see which to deploy
```

### Method 3: Download Model
```
1. Click on run
2. Go to "Files" tab
3. Download model files
4. Deploy manually
```

## 📧 Email Notifications

### Get notified when training completes:
```
1. Go to: https://wandb.ai/settings
2. Click "Notifications"
3. Enable "Run finished"
4. Add your email
```

## 🔗 Share with Team

### Public Link:
```
https://wandb.ai/YOUR_USERNAME/dinesh-ai/runs/xxxxx
```

### Embed in README:
```markdown
[![W&B](https://img.shields.io/badge/W%26B-Dashboard-yellow)](https://wandb.ai/YOUR_USERNAME/dinesh-ai)
```

### Share on Social Media:
- W&B generates beautiful charts
- Click "Share" button
- Copy image or link

## 🆚 W&B vs TensorBoard

| Feature | TensorBoard | W&B |
|---------|-------------|-----|
| **Local monitoring** | ✅ Yes | ✅ Yes |
| **Cloud monitoring** | ❌ No | ✅ Yes |
| **GitHub Actions** | ❌ No | ✅ Yes |
| **Mobile access** | ❌ No | ✅ Yes |
| **Compare runs** | ⚠️ Limited | ✅ Excellent |
| **Share links** | ❌ No | ✅ Yes |
| **Team collaboration** | ❌ No | ✅ Yes |
| **Cost** | Free | Free (public) |

**Recommendation**: Use **both**!
- TensorBoard for local debugging
- W&B for cloud monitoring and GitHub Actions

## 🔧 Advanced: Custom Metrics

### Add your own metrics in code:
```python
import wandb

# Log custom metric
wandb.log({"custom/my_metric": value})

# Log image
wandb.log({"samples/generated_text": wandb.Image(img)})

# Log table
wandb.log({"results": wandb.Table(data=df)})
```

## 🐛 Troubleshooting

### "W&B not logging"
```bash
# Check if WANDB_API_KEY is set
echo $WANDB_API_KEY  # Linux/Mac
echo %WANDB_API_KEY%  # Windows

# If empty, set it:
export WANDB_API_KEY=wandb_xxxxx  # Linux/Mac
set WANDB_API_KEY=wandb_xxxxx     # Windows
```

### "Run not showing in dashboard"
- Wait 30 seconds for sync
- Check internet connection
- Verify API key is correct

### "GitHub Actions not logging"
- Verify WANDB_API_KEY secret is added
- Check workflow logs for errors
- Ensure wandb is in requirements.txt

## 📚 Example Workflow

### Day 1: Initial Training
```
1. Push code to GitHub
2. GitHub Actions starts training
3. Open W&B link from Actions logs
4. Watch live progress on phone
5. Training completes after 3 hours
6. Check final metrics
7. Star the run if good
```

### Day 2: Improve Model
```
1. Change config (increase vocab_size)
2. Push to GitHub
3. New training starts
4. Compare with Day 1 run
5. See improvement: vocab_match 45% → 62%
6. Deploy new model
```

### Day 3: Share Progress
```
1. Open W&B dashboard
2. Click "Share" button
3. Copy public link
4. Post on Twitter/LinkedIn
5. Show your AI learning progress!
```

## 🎓 Learn More

- **W&B Docs**: https://docs.wandb.ai
- **Quickstart**: https://docs.wandb.ai/quickstart
- **Examples**: https://wandb.ai/gallery
- **Community**: https://wandb.ai/community

## 💡 Pro Tips

1. **Tag your runs**: Add tags like "experiment", "production", "test"
2. **Use notes**: Document what you changed in each run
3. **Create reports**: Share progress with beautiful reports
4. **Set up alerts**: Get notified when metrics cross thresholds
5. **Use sweeps**: Automatically try different hyperparameters

---

**Ready?** Add WANDB_API_KEY to GitHub Secrets and watch your training live! 🚀

## 🔗 Quick Links

- **Your Dashboard**: https://wandb.ai/YOUR_USERNAME/dinesh-ai
- **GitHub Secrets**: https://github.com/YOUR_USERNAME/Dinesh-AI/settings/secrets/actions
- **Get API Key**: https://wandb.ai/authorize
- **Documentation**: https://docs.wandb.ai
