# 🤖 Dinesh AI - Automated Deployment

## 🎯 What This Does

- ✅ Trains model every 6 hours automatically (GitHub Actions CPU)
- ✅ Deploys new version every Sunday
- ✅ Sends email to dineshganji372@gmail.com after deployment
- ✅ Free Streamlit Cloud hosting with public URL
- ✅ Collects data from Wikipedia, ArXiv, Gutenberg, Reddit, HackerNews, News
- ✅ Free model storage (Hugging Face)
- ✅ Centralized configuration (config.yaml)
- ✅ Optimized for human-like responses
- ✅ $0/month cost

## 🚀 Quick Setup

### Required Accounts

You need to create 4 accounts/tokens:

#### 1. Hugging Face Token (Model Storage)
```
1. Go to: https://huggingface.co/settings/tokens
2. Click "New token"
3. Name: dinesh-ai-token
4. Type: Write
5. Copy token: hf_xxxxxxxxxxxxx
```

#### 2. GitHub Repository
```
1. Go to: https://github.com/new
2. Name: Dinesh-AI
3. Public
4. Create repository
```

#### 3. Gmail App Password (Email Notifications)
```
1. Go to: https://myaccount.google.com/security
2. Enable "2-Step Verification"
3. Search "App passwords"
4. Select app: Mail
5. Generate
6. Copy 16-character password: xxxx xxxx xxxx xxxx
```

#### 4. Streamlit Cloud (Free Hosting)
```
1. Go to: https://streamlit.io/cloud
2. Sign in with GitHub
3. No token needed
```

### Setup Steps

#### Step 1: Add GitHub Secrets
```
Go to: https://github.com/yourusername/Dinesh-AI/settings/secrets/actions

Add these 4 secrets:
1. HF_TOKEN = hf_xxxxxxxxxxxxx (from Hugging Face)
2. HF_REPO = yourusername/dinesh-ai
3. EMAIL_USERNAME = dineshganji372@gmail.com
4. EMAIL_PASSWORD = xxxx xxxx xxxx xxxx (Gmail app password)
```

#### Step 2: Push Code
```bash
cd Dinesh-AI
git init
git add .
git commit -m "Automated deployment setup"
git branch -M main
git remote add origin https://github.com/yourusername/Dinesh-AI.git
git push -u origin main
```

#### Step 3: Deploy Streamlit
```
1. Go to: https://share.streamlit.io
2. Click "New app"
3. Select: yourusername/Dinesh-AI
4. Branch: main
5. File: app.py
6. Advanced → Secrets:
   HF_REPO = "yourusername/dinesh-ai"
7. Deploy
```

Done! Your app is live at: `https://yourusername-dinesh-ai.streamlit.app`

## 📅 Automation Schedule

| Task | Frequency | Platform | What Happens |
|------|-----------|----------|--------------|
| Data Collection | Every 6 hours | GitHub Actions | Collects from Wikipedia, ArXiv, Gutenberg |
| Model Training | Every 6 hours | GitHub Actions (CPU) | Trains/fine-tunes on new data (2-3 hours) |
| Model Upload | Every 6 hours | GitHub Actions | Uploads to Hugging Face |
| Weekly Deployment | Every Sunday | GitHub Actions | Creates version backup |
| Email Notification | Every Sunday | GitHub Actions | Sends deployment email |
| Streamlit Update | On push | Streamlit Cloud | Auto-redeploys app |

## 🔧 Manual Controls

### Trigger Training Now
```
GitHub → Actions → Continuous Training → Run workflow
```

### Trigger Deployment Now
```
GitHub → Actions → Weekly Deployment → Run workflow
```

### Force App Redeploy
```bash
git commit --allow-empty -m "Redeploy"
git push
```

## 📊 Monitoring

- **Training Logs**: GitHub → Actions → Continuous Training
- **Deployment Logs**: GitHub → Actions → Weekly Deployment
- **Model Versions**: https://huggingface.co/yourusername/dinesh-ai
- **Live App**: https://yourusername-dinesh-ai.streamlit.app
- **Email**: Check dineshganji372@gmail.com every Sunday

## 📧 Email Notifications

You'll receive emails with:
- Deployment date and time
- Model version
- App URL
- Hugging Face repo link

## 💰 Cost Breakdown

| Service | Usage | Cost |
|---------|-------|------|
| GitHub Actions | 2,000 min/month (CPU) | $0 |
| Hugging Face | Unlimited storage | $0 |
| Streamlit Cloud | 1 app | $0 |
| Gmail | Unlimited emails | $0 |
| **Total** | | **$0/month** |

## 🎉 What You Get

1. **Public URL**: Share your AI with anyone
2. **24/7 Training**: Continuously learning
3. **Weekly Updates**: New model every Sunday
4. **Email Alerts**: Know when deployed
5. **Version History**: All models saved
6. **Zero Maintenance**: Fully automated

## 📚 Documentation

- `PROJECT_SUMMARY.md` - Complete project overview and architecture
- `SETUP_CHECKLIST.md` - Step-by-step setup guide
- `config.yaml` - Central configuration file (all settings)
- `docs/` - Detailed documentation (13 guides)

## ⚙️ Configuration

All settings are in `config.yaml`:
- Model parameters (vocab_size, layers, temperature, etc.)
- Data source limits and rate limits
- Training hyperparameters
- App UI settings (themes, prompts)
- System configuration

**No hardcoded values** - everything is configurable!

## 🔗 Important Links

- Hugging Face Tokens: https://huggingface.co/settings/tokens
- GitHub Secrets: https://github.com/yourusername/Dinesh-AI/settings/secrets/actions
- Gmail App Passwords: https://myaccount.google.com/apppasswords
- Streamlit Cloud: https://share.streamlit.io
- Project Summary: See `PROJECT_SUMMARY.md`

---

**Ready?** Follow `SETUP_CHECKLIST.md` for step-by-step instructions!