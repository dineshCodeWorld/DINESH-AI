# ✅ SETUP CHECKLIST - CPU Training (Free)

## What You're Setting Up

✅ GitHub Actions - Trains every 6 hours on CPU
✅ Hugging Face - Unlimited free storage  
✅ Streamlit Cloud - Free hosting with public URL
✅ Email Notifications - Weekly deployment emails
✅ Fully Automated - Zero manual intervention
✅ $0/month cost

## Step 1: Create Accounts (15 minutes)

### 1. Hugging Face (5 min)
- [ ] Go to: https://huggingface.co/join
- [ ] Create account
- [ ] Go to: https://huggingface.co/settings/tokens
- [ ] Create new token (WRITE permission)
- [ ] Copy token: `hf_xxxxxxxxxxxxx`
- [ ] Go to: https://huggingface.co/new
- [ ] Create repo: name=`dinesh-ai`, type=Model, public

### 2. GitHub (3 min)
- [ ] Go to: https://github.com/new
- [ ] Name: `Dinesh-AI`
- [ ] Public repository
- [ ] Create

### 3. Gmail App Password (5 min)
- [ ] Go to: https://myaccount.google.com/security
- [ ] Enable "2-Step Verification"
- [ ] Search "App passwords"
- [ ] Generate for Mail
- [ ] Copy 16-character password: `xxxx xxxx xxxx xxxx`

### 4. Streamlit Cloud (2 min)
- [ ] Go to: https://streamlit.io/cloud
- [ ] Sign in with GitHub
- [ ] Done (no additional setup)

## Step 2: Add GitHub Secrets (3 minutes)

- [ ] Go to: `https://github.com/yourusername/Dinesh-AI/settings/secrets/actions`
- [ ] Click "New repository secret"
- [ ] Add these 4 secrets:

```
Name: HF_TOKEN
Value: hf_xxxxxxxxxxxxx (your Hugging Face token)

Name: HF_REPO  
Value: yourusername/dinesh-ai

Name: EMAIL_USERNAME
Value: dineshganji372@gmail.com

Name: EMAIL_PASSWORD
Value: xxxx xxxx xxxx xxxx (your Gmail app password)
```

## Step 3: Push Code to GitHub (2 minutes)

```bash
cd Dinesh-AI

git init
git add .
git commit -m "Initial deployment"
git branch -M main
git remote add origin https://github.com/yourusername/Dinesh-AI.git
git push -u origin main
```

## Step 4: Deploy to Streamlit (5 minutes)

- [ ] Go to: https://share.streamlit.io
- [ ] Click "New app"
- [ ] Repository: `yourusername/Dinesh-AI`
- [ ] Branch: `main`
- [ ] Main file: `app.py`
- [ ] Click "Advanced settings"
- [ ] Add secret:
```
HF_REPO = "yourusername/dinesh-ai"
```
- [ ] Click "Deploy"

Your app will be live at: `https://yourusername-dinesh-ai.streamlit.app`

## Step 5: Verify Automation (5 minutes)

### Test Training
- [ ] Go to: `https://github.com/yourusername/Dinesh-AI/actions`
- [ ] Click "Continuous Training"
- [ ] Click "Run workflow" → "Run workflow"
- [ ] Wait 2-3 hours for training to complete
- [ ] Check logs for success

### Test Deployment
- [ ] Go to: `https://github.com/yourusername/Dinesh-AI/actions`
- [ ] Click "Weekly Deployment"
- [ ] Click "Run workflow" → "Run workflow"
- [ ] Check email: dineshganji372@gmail.com
- [ ] Should receive deployment notification

## What Happens Now

✅ **Every 6 hours**: GitHub Actions collects data, trains model, uploads to Hugging Face
✅ **Every Sunday**: Creates version backup, sends email notification
✅ **On every push**: Streamlit auto-redeploys with latest code
✅ **Manual trigger**: Run workflows anytime from GitHub Actions

## Monitoring

- **Training logs**: https://github.com/yourusername/Dinesh-AI/actions
- **Model versions**: https://huggingface.co/yourusername/dinesh-ai
- **Live app**: https://yourusername-dinesh-ai.streamlit.app
- **Email**: dineshganji372@gmail.com (every Sunday)

## Troubleshooting

**Training fails:**
- Check GitHub Actions logs
- Verify HF_TOKEN has write permission
- Verify HF_REPO name is correct

**Email not received:**
- Check spam folder
- Verify EMAIL_PASSWORD is correct (16-char app password)
- Verify 2-Step Verification enabled

**Streamlit not loading model:**
- Check Streamlit logs
- Verify HF_REPO secret is set correctly
- Manually reboot app in Streamlit dashboard

## Cost

**Total: $0/month**
- GitHub Actions: 2,000 minutes/month free
- Hugging Face: Unlimited storage
- Streamlit Cloud: Free tier
- Gmail: Free

## Next Steps

1. Complete all checkboxes above
2. Wait for first training run (or trigger manually)
3. Check email on Sunday for deployment notification
4. Access your app at Streamlit URL

---

**Total setup time: ~30 minutes**
**Everything runs automatically after setup!**
