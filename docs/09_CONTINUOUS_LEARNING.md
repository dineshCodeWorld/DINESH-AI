# 🔄 Continuous Learning

24/7 automated training system.

## Overview

Enables model to:
- Collect new data automatically
- Train incrementally
- Create weekly backups
- Run indefinitely

## Starting System

```bash
python scripts/continuous_train.py
```

## Schedule

**Data Collection**: Every 6 hours
- New Wikipedia articles
- Latest ArXiv papers
- New Gutenberg books
- Skips duplicates (MD5 hash)

**Model Training**: Every 24 hours
- Fine-tunes on new data only
- Lower learning rate (0.0001)
- 3 epochs (vs 10 for full training)
- 20-30 minutes (vs 2-3 hours)

**Backups**: Every 7 days
- Creates versioned backup
- Keeps last 4 versions
- Enables rollback

## How It Works

```python
while True:
    # Every 6 hours
    if time_for_collection:
        collect_new_data()
    
    # Every 24 hours
    if time_for_training:
        fine_tune_model()
    
    # Every 7 days
    if time_for_backup:
        create_backup()
    
    sleep(1 hour)
```

## Fine-Tuning vs Full Training

| Aspect | Full | Fine-Tune |
|--------|------|-----------|
| Data | 1,500 items | New only |
| Learning rate | 0.001 | 0.0001 |
| Epochs | 10 | 3 |
| Time | 2-3 hours | 20-30 min |
| Init | Random | Load existing |

## Monitoring

```bash
# Check logs
type logs\continuous_learning.log

# Expected output:
[10:00] Collected 45 new items
[16:00] Collected 38 new items
[10:00] Fine-tuning on 135 items...
[10:28] Complete (28 min)
```

## Configuration

```yaml
CONTINUOUS_LEARNING:
  collection_interval_hours: 6
  training_interval_hours: 24
  backup_interval_days: 7
  max_backups: 4
```

## Growth Over Time

- Week 1: 1,500 → 1,780 items
- Month 1: ~2,700 items
- Year 1: ~15,500 items

## Cloud Deployment

For true 24/7:

```bash
# AWS EC2
screen -S continuous
python3 scripts/continuous_train.py
# Detach: Ctrl+A, D

# Or systemd service
sudo systemctl enable dinesh-ai
sudo systemctl start dinesh-ai
```

## Common Issues

**Process stops**: Use screen/tmux or systemd
**Disk full**: Reduce max_backups
**Quality degrades**: Lower learning rate
**No new data**: Sources exhausted

## Related Docs

- [Fine-Tuning](10_FINE_TUNING.md)
- [Model Safety](11_MODEL_SAFETY.md)
- [Deployment Guide](08_DEPLOYMENT_GUIDE.md)
