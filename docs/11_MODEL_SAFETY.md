# 🔒 Model Safety

Backup and recovery system for Dinesh AI models.

## 🎯 Overview

Model safety ensures:
- No knowledge loss during updates
- Recovery from failed training
- Version history tracking
- Rollback capability

## 💾 Backup System

### Automatic Backups

**When backups are created:**
1. Before fine-tuning (safety backup)
2. Every 7 days (weekly versions)
3. Before major updates
4. Manual backups on demand

**Backup naming:**
```
models/
├── dinesh_ai_model.pth                    # Current (latest)
├── dinesh_ai_model_backup.pth             # Pre-update backup
├── dinesh_ai_model_v2024-02-26.pth        # Weekly version
├── dinesh_ai_model_v2024-03-04.pth        # Weekly version
└── dinesh_ai_model_v2024-03-11.pth        # Weekly version
```

### Manual Backup

```bash
# Windows
copy models\dinesh_ai_model.pth models\dinesh_ai_model_backup.pth

# Python
import shutil
shutil.copy('models/dinesh_ai_model.pth', 'models/dinesh_ai_model_backup.pth')
```

## 🔄 Recovery Process

### Scenario 1: Fine-tuning degraded quality

**Problem**: Model quality worse after fine-tuning

**Solution**: Restore pre-update backup

```bash
# Restore backup
copy models\dinesh_ai_model_backup.pth models\dinesh_ai_model.pth

# Test
streamlit run app.py
```

### Scenario 2: Corrupted model file

**Problem**: Model won't load, file corrupted

**Solution**: Restore latest weekly version

```bash
# Find latest version
dir models\dinesh_ai_model_v*.pth

# Restore
copy models\dinesh_ai_model_v2024-03-11.pth models\dinesh_ai_model.pth
```

### Scenario 3: Accidental deletion

**Problem**: Model file deleted

**Solution**: Restore from any backup

```bash
# Check available backups
dir models\*.pth

# Restore most recent
copy models\dinesh_ai_model_backup.pth models\dinesh_ai_model.pth
```

## 📊 Version Management

### Keeping Multiple Versions

**Configuration:**
```yaml
MODEL_SAFETY:
  max_backups: 4              # Keep last 4 weekly versions
  backup_interval_days: 7     # Create version every 7 days
  auto_backup_before_train: true
```

**Storage calculation:**
- Model size: 380MB
- 4 versions: 1.5GB
- Current + backup: 760MB
- Total: ~2.3GB

### Cleanup Old Versions

```python
import os
import glob

# Get all version files
versions = sorted(glob.glob('models/dinesh_ai_model_v*.pth'))

# Keep only last 4
if len(versions) > 4:
    for old_version in versions[:-4]:
        os.remove(old_version)
        print(f"Removed old version: {old_version}")
```

## 🧪 Testing After Updates

### Quality Validation

```python
# Test prompts
test_prompts = [
    "What is artificial intelligence?",
    "Explain machine learning",
    "What is Python programming?"
]

# Generate responses
for prompt in test_prompts:
    response = model.generate(prompt)
    print(f"Q: {prompt}")
    print(f"A: {response}\n")
```

### Automated Testing

```python
def validate_model(model_path):
    model = load_model(model_path)
    
    # Test 1: Model loads successfully
    assert model is not None
    
    # Test 2: Can generate text
    output = model.generate("Test prompt")
    assert len(output) > 0
    
    # Test 3: Quality check
    score = quality_score(output)
    assert score > 0.7
    
    return True
```

## 🚨 Common Issues

### Issue 1: Backup file too large

**Cause**: Model size 380MB per backup

**Fix**: Reduce max_backups or compress

```python
import gzip
import shutil

# Compress backup
with open('models/dinesh_ai_model.pth', 'rb') as f_in:
    with gzip.open('models/dinesh_ai_model.pth.gz', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
```

### Issue 2: Disk space full

**Cause**: Too many backups

**Fix**: Clean old versions

```bash
# Delete versions older than 30 days
forfiles /p models /m dinesh_ai_model_v*.pth /d -30 /c "cmd /c del @path"
```

### Issue 3: Backup corrupted

**Cause**: Interrupted copy, disk error

**Fix**: Verify checksums

```python
import hashlib

def verify_backup(original, backup):
    def file_hash(path):
        with open(path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    return file_hash(original) == file_hash(backup)
```

## 📁 Backup Storage

### Local Storage

```
models/
├── dinesh_ai_model.pth           # 380MB
├── dinesh_ai_model_backup.pth    # 380MB
├── dinesh_ai_model_v*.pth        # 380MB × 4 = 1.5GB
└── tokenizer.json                # 2MB
Total: ~2.3GB
```

### Cloud Storage (Optional)

**AWS S3:**
```bash
# Upload backup
aws s3 cp models/dinesh_ai_model.pth s3://my-bucket/backups/

# Download backup
aws s3 cp s3://my-bucket/backups/dinesh_ai_model.pth models/
```

**Google Drive:**
```python
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

# Upload
file = drive.CreateFile({'title': 'dinesh_ai_model.pth'})
file.SetContentFile('models/dinesh_ai_model.pth')
file.Upload()
```

## 🎯 Best Practices

1. **Always backup before fine-tuning**
2. **Test model after updates**
3. **Keep at least 2 versions**
4. **Monitor disk space**
5. **Verify backup integrity**
6. **Document version changes**

## 🔧 Backup Scripts

### Create Backup

```python
import shutil
from datetime import datetime

def create_backup():
    timestamp = datetime.now().strftime('%Y-%m-%d')
    source = 'models/dinesh_ai_model.pth'
    backup = f'models/dinesh_ai_model_v{timestamp}.pth'
    
    shutil.copy(source, backup)
    print(f"Backup created: {backup}")
```

### Restore Backup

```python
def restore_backup(version_date):
    backup = f'models/dinesh_ai_model_v{version_date}.pth'
    target = 'models/dinesh_ai_model.pth'
    
    if os.path.exists(backup):
        shutil.copy(backup, target)
        print(f"Restored from: {backup}")
    else:
        print(f"Backup not found: {backup}")
```

## 📚 Related Documentation

- [Fine-Tuning System](10_FINE_TUNING.md) - When backups are created
- [Continuous Learning](09_CONTINUOUS_LEARNING.md) - Automated backups
- [Troubleshooting](12_TROUBLESHOOTING.md) - Recovery procedures

---

**Create backup now**: `copy models\dinesh_ai_model.pth models\dinesh_ai_model_backup.pth`
