# ⚙️ Configuration Guide

Complete guide to configuring Dinesh AI for your needs.

## 📋 Overview

Dinesh AI uses two configuration files:
- **config.yaml** - Production training (full model)
- **config.local.yaml** - Local testing (minimal model)

## 🎯 Configuration Files

### config.yaml (Production)

**Purpose:** Full-scale training for production use

**Key settings:**
- 100M parameters
- 50,000 vocabulary
- 1,600 training items
- 3 epochs
- ~2-3 hours training time

### config.local.yaml (Local Testing)

**Purpose:** Quick testing and development

**Key settings:**
- 2M parameters
- 5,000 vocabulary
- 35 training items
- 2 epochs
- ~10-15 minutes training time

## 📊 Configuration Sections

### 1. Model Configuration

```yaml
model:
  vocab_size: 50000          # Vocabulary size
  d_model: 768               # Model dimension
  num_layers: 12             # Transformer layers
  num_heads: 12              # Attention heads
  d_ff: 3072                 # Feed-forward dimension
  max_length: 512            # Max sequence length
  dropout: 0.1               # Dropout rate
  temperature: 0.7           # Generation temperature
  top_k: 50                  # Top-K sampling
  top_p: 0.9                 # Nucleus sampling
  max_new_tokens: 100        # Max generated tokens
```

**Parameters explained:**

- **vocab_size**: Number of unique tokens
  - Larger = better language coverage, slower
  - Minimum: 1,000 (basic)
  - Recommended: 50,000 (production)

- **d_model**: Hidden dimension size
  - Larger = more capacity, slower
  - Must be divisible by num_heads
  - Common: 128, 256, 512, 768, 1024

- **num_layers**: Number of transformer blocks
  - More = better quality, slower
  - Minimum: 1
  - Recommended: 6-12

- **num_heads**: Multi-head attention heads
  - More = better attention, slower
  - Must divide d_model evenly
  - Common: 4, 8, 12, 16

- **d_ff**: Feed-forward network size
  - Usually 4× d_model
  - Larger = more capacity

- **max_length**: Maximum input sequence
  - Longer = more context, more memory
  - Common: 128, 256, 512, 1024

### 2. Training Configuration

```yaml
training:
  batch_size: 16             # Samples per batch
  learning_rate: 0.00002     # Learning rate
  weight_decay: 0.01         # Weight decay (regularization)
  epochs: 3                  # Training epochs
  warmup_steps: 1000         # Learning rate warmup
  optimizer: "adamw"         # Optimizer type
  scheduler: "cosine"        # LR scheduler
  gradient_clip: 1.0         # Gradient clipping
  device: "cpu"              # cpu or cuda
  mixed_precision: false     # FP16 training
  save_interval: 500         # Checkpoint interval
  eval_interval: 500         # Evaluation interval
  keep_best_checkpoints: 3   # Number to keep
  early_stopping_patience: 5 # Early stopping
```

**Parameters explained:**

- **batch_size**: Samples processed together
  - Larger = faster, more memory
  - Reduce if out of memory
  - Common: 4, 8, 16, 32

- **learning_rate**: Step size for updates
  - Too high = unstable
  - Too low = slow learning
  - Typical: 0.00001 - 0.0001

- **epochs**: Complete passes through data
  - More = better learning, longer
  - Typical: 3-5

- **device**: Hardware to use
  - "cpu" = slower, always works
  - "cuda" = 30-50× faster, needs GPU

### 3. Data Configuration

```yaml
data:
  train_split: 0.9           # Training data %
  val_split: 0.05            # Validation data %
  test_split: 0.05           # Test data %
  max_samples: 1600          # Max training samples
  min_length: 50             # Min text length
  max_length: 10000          # Max text length
  shuffle: true              # Shuffle data
  bpe_vocab_size: 50000      # BPE vocabulary
  bpe_min_frequency: 2       # Min token frequency
```

### 4. Data Sources Configuration

```yaml
data_sources:
  wikipedia:
    enabled: true
    limit: 1000              # Articles to collect
    categories:
      - Technology
      - Science
      - History
      - Mathematics
  
  arxiv:
    enabled: true
    limit: 500               # Papers to collect
    sort_by: "submittedDate"
    categories:
      - cs.AI
      - cs.LG
      - cs.CL
  
  gutenberg:
    enabled: true
    limit: 100               # Books to collect
    mirror: "https://www.gutenberg.org"
```

### 5. Continuous Learning Configuration

```yaml
continuous:
  enabled: true              # Enable 24/7 learning
  collection_interval_hours: 6   # Collect every 6 hours
  fine_tune_interval_hours: 24   # Train every 24 hours
  sample_size: 170           # Items per collection
  retention_days: 30         # Keep data for days
  auto_cleanup: true         # Auto delete old data
  weekly_versioning: true    # Create weekly versions
  auto_deploy: false         # Auto deploy models
```

### 6. Deployment Configuration

```yaml
deployment:
  push_to_hub: false         # Push to Hugging Face
  hf_repo_name: "dinesh-ai"  # HF repository name
  hf_private: true           # Private repository
  version_format: "v{number}-{date}"
  archive_old_versions: true
  targets:
    hugging_face: false
    local: true
    oracle_cloud: false
```

### 7. Logging Configuration

```yaml
logging:
  level: "INFO"              # DEBUG, INFO, WARNING, ERROR
  format: "%(asctime)s - %(levelname)s - %(message)s"
  log_file: "logs/training.log"
  max_file_size: 10485760    # 10MB
  backup_count: 5
  console_output: true
  tensorboard: false
  wandb: false
```

## 🎛️ Common Configurations

### Minimal (Fast Testing)
```yaml
model:
  d_model: 128
  num_layers: 2
  num_heads: 2
  vocab_size: 5000

training:
  batch_size: 4
  epochs: 2
  
data_sources:
  wikipedia:
    limit: 20
  arxiv:
    limit: 10
  gutenberg:
    limit: 5
```
**Time:** 10-15 minutes
**Model:** ~2M parameters, ~5MB

### Small (Quick Training)
```yaml
model:
  d_model: 256
  num_layers: 4
  num_heads: 4
  vocab_size: 10000

training:
  batch_size: 8
  epochs: 3
  
data_sources:
  wikipedia:
    limit: 100
  arxiv:
    limit: 50
  gutenberg:
    limit: 20
```
**Time:** 30-45 minutes
**Model:** ~10M parameters, ~40MB

### Medium (Balanced)
```yaml
model:
  d_model: 512
  num_layers: 8
  num_heads: 8
  vocab_size: 30000

training:
  batch_size: 12
  epochs: 3
  
data_sources:
  wikipedia:
    limit: 500
  arxiv:
    limit: 250
  gutenberg:
    limit: 50
```
**Time:** 1-1.5 hours
**Model:** ~50M parameters, ~200MB

### Large (Production)
```yaml
model:
  d_model: 768
  num_layers: 12
  num_heads: 12
  vocab_size: 50000

training:
  batch_size: 16
  epochs: 3
  
data_sources:
  wikipedia:
    limit: 1000
  arxiv:
    limit: 500
  gutenberg:
    limit: 100
```
**Time:** 2-3 hours
**Model:** ~100M parameters, ~380MB

## 🔧 Using Configurations

### Method 1: Command Line Flag
```bash
python scripts/train.py --config config.local.yaml
```

### Method 2: --local Shortcut
```bash
python scripts/train.py --local
```
Automatically uses config.local.yaml

### Method 3: Environment Variable
```bash
export DINESH_AI_CONFIG=config.local.yaml
python scripts/train.py
```

### Method 4: Default
```bash
python scripts/train.py
```
Uses config.yaml by default

## 💡 Configuration Tips

### For Fast Testing
- Use config.local.yaml
- Reduce data collection limits
- Decrease epochs to 1-2
- Use smaller model dimensions

### For Best Quality
- Use config.yaml
- Increase data collection limits
- Train for 3-5 epochs
- Use larger model dimensions
- Enable GPU if available

### For Memory Constraints
- Reduce batch_size
- Reduce d_model
- Reduce max_length
- Use gradient checkpointing

### For GPU Training
```yaml
training:
  device: "cuda"
  mixed_precision: true  # FP16 for 2× speedup
  batch_size: 32         # Larger batches
```

## 🎯 Parameter Relationships

### Model Size Calculation
```
Parameters ≈ vocab_size × d_model + 
             num_layers × (4 × d_model²)
```

### Memory Usage
```
Memory ≈ batch_size × max_length × d_model × 4 bytes
```

### Training Time
```
Time ≈ data_samples × epochs / (batch_size × speed)
```

## ✅ Validation

### Check Configuration
```bash
python -c "import yaml; yaml.safe_load(open('config.yaml'))"
```

### Verify Settings
```bash
python scripts/train.py --config config.yaml --dry-run
```

## 🐛 Common Issues

### "d_model must be divisible by num_heads"
Ensure d_model % num_heads == 0

### "Out of memory"
Reduce batch_size or d_model

### "CUDA out of memory"
Reduce batch_size or use CPU

### "Invalid YAML"
Check indentation and syntax

---

**Next:** [Architecture Overview](04_ARCHITECTURE.md)

*Last Updated: February 26, 2026*
