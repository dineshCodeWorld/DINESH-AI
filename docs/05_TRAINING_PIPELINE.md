# 🔄 Training Pipeline

Complete training workflow from data to deployed model.

## Pipeline Stages

### 1. Data Collection (10% - 20 min)
- Wikipedia: 800 articles
- ArXiv: 500 papers  
- Gutenberg: 200 books
- Total: 1,500 items

### 2. Preprocessing (5% - 5 min)
- Clean text
- Remove duplicates
- Filter short texts
- Normalize formatting

### 3. Tokenization (5% - 10 min)
- Train BPE tokenizer
- Build 50K vocabulary
- min_frequency=1 (critical!)
- Save tokenizer.json

### 4. Model Training (75% - 2 hours)
- Initialize transformer
- 10 epochs
- Batch size: 8
- Save checkpoints

### 5. Deployment (5% - 5 min)
- Save final model
- Create model_info.json
- Validate loading

## Running Pipeline

```bash
python scripts/train.py
```

## Output Files

```
models/
├── dinesh_ai_model.pth (380MB)
├── tokenizer.json (2MB)
└── model_info.json

data/
├── raw/ (Wikipedia, ArXiv, Gutenberg)
├── processed/train_data.json
└── collected_hashes.json

logs/
└── training_*.log
```

## Key Settings

```yaml
MODEL:
  vocab_size: 50000
  d_model: 768
  num_layers: 12
  num_heads: 12

TRAINING:
  batch_size: 8
  learning_rate: 0.001
  epochs: 10
```

## Common Issues

**Vocab too small (< 5000)**: Set min_frequency=1
**Out of memory**: Reduce batch_size to 4
**Slow training**: Use GPU if available

## Related Docs

- [Data Collection](06_DATA_COLLECTION.md)
- [Configuration](03_CONFIGURATION.md)
- [Troubleshooting](12_TROUBLESHOOTING.md)
