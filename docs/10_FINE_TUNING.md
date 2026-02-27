# 🎯 Fine-Tuning System

Incremental learning without forgetting.

## What is Fine-Tuning?

**Full Training**: Train from scratch on all data
**Fine-Tuning**: Update existing model with new data only

## Why Fine-Tune?

### Benefits:
- 20-30 min vs 2-3 hours
- Only needs new data
- Preserves existing knowledge
- Efficient for daily updates

## Comparison

| Aspect | Full Training | Fine-Tuning |
|--------|---------------|-------------|
| Time | 2-3 hours | 20-30 min |
| Data | All 1,500 | New only |
| Learning Rate | 0.001 | 0.0001 |
| Epochs | 10 | 3 |
| Model Init | Random | Load existing |

## How It Works

### 1. Load Existing Model
```python
model = TransformerModel(...)
model.load_state_dict(torch.load('models/dinesh_ai_model.pth'))
```

### 2. Prepare New Data
```python
new_data = collect_new_data()  # Only 45 items vs 1,500
```

### 3. Lower Learning Rate
```python
optimizer = Adam(model.parameters(), lr=0.0001)  # 10x lower
```

**Why?** Prevents catastrophic forgetting

### 4. Fewer Epochs
```python
for epoch in range(3):  # vs 10 for full training
    train_on_new_data()
```

### 5. Save Updated Model
```python
torch.save(model.state_dict(), 'models/dinesh_ai_model.pth')
```

## Catastrophic Forgetting

**Problem**: Model forgets old knowledge when learning new

**Solution**: Lower learning rate + fewer epochs

```python
# BAD: Same as full training
lr = 0.001  # Too high, forgets old knowledge

# GOOD: Much lower
lr = 0.0001  # Preserves old, adds new
```

## Process

```
1. Load model (380MB)
2. Collect new data (45 items)
3. Skip duplicates (12)
4. Preprocess (33 items)
5. Tokenize
6. Fine-tune (3 epochs)
7. Save updated model
8. Validate quality
```

## Configuration

```yaml
FINE_TUNING:
  learning_rate: 0.0001
  epochs: 3
  batch_size: 8
  max_seq_length: 512
```

## When to Use

**Fine-tune when**:
- Adding < 500 new items
- Daily/weekly updates
- Time/compute limited

**Full retrain when**:
- Major data changes (> 1,000 items)
- Architecture changes
- Quality degradation

## Common Issues

**Quality degrades**: Lower learning rate to 0.00005
**Doesn't learn new**: Increase to 0.0002, more epochs
**Too slow**: Reduce epochs to 2
**Out of memory**: Reduce batch_size to 4

## Monitoring

```
Epoch 1/3: Loss=2.1 (learning)
Epoch 2/3: Loss=1.9 (refining)
Epoch 3/3: Loss=1.8 (converging)
```

**Good**: Loss decreases
**Bad**: Loss increases (lr too high)

## Manual Fine-Tuning

```python
from src.core.model_trainer import ModelTrainer

trainer = ModelTrainer('config.yaml')
trainer.load_model('models/dinesh_ai_model.pth')
trainer.fine_tune(new_data, epochs=3, lr=0.0001)
trainer.save_model('models/dinesh_ai_model.pth')
```

## Related Docs

- [Training Pipeline](05_TRAINING_PIPELINE.md)
- [Continuous Learning](09_CONTINUOUS_LEARNING.md)
- [Model Safety](11_MODEL_SAFETY.md)
