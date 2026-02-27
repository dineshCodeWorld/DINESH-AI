# 🏗️ Architecture Overview

Understanding how Dinesh AI works under the hood.

## 🎯 System Overview

Dinesh AI is a custom GPT-style language model built from scratch with:
- **Custom transformer architecture** (not pretrained)
- **Multi-source data collection** (Wikipedia, ArXiv, Gutenberg)
- **Incremental learning** (fine-tuning on new data)
- **Automatic deduplication** (prevents duplicate training)
- **Model versioning** (weekly snapshots)
- **Web interface** (Streamlit-based chat)

## 📊 High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Dinesh AI System                      │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  ┌──────────────┐    ┌──────────────┐    ┌───────────┐ │
│  │ Data Sources │───▶│  Collection  │───▶│Processing │ │
│  │              │    │              │    │           │ │
│  │ • Wikipedia  │    │ • Dedup      │    │ • Clean   │ │
│  │ • ArXiv      │    │ • Schedule   │    │ • Filter  │ │
│  │ • Gutenberg  │    │ • Parallel   │    │ • Format  │ │
│  └──────────────┘    └──────────────┘    └───────────┘ │
│         │                                       │        │
│         └───────────────────┬───────────────────┘        │
│                             ▼                            │
│                    ┌──────────────┐                      │
│                    │ Tokenization │                      │
│                    │              │                      │
│                    │ • BPE        │                      │
│                    │ • 5K-50K     │                      │
│                    └──────────────┘                      │
│                             │                            │
│                             ▼                            │
│                    ┌──────────────┐                      │
│                    │   Training   │                      │
│                    │              │                      │
│                    │ • From       │                      │
│                    │   Scratch    │                      │
│                    │ • Fine-tune  │                      │
│                    └──────────────┘                      │
│                             │                            │
│                             ▼                            │
│         ┌──────────────────────────────────┐            │
│         │         Model Storage            │            │
│         │                                  │            │
│         │ • Daily models                   │            │
│         │ • Weekly versions                │            │
│         │ • Automatic backups              │            │
│         └──────────────────────────────────┘            │
│                             │                            │
│                             ▼                            │
│                    ┌──────────────┐                      │
│                    │ Web Interface│                      │
│                    │              │                      │
│                    │ • Chat       │                      │
│                    │ • Testing    │                      │
│                    │ • Info       │                      │
│                    └──────────────┘                      │
└─────────────────────────────────────────────────────────┘
```

## 🧠 Model Architecture

### Custom GPT Transformer

```
Input Text
    │
    ▼
┌─────────────────┐
│ Tokenization    │  BPE tokenizer
│ (5K-50K vocab)  │
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ Token Embedding │  vocab_size × d_model
│ + Positional    │  max_length × d_model
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ Transformer     │  ┌──────────────────┐
│ Decoder Block 1 │──│ Self-Attention   │
│                 │  │ (Multi-head)     │
│                 │  ├──────────────────┤
│                 │  │ Feed-Forward     │
│                 │  │ Network          │
│                 │  └──────────────────┘
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ Transformer     │  (Repeat num_layers times)
│ Decoder Block N │
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ Layer Norm      │
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ Output          │  d_model → vocab_size
│ Projection      │
└─────────────────┘
    │
    ▼
Generated Text
```

### Transformer Decoder Block

```
Input (batch, seq_len, d_model)
    │
    ▼
┌─────────────────────────────────┐
│ Layer Normalization             │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│ Multi-Head Self-Attention       │
│                                 │
│ ┌─────┐ ┌─────┐ ┌─────┐        │
│ │  Q  │ │  K  │ │  V  │        │
│ └─────┘ └─────┘ └─────┘        │
│    │       │       │            │
│    └───────┴───────┘            │
│           │                     │
│    ┌──────▼──────┐              │
│    │  Attention  │              │
│    │  (Causal)   │              │
│    └─────────────┘              │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│ Residual Connection + Dropout   │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│ Layer Normalization             │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│ Feed-Forward Network            │
│                                 │
│ Linear(d_model → d_ff)          │
│         ↓                       │
│       GELU                      │
│         ↓                       │
│ Linear(d_ff → d_model)          │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│ Residual Connection + Dropout   │
└─────────────────────────────────┘
    │
    ▼
Output (batch, seq_len, d_model)
```

## 🔄 Training Pipeline

### Complete Workflow

```
1. Data Collection
   ├─ Wikipedia API → Articles
   ├─ ArXiv API → Papers
   └─ Gutenberg → Books
   
2. Deduplication
   ├─ MD5 hash each item
   ├─ Check seen_content_hashes.json
   └─ Skip duplicates
   
3. Preprocessing
   ├─ Clean text
   ├─ Remove HTML/special chars
   ├─ Filter by length
   └─ Merge with existing data
   
4. Tokenization
   ├─ Train BPE tokenizer
   ├─ Build vocabulary (5K-50K)
   └─ Save tokenizer.json
   
5. Model Training
   ├─ Load previous model (if exists)
   ├─ Fine-tune on new data
   ├─ Save checkpoints
   └─ Save final model
   
6. Deployment
   ├─ Create model card
   ├─ Save configuration
   ├─ Create backups
   └─ Ready for use
```

### Training Modes

**From Scratch (Day 1):**
```python
model = CustomGPT(config)
model.train(data)
model.save()
```

**Fine-Tuning (Day 2+):**
```python
model = load_previous_model()
model.fine_tune(new_data, lr=1e-5, epochs=1)
model.save()
```

## 💾 Data Flow

### Collection → Training

```
Wikipedia/ArXiv/Gutenberg
         │
         ▼
    [Raw Data]
    data/raw/
         │
         ▼
  [Deduplication]
  MD5 hashing
         │
         ▼
   [Processing]
   Clean & filter
         │
         ▼
  [Processed Data]
  data/processed/
         │
         ▼
  [Tokenization]
  BPE encoding
         │
         ▼
   [Training]
   Model learning
         │
         ▼
  [Trained Model]
  models/model_*/
```

### Continuous Learning

```
Day 1: Collect → Train from scratch → Save v1
Day 2: Collect → Fine-tune v1 → Save v2
Day 3: Collect → Fine-tune v2 → Save v3
...
Week 1: Create version snapshot → v1-YYYY-MM-DD
```

## 🔧 Key Components

### 1. Data Collector (`src/data/data_collector.py`)
- Fetches from multiple sources
- MD5-based deduplication
- Random sampling for freshness
- Parallel collection

### 2. Data Preprocessor (`src/data/data_preprocessor.py`)
- Text cleaning
- Length filtering
- Format standardization
- Merging with existing data

### 3. Custom Model (`src/core/custom_model.py`)
- Transformer architecture
- Multi-head attention
- Causal masking
- Text generation

### 4. Model Trainer (`src/core/model_trainer.py`)
- Training from scratch
- Fine-tuning
- Checkpoint management
- Loss tracking

### 5. Continuous Trainer (`src/continuous/continuous_trainer.py`)
- Scheduled collection (every 6 hours)
- Scheduled training (every 24 hours)
- Model search & backup
- Parallel execution

### 6. Web Interface (`app.py`)
- Streamlit-based UI
- Model loading
- Text generation
- Statistics display

## 📊 Model Specifications

### Local Configuration
```
Parameters: ~2M
Layers: 2
Dimension: 128
Heads: 2
FFN: 512
Vocabulary: 5,000
Max Length: 128
Size: ~5MB
```

### Production Configuration
```
Parameters: ~100M
Layers: 12
Dimension: 768
Heads: 12
FFN: 3,072
Vocabulary: 50,000
Max Length: 512
Size: ~380MB
```

## 🎯 Design Decisions

### Why Custom Architecture?
- **Full control** over model design
- **Learning experience** - understand transformers deeply
- **Customization** - optimize for specific needs
- **No black boxes** - complete transparency

### Why From Scratch?
- **Educational value** - learn by building
- **Flexibility** - modify as needed
- **Understanding** - know every component
- **Innovation** - implement new ideas

### Why Fine-Tuning?
- **Incremental learning** - don't forget old knowledge
- **Efficiency** - faster than retraining
- **Continuous improvement** - always learning
- **Resource-friendly** - less computation

### Why Deduplication?
- **Efficiency** - don't process same data twice
- **Quality** - avoid overfitting on duplicates
- **Storage** - save disk space
- **Speed** - faster training

## 🔐 Technical Details

### Attention Mechanism
```python
scores = Q @ K.T / sqrt(d_k)
scores = mask_future(scores)  # Causal masking
attention = softmax(scores)
output = attention @ V
```

### Causal Masking
```
Prevents attending to future tokens:
[1, 0, 0, 0]
[1, 1, 0, 0]
[1, 1, 1, 0]
[1, 1, 1, 1]
```

### BPE Tokenization
```
"artificial intelligence" →
["art", "ificial", " intel", "ligence"]
```

### Parameter Count
```
Embeddings: vocab_size × d_model
Positional: max_length × d_model
Per Layer: 4 × d_model² (attention + FFN)
Output: d_model × vocab_size
Total: ~100M for production config
```

## 📈 Performance Characteristics

### Training Speed
- CPU: ~40 min/epoch (production)
- GPU: ~5-10 min/epoch (production)
- Local: ~5 min/epoch

### Inference Speed
- CPU: ~1-2 sec/response
- GPU: ~0.1-0.5 sec/response

### Memory Usage
- Training: ~4GB (production)
- Inference: ~500MB (production)
- Local: ~500MB total

---

**Next:** [Training Pipeline](05_TRAINING_PIPELINE.md)

*Last Updated: February 26, 2026*
