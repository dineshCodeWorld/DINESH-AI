# 📚 Data Collection

Multi-source data collection system.

## Data Sources

### 1. Wikipedia (800 articles)
**Purpose**: General knowledge, facts

**Categories**:
- Technology, Science, History
- Geography, Mathematics
- Philosophy, Psychology

**API**: https://en.wikipedia.org/w/api.php

**Key Fix**: User-Agent header required
```python
headers = {'User-Agent': 'DineshAI/1.0'}
```

### 2. ArXiv (500 papers)
**Purpose**: Scientific/technical content

**Categories**:
- cs.AI, cs.LG, cs.CL
- cs.CV, cs.NE, stat.ML

**API**: http://export.arxiv.org/api/query

### 3. Gutenberg (200 books)
**Purpose**: Conversational language, dialogue

**Why**: Wikipedia/ArXiv are formal. Novels contain conversations teaching casual language patterns.

**API**: https://gutendex.com/books/

## Deduplication

MD5 hash tracking prevents duplicates:

```python
content_hash = hashlib.md5(content.encode()).hexdigest()
if content_hash not in collected_hashes:
    collect(content)
```

Saved in: `data/collected_hashes.json`

## Collection Stats

| Source | Count | Size | Time |
|--------|-------|------|------|
| Wikipedia | 800 | 1.6MB | 10-15 min |
| ArXiv | 500 | 7.5MB | 5-8 min |
| Gutenberg | 200 | 40MB | 15-20 min |
| **Total** | **1,500** | **~50MB** | **30-45 min** |

## Storage Structure

```
data/
├── raw/
│   ├── wikipedia_*.json
│   ├── arxiv_*.json
│   └── gutenberg_*.json
├── processed/train_data.json
└── collected_hashes.json
```

## Configuration

```yaml
DATA_SOURCES:
  wikipedia:
    limit: 800
    categories: [Technology, Science, ...]
  arxiv:
    limit: 500
    categories: [cs.AI, cs.LG, ...]
  gutenberg:
    limit: 200
```

## Common Issues

**Wikipedia 403**: Missing User-Agent (fixed)
**Collection stops early**: Loop issue (fixed)
**ArXiv only 10 results**: max_results too low (fixed)

## Related Docs

- [Training Pipeline](05_TRAINING_PIPELINE.md)
- [Configuration](03_CONFIGURATION.md)
