# 🌐 Web Interface

Streamlit-based chat interface for Dinesh AI.

## Starting Interface

```bash
streamlit run app.py
```

Opens at: **http://localhost:8501**

## Features

### 1. Model Selection
Dropdown to choose from available models:
- dinesh_ai_model (current)
- Previous versions

### 2. Chat Input
- Text box for prompts
- Generate button
- Clear conversation button

### 3. Generation Settings

**Temperature** (0.1 - 2.0):
- 0.3: Focused, deterministic
- 0.7: Balanced (default)
- 1.5: Creative, random

**Max Length** (50 - 500):
- Number of tokens to generate
- Default: 200

### 4. Response Display
Shows:
- Generated text
- Generation time
- Token count

## Example Usage

**Prompt**: "Explain artificial intelligence"

**Response**: "Artificial intelligence is the simulation of human intelligence processes by machines..."

**Stats**: 2.3s, 127 tokens

## Customization

Edit `app.py`:

```python
st.set_page_config(
    page_title="Dinesh AI",
    page_icon="🤖"
)

# Default settings
temperature = st.slider("Temperature", 0.1, 2.0, 0.7)
max_length = st.slider("Max Length", 50, 500, 200)
```

## Troubleshooting

**Model not found**: Check `models/dinesh_ai_model.pth` exists
**Slow generation**: Reduce max_length
**Poor quality**: Adjust temperature
**Port in use**: Change port with `--server.port 8502`

## Network Access

Access from other devices:

```bash
# Get your IP
ipconfig

# Start with network access
streamlit run app.py --server.address 0.0.0.0

# Access from other device
http://YOUR_IP:8501
```

## Related Docs

- [Quick Start](01_QUICK_START.md)
- [Deployment Guide](08_DEPLOYMENT_GUIDE.md)
- [Troubleshooting](12_TROUBLESHOOTING.md)
