import streamlit as st
import torch
import os
import json
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.core.custom_model import CustomGPT

st.set_page_config(page_title="Dinesh AI", page_icon="🤖", layout="wide")

@st.cache_resource
def download_and_load_model():
    """Download latest model from Hugging Face and load it"""
    try:
        repo_id = os.environ.get('HF_REPO', 'yourusername/dinesh-ai')
        
        st.info(f"Downloading model from {repo_id}...")
        
        # Download model files
        model_path = hf_hub_download(
            repo_id=repo_id,
            filename="dinesh_ai_model.pth",
            cache_dir="models"
        )
        
        tokenizer_path = hf_hub_download(
            repo_id=repo_id,
            filename="tokenizer.json",
            cache_dir="models"
        )
        
        config_path = hf_hub_download(
            repo_id=repo_id,
            filename="model_config.json",
            cache_dir="models"
        )
        
        # Load config
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Load tokenizer
        tokenizer = Tokenizer.from_file(tokenizer_path)
        
        # Initialize model with config
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = CustomGPT(
            vocab_size=config["vocab_size"],
            d_model=config["d_model"],
            num_layers=config["num_layers"],
            num_heads=config.get("num_heads", config["d_model"] // 64),
            d_ff=config.get("d_ff", config["d_model"] * 4),
            max_seq_len=config["max_seq_len"],
            device=str(device)
        )
        
        # Load weights
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        
        st.success("Model loaded successfully!")
        return model, tokenizer, device
        
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Make sure HF_REPO environment variable is set correctly")
        return None, None, None

st.title("🤖 Dinesh AI")
st.markdown("*Custom GPT trained from scratch*")

with st.sidebar:
    st.header("⚙️ Settings")
    temperature = st.slider("Temperature", 0.1, 2.0, 0.7, 0.1)
    max_length = st.slider("Max Length", 50, 300, 150)

model, tokenizer, device = download_and_load_model()

if model and tokenizer:
    user_input = st.text_input("Ask me anything:", placeholder="What is AI?")
    
    if st.button("Generate") and user_input:
        with st.spinner("Generating..."):
            # Add your generation logic here
            st.info(f"Response for: {user_input}")
else:
    st.error("Model not available. Check HF_REPO environment variable.")

st.divider()
st.markdown("*Dinesh AI © 2024 | Continuously Learning*")
