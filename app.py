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

@st.cache_resource(show_spinner="Downloading model from Hugging Face...")
def download_and_load_model():
    """Download latest model from Hugging Face and load it"""
    try:
        repo_id = os.environ.get('HF_REPO')
        
        if not repo_id:
            st.error("HF_REPO not set in secrets!")
            return None, None, None
        
        # Download model files
        model_path = hf_hub_download(
            repo_id=repo_id,
            filename="dinesh_ai_model.pth",
            cache_dir="models",
            token=None
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
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.to(device)
        model.eval()
        
        return model, tokenizer, device
        
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Make sure HF_REPO environment variable is set correctly")
        return None, None, None

st.title("🤖 Dinesh AI")
st.markdown("*Custom GPT trained from scratch*")

# Show loading message
with st.spinner("Loading model from Hugging Face..."):
    model, tokenizer, device = download_and_load_model()

with st.sidebar:
    st.header("⚙️ Settings")
    temperature = st.slider("Temperature", 0.1, 2.0, 0.7, 0.1)
    max_length = st.slider("Max Length", 10, 64, 50)  # Limited to model's max_seq_len
    
    st.divider()
    st.subheader("📊 Model Info")
    if model:
        st.metric("Parameters", f"{model.count_parameters():,}")
        st.metric("Vocab Size", f"{model.vocab_size:,}")
        st.metric("Layers", model.num_layers)
        st.metric("Model Dim", model.d_model)
        st.metric("Device", str(device).upper())
        st.info(f"🤗 Model: alien-x/dinesh-ai")

if model and tokenizer:
    user_input = st.text_input("Ask me anything:", placeholder="What is AI?")
    
    if st.button("Generate") and user_input:
        with st.spinner("Generating..."):
            try:
                # Encode input
                encoded = tokenizer.encode(user_input)
                input_ids = encoded.ids
                
                # Limit input length
                if len(input_ids) > 100:
                    input_ids = input_ids[:100]
                
                input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
                
                st.info(f"Input tokens: {len(input_ids)}")
                
                # Generate
                with torch.no_grad():
                    output_ids = model.generate(
                        input_tensor,
                        max_length=min(max_length, 150),
                        temperature=temperature,
                        top_p=0.9,
                        top_k=50,
                        eos_token_id=2
                    )
                
                # Decode
                generated_ids = output_ids[0].cpu().tolist()
                generated_text = tokenizer.decode(generated_ids)
                
                # Clean up BPE artifacts
                generated_text = generated_text.replace('Ġ', ' ').strip()
                
                st.success("Response:")
                st.write(generated_text)
                
            except Exception as e:
                st.error(f"Generation error: {e}")
                import traceback
                st.code(traceback.format_exc())
else:
    st.error("Model not available. Check HF_REPO environment variable.")

st.divider()
st.markdown("*Dinesh AI © 2024 | Continuously Learning*")
