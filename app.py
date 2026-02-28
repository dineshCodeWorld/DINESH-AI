import streamlit as st
import torch
import os
import json
from huggingface_hub import hf_hub_download, HfApi
from tokenizers import Tokenizer
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
from src.core.custom_model import CustomGPT

st.set_page_config(page_title="Dinesh AI", page_icon="✨", layout="centered")

st.markdown("""
<style>
    #MainMenu, footer, header {visibility: hidden;}
    .stApp {background: #0e1117;}
    .block-container {padding: 2rem 1rem; max-width: 700px;}
    
    .stTextInput input {
        background: #1e1e1e;
        border: 1px solid #333;
        border-radius: 12px;
        color: white;
        padding: 12px 16px;
    }
    
    .stButton button {
        background: linear-gradient(90deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 24px;
        width: 100%;
    }
    
    .msg {
        padding: 16px;
        border-radius: 12px;
        margin: 12px 0;
    }
    
    .user {background: #1e3a5f; color: white;}
    .bot {background: #1e1e1e; color: #e0e0e0;}
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=300)
def get_models():
    try:
        repo_id = os.environ.get('HF_REPO')
        if not repo_id:
            return {"Latest": "dinesh_ai_model.pth"}
        
        api = HfApi()
        files = api.list_repo_files(repo_id=repo_id)
        models = {"Latest": "dinesh_ai_model.pth"}
        
        for f in files:
            if f.startswith('versions/') and f.endswith('.pth'):
                v = f.split('/')[-1].replace('dinesh_ai_model_', '').replace('.pth', '')
                models[v] = f
        
        return models
    except:
        return {"Latest": "dinesh_ai_model.pth"}

@st.cache_resource
def load_model(model_file: str):
    try:
        repo_id = os.environ.get('HF_REPO')
        if not repo_id:
            return None, None, None
        
        model_path = hf_hub_download(repo_id=repo_id, filename=model_file, cache_dir="models")
        tokenizer_path = hf_hub_download(repo_id=repo_id, filename="tokenizer.json", cache_dir="models")
        config_path = hf_hub_download(repo_id=repo_id, filename="model_config.json", cache_dir="models")
        
        with open(config_path) as f:
            config = json.load(f)
        
        tokenizer = Tokenizer.from_file(tokenizer_path)
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
        
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.to(device)
        model.eval()
        return model, tokenizer, device
    except:
        return None, None, None

if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = 'Latest'

# Header
col1, col2 = st.columns([3, 2])
with col1:
    st.title("✨ Dinesh AI")
with col2:
    models = get_models()
    selected = st.selectbox("Model", list(models.keys()), 
                           index=list(models.keys()).index(st.session_state.selected_model))
    if selected != st.session_state.selected_model:
        st.session_state.selected_model = selected
        st.cache_resource.clear()
        st.rerun()

if st.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()

st.markdown("---")

# Load model
model, tokenizer, device = load_model(models[st.session_state.selected_model])

# Display messages
for msg in st.session_state.messages:
    if msg['role'] == 'user':
        st.markdown(f'<div class="msg user">👤 {msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="msg bot">✨ {msg["content"]}</div>', unsafe_allow_html=True)

# Input
if not st.session_state.messages:
    st.markdown("### Ask me anything")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🤖 What is AI?"):
            st.session_state.messages.append({"role": "user", "content": "What is artificial intelligence?"})
            st.rerun()
    with col2:
        if st.button("🌍 Science"):
            st.session_state.messages.append({"role": "user", "content": "Tell me about science"})
            st.rerun()
    with col3:
        if st.button("💻 Tech"):
            st.session_state.messages.append({"role": "user", "content": "Explain technology"})
            st.rerun()

user_input = st.text_input("Type your message", key="input", label_visibility="collapsed")

if st.button("Send") and user_input and model:
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    with st.spinner("Thinking..."):
        try:
            encoded = tokenizer.encode(user_input)
            input_ids = encoded.ids[:100]
            input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
            
            with torch.no_grad():
                output_ids = model.generate(input_tensor, max_length=min(128, model.max_seq_len),
                                          temperature=0.7, top_p=0.9, top_k=50, eos_token_id=2)
            
            text = tokenizer.decode(output_ids[0].cpu().tolist()).replace('Ġ', ' ').strip()
            st.session_state.messages.append({"role": "assistant", "content": text})
        except Exception as e:
            st.session_state.messages.append({"role": "assistant", "content": f"Error: {str(e)}"})
    st.rerun()
