import streamlit as st
import torch
import os
import json
from huggingface_hub import hf_hub_download, HfApi
from tokenizers import Tokenizer
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.core.custom_model import CustomGPT

# Page config
st.set_page_config(
    page_title="Dinesh AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Modern CSS
st.markdown("""
<style>
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main container */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 0rem;
        max-width: 900px;
    }
    
    /* Header */
    .app-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1rem 0;
        border-bottom: 1px solid #e0e0e0;
        margin-bottom: 2rem;
    }
    
    .app-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1a1a1a;
    }
    
    /* Chat container */
    .chat-container {
        height: 60vh;
        overflow-y: auto;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    /* Messages */
    .message {
        margin-bottom: 1.5rem;
        display: flex;
        gap: 1rem;
    }
    
    .message-avatar {
        width: 32px;
        height: 32px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.2rem;
        flex-shrink: 0;
    }
    
    .user-avatar {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .bot-avatar {
        background: #f0f0f0;
    }
    
    .message-content {
        flex: 1;
        padding: 0.75rem 1rem;
        border-radius: 12px;
        line-height: 1.6;
    }
    
    .user-message .message-content {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: auto;
    }
    
    .bot-message .message-content {
        background: #f7f7f8;
        color: #1a1a1a;
    }
    
    /* Input area */
    .input-container {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: white;
        border-top: 1px solid #e0e0e0;
        padding: 1rem;
        z-index: 999;
    }
    
    .stTextInput > div > div > input {
        border-radius: 24px;
        border: 1px solid #e0e0e0;
        padding: 0.75rem 1.5rem;
        font-size: 1rem;
    }
    
    .stButton > button {
        border-radius: 24px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-weight: 500;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* Model selector */
    .stSelectbox {
        border-radius: 8px;
    }
    
    /* Examples */
    .example-card {
        background: #f7f7f8;
        border-radius: 12px;
        padding: 1rem;
        cursor: pointer;
        transition: all 0.3s;
        border: 1px solid transparent;
    }
    
    .example-card:hover {
        border-color: #667eea;
        transform: translateY(-2px);
    }
    
    /* Empty state */
    .empty-state {
        text-align: center;
        padding: 3rem 1rem;
        color: #666;
    }
    
    .empty-state-icon {
        font-size: 4rem;
        margin-bottom: 1rem;
    }
    
    /* Typing indicator */
    .typing-indicator {
        display: flex;
        gap: 0.3rem;
        padding: 1rem;
    }
    
    .typing-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #999;
        animation: typing 1.4s infinite;
    }
    
    .typing-dot:nth-child(2) { animation-delay: 0.2s; }
    .typing-dot:nth-child(3) { animation-delay: 0.4s; }
    
    @keyframes typing {
        0%, 60%, 100% { transform: translateY(0); }
        30% { transform: translateY(-10px); }
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=300)
def get_available_models():
    """Get list of available model versions"""
    try:
        repo_id = os.environ.get('HF_REPO')
        if not repo_id:
            return {"Latest": "dinesh_ai_model.pth"}
        
        api = HfApi()
        files = api.list_repo_files(repo_id=repo_id)
        
        models = {"Latest": "dinesh_ai_model.pth"}
        
        for f in files:
            if f.startswith('versions/') and f.endswith('.pth'):
                version_name = f.split('/')[-1].replace('dinesh_ai_model_', '').replace('.pth', '')
                models[version_name] = f
        
        return models
    except:
        return {"Latest": "dinesh_ai_model.pth"}

@st.cache_resource
def load_model(model_file: str):
    """Load model from Hugging Face"""
    try:
        repo_id = os.environ.get('HF_REPO')
        if not repo_id:
            return None, None, None
        
        model_path = hf_hub_download(repo_id=repo_id, filename=model_file, cache_dir="models")
        tokenizer_path = hf_hub_download(repo_id=repo_id, filename="tokenizer.json", cache_dir="models")
        config_path = hf_hub_download(repo_id=repo_id, filename="model_config.json", cache_dir="models")
        
        with open(config_path, 'r') as f:
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
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = 'Latest'

# Header
col1, col2, col3 = st.columns([2, 2, 1])
with col1:
    st.markdown('<div class="app-title">🤖 Dinesh AI</div>', unsafe_allow_html=True)
with col2:
    available_models = get_available_models()
    selected = st.selectbox(
        "Model",
        options=list(available_models.keys()),
        index=list(available_models.keys()).index(st.session_state.selected_model),
        label_visibility="collapsed"
    )
    if selected != st.session_state.selected_model:
        st.session_state.selected_model = selected
        st.cache_resource.clear()
        st.rerun()
with col3:
    if st.button("🗑️ Clear", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.markdown("---")

# Load model
with st.spinner(f"Loading {st.session_state.selected_model}..."):
    model, tokenizer, device = load_model(available_models[st.session_state.selected_model])

# Chat display
if len(st.session_state.messages) == 0:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-state-icon">💬</div>
        <h2>How can I help you today?</h2>
        <p>Ask me anything about AI, science, technology, or any topic!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Example prompts
    st.markdown("### Try these examples:")
    col1, col2, col3 = st.columns(3)
    
    examples = [
        ("🤖", "What is AI?", "Explain artificial intelligence in simple terms"),
        ("🌍", "Climate change", "Tell me about climate change"),
        ("💡", "Machine learning", "How does machine learning work?")
    ]
    
    for col, (icon, title, prompt) in zip([col1, col2, col3], examples):
        with col:
            if st.button(f"{icon} {title}", use_container_width=True, key=f"ex_{title}"):
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.rerun()
else:
    # Display messages
    for msg in st.session_state.messages:
        if msg['role'] == 'user':
            st.markdown(f"""
            <div class="message user-message">
                <div class="message-content">{msg['content']}</div>
                <div class="message-avatar user-avatar">👤</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="message bot-message">
                <div class="message-avatar bot-avatar">🤖</div>
                <div class="message-content">{msg['content']}</div>
            </div>
            """, unsafe_allow_html=True)

# Input area
st.markdown("<br><br>", unsafe_allow_html=True)
col1, col2 = st.columns([6, 1])

with col1:
    user_input = st.text_input(
        "Message",
        placeholder="Type your message here...",
        label_visibility="collapsed",
        key="input"
    )

with col2:
    send = st.button("Send", use_container_width=True, type="primary")

# Generate response
if send and user_input and model:
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    with st.spinner("Thinking..."):
        try:
            encoded = tokenizer.encode(user_input)
            input_ids = encoded.ids[:100]
            input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
            
            with torch.no_grad():
                output_ids = model.generate(
                    input_tensor,
                    max_length=min(64, model.max_seq_len),
                    temperature=0.7,
                    top_p=0.9,
                    top_k=50,
                    eos_token_id=2
                )
            
            generated_text = tokenizer.decode(output_ids[0].cpu().tolist())
            generated_text = generated_text.replace('Ġ', ' ').strip()
            
            st.session_state.messages.append({"role": "assistant", "content": generated_text})
        except Exception as e:
            st.session_state.messages.append({"role": "assistant", "content": f"Sorry, I encountered an error: {str(e)}"})
    
    st.rerun()

# Footer
st.markdown("<br><br><br>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; color: #999; font-size: 0.9rem; padding: 2rem 0;'>
    Dinesh AI • Trained from scratch • <a href='https://github.com/dineshCodeWorld/DINESH-AI' style='color: #667eea;'>GitHub</a> • 
    <a href='https://huggingface.co/alien-x/dinesh-ai' style='color: #667eea;'>Hugging Face</a>
</div>
""", unsafe_allow_html=True)
