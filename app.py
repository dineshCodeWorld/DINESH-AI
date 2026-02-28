import streamlit as st
import torch
import os
import json
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.core.custom_model import CustomGPT

st.set_page_config(
    page_title="Dinesh AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #e3f2fd;
        margin-left: 20%;
    }
    .bot-message {
        background-color: #f5f5f5;
        margin-right: 20%;
    }
    .message-label {
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        font-size: 1rem;
        border-radius: 0.5rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_available_models():
    """Get list of available model versions from Hugging Face"""
    try:
        from huggingface_hub import HfApi
        repo_id = os.environ.get('HF_REPO')
        if not repo_id:
            return ["latest"]
        
        api = HfApi()
        files = api.list_repo_files(repo_id=repo_id)
        
        # Get all model files
        models = {"latest": "dinesh_ai_model.pth"}
        
        for f in files:
            if f.startswith('versions/') and f.endswith('.pth'):
                # Extract version name
                version_name = f.split('/')[-1].replace('dinesh_ai_model_', '').replace('.pth', '')
                models[version_name] = f
        
        return models
    except:
        return {"latest": "dinesh_ai_model.pth"}

@st.cache_resource
def load_specific_model(model_file: str):
    """Load a specific model version"""
    try:
        repo_id = os.environ.get('HF_REPO')
        
        if not repo_id:
            st.error("❌ HF_REPO not set in secrets!")
            return None, None, None
        
        # Download model files
        model_path = hf_hub_download(
            repo_id=repo_id,
            filename=model_file,
            cache_dir="models",
            token=None
        )
        
        tokenizer_path = hf_hub_download(
            repo_id=repo_id,
            filename="tokenizer.json",
            cache_dir="models",
            token=None
        )
        
        config_path = hf_hub_download(
            repo_id=repo_id,
            filename="model_config.json",
            cache_dir="models",
            token=None
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
        st.error(f"❌ Error loading model: {e}")
        return None, None, None

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'total_tokens' not in st.session_state:
    st.session_state.total_tokens = 0
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = 'latest'
if 'current_model' not in st.session_state:
    st.session_state.current_model = None

# Header
st.markdown('<div class="main-header">🤖 Dinesh AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Custom GPT trained from scratch • Continuously Learning</div>', unsafe_allow_html=True)

# Get available models
available_models = get_available_models()

# Model selector in header
col1, col2, col3 = st.columns([2, 2, 1])
with col2:
    selected_model_key = st.selectbox(
        "🎯 Select Model Version",
        options=list(available_models.keys()),
        index=list(available_models.keys()).index(st.session_state.selected_model) if st.session_state.selected_model in available_models else 0,
        help="Choose which model version to use"
    )
    
    # Load model if selection changed
    if selected_model_key != st.session_state.selected_model:
        st.session_state.selected_model = selected_model_key
        st.session_state.current_model = None  # Force reload
        st.rerun()

# Load model
if st.session_state.current_model is None:
    with st.spinner(f"🔄 Loading {selected_model_key} model..."):
        model, tokenizer, device = load_specific_model(available_models[selected_model_key])
        st.session_state.current_model = (model, tokenizer, device)
else:
    model, tokenizer, device = st.session_state.current_model

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=80)
    st.title("⚙️ Settings")
    
    temperature = st.slider("🌡️ Temperature", 0.1, 2.0, 0.7, 0.1, 
                           help="Higher = more creative, Lower = more focused")
    max_length = st.slider("📏 Max Length", 10, 128, 50,
                          help="Maximum tokens to generate")
    top_k = st.slider("🎯 Top-K", 10, 100, 50,
                     help="Consider top K tokens")
    
    st.divider()
    
    # Model Info
    if model:
        st.subheader("📊 Model Stats")
        st.info(f"🎯 **Active:** {st.session_state.selected_model}")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Parameters", f"{model.count_parameters():,}")
            st.metric("Layers", model.num_layers)
            st.metric("Vocab", f"{model.vocab_size:,}")
        with col2:
            st.metric("Dimension", model.d_model)
            st.metric("Max Length", model.max_seq_len)
            st.metric("Device", str(device).upper())
        
        st.info(f"🤗 **Model:** {os.environ.get('HF_REPO', 'N/A')}")
    
    st.divider()
    
    # Stats
    st.subheader("💬 Session Stats")
    st.metric("Messages", len(st.session_state.chat_history))
    st.metric("Tokens Used", st.session_state.total_tokens)
    
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.session_state.total_tokens = 0
        st.rerun()

# Main chat interface
if model and tokenizer:
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_history:
            if msg['role'] == 'user':
                st.markdown(f"""
                <div class="chat-message user-message">
                    <div class="message-label">👤 You</div>
                    <div>{msg['content']}</div>
                    <div style="font-size: 0.8rem; color: #999; margin-top: 0.5rem;">{msg['time']}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message bot-message">
                    <div class="message-label">🤖 Dinesh AI</div>
                    <div>{msg['content']}</div>
                    <div style="font-size: 0.8rem; color: #999; margin-top: 0.5rem;">{msg['time']} • {msg.get('tokens', 0)} tokens</div>
                </div>
                """, unsafe_allow_html=True)
    
    # Input area
    st.divider()
    col1, col2 = st.columns([5, 1])
    
    with col1:
        user_input = st.text_input("💭 Ask me anything...", 
                                   placeholder="What is artificial intelligence?",
                                   label_visibility="collapsed",
                                   key="user_input")
    
    with col2:
        send_button = st.button("Send 🚀", use_container_width=True)
    
    # Generate response
    if send_button and user_input:
        # Add user message
        st.session_state.chat_history.append({
            'role': 'user',
            'content': user_input,
            'time': datetime.now().strftime("%H:%M")
        })
        
        with st.spinner("🤔 Thinking..."):
            try:
                # Encode input
                encoded = tokenizer.encode(user_input)
                input_ids = encoded.ids
                
                # Limit input length
                if len(input_ids) > 100:
                    input_ids = input_ids[:100]
                
                input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
                
                # Generate
                with torch.no_grad():
                    output_ids = model.generate(
                        input_tensor,
                        max_length=min(max_length, model.max_seq_len),
                        temperature=temperature,
                        top_p=0.9,
                        top_k=top_k,
                        eos_token_id=2
                    )
                
                # Decode
                generated_ids = output_ids[0].cpu().tolist()
                generated_text = tokenizer.decode(generated_ids)
                
                # Clean up BPE artifacts
                generated_text = generated_text.replace('Ġ', ' ').strip()
                
                # Add bot message
                st.session_state.chat_history.append({
                    'role': 'bot',
                    'content': generated_text,
                    'time': datetime.now().strftime("%H:%M"),
                    'tokens': len(generated_ids)
                })
                
                st.session_state.total_tokens += len(generated_ids)
                
            except Exception as e:
                st.error(f"❌ Error: {e}")
        
        st.rerun()
    
    # Example prompts
    if len(st.session_state.chat_history) == 0:
        st.markdown("### 💡 Try these examples:")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🤖 What is AI?"):
                st.session_state.user_input = "What is artificial intelligence?"
                st.rerun()
        
        with col2:
            if st.button("🌍 Tell me about science"):
                st.session_state.user_input = "Tell me about science"
                st.rerun()
        
        with col3:
            if st.button("📚 Explain machine learning"):
                st.session_state.user_input = "Explain machine learning"
                st.rerun()

else:
    st.error("❌ Model not available. Check HF_REPO environment variable.")
    st.info("💡 Make sure to set HF_REPO in Streamlit secrets.")

# Footer
st.divider()
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("🔗 [GitHub](https://github.com/dineshCodeWorld/DINESH-AI)")
with col2:
    st.markdown("🤗 [Hugging Face](https://huggingface.co/alien-x/dinesh-ai)")
with col3:
    st.markdown("📧 dineshganji372@gmail.com")

st.markdown("<div style='text-align: center; color: #999; margin-top: 2rem;'>Dinesh AI © 2026 • Trained from scratch with ❤️</div>", unsafe_allow_html=True)
