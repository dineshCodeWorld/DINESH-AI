import streamlit as st
import torch
import os
import json
from huggingface_hub import hf_hub_download, HfApi, list_repo_commits
from tokenizers import Tokenizer
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
from src.core.custom_model import CustomGPT
from src.config_loader import APP_CONFIG, MODEL_CONFIG

# Load config
page_config = {
    "page_title": APP_CONFIG.get("page_title", "Dinesh AI"),
    "page_icon": APP_CONFIG.get("page_icon", "✨"),
    "layout": APP_CONFIG.get("layout", "wide"),
    "initial_sidebar_state": "expanded"
}
st.set_page_config(**page_config)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = None
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = True

# Get theme colors from config
dark = APP_CONFIG.get("dark_theme", {})
light = APP_CONFIG.get("light_theme", {})

if st.session_state.dark_mode:
    bg, text, accent = dark.get("background", "#0e1117"), dark.get("text", "#e0e0e0"), dark.get("accent", "#667eea")
    user_bg, bot_bg = dark.get("user_msg_bg", "#1e3a5f"), dark.get("bot_msg_bg", "#1e1e1e")
    input_bg, border = dark.get("input_bg", "#1e1e1e"), dark.get("border", "#333")
else:
    bg, text, accent = light.get("background", "#ffffff"), light.get("text", "#1a1a1a"), light.get("accent", "#667eea")
    user_bg, bot_bg = light.get("user_msg_bg", "#e3f2fd"), light.get("bot_msg_bg", "#f5f5f5")
    input_bg, border = light.get("input_bg", "#ffffff"), light.get("border", "#ddd")

# Dynamic theme CSS
theme = f"""
<style>
    #MainMenu, footer, header {{visibility: hidden;}}
    .stApp {{background: {bg}; color: {text};}}
    .block-container {{padding: 2rem 1rem;}}
    
    .msg {{
        padding: 16px;
        border-radius: 12px;
        margin: 12px 0;
        border-left: 4px solid {accent};
    }}
    .user {{background: {user_bg}; color: white;}}
    .bot {{background: {bot_bg}; color: {text};}}
    
    .stTextInput input {{
        background: {input_bg};
        border: 1px solid {border};
        border-radius: 12px;
        color: {text};
        padding: 12px 16px;
    }}
    
    .stButton button {{
        background: linear-gradient(90deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 24px;
    }}
</style>
"""

st.markdown(theme, unsafe_allow_html=True)

@st.cache_data(ttl=APP_CONFIG.get("model_cache_ttl", 60))
def get_models():
    try:
        repo_id = os.environ.get('HF_REPO')
        if not repo_id:
            return {"v0.0": {"file": "dinesh_ai_model.pth", "date": "N/A", "commit": "N/A", "message": "N/A", "size": "N/A", "filename": "N/A"}}
        
        api = HfApi()
        commits = list(list_repo_commits(repo_id=repo_id))
        commits.reverse()
        
        models = {}
        for i, commit in enumerate(commits[:50]):
            try:
                files = api.list_repo_tree(repo_id=repo_id, revision=commit.commit_id)
                model_file = next((f for f in files if f.path == "dinesh_ai_model.pth"), None)
                size_mb = model_file.size / (1024 * 1024) if model_file and hasattr(model_file, 'size') else 0
            except:
                size_mb = 0
            
            version = f"v0.{i}"
            if i == len(commits[:50]) - 1:
                version += " (Latest)"
            models[version] = {
                "file": f"dinesh_ai_model.pth?revision={commit.commit_id}",
                "date": commit.created_at.strftime("%Y-%m-%d %H:%M UTC"),
                "commit": commit.commit_id[:8],
                "message": commit.title,
                "size": f"{size_mb:.1f} MB" if size_mb > 0 else "N/A",
                "filename": "dinesh_ai_model.pth"
            }
        
        models = dict(reversed(list(models.items())))
        return models
    except Exception as e:
        st.sidebar.error(f"Error: {e}")
        return {"v0.0": {"file": "dinesh_ai_model.pth", "date": "N/A", "commit": "N/A", "message": "N/A", "size": "N/A", "filename": "N/A"}}

@st.cache_resource
def load_model(model_info: dict):
    try:
        repo_id = os.environ.get('HF_REPO')
        if not repo_id:
            return None, None, None, None
        
        model_file = model_info["file"]
        revision = None
        if "?revision=" in model_file:
            model_file, revision = model_file.split("?revision=")
        
        model_path = hf_hub_download(repo_id=repo_id, filename=model_file, cache_dir="models", revision=revision)
        tokenizer_path = hf_hub_download(repo_id=repo_id, filename="tokenizer.json", cache_dir="models", revision=revision)
        config_path = hf_hub_download(repo_id=repo_id, filename="model_config.json", cache_dir="models", revision=revision)
        
        with open(config_path) as f:
            config = json.load(f)
        
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        actual_vocab_size = state_dict['token_embedding.weight'].shape[0]
        actual_max_seq_len = state_dict['positional_embedding.weight'].shape[0]
        
        tokenizer = Tokenizer.from_file(tokenizer_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        model = CustomGPT(
            vocab_size=actual_vocab_size,
            d_model=config["d_model"],
            num_layers=config["num_layers"],
            num_heads=config.get("num_heads", config["d_model"] // 64),
            d_ff=config.get("d_ff", config["d_model"] * 4),
            max_seq_len=actual_max_seq_len,
            device=str(device)
        )
        
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        
        config["vocab_size"] = actual_vocab_size
        config["max_seq_len"] = actual_max_seq_len
        
        return model, tokenizer, device, config
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None, None

# Sidebar
with st.sidebar:
    st.title("⚙️ Settings")
    
    if st.button("🌓 Toggle Theme", use_container_width=True):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()
    
    st.divider()
    
    st.subheader("🤖 Model Version")
    models = get_models()
    versions = list(models.keys())
    
    if not st.session_state.selected_model or st.session_state.selected_model not in versions:
        st.session_state.selected_model = versions[-1] if versions else "v0.0"
    
    selected = st.selectbox(
        "Select Version", 
        versions,
        index=versions.index(st.session_state.selected_model) if st.session_state.selected_model in versions else len(versions)-1
    )
    
    if selected != st.session_state.selected_model:
        st.session_state.selected_model = selected
        st.cache_resource.clear()
        st.rerun()
    
    if selected in models and isinstance(models[selected], dict):
        info = models[selected]
        st.caption(f"📅 {info.get('date', 'N/A')}")
        st.caption(f"📦 Size: {info.get('size', 'N/A')}")
        st.caption(f"🔖 Commit: {info.get('commit', 'N/A')}")
        with st.expander("📝 Details"):
            st.text(f"Filename: {info.get('filename', 'N/A')}")
            st.text(f"Message: {info.get('message', 'N/A')}")
            st.text(f"Repository: {os.environ.get('HF_REPO', 'N/A')}")
    
    st.divider()
    
    st.subheader("🎛️ Parameters")
    temperature = st.slider("Temperature", 0.1, 2.0, 
                           float(APP_CONFIG.get("default_temperature", 0.8)), 0.1, 
                           help="Higher = more creative/human-like")
    top_k = st.slider("Top-K", 10, 100, 
                     APP_CONFIG.get("default_top_k", 50), 10,
                     help="Number of top tokens to consider")
    max_length = st.slider("Max Length", 20, 256, 
                          APP_CONFIG.get("default_max_length", 150), 10,
                          help="Maximum tokens to generate")
    
    st.divider()
    
    model, tokenizer, device, config = load_model(models[st.session_state.selected_model])
    
    if model and config:
        st.subheader("📊 Model Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Parameters", f"{config.get('parameter_count', 0):,}")
            st.metric("Layers", config.get('num_layers', 0))
            st.metric("Dimension", config.get('d_model', 0))
        with col2:
            st.metric("Vocab Size", f"{config.get('vocab_size', 0):,}")
            st.metric("Max Length", config.get('max_seq_len', 0))
            st.metric("Device", str(device).upper())
        
        st.info(f"🤗 **Repo:** {os.environ.get('HF_REPO', 'N/A')}")
    
    st.divider()
    
    st.subheader("💬 Session")
    st.metric("Messages", len(st.session_state.messages))
    
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Main area
st.title("✨ Dinesh AI")
st.caption("Custom GPT trained from scratch • Continuously learning")

st.divider()

# Display messages
if not st.session_state.messages:
    st.markdown("### 👋 Hello! Ask me anything")
    
    examples = APP_CONFIG.get("example_prompts", [
        {"icon": "🤖", "title": "What is AI?", "prompt": "What is artificial intelligence?"},
        {"icon": "🌍", "title": "Science", "prompt": "Tell me about quantum physics"},
        {"icon": "💻", "title": "Technology", "prompt": "How does blockchain work?"}
    ])
    
    cols = st.columns(len(examples))
    for col, ex in zip(cols, examples):
        with col:
            if st.button(f"{ex['icon']} **{ex['title']}**", use_container_width=True, key=f"ex_{ex['title']}"):
                st.session_state.messages.append({"role": "user", "content": ex['prompt']})
                st.rerun()
else:
    for msg in st.session_state.messages:
        if msg['role'] == 'user':
            st.markdown(f'<div class="msg user">👤 **You:** {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            version = msg.get('version', st.session_state.selected_model)
            st.markdown(f'<div class="msg bot">✨ **Dinesh AI ({version}):** {msg["content"]}</div>', unsafe_allow_html=True)

# Input
st.divider()
col1, col2 = st.columns([6, 1])

with col1:
    user_input = st.text_input("Type your message", key="input", label_visibility="collapsed", 
                               placeholder="Ask me anything...")

with col2:
    send = st.button("Send", use_container_width=True, type="primary")

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
                    max_length=min(max_length, model.max_seq_len),
                    temperature=temperature, 
                    top_p=float(MODEL_CONFIG.get("top_p", 0.92)), 
                    top_k=top_k, 
                    eos_token_id=2
                )
            
            text = tokenizer.decode(output_ids[0].cpu().tolist()).replace('Ġ', ' ').strip()
            st.session_state.messages.append({"role": "assistant", "content": text, "version": st.session_state.selected_model})
        except Exception as e:
            st.session_state.messages.append({"role": "assistant", "content": f"Error: {str(e)}", "version": st.session_state.selected_model})
    st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #999; font-size: 0.9rem;'>
    Dinesh AI © 2026 • <a href='https://github.com/dineshCodeWorld/DINESH-AI' style='color: #667eea;'>GitHub</a> • 
    <a href='https://huggingface.co/alien-x/dinesh-ai' style='color: #667eea;'>Hugging Face</a>
</div>
""", unsafe_allow_html=True)
