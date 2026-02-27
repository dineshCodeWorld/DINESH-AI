import logging
import os
from pathlib import Path
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from ..config_loader import MODELS_DIR, DEPLOYMENT_CONFIG, LOGGING_CONFIG

logging.basicConfig(**LOGGING_CONFIG)
logger = logging.getLogger(__name__)

class ModelDeployer:
    def __init__(self):
        self.models_dir = MODELS_DIR
        self.hub_repo = DEPLOYMENT_CONFIG.get("repo_name", "dinesh-ai-model")
        self.push_to_hub = DEPLOYMENT_CONFIG.get("push_to_hub", False)
    
    def create_model_card(self, model_path: str, version: str) -> str:
        """Create a model card for Hugging Face Hub"""
        model_card = f"""---
language: en
license: mit
tags:
- text-generation
- fine-tuned
- knowledge-base
---

# Dinesh AI Model - Version {version}

This is a fine-tuned language model trained on diverse knowledge sources including Wikipedia, ArXiv, and Project Gutenberg.

## Model Details

- **Base Model**: distilbert-base-uncased
- **Training Date**: {datetime.now().strftime('%Y-%m-%d')}
- **Version**: {version}
- **Training Data**: Wikipedia, ArXiv papers, Project Gutenberg texts

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("dinesh-ai-model")
model = AutoModelForCausalLM.from_pretrained("dinesh-ai-model")

prompt = "What is artificial intelligence?"
inputs = tokenizer.encode(prompt, return_tensors="pt")
outputs = model.generate(inputs, max_length=100)
print(tokenizer.decode(outputs[0]))
```

## Training Process

This model is continuously trained weekly with new data from various knowledge sources.
Each version represents an improved iteration with more knowledge and better understanding.

## License

MIT License
"""
        
        card_path = Path(model_path) / "README.md"
        with open(card_path, 'w') as f:
            f.write(model_card)
        
        logger.info(f"Model card created at {card_path}")
        return str(card_path)
    
    def prepare_for_deployment(self, model_path: str, version: str = None) -> dict:
        """Prepare model for deployment"""
        if version is None:
            version = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        try:
            model_path = Path(model_path)
            
            # Create model card
            self.create_model_card(str(model_path), version)
            
            # Verify model files exist
            required_files = ["pytorch_model.bin", "config.json", "tokenizer.json"]
            missing_files = [f for f in required_files if not (model_path / f).exists()]
            
            if missing_files:
                logger.warning(f"Missing files: {missing_files}")
            
            deployment_info = {
                "model_path": str(model_path),
                "version": version,
                "timestamp": datetime.now().isoformat(),
                "ready_for_deployment": True,
                "files": list(model_path.glob("*"))
            }
            
            logger.info(f"Model prepared for deployment: {deployment_info}")
            return deployment_info
            
        except Exception as e:
            logger.error(f"Error preparing model for deployment: {str(e)}")
            raise
    
    def push_to_huggingface(self, model_path: str, version: str = None):
        """Push model to Hugging Face Hub"""
        if not self.push_to_hub:
            logger.info("Push to Hub is disabled in configuration")
            return None
        
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            logger.warning("HF_TOKEN not found in environment variables. Skipping Hub push.")
            return None
        
        try:
            if version is None:
                version = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            logger.info(f"Pushing model to Hugging Face Hub: {self.hub_repo}")
            
            # Load model and tokenizer
            model = AutoModelForCausalLM.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # Push to hub
            model.push_to_hub(
                repo_id=self.hub_repo,
                use_auth_token=hf_token,
                commit_message=f"Version {version}: Weekly training update"
            )
            
            tokenizer.push_to_hub(
                repo_id=self.hub_repo,
                use_auth_token=hf_token,
                commit_message=f"Version {version}: Weekly training update"
            )
            
            logger.info(f"Model successfully pushed to {self.hub_repo}")
            return {
                "status": "success",
                "repo": self.hub_repo,
                "version": version,
                "url": f"https://huggingface.co/{self.hub_repo}"
            }
            
        except Exception as e:
            logger.error(f"Error pushing to Hugging Face Hub: {str(e)}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    def create_deployment_package(self, model_path: str, version: str = None) -> str:
        """Create a deployment package"""
        if version is None:
            version = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        try:
            model_path = Path(model_path)
            package_dir = self.models_dir / f"deployment_{version}"
            package_dir.mkdir(exist_ok=True)
            
            # Copy model files
            for file in model_path.glob("*"):
                if file.is_file():
                    import shutil
                    shutil.copy2(file, package_dir / file.name)
            
            # Create deployment info
            deployment_info = {
                "version": version,
                "timestamp": datetime.now().isoformat(),
                "model_path": str(package_dir),
                "status": "ready"
            }
            
            logger.info(f"Deployment package created at {package_dir}")
            return str(package_dir)
            
        except Exception as e:
            logger.error(f"Error creating deployment package: {str(e)}")
            raise

if __name__ == "__main__":
    deployer = ModelDeployer()
    # Example usage
    # deployer.prepare_for_deployment("path/to/model")
    # deployer.push_to_huggingface("path/to/model")
