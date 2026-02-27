import logging
import torch
import torch.nn as nn
import json
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors, trainers
from .custom_model import CustomGPT
from ..config_loader import MODELS_DIR, MODEL_CONFIG, TRAINING_CONFIG, LOGGING_CONFIG

logging.basicConfig(**LOGGING_CONFIG)
logger = logging.getLogger(__name__)

class TextDataset(Dataset):
    """Custom dataset for language modeling"""
    
    def __init__(self, file_path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Read and tokenize text
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Tokenize entire text
        encodings = self.tokenizer.encode(text)
        self.token_ids = encodings.ids
        
        logger.info(f"Tokenized dataset: {len(self.token_ids)} tokens")
    
    def __len__(self):
        return len(self.token_ids) - self.max_length
    
    def __getitem__(self, idx):
        input_ids = self.token_ids[idx:idx + self.max_length]
        target_ids = self.token_ids[idx + 1:idx + self.max_length + 1]
        
        # Pad if necessary
        if len(input_ids) < self.max_length:
            pad_len = self.max_length - len(input_ids)
            input_ids = input_ids + [0] * pad_len
            target_ids = target_ids + [0] * pad_len
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'target_ids': torch.tensor(target_ids, dtype=torch.long)
        }


class SimpleTokenizer:
    """Simple BPE tokenizer for custom model"""
    
    def __init__(self, vocab_size: int = 50000):
        self.vocab_size = vocab_size
        self.tokenizer = None
    
    def train(self, data_file: str):
        """Train tokenizer on data"""
        try:
            logger.info(f"Training tokenizer with vocabulary size {self.vocab_size}")
            
            # Create BPE tokenizer
            self.tokenizer = Tokenizer(models.BPE())
            
            # Tokenizers' normalizer and pre-tokenizer
            self.tokenizer.normalizer = normalizers.Sequence([
                normalizers.NFC(),
                normalizers.Lowercase()
            ])
            self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
            
            # Train on data file
            trainer = trainers.BpeTrainer(
                vocab_size=self.vocab_size,
                min_frequency=1,  # Include all tokens (was 2)
                show_progress=True,
                special_tokens=["<pad>", "<unk>", "<eos>", "<bos>"]
            )
            
            self.tokenizer.train([data_file], trainer=trainer)
            self.tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
            
            logger.info("Tokenizer training completed")
            
        except Exception as e:
            logger.error(f"Error training tokenizer: {str(e)}")
            raise
    
    def encode(self, text: str):
        """Encode text to token IDs"""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained. Call train() first.")
        return self.tokenizer.encode(text)
    
    def decode(self, token_ids: list) -> str:
        """Decode token IDs to text"""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained. Call train() first.")
        return self.tokenizer.decode(token_ids)
    
    def save(self, path: str):
        """Save tokenizer"""
        self.tokenizer.save(path)
    
    def load(self, path: str):
        """Load tokenizer"""
        self.tokenizer = Tokenizer.from_file(path)
    
    @property
    def vocab_size(self):
        if self.tokenizer:
            return self.tokenizer.get_vocab_size()
        return self._vocab_size
    
    @vocab_size.setter
    def vocab_size(self, value):
        self._vocab_size = value


class ModelTrainer:
    def __init__(self):
        self.models_dir = MODELS_DIR
        self.models_dir.mkdir(exist_ok=True)
        self.device = torch.device(TRAINING_CONFIG["device"])
        self.tokenizer = None
        self.model = None
    
    def create_tokenizer_and_model(self, data_file: str):
        """Train tokenizer and create model from scratch"""
        try:
            # Train tokenizer
            logger.info("Creating and training tokenizer...")
            self.tokenizer = SimpleTokenizer(vocab_size=MODEL_CONFIG.get("vocab_size", 50000))
            self.tokenizer.train(data_file)
            
            # Create model from scratch
            logger.info("Creating custom GPT model from scratch...")
            self.model = CustomGPT(
                vocab_size=self.tokenizer.vocab_size,
                d_model=MODEL_CONFIG.get("d_model", 768),
                num_layers=MODEL_CONFIG.get("num_layers", 12),
                num_heads=MODEL_CONFIG.get("num_heads", 12),
                d_ff=MODEL_CONFIG.get("d_ff", 3072),
                max_seq_len=MODEL_CONFIG.get("max_length", 512),
                dropout=MODEL_CONFIG.get("dropout", 0.1),
                device=str(self.device)
            )
            self.model.to(self.device)
            
            param_count = self.model.count_parameters()
            logger.info(f"Model created with {param_count:,} parameters")
            logger.info(f"Tokenizer vocab size: {self.tokenizer.vocab_size}")
            
        except Exception as e:
            logger.error(f"Error creating tokenizer and model: {str(e)}")
            raise
    
    def train(self, data_file: str, output_dir: str = None):
        """Train the custom model from scratch"""
        if output_dir is None:
            output_dir = self.models_dir / f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(exist_ok=True)
        
        try:
            # Create tokenizer and model
            self.create_tokenizer_and_model(data_file)
            
            # Prepare dataset
            logger.info("Preparing dataset...")
            dataset = TextDataset(data_file, self.tokenizer, MODEL_CONFIG.get("max_length", 512))
            dataloader = DataLoader(
                dataset,
                batch_size=MODEL_CONFIG.get("batch_size", 16),
                shuffle=True,
                num_workers=0
            )
            
            # Training setup
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=MODEL_CONFIG.get("learning_rate", 2e-5),
                weight_decay=TRAINING_CONFIG.get("weight_decay", 0.01)
            )
            
            num_epochs = TRAINING_CONFIG.get("epochs", 3)
            total_steps = len(dataloader) * num_epochs
            
            # Learning rate scheduler
            from torch.optim.lr_scheduler import CosineAnnealingLR
            scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)
            
            logger.info(f"Starting training for {num_epochs} epochs...")
            logger.info(f"Total steps: {total_steps}")
            logger.info(f"Estimated time: {total_steps * 0.5 / 60:.1f} minutes")
            
            # Training loop
            self.model.train()
            global_step = 0
            training_start = datetime.now()
            
            for epoch in range(num_epochs):
                epoch_loss = 0.0
                epoch_start = datetime.now()
                progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
                
                for batch_idx, batch in enumerate(progress_bar):
                    input_ids = batch['input_ids'].to(self.device)
                    target_ids = batch['target_ids'].to(self.device)
                    
                    # Forward pass
                    logits = self.model(input_ids)
                    
                    # Compute loss
                    loss = nn.functional.cross_entropy(
                        logits.view(-1, self.model.vocab_size),
                        target_ids.view(-1),
                        ignore_index=0
                    )
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    scheduler.step()
                    
                    epoch_loss += loss.item()
                    global_step += 1
                    
                    # Calculate progress
                    overall_progress = (global_step / total_steps) * 100
                    elapsed = (datetime.now() - training_start).total_seconds()
                    estimated_total = (elapsed / global_step) * total_steps if global_step > 0 else 0
                    remaining = estimated_total - elapsed
                    
                    # Update progress bar
                    progress_bar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'progress': f'{overall_progress:.1f}%',
                        'remaining': f'{remaining/60:.1f}min'
                    })
                    
                    # Save checkpoint
                    if (batch_idx + 1) % TRAINING_CONFIG.get("save_interval", 500) == 0:
                        self._save_checkpoint(output_dir, epoch, global_step)
                
                avg_epoch_loss = epoch_loss / len(dataloader)
                epoch_time = (datetime.now() - epoch_start).total_seconds()
                logger.info(f"Epoch {epoch + 1} completed. Loss: {avg_epoch_loss:.4f} | Time: {epoch_time:.1f}s ({epoch_time/60:.1f}min)")
            
            # Save final model
            self._save_model(output_dir)
            
            logger.info(f"Training completed. Model saved to {output_dir}")
            return str(output_dir)
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise
    
    def _save_checkpoint(self, output_dir: Path, epoch: int, step: int):
        """Save training checkpoint"""
        try:
            checkpoint_dir = output_dir / f"checkpoint-{step}"
            checkpoint_dir.mkdir(exist_ok=True)
            
            torch.save(self.model.state_dict(), checkpoint_dir / "model.pt")
            self.tokenizer.save(str(checkpoint_dir / "tokenizer.json"))
            
            logger.info(f"Checkpoint saved to {checkpoint_dir}")
        except Exception as e:
            logger.warning(f"Error saving checkpoint: {str(e)}")
    
    def _save_model(self, output_dir: Path):
        """Save final model"""
        try:
            # Save model weights
            torch.save(self.model.state_dict(), output_dir / "model.pt")
            
            # Save tokenizer
            self.tokenizer.save(str(output_dir / "tokenizer.json"))
            
            # Save model config
            config = {
                "vocab_size": self.model.vocab_size,
                "d_model": self.model.d_model,
                "num_layers": self.model.num_layers,
                "max_seq_len": self.model.max_seq_len,
                "device": str(self.device),
                "parameter_count": self.model.count_parameters()
            }
            with open(output_dir / "model_config.json", 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"Model saved to {output_dir}")
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    def load_trained_model(self, model_path: str):
        """Load a trained custom model"""
        try:
            model_path = Path(model_path)
            logger.info(f"Loading trained model from {model_path}")
            
            # Load config
            with open(model_path / "model_config.json", 'r') as f:
                config = json.load(f)
            
            # Load tokenizer
            self.tokenizer = SimpleTokenizer()
            self.tokenizer.load(str(model_path / "tokenizer.json"))
            
            # Create model with saved config
            self.model = CustomGPT(
                vocab_size=config["vocab_size"],
                d_model=config["d_model"],
                num_layers=config["num_layers"],
                max_seq_len=config["max_seq_len"],
                device=str(self.device)
            )
            
            # Load weights
            self.model.load_state_dict(torch.load(model_path / "model.pt", map_location=self.device))
            self.model.to(self.device)
            
            logger.info("Trained model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading trained model: {str(e)}")
            raise
    
    def generate_text(self, prompt: str, max_length: int = 100) -> str:
        """Generate text using the model"""
        if not self.model or not self.tokenizer:
            raise ValueError("Model not loaded. Call load_trained_model() first.")
        
        try:
            self.model.eval()
            
            # Encode prompt
            encoded = self.tokenizer.encode(prompt)
            input_ids = torch.tensor([encoded.ids], dtype=torch.long).to(self.device)
            
            # Generate
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids,
                    max_length=max_length,
                    temperature=0.7,
                    top_p=0.9,
                    top_k=50,
                    eos_token_id=3
                )
            
            # Decode
            generated_text = self.tokenizer.decode(output_ids[0].cpu().tolist())
            return generated_text
        except Exception as e:
            logger.error(f"Error generating text: {str(e)}")
            raise
    
    def fine_tune(self, data_file: str, output_dir: str = None, epochs: int = 1, learning_rate: float = 1e-5):
        """Fine-tune existing model on new data"""
        if not self.model or not self.tokenizer:
            raise ValueError("Model not loaded. Call load_trained_model() first.")
        
        if output_dir is None:
            output_dir = self.models_dir / f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(exist_ok=True)
        
        try:
            logger.info(f"Fine-tuning model on new data for {epochs} epoch(s)")
            
            # Prepare dataset
            dataset = TextDataset(data_file, self.tokenizer, MODEL_CONFIG.get("max_length", 512))
            dataloader = DataLoader(
                dataset,
                batch_size=MODEL_CONFIG.get("batch_size", 16),
                shuffle=True,
                num_workers=0
            )
            
            # Fine-tuning setup (lower learning rate)
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=learning_rate,  # Lower LR for fine-tuning
                weight_decay=TRAINING_CONFIG.get("weight_decay", 0.01)
            )
            
            # Fine-tuning loop
            self.model.train()
            
            for epoch in range(epochs):
                epoch_loss = 0.0
                progress_bar = tqdm(dataloader, desc=f"Fine-tuning Epoch {epoch + 1}/{epochs}")
                
                for batch in progress_bar:
                    input_ids = batch['input_ids'].to(self.device)
                    target_ids = batch['target_ids'].to(self.device)
                    
                    # Forward pass
                    logits = self.model(input_ids)
                    
                    # Compute loss
                    loss = nn.functional.cross_entropy(
                        logits.view(-1, self.model.vocab_size),
                        target_ids.view(-1),
                        ignore_index=0
                    )
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
                
                avg_loss = epoch_loss / len(dataloader)
                logger.info(f"Fine-tuning epoch {epoch + 1} completed. Average loss: {avg_loss:.4f}")
            
            # Save fine-tuned model
            self._save_model(output_dir)
            logger.info(f"Fine-tuning completed. Model saved to {output_dir}")
            
            return str(output_dir)
            
        except Exception as e:
            logger.error(f"Error during fine-tuning: {str(e)}")
            raise
    
    def get_model_info(self) -> dict:
        """Get model information"""
        if not self.model:
            return {}
        
        return {
            "model_type": "Custom GPT (from scratch)",
            "device": str(self.device),
            "parameters": self.model.count_parameters(),
            "vocab_size": self.model.vocab_size,
            "d_model": self.model.d_model,
            "num_layers": self.model.num_layers
        }

if __name__ == "__main__":
    trainer = ModelTrainer()
    # Example usage
    # trainer.train("path/to/data.txt")
