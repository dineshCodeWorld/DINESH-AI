import logging
import time
import schedule
import threading
from datetime import datetime
from pathlib import Path
import json
from ..data.data_collector import DataCollector
from ..data.data_preprocessor import DataPreprocessor
from ..core.model_trainer import ModelTrainer
from ..deployment.model_deployer import ModelDeployer
from ..config_loader import DATA_DIR, MODELS_DIR, LOGGING_CONFIG

logging.basicConfig(**LOGGING_CONFIG)
logger = logging.getLogger(__name__)

class ContinuousTrainer:
    """Continuously collect data AND train in parallel 24/7"""
    
    def __init__(self):
        self.collector = DataCollector()
        self.preprocessor = DataPreprocessor()
        self.trainer = ModelTrainer()
        self.deployer = ModelDeployer()
        self.data_dir = DATA_DIR
        self.models_dir = MODELS_DIR
        self.running = False
        self.current_model_path = None
        
    def collect_data_continuously(self):
        """Collect data every 6 hours (runs in parallel with training)"""
        logger.info("🔄 Data collection cycle started")
        
        try:
            # Collect substantial amounts for better training
            # You can adjust these based on your needs
            limits = {
                "wikipedia": 100,   # 100 articles per collection
                "arxiv": 50,        # 50 papers per collection
                "gutenberg": 20     # 20 books per collection
            }
            # Total: 170 items per collection
            # Per day (4 collections): 680 items
            # Per week: ~4,760 items
            
            collected_file = self.collector.collect_all(limits)
            logger.info(f"✅ Data collected: {collected_file}")
            return True
        except Exception as e:
            logger.error(f"❌ Data collection failed: {str(e)}")
            return False
    
    def train_on_new_data(self):
        """Fine-tune model on accumulated data (runs every 24 hours)"""
        logger.info("🧠 Training cycle started")
        
        try:
            # Check if we have new data
            new_data_files = list(self.data_dir.glob("collected_data_*.json"))
            if not new_data_files:
                logger.info("⏭️ No new data to train on")
                return False
            
            # Combine all new data
            all_new_data = []
            for file in new_data_files:
                data = self.preprocessor.load_collected_data(str(file))
                all_new_data.extend(data)
            
            if len(all_new_data) < 10:
                logger.info(f"⏭️ Not enough data ({len(all_new_data)} items), waiting for more")
                return False
            
            logger.info(f"📊 Training on {len(all_new_data)} new items")
            
            # Preprocess
            processed_data = self.preprocessor.preprocess_data(all_new_data)
            
            # Save to training file
            training_file = self.data_dir / "incremental_training_data.txt"
            with open(training_file, 'w', encoding='utf-8') as f:
                for item in processed_data:
                    f.write(item["text"] + "\n\n")
            
            # SAFETY: Find latest model (even if current_model_path is wrong)
            latest_model = self._find_latest_model()
            
            # FINE-TUNE existing model (not train from scratch)
            if latest_model and Path(latest_model).exists():
                logger.info(f"🔄 Fine-tuning existing model: {latest_model}")
                try:
                    # Load existing model
                    self.trainer.load_trained_model(latest_model)
                    # Fine-tune on new data (fewer epochs, lower learning rate)
                    model_path = self.trainer.fine_tune(
                        str(training_file),
                        epochs=1,  # Just 1 epoch for fine-tuning
                        learning_rate=1e-5  # Lower learning rate
                    )
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load model: {e}")
                    logger.info("🆕 Training new model from scratch instead")
                    model_path = self.trainer.train(str(training_file))
            else:
                logger.info("🆕 Training new model from scratch (first time or no previous model found)")
                # First time: train from scratch
                model_path = self.trainer.train(str(training_file))
            
            self.current_model_path = model_path
            
            # SAFETY: Create backup of this model
            self._create_backup(model_path)
            
            # Clean up processed data files
            for file in new_data_files:
                file.unlink()
            
            logger.info(f"✅ Training completed: {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Training failed: {str(e)}")
            return False
    
    def _find_latest_model(self):
        """Find the most recent model (safety fallback)"""
        try:
            # Check current_model_path first
            if self.current_model_path and Path(self.current_model_path).exists():
                return self.current_model_path
            
            # Search for any model in models directory
            model_dirs = list(self.models_dir.glob("model_*"))
            if model_dirs:
                # Get most recent by modification time
                latest = max(model_dirs, key=lambda p: p.stat().st_mtime)
                logger.info(f"📂 Found latest model: {latest.name}")
                return str(latest)
            
            return None
        except Exception as e:
            logger.warning(f"Error finding latest model: {e}")
            return None
    
    def _create_backup(self, model_path: str):
        """Create backup of model (safety feature)"""
        try:
            import shutil
            backup_dir = self.models_dir / "backups"
            backup_dir.mkdir(exist_ok=True)
            
            model_name = Path(model_path).name
            backup_path = backup_dir / model_name
            
            if not backup_path.exists():
                shutil.copytree(model_path, backup_path)
                logger.info(f"💾 Backup created: {backup_path}")
            
            # Keep only last 3 backups (save space)
            backups = sorted(backup_dir.glob("model_*"), key=lambda p: p.stat().st_mtime)
            if len(backups) > 3:
                for old_backup in backups[:-3]:
                    shutil.rmtree(old_backup)
                    logger.info(f"🗑️ Removed old backup: {old_backup.name}")
        except Exception as e:
            logger.warning(f"⚠️ Backup creation failed: {e}")
    
    def create_weekly_version(self):
        """Create official deployable version (runs every Sunday)"""
        logger.info("📦 Creating weekly deployable version")
        
        try:
            if not self.current_model_path:
                logger.warning("⚠️ No model to version")
                return False
            
            # Get version number
            versions_dir = self.models_dir / "versions"
            versions_dir.mkdir(exist_ok=True)
            
            existing_versions = list(versions_dir.glob("v*"))
            next_version_num = len(existing_versions) + 1
            
            # Create version name: v1-24-02-2026
            today = datetime.now().strftime('%d-%m-%Y')
            version_name = f"v{next_version_num}-{today}"
            version_path = versions_dir / version_name
            
            # Copy current model to version
            import shutil
            shutil.copytree(self.current_model_path, version_path, dirs_exist_ok=True)
            
            # Save metadata
            metadata = {
                "version_number": next_version_num,
                "version_name": version_name,
                "date_released": today,
                "created_at": datetime.now().isoformat(),
                "source_model": str(self.current_model_path)
            }
            
            with open(version_path / "version_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Update latest version pointer
            latest_config = {"latest_version": version_name, "updated_at": datetime.now().isoformat()}
            with open(self.models_dir / "latest_version.json", 'w') as f:
                json.dump(latest_config, f, indent=2)
            
            logger.info(f"✅ Version created: {version_name}")
            
            # Deploy to Hugging Face
            logger.info("🚀 Deploying to Hugging Face Hub")
            self.deployer.push_to_huggingface(str(version_path), version_name)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Version creation failed: {str(e)}")
            return False
    
    def schedule_tasks(self):
        """Schedule all tasks"""
        # Data collection every 6 hours
        schedule.every(6).hours.do(self.collect_data_continuously)
        
        # Training every 24 hours
        schedule.every(24).hours.do(self.train_on_new_data)
        
        # Weekly versioning every Sunday at 00:00
        schedule.every().sunday.at("00:00").do(self.create_weekly_version)
        
        logger.info("✅ Tasks scheduled:")
        logger.info("   📊 Data collection: Every 6 hours")
        logger.info("   🧠 Training: Every 24 hours")
        logger.info("   📦 Versioning: Every Sunday 00:00")
    
    def run(self):
        """Run continuous training 24/7"""
        self.running = True
        self.schedule_tasks()
        
        logger.info("\n" + "="*80)
        logger.info("🚀 CONTINUOUS TRAINER STARTED (24/7)")
        logger.info("="*80)
        logger.info("📊 Collecting data every 6 hours")
        logger.info("🧠 Training model every 24 hours")
        logger.info("📦 Creating versions every Sunday")
        logger.info("="*80 + "\n")
        
        # Run initial collection and training
        logger.info("🎬 Running initial data collection...")
        self.collect_data_continuously()
        
        try:
            while self.running:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        except KeyboardInterrupt:
            logger.info("\n✅ Continuous trainer stopped")
            self.running = False

if __name__ == "__main__":
    trainer = ContinuousTrainer()
    trainer.run()
