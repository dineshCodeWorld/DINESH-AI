import logging
import json
import sys
from pathlib import Path
from datetime import datetime
import yaml
import os
import os

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.data_collector import DataCollector
from src.data.data_preprocessor import DataPreprocessor
from src.core.model_trainer import ModelTrainer
from src.deployment.model_deployer import ModelDeployer

# Initialize W&B if available
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Load configuration from config.yaml
with open(project_root / 'config.yaml', 'r', encoding='utf-8') as f:
    yaml_config = yaml.safe_load(f)

DATA_DIR = project_root / "data"
MODELS_DIR = project_root / "models"
LOGGING_CONFIG = {
    "level": yaml_config.get('logging', {}).get('level', 'INFO'),
    "format": yaml_config.get('logging', {}).get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
}

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, project_root / "logs"]:
    directory.mkdir(exist_ok=True)

logging.basicConfig(**LOGGING_CONFIG)
logger = logging.getLogger(__name__)

class TrainingPipeline:
    def __init__(self):
        # Initialize W&B for cloud monitoring
        self.wandb_run = None
        if WANDB_AVAILABLE and os.environ.get('WANDB_API_KEY'):
            try:
                self.wandb_run = wandb.init(
                    project="dinesh-ai",
                    name=f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    tags=["github-actions"] if os.environ.get('GITHUB_ACTIONS') else ["local"],
                    config=yaml_config
                )
                logger.info(f"✓ W&B enabled: {self.wandb_run.url}")
            except Exception as e:
                logger.error(f"W&B init failed: {e}")
        elif WANDB_AVAILABLE:
            logger.warning("W&B available but WANDB_API_KEY not set")
        else:
            logger.warning("W&B not installed (pip install wandb)")
        
        # Initialize TensorBoard writer
        self.writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir='runs')
            logger.info("TensorBoard enabled: tensorboard --logdir=runs")
        except ImportError:
            logger.warning("TensorBoard not available")
        
        self.data_collector = DataCollector(writer=self.writer)
        self.preprocessor = DataPreprocessor()
        self.trainer = ModelTrainer()
        self.deployer = ModelDeployer()
        self.pipeline_log = []
    
    def run_full_pipeline(self, collect_limits: dict = None, deploy: bool = True, skip_collection: bool = False):
        """Run the complete training pipeline"""
        
        pipeline_start = datetime.now()
        logger.info("=" * 80)
        logger.info("STARTING FULL TRAINING PIPELINE")
        logger.info("=" * 80)
        
        try:
            # Step 1: Collect data (10% of total time) - SKIPPABLE
            if skip_collection:
                logger.info("\n[STEP 1/5] Skipping data collection (using existing data)...")
                logger.info("⏱️  Time saved: ~78 minutes | Progress: 10%")
            else:
                logger.info("\n[STEP 1/5] Collecting data from sources... (Progress: 0%)")
                if self.writer:
                    self.writer.add_text('Pipeline/Stage', 'Data Collection Started', 0)
                
                if collect_limits is None:
                    collect_limits = {
                        "wikipedia": 1000,
                        "arxiv": 500,
                        "gutenberg": 100
                    }
                
                collected_file = self.data_collector.collect_all(collect_limits)
                self.pipeline_log.append({
                    "step": "data_collection",
                    "status": "completed",
                    "file": collected_file,
                    "timestamp": datetime.now().isoformat()
                })
                elapsed = (datetime.now() - pipeline_start).total_seconds()
                logger.info(f"✓ Data collection completed: {collected_file}")
                logger.info(f"⏱️  Time elapsed: {elapsed:.1f}s | Progress: 10%")
                
                if self.wandb_run:
                    wandb.log({"pipeline/progress": 10, "pipeline/stage": "data_collection"})
            
            # Step 2: Preprocess data (5% of total time)
            if skip_collection:
                logger.info("\n[STEP 2/5] Using existing preprocessed data... (Progress: 10%)")
                # Check if existing data file exists
                import json
                existing_data_file = DATA_DIR / "all_training_data.json"
                if not existing_data_file.exists():
                    logger.error("❌ No existing training data found! Cannot skip collection.")
                    raise FileNotFoundError("all_training_data.json not found. Run with collection first.")
                
                with open(existing_data_file, 'r') as f:
                    processed_data = json.load(f)
                processed_file = str(existing_data_file)
            else:
                logger.info("\n[STEP 2/5] Preprocessing collected data... (Progress: 10%)")
                if self.writer:
                    self.writer.add_text('Pipeline/Stage', 'Data Preprocessing', 1)
                
                collected_data = self.preprocessor.load_collected_data(collected_file)
                processed_data = self.preprocessor.preprocess_data(collected_data)
                
                existing_data_file = DATA_DIR / "all_training_data.json"
                if existing_data_file.exists():
                    processed_data = self.preprocessor.merge_with_existing(
                        processed_data, 
                        str(existing_data_file)
                    )
                
                processed_file = self.preprocessor.save_processed_data(
                    processed_data,
                    "all_training_data.json"
                )
            
            self.pipeline_log.append({
                "step": "data_preprocessing",
                "status": "completed",
                "file": processed_file,
                "samples": len(processed_data),
                "timestamp": datetime.now().isoformat()
            })
            elapsed = (datetime.now() - pipeline_start).total_seconds()
            logger.info(f"✓ Data preprocessing completed: {len(processed_data)} samples")
            logger.info(f"⏱️  Time elapsed: {elapsed:.1f}s | Progress: 15%")
            
            if self.wandb_run:
                wandb.log({"pipeline/progress": 15, "data/samples": len(processed_data)})
            
            # Step 3: Convert to text format (5% of total time)
            logger.info("\n[STEP 3/5] Preparing training data... (Progress: 15%)")
            if self.writer:
                self.writer.add_text('Pipeline/Stage', 'Training Data Preparation', 2)
                self.writer.add_scalar('Data/TotalSamples', len(processed_data), 0)
            
            text_file = DATA_DIR / "training_data.txt"
            with open(text_file, 'w', encoding='utf-8') as f:
                for item in processed_data:
                    f.write(item["text"] + "\n\n")
            
            elapsed = (datetime.now() - pipeline_start).total_seconds()
            logger.info(f"✓ Training data prepared: {text_file}")
            logger.info(f"⏱️  Time elapsed: {elapsed:.1f}s | Progress: 20%")
            
            if self.wandb_run:
                wandb.log({"pipeline/progress": 20})
            
            # Step 4: Download latest model for fine-tuning (75% of total time)
            logger.info("\n[STEP 4/6] Downloading latest model from Hugging Face... (Progress: 20%)")
            
            # Try to download existing model
            import os
            from huggingface_hub import hf_hub_download, HfApi
            
            existing_model_dir = None
            hf_repo = os.environ.get('HF_REPO')
            
            if hf_repo:
                try:
                    logger.info(f"Attempting to download latest model from {hf_repo}...")
                    
                    # First try to get the latest version from versions folder
                    api = HfApi()
                    files = api.list_repo_files(repo_id=hf_repo)
                    version_files = [f for f in files if f.startswith('versions/') and f.endswith('.pth')]
                    
                    model_to_download = "dinesh_ai_model.pth"  # Default
                    
                    if version_files:
                        # Sort by version number and timestamp to get latest
                        import re
                        def extract_version(filename):
                            match = re.search(r'_v(\d+)_(\d{4}-\d{2}-\d{2}_\d{6})', filename)
                            if match:
                                return (int(match.group(1)), match.group(2))
                            return (0, '')
                        
                        latest_version_file = max(version_files, key=extract_version)
                        logger.info(f"Found latest version: {latest_version_file}")
                        model_to_download = latest_version_file
                    
                    model_path = hf_hub_download(repo_id=hf_repo, filename=model_to_download, cache_dir="models")
                    tokenizer_path = hf_hub_download(repo_id=hf_repo, filename="tokenizer.json", cache_dir="models")
                    config_path = hf_hub_download(repo_id=hf_repo, filename="model_config.json", cache_dir="models")
                    
                    # Check vocab size compatibility
                    import json
                    with open(config_path) as f:
                        old_config = json.load(f)
                    
                    old_vocab = old_config.get('vocab_size', 0)
                    new_vocab = yaml_config.get('model', {}).get('vocab_size', 8000)
                    
                    if old_vocab != new_vocab:
                        logger.warning(f"Vocab size mismatch: old={old_vocab}, new={new_vocab}")
                        logger.warning("Cannot fine-tune. Will train from scratch.")
                        existing_model_dir = None
                    else:
                        # Create temp directory with downloaded files
                        import shutil
                        existing_model_dir = MODELS_DIR / "downloaded_model"
                        existing_model_dir.mkdir(exist_ok=True)
                        shutil.copy(model_path, existing_model_dir / "model.pt")
                        shutil.copy(tokenizer_path, existing_model_dir / "tokenizer.json")
                        shutil.copy(config_path, existing_model_dir / "model_config.json")
                        logger.info(f"✓ Latest model downloaded: {model_to_download}")
                except Exception as e:
                    logger.warning(f"Could not download existing model: {e}")
                    logger.info("Will train from scratch instead")
            
            # Step 5: Train/Fine-tune model (75% of total time)
            logger.info("\n[STEP 5/6] Training model... (Progress: 30%)")
            logger.info("⚠️  This will take the longest time (65% of total)")
            
            if existing_model_dir and existing_model_dir.exists():
                logger.info("🔄 Fine-tuning existing model with new data...")
                self.trainer.load_trained_model(str(existing_model_dir))
                model_output_dir = self.trainer.fine_tune(
                    str(text_file),
                    epochs=yaml_config.get('training', {}).get('epochs', 3),
                    learning_rate=yaml_config.get('training', {}).get('learning_rate', 0.0001) / 10
                )
            else:
                logger.info("🆕 Training new model from scratch...")
                model_output_dir = self.trainer.train(str(text_file))
            
            self.pipeline_log.append({
                "step": "model_training",
                "status": "completed",
                "model_path": model_output_dir,
                "timestamp": datetime.now().isoformat()
            })
            elapsed = (datetime.now() - pipeline_start).total_seconds()
            logger.info(f"✓ Model training completed: {model_output_dir}")
            logger.info(f"⏱️  Time elapsed: {elapsed:.1f}s ({elapsed/60:.1f} min) | Progress: 95%")
            
            if self.wandb_run:
                wandb.log({"pipeline/progress": 95})
            
            # Step 6: Prepare for deployment (5% of total time)
            logger.info("\n[STEP 6/6] Preparing for deployment... (Progress: 95%)")
            version = datetime.now().strftime('%Y%m%d_%H%M%S')
            deployment_info = self.deployer.prepare_for_deployment(model_output_dir, version)
            
            self.pipeline_log.append({
                "step": "deployment_preparation",
                "status": "completed",
                "deployment_info": deployment_info,
                "timestamp": datetime.now().isoformat()
            })
            logger.info(f"✓ Deployment preparation completed")
            
            if deploy:
                deployment_result = self.deployer.push_to_huggingface(model_output_dir, version)
                if deployment_result and deployment_result.get("status") == "success":
                    logger.info(f"✓ Model deployed successfully: {deployment_result.get('url')}")
                else:
                    logger.warning("Model deployment skipped or failed (HF_TOKEN not set)")
                
                self.pipeline_log.append({
                    "step": "model_deployment",
                    "status": deployment_result.get("status", "skipped") if deployment_result else "skipped",
                    "deployment_info": deployment_result,
                    "timestamp": datetime.now().isoformat()
                })
            
            # Save pipeline log
            pipeline_end = datetime.now()
            total_time = (pipeline_end - pipeline_start).total_seconds()
            
            # Convert WindowsPath to string for JSON serialization
            for step in self.pipeline_log:
                if 'deployment_info' in step and step['deployment_info']:
                    if 'files' in step['deployment_info']:
                        step['deployment_info']['files'] = [str(f) for f in step['deployment_info']['files']]
                    if 'model_path' in step['deployment_info']:
                        step['deployment_info']['model_path'] = str(step['deployment_info']['model_path'])
            
            pipeline_summary = {
                "pipeline_id": datetime.now().strftime('%Y%m%d_%H%M%S'),
                "start_time": pipeline_start.isoformat(),
                "end_time": pipeline_end.isoformat(),
                "total_time_seconds": total_time,
                "status": "completed",
                "steps": self.pipeline_log,
                "model_version": version,
                "data_samples": len(processed_data)
            }
            
            log_file = MODELS_DIR / f"pipeline_log_{version}.json"
            with open(log_file, 'w') as f:
                json.dump(pipeline_summary, f, indent=2)
            
            logger.info("\n" + "=" * 80)
            logger.info("✅ PIPELINE COMPLETED SUCCESSFULLY (100%)")
            logger.info("=" * 80)
            logger.info(f"⏱️  Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes / {total_time/3600:.2f} hours)")
            logger.info(f"📊 Model version: {version}")
            logger.info(f"📊 Data samples: {len(processed_data)}")
            logger.info(f"📄 Pipeline log: {log_file}")
            
            if self.wandb_run:
                wandb.log({"pipeline/progress": 100, "pipeline/total_time": total_time})
                wandb.finish()
            
            if self.writer:
                self.writer.close()
            
            return pipeline_summary
            
        except Exception as e:
            logger.error(f"\n✗ Pipeline failed: {str(e)}")
            self.pipeline_log.append({
                "step": "pipeline_error",
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
            raise

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Dinesh AI Model')
    parser.add_argument('--config', type=str, help='Config file to use (e.g., config.custom.yaml)')
    parser.add_argument('--skip-collection', action='store_true', help='Skip data collection, use existing data')
    args = parser.parse_args()
    
    pipeline = TrainingPipeline()
    
    # Check which config is being used
    from src.config_loader import CONFIG_FILE, DATA_SOURCES
    logger.info(f"Using configuration: {CONFIG_FILE}")
    
    if args.skip_collection:
        logger.info("⚠️  SKIP COLLECTION MODE: Using existing data")
    
    # Get collection limits from config
    collect_limits = {
        "wikipedia": DATA_SOURCES.get("wikipedia", {}).get("limit", 800),
        "arxiv": DATA_SOURCES.get("arxiv", {}).get("limit", 500),
        "gutenberg": DATA_SOURCES.get("gutenberg", {}).get("limit", 200),
        "reddit": DATA_SOURCES.get("reddit", {}).get("limit", 100),
        "hackernews": DATA_SOURCES.get("hackernews", {}).get("limit", 50),
        "news": DATA_SOURCES.get("news", {}).get("limit", 50)
    }
    logger.info(f"Collection limits: {collect_limits}")
    
    # Run pipeline with config limits
    pipeline.run_full_pipeline(
        collect_limits=collect_limits,
        deploy=False,
        skip_collection=args.skip_collection
    )

if __name__ == "__main__":
    main()
