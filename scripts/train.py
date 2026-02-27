import logging
import json
import sys
from pathlib import Path
from datetime import datetime
import yaml

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.data_collector import DataCollector
from src.data.data_preprocessor import DataPreprocessor
from src.core.model_trainer import ModelTrainer
from src.deployment.model_deployer import ModelDeployer

# Load configuration from config.yaml
with open(project_root / 'config.yaml', 'r') as f:
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
        self.data_collector = DataCollector()
        self.preprocessor = DataPreprocessor()
        self.trainer = ModelTrainer()
        self.deployer = ModelDeployer()
        self.pipeline_log = []
    
    def run_full_pipeline(self, collect_limits: dict = None, deploy: bool = True):
        """Run the complete training pipeline"""
        
        pipeline_start = datetime.now()
        logger.info("=" * 80)
        logger.info("STARTING FULL TRAINING PIPELINE")
        logger.info("=" * 80)
        
        try:
            # Step 1: Collect data (10% of total time)
            logger.info("\n[STEP 1/5] Collecting data from sources... (Progress: 0%)")
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
            
            # Step 2: Preprocess data (5% of total time)
            logger.info("\n[STEP 2/5] Preprocessing collected data... (Progress: 10%)")
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
            
            # Step 3: Convert to text format (5% of total time)
            logger.info("\n[STEP 3/5] Preparing training data... (Progress: 15%)")
            text_file = DATA_DIR / "training_data.txt"
            with open(text_file, 'w', encoding='utf-8') as f:
                for item in processed_data:
                    f.write(item["text"] + "\n\n")
            
            elapsed = (datetime.now() - pipeline_start).total_seconds()
            logger.info(f"✓ Training data prepared: {text_file}")
            logger.info(f"⏱️  Time elapsed: {elapsed:.1f}s | Progress: 20%")
            
            # Step 4: Train model (75% of total time)
            logger.info("\n[STEP 4/5] Training model... (Progress: 20%)")
            logger.info("⚠️  This will take the longest time (75% of total)")
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
            
            # Step 5: Prepare for deployment (5% of total time)
            logger.info("\n[STEP 5/5] Preparing for deployment... (Progress: 95%)")
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
    args = parser.parse_args()
    
    pipeline = TrainingPipeline()
    
    # Check which config is being used
    from src.config_loader import CONFIG_FILE, DATA_SOURCES
    logger.info(f"Using configuration: {CONFIG_FILE}")
    
    # Get collection limits from config
    collect_limits = {
        "wikipedia": DATA_SOURCES.get("wikipedia", {}).get("limit", 1000),
        "arxiv": DATA_SOURCES.get("arxiv", {}).get("limit", 500),
        "gutenberg": DATA_SOURCES.get("gutenberg", {}).get("limit", 100)
    }
    logger.info(f"Collection limits: {collect_limits}")
    
    # Run pipeline with config limits
    pipeline.run_full_pipeline(
        collect_limits=collect_limits,
        deploy=False
    )

if __name__ == "__main__":
    main()
