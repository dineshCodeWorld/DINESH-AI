import logging
import time
import schedule
from datetime import datetime, timedelta
from pathlib import Path
from ..data.data_collector import DataCollector
from ..config_loader import DATA_DIR, LOGGING_CONFIG

logging.basicConfig(**LOGGING_CONFIG)
logger = logging.getLogger(__name__)

class ContinuousDataCollector:
    """Continuously collects data 24/7 at scheduled intervals"""
    
    def __init__(self, collection_interval_hours=6):
        """
        Initialize continuous collector
        
        Args:
            collection_interval_hours: Collect data every N hours (default: 6 hours)
        """
        self.collector = DataCollector()
        self.interval_hours = collection_interval_hours
        self.data_dir = DATA_DIR
        self.running = False
    
    def collect_batch(self):
        """Collect a batch of data"""
        try:
            logger.info(f"\n{'='*80}")
            logger.info(f"CONTINUOUS DATA COLLECTION - {datetime.now().isoformat()}")
            logger.info(f"{'='*80}")
            
            # Collect smaller batches more frequently
            limits = {
                "wikipedia": 10,
                "arxiv": 5,
                "gutenberg": 2
            }
            
            collected_file = self.collector.collect_all(limits)
            logger.info(f"✓ Batch collected: {collected_file}")
            
            # Append to continuous collection file
            self.append_to_continuous_file(collected_file)
            
            return True
        except Exception as e:
            logger.error(f"✗ Collection batch failed: {str(e)}")
            return False
    
    def append_to_continuous_file(self, new_file):
        """Append new data to continuous collection file"""
        import json
        
        continuous_file = self.data_dir / "continuous_collection.json"
        
        try:
            # Load new data
            with open(new_file, 'r', encoding='utf-8') as f:
                new_data = json.load(f)
            
            # Load existing continuous data
            existing_data = []
            if continuous_file.exists():
                with open(continuous_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
            
            # Merge
            merged_data = existing_data + new_data
            
            # Save
            with open(continuous_file, 'w', encoding='utf-8') as f:
                json.dump(merged_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✓ Data appended to continuous collection: {len(merged_data)} total items")
            
        except Exception as e:
            logger.error(f"Error appending to continuous file: {str(e)}")
    
    def schedule_collections(self):
        """Schedule data collection every N hours"""
        schedule.every(self.interval_hours).hours.do(self.collect_batch)
        logger.info(f"✓ Scheduled data collection every {self.interval_hours} hours")
    
    def run(self):
        """Run continuous collection (blocking)"""
        self.running = True
        self.schedule_collections()
        
        logger.info(f"\n{'='*80}")
        logger.info("CONTINUOUS DATA COLLECTOR STARTED")
        logger.info(f"Collection interval: Every {self.interval_hours} hours")
        logger.info(f"Start time: {datetime.now().isoformat()}")
        logger.info(f"{'='*80}\n")
        
        try:
            while self.running:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        except KeyboardInterrupt:
            logger.info("\n✓ Continuous collector stopped")
            self.running = False

if __name__ == "__main__":
    # Run continuous collection every 6 hours
    collector = ContinuousDataCollector(collection_interval_hours=6)
    collector.run()
