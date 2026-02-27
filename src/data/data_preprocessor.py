import json
import logging
import re
from pathlib import Path
from typing import List, Dict
from datasets import Dataset
from ..config_loader import DATA_DIR, LOGGING_CONFIG

logging.basicConfig(**LOGGING_CONFIG)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self):
        self.data_dir = DATA_DIR
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\-\:\;]', '', text)
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        return text.strip()
    
    def load_collected_data(self, filepath: str) -> List[Dict]:
        """Load collected data from JSON file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Loaded {len(data)} items from {filepath}")
            return data
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return []
    
    def preprocess_data(self, data: List[Dict]) -> List[Dict]:
        """Preprocess collected data"""
        processed = []
        
        for item in data:
            try:
                cleaned_content = self.clean_text(item.get("content", ""))
                
                if len(cleaned_content) > 50:  # Filter out very short texts
                    processed.append({
                        "text": cleaned_content,
                        "source": item.get("source", "unknown"),
                        "title": item.get("title", ""),
                        "url": item.get("url", "")
                    })
            except Exception as e:
                logger.warning(f"Error processing item: {str(e)}")
                continue
        
        logger.info(f"Preprocessed {len(processed)} items from {len(data)} total")
        return processed
    
    def create_dataset(self, data: List[Dict]) -> Dataset:
        """Create HuggingFace dataset from preprocessed data"""
        texts = [item["text"] for item in data]
        dataset = Dataset.from_dict({"text": texts})
        logger.info(f"Created dataset with {len(dataset)} samples")
        return dataset
    
    def merge_with_existing(self, new_data: List[Dict], existing_file: str = None) -> List[Dict]:
        """Merge new data with existing training data"""
        merged = new_data.copy()
        
        if existing_file and Path(existing_file).exists():
            try:
                existing_data = self.load_collected_data(existing_file)
                merged.extend(existing_data)
                logger.info(f"Merged {len(new_data)} new items with {len(existing_data)} existing items")
            except Exception as e:
                logger.warning(f"Could not merge with existing data: {str(e)}")
        
        return merged
    
    def save_processed_data(self, data: List[Dict], filename: str = "processed_data.json") -> str:
        """Save processed data to file"""
        filepath = self.data_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Processed data saved to {filepath}")
        return str(filepath)

if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    # Example usage
    # data = preprocessor.load_collected_data("path/to/collected_data.json")
    # processed = preprocessor.preprocess_data(data)
    # preprocessor.save_processed_data(processed)
