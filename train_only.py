import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.core.model_trainer import ModelTrainer

print("Training model from existing data (skipping collection)...\n")

# Path to existing training data
data_file = "data/training_data.txt"

if not Path(data_file).exists():
    print(f"❌ {data_file} not found!")
    print("Run full pipeline first to collect data.")
    exit(1)

print(f"✓ Found training data: {data_file}")
print("Starting training...\n")

trainer = ModelTrainer()
output_dir = trainer.train(data_file)

print(f"\n✓ Training complete!")
print(f"Model saved to: {output_dir}")
