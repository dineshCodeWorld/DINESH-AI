import os
from pathlib import Path
import sys

sys.stdout.reconfigure(encoding='utf-8')

data_dir = Path("data/raw")

if not data_dir.exists():
    print("❌ data/raw folder not found")
    exit(1)

total_size = 0
file_count = 0

print("📊 Training Data Analysis\n")

for file in data_dir.glob("*.txt"):
    size = file.stat().st_size
    total_size += size
    file_count += 1
    print(f"  {file.name}: {size:,} bytes ({size/1024:.1f} KB)")

print(f"\n📦 Total: {file_count} files, {total_size:,} bytes ({total_size/1024/1024:.2f} MB)")

if total_size < 1_000_000:
    print("\n⚠️  WARNING: Less than 1 MB of data - model will produce gibberish")
    print("   Need at least 10-50 MB for basic coherence")
elif total_size < 10_000_000:
    print("\n⚠️  Data is small - expect poor quality outputs")
    print("   Recommended: 50+ MB for decent results")
else:
    print("\n✅ Data size looks reasonable")
