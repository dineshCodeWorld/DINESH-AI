import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data.data_collector import DataCollector

print("Testing data collection from all sources...\n")

collector = DataCollector()

# Test each source with small limits
print("1. Testing Wikipedia (5 articles)...")
wiki = collector.collect_from_wikipedia_api(5)
print(f"   ✓ Collected {len(wiki)} articles\n")

print("2. Testing ArXiv (5 papers)...")
arxiv = collector.collect_from_arxiv(5)
print(f"   ✓ Collected {len(arxiv)} papers\n")

print("3. Testing Gutenberg (5 books)...")
gutenberg = collector.collect_from_gutenberg(5)
print(f"   ✓ Collected {len(gutenberg)} books\n")

print("4. Testing Reddit (5 posts)...")
reddit = collector.collect_from_reddit(5)
print(f"   ✓ Collected {len(reddit)} posts\n")

print("5. Testing HackerNews (5 stories)...")
hn = collector.collect_from_hackernews(5)
print(f"   ✓ Collected {len(hn)} stories\n")

print("6. Testing News (5 articles)...")
news = collector.collect_from_common_crawl(5)
print(f"   ✓ Collected {len(news)} articles\n")

print("=" * 50)
print(f"TOTAL: {len(wiki) + len(arxiv) + len(gutenberg) + len(reddit) + len(hn) + len(news)} items")
print("=" * 50)

if len(gutenberg) == 0:
    print("\n⚠️  Gutenberg returned 0 - API might be down")
if len(reddit) == 0:
    print("\n⚠️  Reddit returned 0 - might be rate limited")
if len(hn) < 3:
    print("\n⚠️  HackerNews returned few items")
if len(news) == 0:
    print("\n⚠️  News returned 0 - RSS feeds might be blocked")
