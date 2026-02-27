import os
import json
import logging
from pathlib import Path
from datetime import datetime
import requests
from bs4 import BeautifulSoup
from typing import List, Dict
import xml.etree.ElementTree as ET
import random
import hashlib
from ..config_loader import DATA_DIR, DATA_SOURCES, LOGGING_CONFIG

logging.basicConfig(**LOGGING_CONFIG)
logger = logging.getLogger(__name__)

class DataCollector:
    def __init__(self):
        self.data_dir = DATA_DIR
        self.data_dir.mkdir(exist_ok=True)
        self.collected_data = []
        self.seen_hashes_file = self.data_dir / "seen_content_hashes.json"
        self.seen_hashes = self._load_seen_hashes()
    
    def _load_seen_hashes(self) -> set:
        """Load previously seen content hashes to avoid duplicates"""
        if self.seen_hashes_file.exists():
            try:
                with open(self.seen_hashes_file, 'r') as f:
                    return set(json.load(f))
            except:
                return set()
        return set()
    
    def _save_seen_hashes(self):
        """Save seen content hashes"""
        with open(self.seen_hashes_file, 'w') as f:
            json.dump(list(self.seen_hashes), f)
    
    def _get_content_hash(self, content: str) -> str:
        """Generate hash of content for deduplication"""
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def _is_duplicate(self, content: str) -> bool:
        """Check if content was already collected"""
        content_hash = self._get_content_hash(content)
        if content_hash in self.seen_hashes:
            return True
        self.seen_hashes.add(content_hash)
        return False
        
    def collect_from_wikipedia_api(self, limit: int = 100) -> List[Dict]:
        """Collect articles from Wikipedia API"""
        logger.info(f"Starting Wikipedia data collection (target: {limit})...")
        articles = []
        
        # Add User-Agent header to avoid 403 errors
        headers = {
            'User-Agent': 'DineshAI/1.0 (Educational Project; Python/requests)'
        }
        
        try:
            url = "https://en.wikipedia.org/w/api.php"
            
            # Keep collecting until we reach the limit
            while len(articles) < limit:
                remaining = limit - len(articles)
                batch_size = min(remaining, 20)  # API limit per request
                
                params = {
                    "action": "query",
                    "format": "json",
                    "list": "random",
                    "rnnamespace": "0",
                    "rnlimit": batch_size
                }
                
                response = requests.get(url, params=params, headers=headers, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    for page in data.get("query", {}).get("random", []):
                        if len(articles) >= limit:
                            break
                            
                        page_id = page["id"]
                        title = page["title"]
                        
                        # Get page content
                        content_params = {
                            "action": "query",
                            "format": "json",
                            "pageids": page_id,
                            "prop": "extracts",
                            "explaintext": True
                        }
                        content_response = requests.get(url, params=content_params, headers=headers, timeout=10)
                        if content_response.status_code == 200:
                            content_data = content_response.json()
                            pages = content_data.get("query", {}).get("pages", {})
                            if str(page_id) in pages:
                                extract = pages[str(page_id)].get("extract", "")
                                if extract and len(extract) > 100 and not self._is_duplicate(extract):
                                    articles.append({
                                        "source": "wikipedia",
                                        "title": title,
                                        "content": extract[:5000],
                                        "url": f"https://en.wikipedia.org/?curid={page_id}",
                                        "timestamp": datetime.now().isoformat()
                                    })
                                    if len(articles) % 50 == 0:
                                        logger.info(f"Progress: {len(articles)}/{limit} articles")
                else:
                    logger.error(f"Wikipedia API returned status {response.status_code}")
                    break
            
            logger.info(f"Collected {len(articles)} Wikipedia articles")
            return articles
            
        except Exception as e:
            logger.error(f"Error collecting from Wikipedia: {str(e)}")
            return articles
    
    def collect_from_arxiv(self, limit: int = 50) -> List[Dict]:
        """Collect research papers from ArXiv"""
        logger.info(f"Starting ArXiv data collection (target: {limit})...")
        papers = []
        
        try:
            base_url = "http://export.arxiv.org/api/query?"
            categories = ["cs.AI", "cs.LG", "cs.CL", "cs.CV", "physics.gen-ph", "math.CO"]
            
            per_category = limit // len(categories) + 1
            
            for category in categories:
                if len(papers) >= limit:
                    break
                    
                # Use random start position
                random_start = random.randint(0, 1000)
                query = f"cat:{category}"
                params = {
                    "search_query": query,
                    "start": random_start,
                    "max_results": min(per_category, 100),  # Increased from 10
                    "sortBy": "submittedDate",
                    "sortOrder": "descending"
                }
                
                response = requests.get(base_url, params=params, timeout=15)
                if response.status_code == 200:
                    root = ET.fromstring(response.content)
                    
                    for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
                        if len(papers) >= limit:
                            break
                            
                        title_elem = entry.find("{http://www.w3.org/2005/Atom}title")
                        summary_elem = entry.find("{http://www.w3.org/2005/Atom}summary")
                        id_elem = entry.find("{http://www.w3.org/2005/Atom}id")
                        
                        if title_elem is not None and summary_elem is not None:
                            content = summary_elem.text.strip()
                            if len(content) > 100 and not self._is_duplicate(content):
                                papers.append({
                                    "source": "arxiv",
                                    "title": title_elem.text.strip(),
                                    "content": content[:5000],  # Increased from 2000
                                    "url": id_elem.text if id_elem is not None else "",
                                    "timestamp": datetime.now().isoformat()
                                })
                                
                if len(papers) % 50 == 0 and len(papers) > 0:
                    logger.info(f"Progress: {len(papers)}/{limit} papers")
            
            logger.info(f"Collected {len(papers)} ArXiv papers")
            return papers
            
        except Exception as e:
            logger.error(f"Error collecting from ArXiv: {str(e)}")
            return papers
    
    def collect_from_gutenberg(self, limit: int = 10) -> List[Dict]:
        """Collect texts from Project Gutenberg"""
        logger.info(f"Starting Project Gutenberg data collection (target: {limit})...")
        texts = []
        
        try:
            url = "https://gutendex.com/books"
            
            # Collect from multiple pages until we reach limit
            page = 1
            while len(texts) < limit and page <= 10:  # Max 10 pages
                params = {"page": page}
                
                response = requests.get(url, params=params, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    books = data.get("results", [])
                    
                    if not books:
                        break
                    
                    for book in books:
                        if len(texts) >= limit:
                            break
                            
                        title = book.get("title", "Unknown")
                        authors = book.get('authors', [])
                        author_name = authors[0].get('name', 'Unknown') if authors else 'Unknown'
                        
                        # Get actual book text if available
                        text_url = None
                        for format_type, format_url in book.get('formats', {}).items():
                            if 'text/plain' in format_type:
                                text_url = format_url
                                break
                        
                        # Create content from metadata and sample
                        content = f"Title: {title}. Author: {author_name}. "
                        subjects = book.get('subjects', [])
                        if subjects:
                            content += f"Subjects: {', '.join(subjects[:3])}. "
                        
                        if not self._is_duplicate(title):
                            texts.append({
                                "source": "gutenberg",
                                "title": title,
                                "content": content[:2000],
                                "url": text_url or book.get("formats", {}).get("text/html", ""),
                                "timestamp": datetime.now().isoformat()
                            })
                    
                    page += 1
                else:
                    break
            
            logger.info(f"Collected {len(texts)} Gutenberg texts")
            return texts
            
        except Exception as e:
            logger.error(f"Error collecting from Gutenberg: {str(e)}")
            return texts
    
    def save_collected_data(self, data: List[Dict], filename: str = None) -> str:
        """Save collected data to JSON file"""
        if not filename:
            filename = f"collected_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = self.data_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Data saved to {filepath}")
        return str(filepath)
    
    def collect_all(self, limits: Dict = None) -> str:
        """Collect data from all sources"""
        if limits is None:
            limits = {"wikipedia": 50, "arxiv": 30, "gutenberg": 10}
        
        all_data = []
        
        all_data.extend(self.collect_from_wikipedia_api(limits.get("wikipedia", 50)))
        all_data.extend(self.collect_from_arxiv(limits.get("arxiv", 30)))
        all_data.extend(self.collect_from_gutenberg(limits.get("gutenberg", 10)))
        
        # Save seen hashes for future deduplication
        self._save_seen_hashes()
        
        logger.info(f"Total data collected: {len(all_data)} items (after deduplication)")
        logger.info(f"Total unique items seen: {len(self.seen_hashes)}")
        return self.save_collected_data(all_data)

if __name__ == "__main__":
    collector = DataCollector()
    collector.collect_all()
