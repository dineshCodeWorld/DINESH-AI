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
import time
import re
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
                
                time.sleep(0.5)  # Rate limit protection
                response = requests.get(url, params=params, headers=headers, timeout=15)
                if response.status_code == 429:
                    logger.warning("Wikipedia rate limit hit, waiting 10 seconds...")
                    time.sleep(10)
                    continue
                elif response.status_code == 200:
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
                        time.sleep(0.3)  # Rate limit protection
                        content_response = requests.get(url, params=content_params, headers=headers, timeout=15)
                        if content_response.status_code == 429:
                            logger.warning("Wikipedia rate limit on content, waiting 5 seconds...")
                            time.sleep(5)
                            continue
                        elif content_response.status_code == 200:
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
                
                time.sleep(1)  # ArXiv rate limit (1 req/sec)
                try:
                    response = requests.get(base_url, params=params, timeout=20)
                except requests.exceptions.Timeout:
                    logger.warning(f"ArXiv timeout for {category}, retrying...")
                    time.sleep(3)
                    try:
                        response = requests.get(base_url, params=params, timeout=30)
                    except:
                        logger.error(f"ArXiv failed for {category} after retry")
                        continue
                
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
            retries = 0
            max_retries = 3
            
            while len(texts) < limit and page <= 10:  # Max 10 pages
                params = {"page": page}
                
                time.sleep(1)  # Rate limit protection
                try:
                    response = requests.get(url, params=params, timeout=20)
                except requests.exceptions.Timeout:
                    if retries < max_retries:
                        retries += 1
                        logger.warning(f"Gutenberg timeout, retry {retries}/{max_retries}...")
                        time.sleep(5)
                        continue
                    else:
                        logger.error("Gutenberg max retries reached")
                        break
                
                if response.status_code == 200:
                    retries = 0  # Reset on success
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
    
    def collect_from_reddit(self, limit: int = 100) -> List[Dict]:
        """Collect posts from Reddit (no API key needed)"""
        logger.info(f"Starting Reddit data collection (target: {limit})...")
        posts = []
        
        try:
            subreddits = ['AskReddit', 'explainlikeimfive', 'todayilearned', 'science', 'technology']
            headers = {'User-Agent': 'DineshAI/1.0'}
            
            for subreddit in subreddits:
                if len(posts) >= limit:
                    break
                
                time.sleep(2)
                url = f"https://www.reddit.com/r/{subreddit}/top.json?limit=20&t=week"
                
                try:
                    response = requests.get(url, headers=headers, timeout=15)
                    if response.status_code == 200:
                        data = response.json()
                        
                        for post in data.get('data', {}).get('children', []):
                            if len(posts) >= limit:
                                break
                            
                            post_data = post.get('data', {})
                            title = post_data.get('title', '')
                            selftext = post_data.get('selftext', '')
                            
                            content = f"{title}. {selftext}"
                            
                            if len(content) > 100 and not self._is_duplicate(content):
                                posts.append({
                                    "source": "reddit",
                                    "title": title,
                                    "content": content[:3000],
                                    "url": f"https://reddit.com{post_data.get('permalink', '')}",
                                    "timestamp": datetime.now().isoformat()
                                })
                except Exception as e:
                    logger.warning(f"Error fetching r/{subreddit}: {e}")
                    continue
            
            logger.info(f"Collected {len(posts)} Reddit posts")
            return posts
            
        except Exception as e:
            logger.error(f"Error collecting from Reddit: {str(e)}")
            return posts
    
    def collect_from_hackernews(self, limit: int = 50) -> List[Dict]:
        """Collect stories from Hacker News"""
        logger.info(f"Starting HackerNews data collection (target: {limit})...")
        stories = []
        
        try:
            # Get top stories
            response = requests.get('https://hacker-news.firebaseio.com/v0/topstories.json', timeout=10)
            if response.status_code == 200:
                story_ids = response.json()[:limit * 2]  # Get extra in case some fail
                
                for story_id in story_ids:
                    if len(stories) >= limit:
                        break
                    
                    time.sleep(0.5)
                    story_response = requests.get(
                        f'https://hacker-news.firebaseio.com/v0/item/{story_id}.json',
                        timeout=10
                    )
                    
                    if story_response.status_code == 200:
                        story = story_response.json()
                        title = story.get('title', '')
                        text = story.get('text', '')
                        url = story.get('url', '')
                        
                        content = f"{title}. {text}"
                        
                        if len(content) > 50 and not self._is_duplicate(content):
                            stories.append({
                                "source": "hackernews",
                                "title": title,
                                "content": content[:2000],
                                "url": url or f"https://news.ycombinator.com/item?id={story_id}",
                                "timestamp": datetime.now().isoformat()
                            })
            
            logger.info(f"Collected {len(stories)} HackerNews stories")
            return stories
            
        except Exception as e:
            logger.error(f"Error collecting from HackerNews: {str(e)}")
            return stories
    
    def collect_from_common_crawl(self, limit: int = 50) -> List[Dict]:
        """Collect from Common Crawl News dataset"""
        logger.info(f"Starting Common Crawl data collection (target: {limit})...")
        articles = []
        
        try:
            # Use Common Crawl News API
            url = "https://data.commoncrawl.org/crawl-data/CC-NEWS/index.html"
            headers = {'User-Agent': 'DineshAI/1.0'}
            
            # For simplicity, collect from news sites via RSS
            rss_feeds = [
                'https://rss.nytimes.com/services/xml/rss/nyt/Technology.xml',
                'https://feeds.bbci.co.uk/news/technology/rss.xml',
            ]
            
            for feed_url in rss_feeds:
                if len(articles) >= limit:
                    break
                
                time.sleep(2)
                try:
                    response = requests.get(feed_url, headers=headers, timeout=15)
                    if response.status_code == 200:
                        root = ET.fromstring(response.content)
                        
                        for item in root.findall('.//item'):
                            if len(articles) >= limit:
                                break
                            
                            title_elem = item.find('title')
                            desc_elem = item.find('description')
                            link_elem = item.find('link')
                            
                            if title_elem is not None and desc_elem is not None:
                                title = title_elem.text or ''
                                desc = desc_elem.text or ''
                                
                                # Clean HTML tags
                                desc = re.sub(r'<[^>]+>', '', desc)
                                content = f"{title}. {desc}"
                                
                                if len(content) > 100 and not self._is_duplicate(content):
                                    articles.append({
                                        "source": "news",
                                        "title": title,
                                        "content": content[:2000],
                                        "url": link_elem.text if link_elem is not None else "",
                                        "timestamp": datetime.now().isoformat()
                                    })
                except Exception as e:
                    logger.warning(f"Error fetching feed: {e}")
                    continue
            
            logger.info(f"Collected {len(articles)} news articles")
            return articles
            
        except Exception as e:
            logger.error(f"Error collecting from news feeds: {str(e)}")
            return articles
    
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
            limits = {
                "wikipedia": 50,
                "arxiv": 30,
                "gutenberg": 10,
                "reddit": 100,
                "hackernews": 50,
                "news": 50
            }
        
        all_data = []
        
        all_data.extend(self.collect_from_wikipedia_api(limits.get("wikipedia", 50)))
        all_data.extend(self.collect_from_arxiv(limits.get("arxiv", 30)))
        all_data.extend(self.collect_from_gutenberg(limits.get("gutenberg", 10)))
        all_data.extend(self.collect_from_reddit(limits.get("reddit", 100)))
        all_data.extend(self.collect_from_hackernews(limits.get("hackernews", 50)))
        all_data.extend(self.collect_from_common_crawl(limits.get("news", 50)))
        
        # Save seen hashes for future deduplication
        self._save_seen_hashes()
        
        logger.info(f"Total data collected: {len(all_data)} items (after deduplication)")
        logger.info(f"Total unique items seen: {len(self.seen_hashes)}")
        return self.save_collected_data(all_data)

if __name__ == "__main__":
    collector = DataCollector()
    collector.collect_all()
