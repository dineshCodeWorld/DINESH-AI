import sys
sys.stdout.reconfigure(encoding='utf-8')

import requests
import xml.etree.ElementTree as ET

feeds = [
    'https://rss.nytimes.com/services/xml/rss/nyt/Technology.xml',
    'https://feeds.bbci.co.uk/news/technology/rss.xml',
]

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
}

for feed in feeds:
    print(f"\nTesting: {feed}")
    try:
        response = requests.get(feed, headers=headers, timeout=15)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            root = ET.fromstring(response.content)
            items = root.findall('.//item')
            print(f"Items found: {len(items)}")
            
            if items:
                title = items[0].find('title')
                print(f"Sample: {title.text if title is not None else 'No title'}")
        else:
            print(f"Error: {response.text[:200]}")
    except Exception as e:
        print(f"Exception: {e}")
