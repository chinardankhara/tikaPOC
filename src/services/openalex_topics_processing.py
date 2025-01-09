import requests
from typing import Dict, List, Any
import json
from pathlib import Path
import time
from collections import defaultdict

def fetch_openalex_topics() -> List[Dict[str, Any]]:
    """Fetches all topics from OpenAlex API using cursor-based pagination."""
    base_url = "https://api.openalex.org/topics"
    topics: List[Dict[str, Any]] = []
    cursor = "*"
    per_page = 200
    
    while cursor:
        params = {
            "per-page": per_page,
            "cursor": cursor,
            "mailto": "chinardankhara@gmail.com"  
        }
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        
        data = response.json()
        results = data.get("results", [])
        
        if not results:
            break
            
        topics.extend(results)
        
        cursor = data.get("meta", {}).get("next_cursor")

    
    return topics
def create_keyword_mapping(topics: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, str]]]:
    """Creates a reverse mapping from keywords to topics."""
    keyword_map = defaultdict(list)
    
    for topic in topics:
        topic_info = {
            "topic_name": topic["display_name"],
            "description": topic.get("description", ""),
            "topic_id": topic["id"]
        }
        
        # Add mapping for each keyword using dict.get() with empty list default
        for keyword in topic.get("keywords", []):
            keyword_map[keyword.lower()].append(topic_info)
    
    return dict(keyword_map)

def main() -> None:
    # Create data directory if it doesn't exist
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Fetch topics and save raw data
    print("Fetching OpenAlex topics...")
    topics = fetch_openalex_topics()
    
    with open(data_dir / "openalex_topics_raw.json", "w", encoding="utf-8") as f:
        json.dump(topics, f, indent=2, ensure_ascii=False)
    
    # Create and save keyword mapping
    print("Creating keyword mapping...")
    keyword_map = create_keyword_mapping(topics)
    
    with open(data_dir / "keyword_to_topics_map.json", "w", encoding="utf-8") as f:
        json.dump(keyword_map, f, indent=2, ensure_ascii=False)
    
    print(f"Processed {len(topics)} topics and {len(keyword_map)} unique keywords")

if __name__ == "__main__":
    main()