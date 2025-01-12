from typing import List, Dict, Any, Iterator
import psycopg2
from psycopg2.extras import execute_values
import numpy as np
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel
from itertools import islice
from contextlib import contextmanager
import os
from dotenv import load_dotenv
import json

# Database connection function
@contextmanager
def get_db_connection():
    """
    Context manager for database connections.
    Usage:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(...)
    """
    load_dotenv()  # Load environment variables
    
    conn = psycopg2.connect(
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT", "5432")
    )
    try:
        yield conn
    finally:
        conn.close()

class SciBertEmbedder:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
        self.model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased").to(self.device)
        self.model.eval()

    @torch.no_grad()
    def get_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """Get embeddings for a batch of texts using SciBERT."""
        # Tokenize texts
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)

        # Get model outputs
        outputs = self.model(**encoded)
        
        # Use [CLS] token embeddings as sentence embeddings
        embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        return embeddings

def batch_iterator(iterable: Iterator, batch_size: int) -> Iterator:
    """Creates batches from an iterator."""
    iterator = iter(iterable)
    while batch := list(islice(iterator, batch_size)):
        yield batch

def process_topics_batch(
    topics: List[Dict[str, Any]], 
    embedder: SciBertEmbedder,
    batch_size: int = 64
) -> None:
    """Process topics in batches, handling both embeddings and DB insertions."""
    
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            # First, insert all topics in one go
            topic_data = [
                (topic["id"], topic["display_name"], topic.get("description", ""))
                for topic in topics
            ]
            
            print("Inserting topics...")
            execute_values(
                cur,
                "INSERT INTO topics (id, display_name, description) VALUES %s ON CONFLICT DO NOTHING",
                topic_data
            )
            conn.commit()

            # Collect and deduplicate keywords
            unique_keywords = {
                keyword.lower()
                for topic in topics
                for keyword in topic.get("keywords", [])
            }
            keyword_list = list(unique_keywords)

            # Process keywords in batches
            keyword_id_map = {}
            total_batches = (len(keyword_list) + batch_size - 1) // batch_size

            print(f"Processing {len(keyword_list)} unique keywords in {total_batches} batches")

            for keyword_batch in tqdm(batch_iterator(keyword_list, batch_size), 
                                    total=total_batches, 
                                    desc="Processing keyword batches"):
                # Get embeddings for the batch using the passed embedder
                embeddings = embedder.get_embeddings_batch(keyword_batch)
                
                # Prepare batch data
                keyword_data = [
                    (keyword, embedding.astype(float).tolist())  # Convert numpy array to list
                    for keyword, embedding in zip(keyword_batch, embeddings)
                ]

                # Insert keywords and get their IDs
                execute_values(
                    cur,
                    """
                    INSERT INTO keywords (keyword, embedding) 
                    VALUES %s 
                    ON CONFLICT (keyword) DO UPDATE 
                    SET embedding = EXCLUDED.embedding 
                    RETURNING id, keyword
                    """,
                    keyword_data
                )
                
                # Store keyword to id mapping
                for row in cur.fetchall():
                    keyword_id_map[row[1]] = row[0]
                
                conn.commit()

            # Process topic-keyword relationships in batches
            topic_keyword_relations = [
                (topic["id"], keyword_id_map[keyword.lower()])
                for topic in topics
                for keyword in topic.get("keywords", [])
            ]

            print("Inserting topic-keyword relationships...")
            for batch in batch_iterator(topic_keyword_relations, batch_size):
                execute_values(
                    cur,
                    """
                    INSERT INTO topic_keywords (topic_id, keyword_id) 
                    VALUES %s 
                    ON CONFLICT DO NOTHING
                    """,
                    batch
                )
                conn.commit()

def sanity_check_data() -> None:
    """Run basic sanity checks on the inserted data."""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            # Check topic count
            cur.execute("SELECT COUNT(*) FROM topics")
            topic_count = cur.fetchone()[0]
            print(f"Total topics: {topic_count}")
            
            # Check keyword count
            cur.execute("SELECT COUNT(*) FROM keywords")
            keyword_count = cur.fetchone()[0]
            print(f"Total keywords: {keyword_count}")
            
            # Check topic-keyword relationships
            cur.execute("SELECT COUNT(*) FROM topic_keywords")
            relationship_count = cur.fetchone()[0]
            print(f"Total topic-keyword relationships: {relationship_count}")

def main() -> None:
    # Load your topics from JSON
    with open("../../data/openalex_topics_raw.json", "r") as f:
        topics = json.load(f)
    
    # Initialize embedder
    print("Initializing SciBERT model...")
    embedder = SciBertEmbedder()
    
    process_topics_batch(
        topics,
        embedder,
        batch_size=64
    )

    # Basic sanity check
    print("\nRunning basic sanity checks...")
    sanity_check_data()

if __name__ == "__main__":
    main()
