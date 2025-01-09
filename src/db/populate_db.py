from typing import List, Dict, Any
import torch
import psycopg2
from psycopg2.extras import execute_values
from transformers import AutoModel
from optimum.bettertransformer import BetterTransformer
import numpy as np
from tqdm import tqdm

def get_embedding_model() -> AutoModel:
    """Initialize SciBERT model with better-transformer optimization"""
    model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")
    model = BetterTransformer.transform(model)
    if torch.cuda.is_available():
        model = model.to("cuda")
    return model

def get_embeddings_batch(texts: List[str], model: AutoModel, batch_size: int = 32) -> np.ndarray:
    """Get embeddings for a batch of texts"""
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        with torch.no_grad():
            batch_embeddings = model.encode(batch, convert_to_numpy=True)
        embeddings.extend(batch_embeddings)
    return np.array(embeddings)

def populate_database(topics: List[Dict[str, Any]], connection_string: str) -> None:
    """Populate the database with topics and their keyword embeddings"""
    conn = psycopg2.connect(connection_string)
    cur = conn.cursor()
    
    try:
        # Initialize model
        print("Initializing SciBERT model...")
        model = get_embedding_model()
        
        # First, insert all topics
        print("Inserting topics...")
        topic_data = [(
            topic["id"],
            topic["display_name"],
            topic.get("description", "")
        ) for topic in topics]
        
        execute_values(
            cur,
            "INSERT INTO topics (id, display_name, description) VALUES %s",
            topic_data
        )
        
        # Process keywords and get unique ones
        unique_keywords = set()
        for topic in topics:
            for keyword in topic.get("keywords", []):
                unique_keywords.add(keyword.lower())
        
        # Get embeddings for all unique keywords in batches
        print("Generating embeddings for keywords...")
        keyword_list = list(unique_keywords)
        embeddings = get_embeddings_batch(keyword_list, model)
        
        # Insert keywords with their embeddings
        print("Inserting keywords and embeddings...")
        execute_values(
            cur,
            "INSERT INTO keywords (keyword, embedding) VALUES %s RETURNING id, keyword",
            [(k, e) for k, e in zip(keyword_list, embeddings)]
        )
        
        # Create mapping of keyword to id
        keyword_ids = {}
        for row in cur.fetchall():
            keyword_ids[row[1]] = row[0]
        
        # Insert topic-keyword relationships
        print("Creating topic-keyword relationships...")
        topic_keyword_relations = []
        for topic in topics:
            topic_id = topic["id"]
            for keyword in topic.get("keywords", []):
                keyword_id = keyword_ids[keyword.lower()]
                topic_keyword_relations.append((topic_id, keyword_id))
        
        execute_values(
            cur,
            "INSERT INTO topic_keywords (topic_id, keyword_id) VALUES %s",
            topic_keyword_relations
        )
        
        conn.commit()
        print("Database population completed successfully!")
        
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        cur.close()
        conn.close()