from typing import List, Dict, Any, Set
import numpy as np
from psycopg2.extras import execute_values
import torch
from transformers import AutoTokenizer, AutoModel

class TopicSearcher:
    def __init__(self) -> None:
        """Initialize the searcher with SciBERT model."""
        self.tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
        self.model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")
        self.model.eval()  # Set to evaluation mode

    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a single text string."""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=768,
            truncation=True,
            padding=True
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use CLS token embedding
            embedding = outputs.last_hidden_state[0, 0, :].numpy()
        
        return embedding

    def search_topics(
        self,
        query: str,
        excluded_topic_ids: Set[str] = set(),
        n_similar_keywords: int = 10,
        n_topics: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Search for topics based on query, excluding specified topic IDs.
        
        Args:
            query: Search query
            excluded_topic_ids: Set of topic IDs to exclude
            n_similar_keywords: Number of similar keywords to consider
            n_topics: Number of topics to return
            
        Returns:
            List of topic dictionaries with id, display_name, and description
        """
        from src.db.connection import get_db_connection
        
        # Get query embedding
        query_embedding = self.get_embedding(query)
        
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Using a CTE for clarity and efficiency
                cur.execute(
                    """
                    WITH similar_keywords AS (
                        -- Find similar keywords
                        SELECT 
                            k.keyword,
                            k.id as keyword_id,
                            embedding <=> %s::vector as similarity
                        FROM keywords k
                        ORDER BY similarity
                        LIMIT %s
                    ),
                    topic_scores AS (
                        -- Get topics and their scores
                        SELECT 
                            t.id,
                            t.display_name,
                            t.description,
                            COUNT(DISTINCT sk.keyword_id) as matching_keywords,
                            AVG(sk.similarity) as avg_similarity
                        FROM similar_keywords sk
                        JOIN topic_keywords tk ON sk.keyword_id = tk.keyword_id
                        JOIN topics t ON tk.topic_id = t.id
                        WHERE t.id != ALL(%s)  -- Exclude specified topics
                        GROUP BY t.id, t.display_name, t.description
                    )
                    -- Final ranking and selection
                    SELECT 
                        id,
                        display_name,
                        description,
                        matching_keywords,
                        avg_similarity
                    FROM topic_scores
                    ORDER BY 
                        matching_keywords DESC,
                        avg_similarity ASC
                    LIMIT %s
                    """,
                    (
                        query_embedding.tolist(),
                        n_similar_keywords,
                        list(excluded_topic_ids) if excluded_topic_ids else [],
                        n_topics
                    )
                )
                
                results = [
                    {
                        "id": row[0],
                        "display_name": row[1],
                        "description": row[2],
                        "matching_keywords": row[3],
                        "similarity_score": float(row[4])
                    }
                    for row in cur.fetchall()
                ]
                
                return results
