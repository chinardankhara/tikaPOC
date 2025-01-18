from typing import List, Dict, Any, Set
import numpy as np
import psycopg2
from psycopg2.extensions import connection
import torch
from transformers import AutoTokenizer, AutoModel
import sys
import os
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set up Key Vault client
vault_url = "https://tikasecrets.vault.azure.net/"
credential = DefaultAzureCredential()
secret_client = SecretClient(vault_url=vault_url, credential=credential)

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

def get_db_connection() -> connection:
    """
    Create a database connection using secrets from Azure Key Vault.
    """
    try:
        # First try to get all secrets
        try:
            host = secret_client.get_secret("DB-HOST").value
            db_name = secret_client.get_secret("DATABASE-NAME").value
            user = secret_client.get_secret("DB-USER").value
            password = secret_client.get_secret("DB-PASSWORD").value
            port = secret_client.get_secret("DB-PORT").value
        except Exception as e:
            print(f"Failed to retrieve secrets from Key Vault: {str(e)}")
            raise

        print("Retrieved connection details:")
        print(f"Host: {host}")
        print(f"Database Name: {db_name}")  # Let's explicitly see the database name
        print(f"User: {user}")
        print(f"Port: {port}")
        
        # Now try to connect
        try:
            conn = psycopg2.connect(
                host=host,
                database=db_name,
                user=user,
                password=password,
                port=port
            )
            print("Connection successful!")
            return conn
        except psycopg2.Error as e:
            print(f"PostgreSQL Error: {e.pgcode} - {e.pgerror}")
            raise
            
    except Exception as e:
        print(f"Connection error details: {type(e).__name__}: {str(e)}")
        raise ConnectionError(f"Failed to connect to database: {str(e)}") 