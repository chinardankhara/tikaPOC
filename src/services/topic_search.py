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

    def get_embedding(self, text: str, topic_name: str = None) -> np.ndarray:
        """Get embedding for a text string, optionally prefixed with topic."""
        if topic_name:
            # Format: "TOPIC: keyword" to create contextual embeddings
            text_to_embed = f"{topic_name}: {text}"
        else:
            text_to_embed = text
            
        inputs = self.tokenizer(
            text_to_embed,
            return_tensors="pt",
            max_length=768,
            truncation=True,
            padding=True
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state[0, 0, :].numpy()
        
        return embedding

    def search_topics(
        self,
        query: str,
        excluded_topic_ids: Set[str] = set(),
        n_similar: int = 20,
        n_topics: int = 3
    ) -> List[Dict[str, Any]]:
        """Search using hybrid approach combining dense vectors and keyword matching."""
        query_embedding = self.get_embedding(query)
        
        # Debug print
        print(f"Query: {query}")
        print(f"Embedding shape: {query_embedding.shape}")
        print(f"Embedding sample: {query_embedding[:5]}")
        
        search_query = """
        WITH vector_scores AS (
            SELECT 
                t.id,
                t.display_name,
                t.description,
                k.keyword,
                1.0 - (k.embedding <=> %s::vector) as vector_similarity,  -- Convert distance to similarity
                ROW_NUMBER() OVER (PARTITION BY t.id ORDER BY k.embedding <=> %s::vector) as dense_rank
            FROM keywords k
            JOIN topics t ON k.topic_id = t.id
            WHERE t.id != ALL(%s)
        ),
        keyword_scores AS (
            SELECT 
                t.id,
                t.display_name,
                t.description,
                k.keyword,
                similarity(k.keyword, %s) as keyword_similarity,
                ROW_NUMBER() OVER (PARTITION BY t.id ORDER BY similarity(k.keyword, %s) DESC) as keyword_rank
            FROM keywords k
            JOIN topics t ON k.topic_id = t.id
            WHERE t.id != ALL(%s)
        ),
        combined_scores AS (
            SELECT 
                v.id,
                v.display_name,
                v.description,
                v.keyword as matching_keyword,
                -- Weighted combination of scores
                (v.vector_similarity * 0.7 + k.keyword_similarity * 0.3) as combined_score,
                v.vector_similarity,
                k.keyword_similarity
            FROM vector_scores v
            JOIN keyword_scores k ON v.id = k.id
            WHERE v.dense_rank = 1 AND k.keyword_rank = 1
        )
        SELECT 
            id,
            display_name,
            description,
            matching_keyword,
            combined_score,
            vector_similarity,
            keyword_similarity
        FROM combined_scores
        ORDER BY combined_score DESC
        LIMIT %s
        """
        
        query_params = (
            query_embedding.tolist(),
            query_embedding.tolist(),
            list(excluded_topic_ids) if excluded_topic_ids else [],
            query,
            query,
            list(excluded_topic_ids) if excluded_topic_ids else [],
            n_topics
        )
        
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # First, let's check what we're working with
                cur.execute("SELECT COUNT(*) FROM keywords WHERE embedding IS NOT NULL")
                embedding_count = cur.fetchone()[0]
                
                # Execute with EXPLAIN ANALYZE
                cur.execute(search_query, query_params)
                
                # Execute actual query
                cur.execute(search_query, query_params)
                
                results = [
                    {
                        "id": row[0],
                        "display_name": row[1],
                        "description": row[2],
                        "matching_keyword": row[3],
                        "score": float(row[4]),
                        "vector_similarity": float(row[5]),
                        "keyword_similarity": float(row[6])
                    }
                    for row in cur.fetchall()
                ]
                
                # Debug print results
                print("\nSearch Results:")
                for r in results:
                    print(f"Topic: {r['display_name']}")
                    print(f"Vector Similarity: {r['vector_similarity']}")
                    print(f"Keyword Similarity: {r['keyword_similarity']}")
                    print(f"Combined Score: {r['score']}\n")
                
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