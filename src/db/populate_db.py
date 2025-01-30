from typing import List, Dict, Any
import psycopg2
from psycopg2.extras import execute_values
import numpy as np
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel
import json
from pathlib import Path
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient

class SciBertEmbedder:
    def __init__(self):
        # Check if MPS is available (Apple Silicon)
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("Using MPS (Metal Performance Shaders) device")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("Using CUDA device")
        else:
            self.device = torch.device("cpu")
            print("Using CPU device")
            
        self.tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
        self.model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased").to(self.device)
        self.model.eval()

    def get_embedding(self, text: str, topic_name: str = None) -> np.ndarray:
        """Get embedding for a text string with topic context."""
        if topic_name:
            text_to_embed = f"{topic_name}: {text}"
        else:
            text_to_embed = text
            
        inputs = self.tokenizer(
            text_to_embed,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Move to CPU before converting to numpy
            embedding = outputs.last_hidden_state[0, 0, :].cpu().numpy()
        
        return embedding

def get_db_connection():
    """Get database connection using Azure Key Vault credentials."""
    vault_url = "https://tikasecrets.vault.azure.net/"
    credential = DefaultAzureCredential()
    secret_client = SecretClient(vault_url=vault_url, credential=credential)
    
    try:
        host = secret_client.get_secret("DB-HOST").value
        db_name = secret_client.get_secret("DATABASE-NAME").value
        user = secret_client.get_secret("DB-USER").value
        password = secret_client.get_secret("DB-PASSWORD").value
        port = secret_client.get_secret("DB-PORT").value
        
        conn = psycopg2.connect(
            host=host,
            database=db_name,
            user=user,
            password=password,
            port=port
        )
        return conn
    except Exception as e:
        print(f"Connection error: {str(e)}")
        raise

def process_topics_batch(topics: List[Dict[str, Any]], embedder: SciBertEmbedder, batch_size: int = 64) -> None:
    """Process topics and keywords with embeddings."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            # First, insert all topics
            print("\nInserting topics...")
            topic_data = [
                (topic["id"], topic["display_name"], topic.get("description", ""))
                for topic in topics
            ]
            execute_values(
                cur,
                "INSERT INTO topics (id, display_name, description) VALUES %s ON CONFLICT DO NOTHING",
                topic_data
            )
            conn.commit()
            
            # Process topic-keyword pairs with embeddings
            print("\nProcessing topic-keyword pairs...")
            total_pairs = sum(len(topic.get("keywords", [])) for topic in topics)
            
            with tqdm(total=total_pairs, desc="Processing pairs") as pbar:
                for topic in topics:
                    topic_id = topic["id"]
                    keywords = topic.get("keywords", [])
                    
                    for i in range(0, len(keywords), batch_size):
                        batch = keywords[i:i + batch_size]
                        pairs = [
                            (
                                keyword.lower(),
                                topic_id,
                                embedder.get_embedding(
                                    keyword.lower(), 
                                    topic["display_name"]
                                ).tolist()
                            )
                            for keyword in batch
                        ]
                        
                        execute_values(
                            cur,
                            """
                            INSERT INTO keywords (keyword, topic_id, embedding)
                            VALUES %s
                            ON CONFLICT (keyword, topic_id) 
                            DO UPDATE SET embedding = EXCLUDED.embedding
                            """,
                            pairs
                        )
                        conn.commit()
                        pbar.update(len(batch))
                        
    finally:
        conn.close()

def main() -> None:
    # Load topics from JSON
    current_dir = Path(__file__).parent.absolute()
    project_root = current_dir.parent.parent
    data_path = project_root / "data" / "openalex_topics_raw.json"
    
    print(f"Loading data from: {data_path}")
    with open(data_path, "r") as f:
        topics = json.load(f)
    
    print(f"Loaded {len(topics)} topics")
    
    # Initialize embedder
    print("\nInitializing SciBERT model...")
    embedder = SciBertEmbedder()
    
    # Process topics and generate embeddings
    process_topics_batch(topics, embedder)
    
    # Print final statistics
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT 
                    COUNT(DISTINCT topic_id) as topics,
                    COUNT(*) as total_pairs,
                    COUNT(embedding) as pairs_with_embeddings
                FROM keywords
            """)
            topics, pairs, with_emb = cur.fetchone()
            print("\nFinal Statistics:")
            print(f"Topics processed: {topics}")
            print(f"Total topic-keyword pairs: {pairs}")
            print(f"Pairs with embeddings: {with_emb}")
    finally:
        conn.close()

if __name__ == "__main__":
    main()
