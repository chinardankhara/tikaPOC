import os
from typing import Any
import psycopg2
from psycopg2.extensions import connection
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def get_db_connection() -> connection:
    """
    Create a database connection using environment variables.
    
    Required environment variables:
    - DB_HOST: Database host
    - DB_NAME: Database name
    - DB_USER: Database user
    - DB_PASSWORD: Database password
    - DB_PORT: Database port (default: 5432)
    
    Returns:
        psycopg2 connection object
    """
    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST"),
            database=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            port=os.getenv("DB_PORT", "5432")
        )
        return conn
    except Exception as e:
        raise ConnectionError(f"Failed to connect to database: {str(e)}") 