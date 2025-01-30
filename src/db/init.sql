-- Create database (if not exists)
-- CREATE DATABASE openalex_vector_kg;

-- Connect to database and create extensions
-- \c openalex_vector_kg

-- Create extension for vector similarity search
CREATE EXTENSION IF NOT EXISTS vector;
-- Enable trigram extension for better text matching
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Create tables with proper relationships
CREATE TABLE topics (
    id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create keywords table with context and embeddings
CREATE TABLE keywords (
    id BIGSERIAL PRIMARY KEY,
    keyword TEXT NOT NULL,
    topic_id TEXT REFERENCES topics(id),
    embedding vector(768),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (keyword, topic_id)
);

-- Create indices for performance
CREATE INDEX ON keywords USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);
CREATE INDEX idx_keywords_trgm ON keywords USING gin (keyword gin_trgm_ops);

-- Create indices for faster joins and lookups
CREATE INDEX idx_keywords_topic_id ON keywords(topic_id);
CREATE INDEX idx_keywords_keyword ON keywords(keyword);

-- Create RRF function for hybrid search
CREATE OR REPLACE FUNCTION rrf(dense_rank float, keyword_rank float, k float default 60.0) 
RETURNS float AS $$
BEGIN
    -- RRF formula: 1/(k + r) where k is typically 60
    RETURN 1.0/(k + dense_rank) + 1.0/(k + keyword_rank);
END;
$$ LANGUAGE plpgsql;

-- This preserves the keywords but clears embeddings
UPDATE keywords SET embedding = NULL;