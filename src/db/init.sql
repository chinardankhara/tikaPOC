-- Create database
CREATE DATABASE openalex_topics;

-- Connect to database and create extension
\c openalex_topics
CREATE EXTENSION vector;

-- Create tables with proper relationships
CREATE TABLE topics (
    id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE keywords (
    id BIGSERIAL PRIMARY KEY,
    keyword TEXT NOT NULL UNIQUE,
    embedding vector(768),  
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Junction table for many-to-many relationship
CREATE TABLE topic_keywords (
    topic_id TEXT REFERENCES topics(id),
    keyword_id BIGINT REFERENCES keywords(id),
    PRIMARY KEY (topic_id, keyword_id)
);

-- Create an index for vector similarity search
CREATE INDEX ON keywords USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);  -- Number of lists can be adjusted based on your data size