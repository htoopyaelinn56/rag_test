DROP TABLE document_chunks;

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create the document_chunks table
CREATE TABLE document_chunks (
                                 id SERIAL PRIMARY KEY,
                                 chunk_index INTEGER NOT NULL,
                                 chunk_text TEXT NOT NULL,
                                 contextualized_text TEXT NOT NULL,
                                 chunk_tokens INTEGER NOT NULL,
                                 contextualized_tokens INTEGER NOT NULL,
                                 embedding vector(768),  -- Adjust dimension based on your embedding model
                                 created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Add a unique constraint to prevent duplicate chunks
                                 UNIQUE(chunk_index)
);

-- Create indexes for better query performance
CREATE INDEX idx_chunk_index ON document_chunks(chunk_index);

-- Create a vector similarity search index using HNSW (Hierarchical Navigable Small World)
-- This significantly speeds up similarity searches
CREATE INDEX idx_embedding ON document_chunks
    USING hnsw (embedding vector_cosine_ops);
