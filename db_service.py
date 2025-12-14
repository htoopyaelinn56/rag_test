import os
import psycopg2
from psycopg2 import sql
from typing import Iterable, Optional, List, Dict

# Basic connection configuration with sensible defaults; allow override via env vars
DB_NAME = os.getenv("DOC_DB_NAME", "rag_test")
DB_USER = os.getenv("DOC_DB_USER", "postgres")
DB_PASSWORD = os.getenv("DOC_DB_PASSWORD", "password")
DB_HOST = os.getenv("DOC_DB_HOST", "localhost")
DB_PORT = int(os.getenv("DOC_DB_PORT", "5432"))

from embedding_service import embed_text


def get_connection():
    """
    Create and return a new psycopg2 connection to the Postgres database.
    Defaults to local Postgres on localhost:5432 with user 'postgres' and password 'password'.
    """
    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT,
    )
    # Let callers control transactions; we'll explicitly commit/rollback around operations
    conn.autocommit = False
    return conn


def _embedding_to_sql_array(embedding: Iterable[float]) -> str:
    """
    Convert a Python iterable of numbers to a SQL ARRAY[...] literal string.
    This is used to cast into pgvector via ARRAY[...]::vector.
    """
    # Ensure floats for safety and join; avoid SQL injection by not accepting strings here
    vals = ",".join(str(float(x)) for x in embedding)
    return f"ARRAY[{vals}]"


def insert_document_chunk(
    chunk_index: int,
    chunk_text: str,
    contextualized_text: str,
    chunk_tokens: int,
    contextualized_tokens: int,
    embedding: Optional[Iterable[float]] = None,
) -> Optional[int]:
    """
    Insert a row into document_chunks. Returns inserted row id on success, or None if skipped by conflict.

    - On conflict of UNIQUE(chunk_index), does nothing and returns None.
    - If embedding is provided, it is inserted via ARRAY[...]::vector cast (expects pgvector extension installed).
    """
    conn = get_connection()
    cur = conn.cursor()
    try:
        if embedding is None:
            query = sql.SQL(
                """
                INSERT INTO document_chunks (chunk_index, chunk_text, contextualized_text,
                                             chunk_tokens, contextualized_tokens)
                VALUES (%s, %s, %s, %s, %s) ON CONFLICT (chunk_index) DO NOTHING
                RETURNING id
                """
            )
            params = (
                chunk_index,
                chunk_text,
                contextualized_text,
                chunk_tokens,
                contextualized_tokens,
            )
        else:
            # Build ARRAY[...]::vector expression for embedding
            array_literal = _embedding_to_sql_array(embedding)
            query = sql.SQL(
                """
                INSERT INTO document_chunks (chunk_index, chunk_text, contextualized_text,
                                             chunk_tokens, contextualized_tokens, embedding)
                VALUES (%s, %s, %s, %s, %s, {array}::vector) ON CONFLICT (chunk_index) DO NOTHING
                RETURNING id
                """
            ).format(array=sql.SQL(array_literal))
            params = (
                chunk_index,
                chunk_text,
                contextualized_text,
                chunk_tokens,
                contextualized_tokens,
            )

        cur.execute(query, params)
        row = cur.fetchone()
        # If conflict occurred, RETURNING yields no row
        inserted_id = row[0] if row else None
        conn.commit()
        return inserted_id
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()


def retrieve_context(query: str, top_k: int = 3, threshold: float = 0.55) -> List[Dict]:
    """
    Retrieve relevant context chunks from vector database.

    Args:
        query: User's question
        top_k: Number of chunks to retrieve
        threshold: Minimum similarity threshold

    Returns:
        List of relevant chunks
    """
    conn = get_connection()
    if not conn:
        raise Exception("Database connection not established. Call connect() first.")

    # Generate embedding for the query
    query_embedding = embed_text(query)
    cursor = conn.cursor()

    # Query similar chunks
    sql_query = """
                SELECT id,
                       chunk_index,
                       chunk_text,
                       contextualized_text,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM document_chunks
                WHERE 1 - (embedding <=> %s::vector) > %s
                ORDER BY embedding <=> %s::vector
                    LIMIT %s \
                """

    cursor.execute(sql_query, (query_embedding, query_embedding, threshold, query_embedding, top_k))

    results = []
    for row in cursor.fetchall():
        results.append({
            'id': row[0],
            'chunk_index': row[1],
            'chunk_text': row[2],
            'contextualized_text': row[3],
            'similarity': row[4]
        })

    cursor.close()
    return results
