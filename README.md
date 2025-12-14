# Docling Test — RAG Chatbot

A simple Retrieval-Augmented Generation (RAG) chatbot that indexes local documents into Postgres, retrieves relevant chunks, and answers questions using Google GenAI.

## Prerequisites
- Python 3.11+ (project shows Python 3.13 bytecode, use 3.11–3.13)
- PostgreSQL 13+
- Google GenAI API key (Gemini)

## Quick Start

1) Clone and enter the project directory

```zsh
cd /path/to/rag_test
```

2) Create and fill in `.env`

```zsh
cp .env.example .env  # if present; otherwise create a new .env
```

Required entries in `.env`:
- `GEMINI_API_KEY` — your Google GenAI API key
- Database settings (choose either individual vars or a single URL):
  - Individual vars: `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`
  - Or: `DATABASE_URL=postgresql://user:password@host:port/dbname`

Example:
```env
GEMINI_API_KEY=your_api_key_here
DB_HOST=127.0.0.1
DB_PORT=5432
DB_NAME=docling
DB_USER=postgres
DB_PASSWORD=postgres
```

3) Install dependencies

```zsh
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install google-genai python-dotenv psycopg2-binary
```

4) Initialize the database schema

Apply the schema to your Postgres instance.

```zsh
psql "host=$DB_HOST port=$DB_PORT dbname=$DB_NAME user=$DB_USER password=$DB_PASSWORD" -f schema.sql
# Or using DATABASE_URL
psql "$DATABASE_URL" -f schema.sql
```

5) Bootstrap data and embeddings

Run the bootstrap step to ingest documents and create embeddings.

```zsh
python bootstrap.py
```

6) Start the chatbot

```zsh
python chatbot.py
```

Type your questions. Use `quit`, `exit`, or `q` to end the session.

## Notes
- Ensure your Postgres server is running and reachable by the connection settings.
- `db_service.py` controls database connections and retrieval logic.
- `embedding_service.py` and `chunker.py` handle splitting text and generating embeddings.
- If you get an API error, double-check `GEMINI_API_KEY` and network access.

## Troubleshooting
- "I couldn't find any relevant information" — verify that `bootstrap.py` ran successfully and the tables contain data.
- Postgres connection errors — confirm credentials and that the DB is reachable.
- API key errors — confirm `GEMINI_API_KEY` is set in `.env` and the virtualenv is activated.

