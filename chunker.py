from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import AutoTokenizer

from db_service import insert_document_chunk
from embedding_service import embed_text

EMBED_MODEL_ID = "/Volumes/HPLSSD/ai_models/embeddinggemma"
MAX_TOKENS = 512

tokenizer = HuggingFaceTokenizer(
    tokenizer=AutoTokenizer.from_pretrained(EMBED_MODEL_ID),
    max_tokens=MAX_TOKENS,  # optional, by default derived from `tokenizer` for HF case
)


def process_chunk(dl_doc):
    print("Begin chunking...")
    chunker = HybridChunker(
        tokenizer=tokenizer,
        merge_peers=True,  # optional, defaults to True
    )
    chunk_iter = chunker.chunk(dl_doc=dl_doc)
    chunks = list(chunk_iter)

    for i, chunk in enumerate(chunks):
        print(f"=== {i} ===")
        txt_tokens = tokenizer.count_tokens(chunk.text)
        print(f"chunk.text ({txt_tokens} tokens):\n{chunk.text!r}")

        ser_txt = chunker.contextualize(chunk=chunk)
        ser_tokens = tokenizer.count_tokens(ser_txt)
        print(f"chunker.contextualize(chunk) ({ser_tokens} tokens):\n{ser_txt!r}")

        # Compute embedding based on contextualized text
        try:
            embedding = embed_text(ser_txt)
        except Exception as e:
            print(f"Embedding failed for chunk {i}: {e}")
            embedding = None

        # Persist to Postgres
        try:
            inserted_id = insert_document_chunk(
                chunk_index=i,
                chunk_text=chunk.text,
                contextualized_text=ser_txt,
                chunk_tokens=txt_tokens,
                contextualized_tokens=ser_tokens,
                embedding=embedding,
            )
            if inserted_id is None:
                print(f"Insert skipped due to conflict for chunk_index={i}")
            else:
                print(f"Inserted chunk_id={inserted_id} for chunk_index={i}")
        except Exception as e:
            print(f"DB insert failed for chunk {i}: {e}")

        print()
