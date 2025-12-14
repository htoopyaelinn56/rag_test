from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import AutoTokenizer, AutoModel
import torch

from db_service import insert_document_chunk

EMBED_MODEL_ID = "/Volumes/HPLSSD/ai_models/embeddinggemma"
MAX_TOKENS = 512

# Initialize tokenizer and model once
_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_ID)
_model = AutoModel.from_pretrained(EMBED_MODEL_ID)
_model.eval()

# If available, use MPS/GPU for speed; otherwise CPU
_device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
_model.to(_device)

tokenizer = HuggingFaceTokenizer(
    tokenizer=_tokenizer,
    max_tokens=MAX_TOKENS,  # optional, by default derived from `tokenizer` for HF case
)


def _embed_text(text: str):
    """Compute a 768-d embedding for the given text using the HF model; mean-pool last hidden state."""
    # Tokenize with truncation to MAX_TOKENS
    inputs = _tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_TOKENS,
        padding=False,
    )
    inputs = {k: v.to(_device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = _model(**inputs)
        # outputs.last_hidden_state: [batch, seq_len, hidden]
        last_hidden = outputs.last_hidden_state  # shape [1, L, H]
        # Mean pool over tokens (exclude padding since we used no padding)
        emb = last_hidden.mean(dim=1).squeeze(0)  # shape [H]
        # Move to CPU and convert to python list of floats
        vec = emb.detach().cpu().numpy().astype(float).tolist()
        return vec


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
            embedding = _embed_text(ser_txt)
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
