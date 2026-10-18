# 📝 RAG Pipeline — Reusable Code Templates

Copy-paste these self-contained code blocks into your own projects.
Each block handles one stage of the RAG pipeline and works independently.

---

## Table of Contents

1. [Stage 1: Chunking (Text Splitting)](#stage-1-chunking-text-splitting)
2. [Stage 2: Embeddings (Text to Vectors)](#stage-2-embeddings-text-to-vectors)
3. [Stage 3: Vector Store (ChromaDB)](#stage-3-vector-store-chromadb)
4. [Stage 4: Retrieval (Similarity Search)](#stage-4-retrieval-similarity-search)
5. [Stage 5: Prompt Augmentation](#stage-5-prompt-augmentation)
6. [Stage 6: LLM Call (Local GGUF Model)](#stage-6-llm-call-local-gguf-model)
7. [Stage 6 Alt: LLM Call (OpenAI API)](#stage-6-alt-llm-call-openai-api)
8. [Stage 6 Alt: LLM Call (LM Studio Server)](#stage-6-alt-llm-call-lm-studio-server)
9. [Full Pipeline (All Stages Combined)](#full-pipeline-all-stages-combined)

---

## Stage 1: Chunking (Text Splitting)

**What it does:** Breaks a long document into small overlapping pieces that fit
within an embedding model's token limit and improve search precision.

**When to use:** Whenever you ingest a new document into your RAG system.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_text(text: str, chunk_size: int = 400, chunk_overlap: int = 50) -> list[str]:
    """
    Splits text into overlapping chunks.

    Args:
        text:          The full document text.
        chunk_size:    Maximum characters per chunk (tune based on your embedding model).
        chunk_overlap: Characters of overlap between consecutive chunks.

    Returns:
        List of text chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_text(text)
    print(f"Created {len(chunks)} chunks (size={chunk_size}, overlap={chunk_overlap})")
    return chunks


# ── Usage ─────────────────────────────────────────────────────────────────────
document = open("my_document.txt").read()
chunks = chunk_text(document, chunk_size=400, chunk_overlap=50)

for i, chunk in enumerate(chunks):
    print(f"Chunk {i} ({len(chunk)} chars): {chunk[:60]}...")
```

---

## Stage 2: Embeddings (Text to Vectors)

**What it does:** Converts text into a list of numbers (a vector) that captures
the semantic meaning of the text. Similar sentences produce similar vectors.

**When to use:** After chunking, before storing in a vector database.

```python
from sentence_transformers import SentenceTransformer
import numpy as np


def load_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """Loads a local sentence embedding model (downloads once, ~90MB)."""
    print(f"Loading embedding model: {model_name}...")
    model = SentenceTransformer(model_name)
    print(f"✅ Model ready (vector size: {model.get_sentence_embedding_dimension()})")
    return model


def embed_texts(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    """
    Converts a list of text strings into embedding vectors.

    Returns:
        numpy array of shape (num_texts, vector_dimension)
    """
    embeddings = model.encode(texts, show_progress_bar=False)
    print(f"✅ Embedded {len(texts)} texts → shape {embeddings.shape}")
    return embeddings


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Computes cosine similarity between two vectors (1.0 = identical meaning)."""
    return float(np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b)))


# ── Usage ─────────────────────────────────────────────────────────────────────
model = load_embedding_model()

# Embed chunks from Stage 1
chunk_vectors = embed_texts(model, chunks)

# Compare two sentences
vec1 = model.encode("Why is Mars red?")
vec2 = model.encode("Iron oxide gives Mars its reddish color.")
print(f"Similarity: {cosine_similarity(vec1, vec2):.4f}")  # → ~0.78
```

---

## Stage 3: Vector Store (ChromaDB)

**What it does:** Stores text chunks alongside their embedding vectors in a
searchable database. ChromaDB runs in-memory (or on disk) with zero setup.

**When to use:** After embedding your chunks, to enable fast similarity search.

```python
import chromadb


def create_collection(name: str = "my_collection") -> chromadb.Collection:
    """Creates an in-memory ChromaDB collection with cosine similarity indexing."""
    client = chromadb.EphemeralClient()  # Use PersistentClient("./db") for disk
    try:
        client.delete_collection(name)
    except Exception:
        pass
    collection = client.create_collection(name=name, metadata={"hnsw:space": "cosine"})
    print(f"✅ Collection '{name}' created")
    return collection


def store_chunks(
    collection: chromadb.Collection,
    chunks: list[str],
    embeddings,  # numpy array
    doc_id: str = "doc_0",
) -> None:
    """Indexes chunks with their vectors and metadata into ChromaDB."""
    collection.add(
        ids=[f"{doc_id}_chunk_{i}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings.tolist(),
        metadatas=[{"doc_id": doc_id, "chunk_index": i} for i in range(len(chunks))],
    )
    print(f"✅ Stored {collection.count()} chunks in ChromaDB")


# ── Usage ─────────────────────────────────────────────────────────────────────
collection = create_collection("my_knowledge_base")
store_chunks(collection, chunks, chunk_vectors, doc_id="mars_doc")
```

---

## Stage 4: Retrieval (Similarity Search)

**What it does:** Takes a user question, embeds it, and finds the top-K most
semantically similar chunks from the vector store.

**When to use:** At query time, to find relevant evidence before calling the LLM.

```python
def retrieve_chunks(
    collection: chromadb.Collection,
    model: SentenceTransformer,
    query: str,
    k: int = 3,
) -> list[dict]:
    """
    Retrieves the top-k most similar chunks to the query.

    Returns:
        List of dicts with keys: 'text', 'similarity', 'metadata'
    """
    query_vector = model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    retrieved = []
    for i in range(len(results["documents"][0])):
        distance = results["distances"][0][i]
        similarity = 1.0 - distance  # ChromaDB cosine distance → similarity

        retrieved.append({
            "text": results["documents"][0][i],
            "similarity": round(similarity, 4),
            "metadata": results["metadatas"][0][i],
        })

    return retrieved


# ── Usage ─────────────────────────────────────────────────────────────────────
question = "Why is Mars called the Red Planet?"
matches = retrieve_chunks(collection, model, question, k=3)

for i, match in enumerate(matches):
    print(f"[Match {i+1}] Similarity: {match['similarity']}")
    print(f"  Text: {match['text'][:80]}...")
```

---

## Stage 5: Prompt Augmentation

**What it does:** Combines retrieved chunks and the user question into a
structured prompt that instructs the LLM to answer using only the provided context.

**When to use:** After retrieval, before sending to any LLM.

```python
def build_rag_prompt(
    question: str,
    retrieved_chunks: list[dict],
    system_instruction: str = None,
) -> tuple[str, str]:
    """
    Builds a system prompt and user prompt for RAG.

    Returns:
        (system_prompt, user_prompt) tuple ready for any LLM.
    """
    system_prompt = system_instruction or (
        "You are a helpful assistant. Answer the user's question using ONLY "
        "the context excerpts below. Be concise and factual. "
        "If the answer is not in the context, say so clearly."
    )

    context = "\n\n".join(
        f"[Excerpt {i+1}]\n{chunk['text']}"
        for i, chunk in enumerate(retrieved_chunks)
    )

    user_prompt = f"""Context:

{context}

---
Question: {question}

Answer:"""

    return system_prompt, user_prompt


# ── Usage ─────────────────────────────────────────────────────────────────────
system_prompt, user_prompt = build_rag_prompt(question, matches)

# Print for inspection or clipboard copy
print("SYSTEM:", system_prompt)
print("\nUSER:", user_prompt)
```

---

## Stage 6: LLM Call (Local GGUF Model)

**What it does:** Loads a GGUF model file directly using `llama-cpp-python` and
runs inference locally. No internet, no API keys, 100% private.

**When to use:** When you want fully offline, free inference.

**Install:** `pip install llama-cpp-python`

```python
from llama_cpp import Llama


def load_local_llm(
    model_path: str,
    n_ctx: int = 2048,
    verbose: bool = False,
) -> Llama:
    """Loads a local GGUF model into memory."""
    print(f"Loading model: {model_path.split('/')[-1]}...")
    llm = Llama(model_path=model_path, n_ctx=n_ctx, verbose=verbose)
    print("✅ Local LLM ready")
    return llm


def ask_local_llm(
    llm: Llama,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 400,
    temperature: float = 0.1,
) -> str:
    """Sends a chat completion request to the local model."""
    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response["choices"][0]["message"]["content"].strip()


# ── Usage ─────────────────────────────────────────────────────────────────────
MODEL_PATH = "/path/to/your/model.gguf"  # Update this path

llm = load_local_llm(MODEL_PATH)
answer = ask_local_llm(llm, system_prompt, user_prompt)
print(f"Answer: {answer}")
```

---

## Stage 6 Alt: LLM Call (OpenAI API)

**What it does:** Sends the augmented prompt to the OpenAI API.

**When to use:** When you have an API key and want cloud-hosted generation.

```python
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


def ask_openai(
    system_prompt: str,
    user_prompt: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    max_tokens: int = 400,
) -> dict:
    """Sends prompt to OpenAI and returns the answer and token usage."""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    return {
        "answer": response.choices[0].message.content,
        "tokens_used": response.usage.total_tokens,
    }


# ── Usage ─────────────────────────────────────────────────────────────────────
result = ask_openai(system_prompt, user_prompt)
print(f"Answer: {result['answer']}")
print(f"Tokens: {result['tokens_used']}")
```

---

## Stage 6 Alt: LLM Call (LM Studio Server)

**What it does:** Connects to LM Studio's local OpenAI-compatible API server.
Requires LM Studio to be running with "Start Server" enabled.

**When to use:** When you prefer managing models in LM Studio's GUI.

```python
from openai import OpenAI


def ask_lmstudio(
    system_prompt: str,
    user_prompt: str,
    base_url: str = "http://localhost:1234/v1",
    max_tokens: int = 400,
    temperature: float = 0.1,
) -> str:
    """Sends prompt to LM Studio's local API and returns the answer."""
    client = OpenAI(base_url=base_url, api_key="lm-studio")

    response = client.chat.completions.create(
        model="local-model",
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    return response.choices[0].message.content.strip()


# ── Usage ─────────────────────────────────────────────────────────────────────
# 1. Open LM Studio → load a model → click "Start Server"
# 2. Then run:
answer = ask_lmstudio(system_prompt, user_prompt)
print(f"Answer: {answer}")
```

---

## Full Pipeline (All Stages Combined)

A minimal end-to-end script that ties all stages together in ~40 lines:

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb

# ── CONFIG ────────────────────────────────────────────────────────────────────
DOCUMENT_PATH = "my_document.txt"
QUESTION      = "What is the main topic of this document?"
TOP_K         = 3
MODEL_PATH    = "/path/to/your/model.gguf"   # Set to None to skip LLM call

# ── STAGE 1: CHUNK ───────────────────────────────────────────────────────────
text = open(DOCUMENT_PATH).read()
chunks = RecursiveCharacterTextSplitter(
    chunk_size=400, chunk_overlap=50
).split_text(text)
print(f"[1] Chunked into {len(chunks)} pieces")

# ── STAGE 2: EMBED ───────────────────────────────────────────────────────────
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
chunk_vectors = embed_model.encode(chunks)
print(f"[2] Embedded → {chunk_vectors.shape}")

# ── STAGE 3: STORE ───────────────────────────────────────────────────────────
client = chromadb.EphemeralClient()
collection = client.get_or_create_collection("docs", metadata={"hnsw:space": "cosine"})
collection.add(
    ids=[f"c_{i}" for i in range(len(chunks))],
    documents=chunks,
    embeddings=chunk_vectors.tolist(),
)
print(f"[3] Stored {collection.count()} chunks")

# ── STAGE 4: RETRIEVE ────────────────────────────────────────────────────────
q_vec = embed_model.encode(QUESTION).tolist()
results = collection.query(query_embeddings=[q_vec], n_results=TOP_K,
                           include=["documents", "distances"])
docs = results["documents"][0]
scores = [round(1 - d, 4) for d in results["distances"][0]]
print(f"[4] Retrieved {TOP_K} chunks (scores: {scores})")

# ── STAGE 5: AUGMENT ─────────────────────────────────────────────────────────
context = "\n\n".join(f"[{i+1}] {d}" for i, d in enumerate(docs))
system_prompt = "Answer using ONLY the context. If not found, say so."
user_prompt = f"Context:\n{context}\n\nQuestion: {QUESTION}\nAnswer:"
print(f"[5] Prompt built ({len(user_prompt)} chars)")

# ── STAGE 6: GENERATE ────────────────────────────────────────────────────────
if MODEL_PATH:
    from llama_cpp import Llama
    llm = Llama(model_path=MODEL_PATH, n_ctx=2048, verbose=False)
    resp = llm.create_chat_completion(messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ], max_tokens=400)
    print(f"[6] Answer: {resp['choices'][0]['message']['content']}")
else:
    print(f"[6] No model path set. Copy prompt above into any LLM chat window.")
```
