"""
RAG Concepts — Local First with ChromaDB
========================================

Learn Retrieval-Augmented Generation (RAG) from scratch on your local machine:
No credentials. No cloud services. Runs 100% locally.

Pipeline Architecture:
  Source Text ──► [1] Chunk  ──► [2] Embed  ──► [3] Store in ChromaDB
                                                         │
  Question    ──► [4] Embed  ──► [5] Similarity Search ──┘
                                         │
                                [6] Augment Prompt  ──►  [7] LLM  ──►  Answer
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb

from knowledge_base import load_knowledge_base, SAMPLE_QUESTIONS, TEXT_FILE_PATH


# ==============================================================================
# Step 1: Chunking Logic
# ==============================================================================
def split_document(text: str, chunk_size: int = 400, chunk_overlap: int = 50) -> List[str]:
    """
    Splits source text into overlapping character chunks.
    
    Why Chunking is needed:
    - Embedding models have maximum token limits (~512 tokens).
    - A focused chunk matches a specific user question with much higher precision
      than an entire document where the answer is buried across multiple pages.
    - Chunk overlap (e.g. 50 characters) ensures sentences split across boundaries
      do not lose their semantic context.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_text(text)


# ==============================================================================
# Step 2: Embedding Logic & Cosine Similarity
# ==============================================================================
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calculates cosine similarity between two 1D vectors:
      1.0  = identical direction (highest semantic similarity)
      0.0  = perpendicular (unrelated concepts)
     -1.0  = opposite direction
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def demonstrate_similarity(embed_model: SentenceTransformer) -> None:
    """
    Demonstrates how semantic similarity works using vector math:
    The model computes cosine similarity to measure how closely related two sentences are.
    """
    sentences = [
        "Why is Mars called the Red Planet?",                          # Target question
        "Iron oxide on the Martian surface gives it a reddish color.",  # Highly relevant chunk
        "Valles Marineris is an immense system of canyons.",           # Related topic (Mars geography)
        "Espresso machines brew coffee under high pressure.",          # Completely unrelated topic
        "Curiosity discovered evidence of ancient freshwater lakes."    # Different Mars topic (rovers)
    ]

    question = sentences[0]
    q_vec = embed_model.encode(question)

    print("\n" + "=" * 65)
    print("LEARNER DEMO: How Cosine Similarity Measures Meaning")
    print(f"Reference question: \"{question}\"")
    print("=" * 65)

    for s in sentences[1:]:
        s_vec = embed_model.encode(s)
        score = cosine_similarity(q_vec, s_vec)
        bar = "█" * int(max(0, score) * 30)
        print(f"  Similarity: {score:.4f}  {bar}")
        print(f"  Sentence:   \"{s}\"")
        print()


# ==============================================================================
# Step 3: ChromaDB Vector Store Setup & Population
# ==============================================================================
def initialize_chromadb(collection_name: str = "mars_knowledge_base") -> chromadb.Collection:
    """
    Initializes an ephemeral in-memory ChromaDB collection.
    EphemeralClient lives in RAM during script execution.
    """
    chroma_client = chromadb.EphemeralClient()
    try:
        chroma_client.delete_collection(collection_name)
    except Exception:
        pass

    collection = chroma_client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}  # Index using cosine distance
    )
    return collection


def index_chunks(
    collection: chromadb.Collection,
    chunks: List[str],
    chunk_embeddings: np.ndarray,
    doc_id: str = "mars_doc"
) -> None:
    """Stores text chunks, their numerical vectors, and metadata in ChromaDB."""
    ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"doc_id": doc_id, "chunk_index": i} for i in range(len(chunks))]

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=chunk_embeddings.tolist(),
        metadatas=metadatas
    )


# ==============================================================================
# Step 4 & 5: Retrieval / Similarity Search
# ==============================================================================
def search_similar_chunks(
    collection: chromadb.Collection,
    embed_model: SentenceTransformer,
    query: str,
    k: int = 3
) -> List[Dict[str, Any]]:
    """
    Steps 4 & 5:
      1. Embeds the user question into a vector.
      2. Queries ChromaDB for the closest 'k' vectors.
      3. Converts distance into a similarity score (similarity = 1 - distance).
    """
    q_vec = embed_model.encode(query).tolist()
    results = collection.query(
        query_embeddings=[q_vec],
        n_results=k,
        include=["documents", "metadatas", "distances"]
    )

    retrieved = []
    for i in range(len(results["documents"][0])):
        dist = results["distances"][0][i]
        similarity = 1.0 - dist  # ChromaDB cosine distance = 1 - cosine_similarity
        doc = results["documents"][0][i]
        meta = results["metadatas"][0][i]
        retrieved.append({
            "text": doc,
            "similarity": similarity,
            "metadata": meta
        })
    return retrieved


# ==============================================================================
# Step 6: Augment Prompt
# ==============================================================================
def build_augmented_prompt(retrieved_chunks: List[str], question: str) -> Tuple[str, str]:
    """
    Step 6: Prompt Augmentation.
    Injects retrieved chunks as authoritative context into the prompt, grounding the LLM.
    """
    context = "\n\n".join(
        f"[Excerpt {i+1}]\n{text}"
        for i, text in enumerate(retrieved_chunks)
    )

    system_prompt = (
        "You are an expert scientific assistant. Answer the user's question "
        "using ONLY the context excerpts provided below. Be concise, direct, and factual.\n"
        "If the answer cannot be found in the context excerpts, clearly state: "
        "'I cannot find the answer to this question in the provided context.'"
    )

    user_prompt = f"""Context Excerpts:
{context}

---
Question: {question}

Answer:"""

    return system_prompt, user_prompt


# ==============================================================================
# Step 7: Call LLM (or fallback)
# ==============================================================================
def generate_answer(system_prompt: str, user_prompt: str, question: str) -> None:
    """
    Step 7: Generation.
    Sends prompt to OpenAI if OPENAI_API_KEY is available.
    Otherwise prints the prompt so the learner can paste it into ChatGPT or Claude.
    """
    load_dotenv()
    openai_key = os.getenv("OPENAI_API_KEY")

    if openai_key:
        from openai import OpenAI

        client = OpenAI(api_key=openai_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,  # 0 ensures focused, deterministic responses
            max_tokens=400,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        answer = response.choices[0].message.content
        tokens = response.usage.total_tokens

        print("=" * 60)
        print(f"Q: {question}")
        print("=" * 60)
        print(f"A: {answer}")
        print(f"\n(Tokens used: {tokens} — approx ${tokens * 0.00000015:.6f})")
    else:
        print("\n" + "=" * 65)
        print("📢 NO API KEY DETECTED — COPY & PASTE PROMPT INTO ANY LLM CHAT WINDOW")
        print("=" * 65)
        print("You can test this RAG prompt for FREE! Copy everything between the")
        print("dashed lines and paste it into ChatGPT (chatgpt.com), Claude (claude.ai),")
        print("Gemini (gemini.google.com), or your favorite LLM chat window:\n")
        print("------------------------- 📋 CLIPBOARD COPY START -------------------------")
        print(f"System Instruction:\n{system_prompt}\n")
        print(user_prompt)
        print("-------------------------- 📋 CLIPBOARD COPY END --------------------------")
        print("\n💡 Tip: Notice how the prompt contains only the relevant excerpts,")
        print("preventing the model from guessing or making up facts!")
        print("=" * 65)


# ==============================================================================
# Helper to Run Full Pipeline End-to-End
# ==============================================================================
def ask(
    question: str,
    embed_model: SentenceTransformer,
    collection: chromadb.Collection,
    k: int = 3
) -> None:
    """Demonstrates the full end-to-end question answering pipeline."""
    load_dotenv()
    openai_key = os.getenv("OPENAI_API_KEY")

    # 1. Embed question
    q_vec = embed_model.encode(question).tolist()

    # 2. Retrieve top-k chunks
    res = collection.query(
        query_embeddings=[q_vec],
        n_results=k,
        include=["documents", "distances"]
    )
    docs = res["documents"][0]
    scores = [round(1.0 - d, 4) for d in res["distances"][0]]

    # 3. Print retrieved matches
    print(f"\nQuestion: {question}")
    print(f"Retrieved {k} chunks (similarity scores: {scores})")

    # 4. Augment prompt
    context = "\n\n".join(f"[{i+1}] {t}" for i, t in enumerate(docs))
    user_msg = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"

    # 5. Generate answer
    if openai_key:
        from openai import OpenAI
        r = OpenAI(api_key=openai_key).chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            max_tokens=300,
            messages=[
                {"role": "system", "content": "Answer using only the context provided. If not in context, state so."},
                {"role": "user", "content": user_msg}
            ]
        )
        print(f"\nAnswer: {r.choices[0].message.content}")
    else:
        print("\n📋 PROMPT READY TO PASTE INTO ANY LLM CHAT WINDOW:")
        print("┌" + "─" * 64 + "┐")
        for line in user_msg.splitlines():
            print(f"  {line}")
        print("└" + "─" * 64 + "┘")
        print("💡 Paste this into ChatGPT, Claude, or Gemini to see the answer!")

    print("-" * 60)


# ==============================================================================
# Main Interactive / CLI Runner
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Local RAG Explainer: Learn Chunking, Embeddings, ChromaDB, and Retrieval."
    )
    parser.add_argument(
        "--file", "-f",
        type=str,
        default=None,
        help="Path to a custom .txt file (defaults to knowledge_base.txt)."
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        default=None,
        help="Custom question to test with the RAG pipeline."
    )
    parser.add_argument(
        "--top_k", "-k",
        type=int,
        default=3,
        help="Number of chunks to retrieve (default: 3)."
    )
    parser.add_argument(
        "--demo-similarity",
        action="store_true",
        help="Run the mathematical cosine similarity demo."
    )
    args = parser.parse_args()

    print("=" * 60)
    print("🎓 RAG Explainer: Retrieval-Augmented Generation from Scratch")
    print("=" * 60)

    # ── Step 1: Chunking ──────────────────────────────────────────────────────
    doc_path = Path(args.file) if args.file else TEXT_FILE_PATH
    document_text = load_knowledge_base(doc_path)
    print(f"\n[Step 1] Loading document ({len(document_text)} characters)...")
    chunks = split_document(document_text, chunk_size=400, chunk_overlap=50)
    print(f"✅ Split into {len(chunks)} chunks (size=400 chars, overlap=50 chars):")
    for i, chunk in enumerate(chunks):
        first_line = chunk.strip().split("\n")[0]
        print(f"   • Chunk {i} ({len(chunk)} chars): {first_line[:65]}...")

    # ── Step 2: Embedding ─────────────────────────────────────────────────────
    print("\n[Step 2] Loading local embedding model (all-MiniLM-L6-v2)...")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    print("✅ Model loaded (384-dimensional dense vectors).")

    if args.demo_similarity:
        demonstrate_similarity(embed_model)

    print(f"\nGenerating vector embeddings for {len(chunks)} chunks...")
    chunk_embeddings = embed_model.encode(chunks, show_progress_bar=False)
    print(f"✅ Embeddings calculated. Array shape: {chunk_embeddings.shape}")

    # ── Step 3: Indexing in ChromaDB ──────────────────────────────────────────
    print("\n[Step 3] Storing chunks and vectors in local ChromaDB collection...")
    collection = initialize_chromadb("mars_knowledge_base")
    index_chunks(collection, chunks, chunk_embeddings, doc_id="mars_doc")
    print(f"✅ Indexed {collection.count()} chunks in ChromaDB.")

    # ── Step 4 & 5: Retrieval ─────────────────────────────────────────────────
    user_question = args.query if args.query else SAMPLE_QUESTIONS[0]["question"]
    k = args.top_k

    print("\n[Step 4 & 5] Performing Semantic Search in ChromaDB...")
    print(f"Target Question: \"{user_question}\"")
    results = search_similar_chunks(collection, embed_model, user_question, k=k)

    print("=" * 60)
    print(f"Top {k} Retrieved Matches:")
    print("=" * 60)
    retrieved_texts = []
    for i, res in enumerate(results):
        sim = res["similarity"]
        text = res["text"]
        meta = res["metadata"]
        retrieved_texts.append(text)
        bar = "█" * int(max(0, sim) * 30)
        print(f"\n[Match {i+1}]  Similarity: {sim:.4f}  {bar}")
        print(f"Metadata:   {meta}")
        print(f"Chunk Text: {text.strip()}")

    # ── Step 6: Augment Prompt ────────────────────────────────────────────────
    print("\n[Step 6] Building Augmented Prompt with retrieved context...")
    system_prompt, user_prompt = build_augmented_prompt(retrieved_texts, user_question)

    # ── Step 7: Generation ────────────────────────────────────────────────────
    print("\n[Step 7] Generating grounded answer from LLM...")
    generate_answer(system_prompt, user_prompt, user_question)

    # ── Multi-question experiments ────────────────────────────────────────────
    if not args.query:
        print("\n" + "=" * 60)
        print("🧪 RUNNING EXPERIMENTAL QUESTIONS FROM KNOWLEDGE BASE")
        print("=" * 60)
        for item in SAMPLE_QUESTIONS[1:]:
            print(f"\nTopic: {item['topic']}")
            ask(item["question"], embed_model, collection, k=k)


if __name__ == "__main__":
    main()
