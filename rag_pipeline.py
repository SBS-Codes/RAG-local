"""
Modular Local RAG Pipeline using ChromaDB & Sentence-Transformers
=================================================================

This module provides a reusable, self-contained `LocalChromaRAG` class that
encapsulates:
  1. Document ingestion and chunking (via RecursiveCharacterTextSplitter)
  2. Local sentence embedding (via SentenceTransformer)
  3. Vector store indexing and semantic search (via ChromaDB)
  4. Prompt augmentation and answer generation (via OpenAI API or prompt formatting)
"""

import os
from typing import List, Dict, Any, Optional, Tuple
import chromadb
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


class LocalChromaRAG:
    """
    A lightweight, local-first RAG pipeline using ChromaDB and Sentence-Transformers.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        collection_name: str = "rag_collection",
        persist_directory: Optional[str] = None
    ):
        """
        Initializes embedding model and ChromaDB client.

        :param model_name: SentenceTransformers model name (default: 'all-MiniLM-L6-v2').
        :param collection_name: ChromaDB collection name.
        :param persist_directory: Path to persist ChromaDB to disk. If None, uses in-memory ephemeral store.
        """
        load_dotenv()
        self.model_name = model_name
        self.collection_name = collection_name
        self.persist_directory = persist_directory

        # Load embedding model
        self.embed_model = SentenceTransformer(self.model_name)

        # Initialize ChromaDB client (Ephemeral in RAM or Persistent on disk)
        if self.persist_directory:
            self.chroma_client = chromadb.PersistentClient(path=self.persist_directory)
        else:
            self.chroma_client = chromadb.EphemeralClient()

        # Get or create collection
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add_document(
        self,
        text: str,
        document_id: str = "doc_0",
        chunk_size: int = 400,
        chunk_overlap: int = 50,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Chunks text, computes embeddings, and stores them in ChromaDB.

        :return: Number of chunks indexed.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks = splitter.split_text(text)
        if not chunks:
            return 0

        # Compute embeddings
        embeddings = self.embed_model.encode(chunks, show_progress_bar=False)

        # Generate unique IDs and metadata per chunk
        ids = [f"{document_id}_chunk_{i}" for i in range(len(chunks))]
        base_meta = metadata or {}
        metadatas = [{**base_meta, "chunk_index": i, "doc_id": document_id} for i in range(len(chunks))]

        self.collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings.tolist(),
            metadatas=metadatas
        )
        return len(chunks)

    def search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """
        Performs vector similarity search against indexed chunks.

        :param query: Question or search query.
        :param k: Number of top matches to return.
        :return: List of dicts containing text, similarity score, and metadata.
        """
        q_vec = self.embed_model.encode(query).tolist()
        results = self.collection.query(
            query_embeddings=[q_vec],
            n_results=k,
            include=["documents", "metadatas", "distances"]
        )

        retrieved = []
        if results.get("documents") and len(results["documents"]) > 0:
            for i in range(len(results["documents"][0])):
                dist = results["distances"][0][i]
                sim = 1.0 - dist  # Cosine similarity
                retrieved.append({
                    "text": results["documents"][0][i],
                    "similarity": sim,
                    "metadata": results["metadatas"][0][i]
                })
        return retrieved

    def build_prompt(
        self,
        query: str,
        retrieved_items: List[Dict[str, Any]],
        system_instruction: Optional[str] = None
    ) -> Tuple[str, str]:
        """
        Formats retrieved chunks and question into system and user prompts.
        """
        system_prompt = system_instruction or (
            "You are an expert AI assistant. Answer the user's question "
            "using ONLY the context excerpts provided below. Be concise and specific. "
            "If the answer cannot be found in the context, clearly state that."
        )

        context = "\n\n".join(
            f"[Excerpt {i+1}]\n{item['text']}"
            for i, item in enumerate(retrieved_items)
        )

        user_prompt = f"""Context:

{context}

---
Question: {query}

Answer:"""
        return system_prompt, user_prompt

    def query_and_answer(
        self,
        query: str,
        k: int = 3,
        system_instruction: Optional[str] = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0
    ) -> Dict[str, Any]:
        """
        Executes end-to-end RAG:
          1. Retrieves top-k relevant chunks
          2. Formats augmented prompt
          3. Calls LLM if OPENAI_API_KEY is available
        """
        retrieved = self.search(query=query, k=k)
        system_prompt, user_prompt = self.build_prompt(
            query=query,
            retrieved_items=retrieved,
            system_instruction=system_instruction
        )

        openai_key = os.getenv("OPENAI_API_KEY")
        llm_response = None
        tokens_used = None

        if openai_key:
            from openai import OpenAI
            client = OpenAI(api_key=openai_key)
            response = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            llm_response = response.choices[0].message.content
            tokens_used = response.usage.total_tokens

        copy_paste_prompt = f"System Instruction:\n{system_prompt}\n\n{user_prompt}"

        return {
            "query": query,
            "retrieved_chunks": retrieved,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "formatted_prompt_for_clipboard": copy_paste_prompt,
            "answer": llm_response,
            "tokens_used": tokens_used
        }
