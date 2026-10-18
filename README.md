# 🚀 Local RAG Explainer: Retrieval-Augmented Generation from Scratch

An educational, beginner-friendly guide and modular codebase for learning **Retrieval-Augmented Generation (RAG)** hands-on.

**No cloud accounts required. No paid services necessary. Runs 100% locally on your laptop.**

---

## 📖 What is RAG? (In Simple Terms)

Large Language Models (like ChatGPT, Claude, or Gemini) are trained on massive amounts of public internet data. However, they have two big limitations:
1. **Knowledge Cutoffs**: They don't know information that was published after they were trained.
2. **Lack of Private Data**: They have never read your personal notes, company manuals, or research papers.
3. **Hallucinations**: When an LLM doesn't know the answer to a question, it might invent convincing facts that are completely wrong.

### The Analogy: Closed-Book vs. Open-Book Exam

* **Traditional LLM (Closed-Book Exam)**: You ask a student a detailed question about a specific chapter in a science book. The student must answer purely from memory. If they forgot or never read it, they might guess.
* **RAG (Open-Book Exam)**: Before answering, the student opens the exact textbook page, reads the relevant paragraphs, and writes down the answer based directly on the text.

**Retrieval-Augmented Generation** gives an LLM an open-book exam!

---

## 🧠 The 6 Core RAG Concepts Explained

```
┌────────────────────────────────────────────────────────────────────────┐
│                       INGESTION PIPELINE (Offline)                     │
│                                                                        │
│   Document ──► [1] Chunking ──► [2] Embeddings ──► [3] Vector Database │
│   (Long text)    (Split text)    (Words to vectors)  (ChromaDB in RAM) │
└────────────────────────────────────────────────────────────────────────┘
                                                              │
┌─────────────────────────────────────────────────────────────▼──────────┐
│                        QUERY PIPELINE (Runtime)                        │
│                                                                        │
│   User Question ──► [4] Embed Question ──► [5] Vector Search           │
│                                                   │ (Top-K Chunks)     │
│                                                   ▼                    │
│   Final Answer ◄─── [7] LLM Response ◄──── [6] Prompt Augmentation     │
└────────────────────────────────────────────────────────────────────────┘
```

### 1. Chunking (Splitting Text)
Embedding models have a strict size limit (often ~512 tokens). More importantly, if you feed a 50-page document into a search engine, the exact answer to a specific question will get lost. 

**Chunking** breaks long documents into small, manageable pieces (e.g., 400 characters) with a slight **overlap** (e.g., 50 characters). The overlap ensures that sentences crossing chunk boundaries don't lose their context.

### 2. Embeddings (Converting Meaning to Numbers)
Computers cannot directly compare the abstract meaning of words. An **embedding model** converts a piece of text into a list of numbers called a **vector** (for example, 384 numbers).

Sentences with similar meanings will point in roughly the same direction in mathematical space:
* `"Why is Mars called the Red Planet?"`
* `"Iron oxide on the surface gives Mars a reddish color."`  
  *(These two produce vectors that are mathematically very close!)*
* `"Espresso machines brew coffee with high pressure."`  
  *(This produces a vector pointing in a completely different direction).*

We measure this alignment using **Cosine Similarity**:
* **`1.0`** = Identical meaning
* **`0.0`** = Completely unrelated
* **`-1.0`** = Opposite meaning

### 3. Vector Database (ChromaDB)
A traditional SQL database looks for exact keyword matches. A **vector database** indexes embedding vectors so you can perform **semantic search** in milliseconds. In this project, we use **ChromaDB**, a lightweight, open-source vector store that runs entirely in your computer's RAM.

### 4. Similarity Search (Retrieval)
When a user asks a question:
1. The question is converted into an embedding vector.
2. The vector database calculates the cosine similarity between the question vector and all stored chunk vectors.
3. The top **$K$** chunks (e.g., the 3 most relevant paragraphs) are retrieved.

### 5. Prompt Augmentation
Instead of sending just the user's question to the LLM, we combine the retrieved chunks and the question into an **augmented prompt**:
```text
You are an expert assistant. Answer using ONLY the context provided below:

[Context Excerpt 1]
Mars appears red due to iron oxide...

[Context Excerpt 2]
Curiosity discovered evidence of ancient lakes...

Question: Why does Mars appear reddish?
Answer:
```

### 6. Grounded Generation
The LLM reads the context excerpts and writes a concise, accurate answer. Because the LLM is instructed to rely solely on the provided excerpts, hallucinations are drastically reduced.

---

## 📁 Project Structure

This project is organized into clean, modular files:

| File | Purpose |
|---|---|
| **[`01_rag_local_chromadb.py`](file:///Users/shashibetadur/Desktop/Agent_007/RAG%20local/01_rag_local_chromadb.py)** | Interactive, step-by-step explainer script demonstrating each concept with rich console visualizations. |
| **[`rag_pipeline.py`](file:///Users/shashibetadur/Desktop/Agent_007/RAG%20local/rag_pipeline.py)** | A reusable, modular Python class (`LocalChromaRAG`) ready to be imported into web apps or APIs. |
| **[`knowledge_base.txt`](file:///Users/shashibetadur/Desktop/Agent_007/RAG%20local/knowledge_base.txt)** | A sample educational document about Mars exploration (no domain-specific jargon). |
| **[`knowledge_base.py`](file:///Users/shashibetadur/Desktop/Agent_007/RAG%20local/knowledge_base.py)** | Helper module to load documents and access curated test questions. |
| **[`01_rag_local_chromadb.ipynb`](file:///Users/shashibetadur/Desktop/Agent_007/RAG%20local/01_rag_local_chromadb.ipynb)** | Jupyter notebook version for cell-by-cell interactive learning. |
| **[`output.md`](file:///Users/shashibetadur/Desktop/Agent_007/RAG%20local/output.md)** | Sample output format showing console logs and prompt copy-pasting for users without an API key. |
| **[`requirements.txt`](file:///Users/shashibetadur/Desktop/Agent_007/RAG%20local/requirements.txt)** | Python dependencies. |
| **[`.env.example`](file:///Users/shashibetadur/Desktop/Agent_007/RAG%20local/.env.example)** | Template for optional OpenAI API key. |

---

## ⚙️ Installation and Setup

### Prerequisites
* Python 3.9, 3.10, or 3.11 installed.

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/local-rag-chromadb.git
cd local-rag-chromadb
```

### 2. Create and Activate a Virtual Environment
* On macOS / Linux:
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```
* On Windows:
  ```powershell
  python -m venv venv
  venv\Scripts\activate
  ```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

> **Note:** The first time you run the script, `sentence-transformers` will automatically download the `all-MiniLM-L6-v2` embedding model (~90MB). This happens only once and is cached locally.

### 4. Optional: Configure OpenAI API Key
If you want the pipeline to automatically generate final answers using OpenAI:
```bash
cp .env.example .env
```
Open `.env` in a text editor and add your key:
```env
OPENAI_API_KEY=sk-your-openai-api-key
```

*(If you do not have an OpenAI API key, don't worry! The program will generate and print the full prompt in your terminal so you can copy and paste it into ChatGPT or Claude for free).*

---

## 💻 How to Use the Programs

Here are the commands to run and experiment with this project:

### 1. Run the Full Explainer Script
Runs the entire educational tutorial, showing chunking, embedding generation, vector indexing, retrieval match bars, and answer generation:
```bash
python 01_rag_local_chromadb.py
```

### 2. Ask a Custom Question
Ask your own question directly from the command line:
```bash
python 01_rag_local_chromadb.py --query "Where can water be found on Mars?"
```

### 3. Retrieve More or Fewer Chunks (Top-K)
Control how many relevant passages are retrieved using `-k` or `--top_k`:
```bash
python 01_rag_local_chromadb.py --query "What did Perseverance discover?" -k 2
```

### 4. Run the Cosine Similarity Math Demo
See how the model scores similarity between questions and related vs. unrelated sentences:
```bash
python 01_rag_local_chromadb.py --demo-similarity
```

### 5. Run Against Your Own Text File
You can point the pipeline to your own `.txt` document:
```bash
python 01_rag_local_chromadb.py --file path/to/my_notes.txt --query "What is the main topic?"
```

### 6. Interactive Learning with Jupyter Notebook
If you prefer a notebook interface:
```bash
jupyter notebook 01_rag_local_chromadb.ipynb
```
*(Or simply open `01_rag_local_chromadb.ipynb` in VS Code and select your Python environment).*

---

## 🧩 Using the Modular `LocalChromaRAG` Class in Your Own Code

You can easily import and reuse the pipeline in any Python application (such as FastAPI, Flask, or Streamlit):

```python
from rag_pipeline import LocalChromaRAG

# 1. Initialize the pipeline
rag = LocalChromaRAG(
    model_name="all-MiniLM-L6-v2", 
    collection_name="my_documents"
)

# 2. Add your text document
sample_text = """
The James Webb Space Telescope (JWST) is an infrared space observatory 
launched in December 2021. It orbits the Sun at the Second Lagrange Point (L2), 
roughly 1.5 million kilometers from Earth.
"""
rag.add_document(sample_text, document_id="jwst_doc", chunk_size=300, chunk_overlap=40)

# 3. Retrieve relevant chunks
matches = rag.search("Where does the JWST orbit?", k=2)
for match in matches:
    print(f"Similarity: {match['similarity']:.4f}")
    print(f"Content: {match['text']}\n")

# 4. Generate an answer (if OPENAI_API_KEY is configured in .env)
result = rag.query_and_answer("Where does the JWST orbit?")
print("Answer:", result["answer"])
```

---

## 🧪 Sample Questions to Test

Here are curated questions included in `knowledge_base.py` to test different aspects of RAG:

| Test Question | Target Concept | What It Demonstrates |
|---|---|---|
| *"Why is Mars called the Red Planet, and what is its atmosphere made of?"* | Surface iron oxide & 95% $CO_2$ | High-precision retrieval from introductory chunks. |
| *"Where does water exist on Mars today, and did it ever have liquid water?"* | Polar ice caps, subsoil ice, and dried riverbeds | Cross-sentence semantic matching. |
| *"What did the Curiosity and Perseverance rovers discover on Mars?"* | Gale Crater freshwater lakes & Jezero Crater cores | Multi-entity fact extraction. |
| *"What is Olympus Mons, and how does it compare to mountains on Earth?"* | Shield volcano 3x taller than Mt. Everest | Numerical and comparative retrieval. |
| *"What are the biggest challenges humans will face on a mission to Mars?"* | Radiation, microgravity, -60°C temperatures, dust storms | Extracting lists of requirements and hurdles. |
| *"What is the diameter of Jupiter's Great Red Spot?"* | Out of Context (Negative Test) | Verifies that the model states it **cannot find the answer** in context rather than hallucinating. |

---

## ❓ Frequently Asked Questions (FAQ)

<details>
<summary><b>1. Do I need a GPU to run this?</b></summary>
No. The default embedding model (<code>all-MiniLM-L6-v2</code>) is lightweight (~90MB) and runs smoothly on standard modern laptop CPUs.
</details>

<details>
<summary><b>2. Do I have to pay for an OpenAI API key?</b></summary>
No. If no API key is provided, the script outputs the complete prompt to your terminal. You can copy it into ChatGPT or Claude's free web interfaces.
</details>

<details>
<summary><b>3. Where is the data stored?</b></summary>
By default, <code>ChromaDB.EphemeralClient()</code> stores embeddings in RAM during execution. You can persist data to disk by passing <code>persist_directory="./chroma_db"</code> when initializing <code>LocalChromaRAG</code>.
</details>

<details>
<summary><b>4. Can I connect this to a local LLM like Ollama?</b></summary>
Yes! You can replace the OpenAI API call in <code>rag_pipeline.py</code> with a local request to <code>http://localhost:11434/api/generate</code> (Ollama) with zero changes to the chunking, embedding, or ChromaDB logic.
</details>

---

## 📄 License
This project is open-source under the MIT License. Feel free to use, modify, and share it for learning and projects!
