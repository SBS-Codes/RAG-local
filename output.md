# 📋 Sample Execution Output (`output.md`)

This document demonstrates the exact terminal output format generated when running the **Local RAG Explainer** pipeline.

It highlights:
1. **Case A (No API Key)**: How the program formats and prints the complete augmented prompt so you can easily copy and paste it into any free LLM chat interface (ChatGPT, Claude, Gemini, etc.).
2. **Case B (With API Key)**: How the program automatically calls the LLM and prints the generated response and token usage.
3. **Experimental Output**: How semantic retrieval handles both fact-based questions and out-of-context questions (hallucination prevention).

---

## 🖥️ Full Console Output Walkthrough

```text
$ python 01_rag_local_chromadb.py
============================================================
🎓 RAG Explainer: Retrieval-Augmented Generation from Scratch
============================================================

[Step 1] Loading document (3321 characters)...
✅ Split into 10 chunks (size=400 chars, overlap=50 chars):
   • Chunk 0 (398 chars): The Exploration of Mars: The Red Planet, Water, and Robotic Missi...
   • Chunk 1 (388 chars): atmospheric pressure at the Martian surface averages less than 1%...
   • Chunk 2 (392 chars): Despite its dry conditions today, evidence gathered by space prob...
   • Chunk 3 (395 chars): Today, water on Mars exists primarily in the form of ice. Vast ic...
   • Chunk 4 (399 chars): Robotic explorers have revolutionized our understanding of Mars. ...
   • Chunk 5 (387 chars): Curiosity discovered chemical evidence of ancient freshwater lake...
   • Chunk 6 (391 chars): In February 2021, the Perseverance rover touched down in Jezero C...
   • Chunk 7 (396 chars): The geography of Mars features some of the most extreme geologic...
   • Chunk 8 (392 chars): Mars also has two small, irregularly shaped moons named Phobos a...
   • Chunk 9 (394 chars): Planning crewed expeditions to Mars presents severe engineering a...

[Step 2] Loading local embedding model (all-MiniLM-L6-v2)...
✅ Model loaded (384-dimensional dense vectors).

Generating vector embeddings for 10 chunks...
✅ Embeddings calculated. Array shape: (10, 384)

[Step 3] Storing chunks and vectors in local ChromaDB collection...
✅ Indexed 10 chunks in ChromaDB.

[Step 4 & 5] Performing Semantic Search in ChromaDB...
Target Question: "Why is Mars called the Red Planet, and what is its atmosphere made of?"
============================================================
Top 3 Retrieved Matches:
============================================================

[Match 1]  Similarity: 0.7842  ███████████████████████
Metadata:   {'chunk_index': 0, 'doc_id': 'mars_doc'}
Chunk Text: The Exploration of Mars: The Red Planet, Water, and Robotic Missions

Mars is the fourth planet from the Sun and the second-smallest planet in the Solar System. It is frequently referred to as the "Red Planet" because the iron oxide (rust) prevalent on its surface gives it a distinct reddish appearance. Mars possesses a very thin atmosphere composed of approximately 95% carbon dioxide

[Match 2]  Similarity: 0.6514  ███████████████████
Metadata:   {'chunk_index': 1, 'doc_id': 'mars_doc'}
Chunk Text: 95% carbon dioxide, 2.6% molecular nitrogen, 1.9% argon, and trace amounts of water vapor and oxygen. Due to this thin atmosphere, atmospheric pressure at the Martian surface averages less than 1% of Earth's pressure, making it impossible for liquid water to exist stably on the surface for extended periods.

[Match 3]  Similarity: 0.4120  ████████████
Metadata:   {'chunk_index': 7, 'doc_id': 'mars_doc'}
Chunk Text: The geography of Mars features some of the most extreme geological formations in the Solar System. Olympus Mons is an enormous shield volcano rising roughly 22 kilometers (nearly 72,000 feet) high, making it nearly three times taller than Mount Everest.

[Step 6] Building Augmented Prompt with retrieved context...

[Step 7] Generating grounded answer from LLM...
```

---

## 📌 Case A: When No API Key is Detected (Free Mode)

When no `OPENAI_API_KEY` is configured in your `.env` file, the program will display the complete input prompt formatted with clear delimiters, ready to copy and paste into any free web chat LLM:

```text
=================================================================
📢 NO API KEY DETECTED — COPY & PASTE PROMPT INTO ANY LLM CHAT WINDOW
=================================================================
You can test this RAG prompt for FREE! Copy everything between the
dashed lines and paste it into ChatGPT (chatgpt.com), Claude (claude.ai),
Gemini (gemini.google.com), or your favorite LLM chat window:

------------------------- 📋 CLIPBOARD COPY START -------------------------
System Instruction:
You are an expert scientific assistant. Answer the user's question using ONLY the context excerpts provided below. Be concise, direct, and factual.
If the answer cannot be found in the context excerpts, clearly state: 'I cannot find the answer to this question in the provided context.'

Context Excerpts:
[Excerpt 1]
The Exploration of Mars: The Red Planet, Water, and Robotic Missions

Mars is the fourth planet from the Sun and the second-smallest planet in the Solar System. It is frequently referred to as the "Red Planet" because the iron oxide (rust) prevalent on its surface gives it a distinct reddish appearance. Mars possesses a very thin atmosphere composed of approximately 95% carbon dioxide

[Excerpt 2]
95% carbon dioxide, 2.6% molecular nitrogen, 1.9% argon, and trace amounts of water vapor and oxygen. Due to this thin atmosphere, atmospheric pressure at the Martian surface averages less than 1% of Earth's pressure, making it impossible for liquid water to exist stably on the surface for extended periods.

[Excerpt 3]
The geography of Mars features some of the most extreme geological formations in the Solar System. Olympus Mons is an enormous shield volcano rising roughly 22 kilometers (nearly 72,000 feet) high, making it nearly three times taller than Mount Everest.

---
Question: Why is Mars called the Red Planet, and what is its atmosphere made of?

Answer:
-------------------------- 📋 CLIPBOARD COPY END --------------------------

💡 Tip: Notice how the prompt contains only the relevant excerpts,
preventing the model from guessing or making up facts!
=================================================================
```

### Expected Output when Pasted into an LLM Window (ChatGPT / Claude / Gemini):

```text
Mars is called the "Red Planet" because iron oxide (rust) on its surface gives it a reddish appearance. 

Its thin atmosphere is composed of:
- Approximately 95% carbon dioxide
- 2.6% molecular nitrogen
- 1.9% argon
- Trace amounts of water vapor and oxygen
```

---

## 🔑 Case B: When an OpenAI API Key IS Provided

If you add `OPENAI_API_KEY=sk-...` to your `.env` file, the program makes an automated API call and directly outputs the answer:

```text
============================================================
Q: Why is Mars called the Red Planet, and what is its atmosphere made of?
============================================================
A: Mars is called the "Red Planet" due to the prevalence of iron oxide (rust) on its surface, which gives it a distinct reddish appearance. Its atmosphere is very thin and primarily composed of approximately 95% carbon dioxide, 2.6% molecular nitrogen, 1.9% argon, and trace amounts of water vapor and oxygen.

(Tokens used: 312 — approx $0.000047)
```

---

## 🧪 Experiments: Testing Fact Retrieval & Hallucination Prevention

Below is the output generated by the experimental test suite in `01_rag_local_chromadb.py`:

### Experiment 1: Factual Retrieval
```text
Question: Where does water exist on Mars today, and did it ever have liquid water?
Retrieved 3 chunks (similarity scores: [0.7612, 0.7231, 0.4109])

📋 PROMPT READY TO PASTE INTO ANY LLM CHAT WINDOW:
┌────────────────────────────────────────────────────────────────┐
  Context:
  [1] Today, water on Mars exists primarily in the form of ice. Vast ice deposits are concentrated in the permanent polar ice caps and trapped beneath the shallow mantle of Martian soil...
  [2] Despite its dry conditions today, evidence gathered by space probes shows that ancient Mars was once much warmer and wetter. Massive outflow channels, dried lake basins, and river valley networks...
  [3] Robotic explorers have revolutionized our understanding of Mars...

  Question: Where does water exist on Mars today, and did it ever have liquid water?
  Answer:
└────────────────────────────────────────────────────────────────┘
💡 Paste this into ChatGPT, Claude, or Gemini to see the answer!
------------------------------------------------------------
```

### Experiment 2: Negative Hallucination Prevention Test
```text
Question: What is the diameter of Jupiter's Great Red Spot?
Retrieved 3 chunks (similarity scores: [0.3812, 0.3105, 0.2891])

📋 PROMPT READY TO PASTE INTO ANY LLM CHAT WINDOW:
┌────────────────────────────────────────────────────────────────┐
  Context:
  [1] The Exploration of Mars: The Red Planet, Water, and Robotic Missions...
  [2] The geography of Mars features some of the most extreme geological formations...
  [3] Planning crewed expeditions to Mars presents severe engineering and biological challenges...

  Question: What is the diameter of Jupiter's Great Red Spot?
  Answer:
└────────────────────────────────────────────────────────────────┘
💡 Paste this into ChatGPT, Claude, or Gemini to see the answer!
------------------------------------------------------------
```

**Expected LLM Response for Experiment 2:**
> *"I cannot find the answer to this question in the provided context."*

*(Notice how the similarity scores for Jupiter were very low (~0.3), and the strict prompt instruction prevented the model from hallucinating!)*
