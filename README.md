# 📄 Chat With Your Documents (RAG Chatbot)

A minimal, modular **Retrieval-Augmented Generation (RAG)** chatbot built with
Streamlit. Upload PDF, DOCX, or TXT files and ask questions grounded in their
content — the bot answers **only** from what you upload, and shows you exactly
which chunks it used.

Powered by:
- **[Groq](https://groq.com/)** for fast, free LLM inference (Llama 3.1/3.3, GPT-OSS)
- **[Sentence-Transformers](https://www.sbert.net/)** for local, free embeddings
- **[Chroma](https://www.trychroma.com/)** as an in-memory vector store

---

## How it works

```
upload → file_loader → chunker → vector_store → rag → llm → screen
```

1. **`file_loader.py`** — extracts raw text from an uploaded `.pdf`, `.docx`,
   or `.txt` file.
2. **`chunker.py`** — splits that text into overlapping, word-based chunks so
   retrieval can target focused passages instead of whole documents.
3. **`vector_store.py`** — embeds each chunk (via Sentence-Transformers) and
   stores it in an in-memory Chroma collection; also handles similarity search.
4. **`rag.py`** — the "Retrieve → Augment" step: takes the top matching chunks
   and assembles a grounded prompt (system instructions + chat history +
   context + question).
5. **`llm.py`** — sends that prompt to Groq and streams the answer back
   token-by-token.
6. **`app.py`** — the Streamlit UI that wires all of the above together, with
   sidebar controls for model, temperature, max tokens, and top-k retrieval.
7. **`config.py`** — every tunable setting (models, chunk size, top-k, system
   prompt) lives here so nothing is hard-coded elsewhere.

---

## Project structure

```
.
├── app.py             # Streamlit entry point — UI + wiring
├── config.py           # Central settings: models, chunk size, prompts
├── file_loader.py       # Bytes → text (PDF / DOCX / TXT)
├── chunker.py          # Text → overlapping word chunks
├── vector_store.py       # Chunk embeddings + similarity search (Chroma)
├── rag.py             # Builds the grounded prompt sent to the LLM
├── llm.py             # Groq client + streaming completion
├── .env               # GROQ_API_KEY (not committed with a real key)
└── README.md
```

---

## Features

- 📎 Upload multiple **PDF, DOCX, or TXT** files at once
- ✂️ Automatic chunking with configurable size/overlap
- 🔍 Semantic search over your documents (cosine similarity)
- 💬 Multi-turn chat with conversation history
- 🔧 Sidebar controls: model selection, temperature, max tokens, top-k
- 📚 "Sources used" panel showing exactly which chunks (and their relevance
  score) grounded each answer
- 🗑️ One-click "Clear documents & chat" to reset the session
- 🙅 Refuses to hallucinate — if the answer isn't in your documents, it says so

---

## Requirements

- Python 3.9+
- A free [Groq API key](https://console.groq.com/keys)

### Python packages

```
streamlit
python-dotenv
sentence-transformers
chromadb
groq
pypdf
python-docx
```

Install them with:

```bash
pip install streamlit python-dotenv sentence-transformers chromadb groq pypdf python-docx
```

*(If you have a `requirements.txt`, use `pip install -r requirements.txt` instead.)*

---

## Setup

1. **Clone or download this project.**

2. **Create a `.env` file** in the project root with your Groq API key:

   ```
   GROQ_API_KEY=your_key_here
   ```

3. **Install dependencies** (see above).

4. **Run the app:**

   ```bash
   streamlit run app.py
   ```

5. Open the URL Streamlit prints (usually `http://localhost:8501`).

---

## Usage

1. Upload one or more `.pdf`, `.docx`, or `.txt` files using the uploader.
2. Wait for the "Indexed N chunks from *filename*" confirmation.
3. Ask a question in the chat box at the bottom.
4. The assistant streams an answer grounded in your documents, and you can
   expand **"📚 Sources used"** under each reply to see which chunks (and
   their relevance scores) were used.
5. Adjust model, temperature, max answer length, or number of retrieved
   chunks (`top_k`) from the sidebar at any time.
6. Click **"🗑️ Clear documents & chat"** to wipe everything and start fresh.

---

## Configuration

All tunable values live in `config.py`:

| Setting | Purpose | Default |
|---|---|---|
| `GROQ_MODELS` | Models selectable in the sidebar | `llama-3.1-8b-instant`, `llama-3.3-70b-versatile`, `openai/gpt-oss-20b` |
| `DEFAULT_TEMPERATURE` | Default generation temperature | `0.2` |
| `DEFAULT_MAX_TOKENS` | Default max answer length | `512` |
| `EMBED_MODEL_NAME` | Local embedding model | `all-MiniLM-L6-v2` |
| `CHUNK_SIZE_WORDS` | Words per chunk | `120` |
| `CHUNK_OVERLAP_WORDS` | Overlap between chunks | `20` |
| `TOP_K` | Chunks retrieved per question | `4` |
| `COLLECTION_NAME` | Chroma collection name | `uploaded_docs` |
| `SYSTEM_PROMPT` | Instructions that ground the LLM in your docs | see `config.py` |

---

## Notes & limitations

- **In-memory storage:** the vector store uses Chroma's `EphemeralClient`, so
  all indexed documents are lost when the app restarts or you hit "Clear
  documents." To persist documents across runs, swap `EphemeralClient()` for
  `PersistentClient(path=...)` in `vector_store.py`.
- **Scanned PDFs:** if a PDF has no selectable text (i.e., it's a scanned
  image), `file_loader.py` will raise a clear error instead of silently
  returning nothing.
- **No API key:** if `GROQ_API_KEY` isn't set, the app shows an in-UI error
  and stops before the chat interface loads.

---

## License

Add your preferred license here (e.g., MIT).
