"""
docchat/app.py - Chat With Your Documents (Mini-Project 1).

The Streamlit entry point. Notice how SHORT this file is: every hard job lives in
its own module, so app.py only does UI + wiring. That is the payoff of splitting
code into modules (lesson 03).

    upload -> file_loader -> chunker -> vector_store -> rag -> llm -> screen

Run it (from the Day-20 folder):
    streamlit run docchat/app.py

Setup - a .env in this folder (or the one you run from) with:
    GROQ_API_KEY=your_key_here
    """

import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import config
from file_loader import load_text
from chunker import chunk_text
from vector_store import VectorStore
from rag import build_messages
from llm import make_client, stream_answer

load_dotenv()

st.set_page_config(page_title="Chat With Your Documents", page_icon="📄")


@st.cache_resource
def get_client():
    return make_client()

@st.cache_resource
def get_embedder():
    return SentenceTransformer(config.EMBED_MODEL_NAME)

embedder = get_embedder()
client = make_client()

if "store" not in st.session_state:
    st.session_state.store = VectorStore(embedder)
store = st.session_state.store
if "messages" not in st.session_state:
    st.session_state.messages = []
if "indexed" not in st.session_state:
    st.session_state.indexed =[]

st.sidebar.header("⚙️ Model settings")
model = st.sidebar.selectbox("Model",config.GROQ_MODELS)
temperature = st.sidebar.slider(
    "Temperature",0.1,1.0,config.DEFAULT_TEMPERATURE,0.05,
    help = "Low = focused and factual. High = more creative.",
)
max_tokens = st.sidebar.slider(
    "Max answer length (tokens)", 128, 2048, config.DEFAULT_MAX_TOKENS, 64,
)
top_k = st.sidebar.slider(
    "Chunks to retrieve (k)", 1, 8, config.TOP_K,
    help="How many document snippets to feed the model per question.",
)

st.sidebar.divider()
# Wipe documents AND chat so the user can start fresh. reset() empties Chroma.
if st.sidebar.button("🗑️ Clear documents & chat", width="stretch"):
    store.reset()
    st.session_state.indexed = []
    st.session_state.messages = []
    st.rerun()   # rerun immediately so the cleared state is reflected at once.

st.sidebar.caption(f"Indexed chunks: {store.count()}")


# --- Header ------------------------------------------------------------------
st.title("📄 Chat With Your Documents")
st.write("Upload PDF, DOCX or TXT files, then ask questions grounded in them.")

# If there's no API key, explain and stop before the chat UI (Day 19 pattern).
if client is None:
    st.error(
        "No GROQ_API_KEY found. Create a .env file with "
        "`GROQ_API_KEY=your_key_here`, then restart the app."
    )
    st.stop()


# --- Upload + index ----------------------------------------------------------
uploads = st.file_uploader(
    "Upload documents",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True,
)

# Index any file we haven't indexed yet. We track filenames in session_state so
# a rerun (which re-hands us the same uploads) doesn't re-index them every time.
for uploaded in uploads or []:
    if uploaded.name in st.session_state.indexed:
        continue
    try:
        # 1. bytes -> text  (file_loader, which uses lessons 01-02)
        text = load_text(uploaded.name, uploaded.getvalue())
        # 2. text -> overlapping chunks  (chunker)
        chunks = chunk_text(text)
        # 3. chunks -> embeddings in Chroma  (vector_store, from Day 18)
        added = store.add_texts(chunks, source=uploaded.name)
        st.session_state.indexed.append(uploaded.name)
        st.success(f"Indexed {added} chunks from {uploaded.name}")
    except ValueError as err:
        # e.g. scanned PDF with no text, or an unsupported type.
        st.warning(str(err))

# Don't show the chat until there is something to chat about.
if store.count() == 0:
    st.info("Upload at least one document to start chatting.")
    st.stop()


# --- Chat --------------------------------------------------------------------
# Repaint the whole conversation on every rerun (Day 19 pattern).
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

question = st.chat_input("Ask something about your documents")

if question:
    # 1. Show + store the user's question.
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    # 2. RETRIEVE the most relevant chunks for this question (vector_store).
    matches = store.search(question, k=top_k)

    # 3. AUGMENT: build the grounded prompt from those chunks (rag).
    #    We pass the history WITHOUT the just-added question (it's added inside).
    history = st.session_state.messages[:-1]
    messages = build_messages(question, matches, history)

    # 4. GENERATE: stream Groq's answer into an assistant bubble (llm).
    with st.chat_message("assistant"):
        reply = st.write_stream(
            stream_answer(client, model, messages, temperature, max_tokens)
        )
        # Show exactly which chunks grounded the answer - honest, debuggable RAG.
        with st.expander("📚 Sources used"):
            for i, m in enumerate(matches, 1):
                st.markdown(
                    f"**Source {i}** · `{m['source']}` · "
                    f"relevance {m['similarity']:.2f}"
                )
                st.caption(m["document"])

    # 5. Save the reply so it survives the next rerun.
    st.session_state.messages.append({"role": "assistant", "content": reply})


