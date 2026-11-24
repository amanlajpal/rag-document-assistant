import os
import sys
import uuid
import importlib
import streamlit as st
from dotenv import load_dotenv

import src.vector_store
importlib.reload(src.vector_store)
from src.vector_store import get_embedding_function, VectorStoreManager
from src.document_processor import extract_text_from_pdf_stream, chunk_documents
from src.agentic_rag import AgenticRAGAssistant
from src.workflow_rag import WorkflowRAGAssistant
from src.config import (
    DEFAULT_GROQ_MODEL,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_TOP_K,
    DEFAULT_CHROMA_DIR,
)

from src.llm_factory import fetch_available_groq_models, AVAILABLE_GROQ_MODELS

# Load local environment variables if available
load_dotenv()

# Streamlit Page Setup
st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        font-size: 0.8rem;
        font-weight: 600;
        border-radius: 0.375rem;
        background-color: #E0E7FF;
        color: #3730A3;
        margin-right: 0.5rem;
    }
    .source-box {
        background-color: #F8FAFC;
        border-left: 3px solid #3B82F6;
        padding: 0.75rem;
        margin-bottom: 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize Session State
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_manager" not in st.session_state or not hasattr(st.session_state.vector_manager, "get_indexed_files_summary"):
    st.session_state.vector_manager = VectorStoreManager(persist_directory=DEFAULT_CHROMA_DIR)

if "processed_files" not in st.session_state:
    st.session_state.processed_files = []


# ==========================================
# SIDEBAR: Configuration & Document Ingestion
# ==========================================
with st.sidebar:
    st.title("⚙️ Configuration")

    # API Keys Section
    with st.expander("🔑 API Credentials", expanded=True):
        groq_api_key = st.text_input(
            "LLM API Key (Groq or xAI Grok)",
            type="password",
            value=os.getenv("GROQ_API_KEY", ""),
            help="Free Groq key (starts with gsk_ from https://console.groq.com) or xAI Grok key (starts with xai-)",
        )

        active_key = groq_api_key.strip()
        if active_key.startswith("xai-"):
            st.info("🤖 Detected **xAI (Grok)** API Key")
            model_options = ["grok-beta", "grok-2-latest", "grok-vision-beta"]
            selected_model = st.selectbox("xAI Model:", model_options, index=0)
        elif active_key.startswith("gsk_"):
            st.caption("⚡ Detected **Groq (console.groq.com)** API Key")
            discovered_models = fetch_available_groq_models(active_key)
            selected_model = st.selectbox("Groq Model:", discovered_models, index=0)
            custom_model = st.text_input(
                "Or enter custom model ID (optional):",
                value="",
                placeholder="e.g. openai/gpt-oss-120b",
                help="Type any exact model ID from your Groq console if desired.",
            )
            if custom_model.strip():
                selected_model = custom_model.strip()
        else:
            if active_key:
                st.warning("⚠️ Note: Groq keys start with `gsk_` from console.groq.com; xAI Grok keys start with `xai-`.")
            selected_model = st.selectbox("Groq Model:", AVAILABLE_GROQ_MODELS, index=0)

        openai_api_key = st.text_input(
            "OpenAI API Key (Optional)",
            type="password",
            value=os.getenv("OPENAI_API_KEY", ""),
            help="Required only if using OpenAI Embeddings (text-embedding-3-small)",
        )

    # Architecture Mode Selector (Both resume bullet points)
    st.subheader("🧠 RAG Architecture Mode")
    rag_mode = st.radio(
        "Select Implementation:",
        [
            "Agentic RAG (Tool-Calling Agent)",
            "LangGraph 3-Stage Pipeline",
        ],
        help=(
            "Agentic RAG uses an LLM agent with retrieval as a tool and in-memory checkpoints.\n\n"
            "LangGraph 3-Stage Pipeline executes explicit stages: Retrieval -> Context Prep -> Generation."
        ),
    )

    # Embedding Provider Selector
    embedding_choice = st.selectbox(
        "Embedding Provider:",
        ["OpenAI (text-embedding-3-small)", "HuggingFace (Free / Local)"],
        index=0,
        help="OpenAI matches the resume specification. HuggingFace provides a 100% free offline alternative.",
    )
    provider_key = "openai" if "OpenAI" in embedding_choice else "huggingface"

    st.divider()

    # Document Upload Section
    st.subheader("📄 Upload PDF Documents")
    uploaded_files = st.file_uploader(
        "Choose PDF file(s):",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or multiple PDF documents to index into Chroma vector database.",
    )

    if uploaded_files:
        if st.button("🚀 Process & Index PDFs", type="primary", use_container_width=True):
            if provider_key == "openai" and not (openai_api_key or os.getenv("OPENAI_API_KEY")):
                st.error("⚠️ Please provide an OpenAI API Key or switch to HuggingFace embeddings.")
            else:
                with st.spinner("Extracting text and chunking documents..."):
                    all_chunks = []
                    for uploaded_file in uploaded_files:
                        docs = extract_text_from_pdf_stream(uploaded_file)
                        chunks = chunk_documents(
                            docs,
                            chunk_size=DEFAULT_CHUNK_SIZE,
                            chunk_overlap=DEFAULT_CHUNK_OVERLAP,
                        )
                        all_chunks.extend(chunks)

                    st.info(f"Generated {len(all_chunks)} chunks from {len(uploaded_files)} file(s).")

                    # Initialize Vector Store
                    emb_fn = get_embedding_function(
                        provider=provider_key,
                        openai_api_key=openai_api_key,
                    )
                    st.session_state.vector_manager.initialize_store(embedding_function=emb_fn)
                    st.session_state.vector_manager.add_documents(all_chunks)

                    st.session_state.processed_files = [f.name for f in uploaded_files]
                    st.success(f"✅ Successfully indexed {len(all_chunks)} chunks into Chroma DB!")

    # Vector Store & Persistent File Status
    st.divider()
    # Ensure vector store is initialized to read disk contents
    if not st.session_state.vector_manager.vector_store:
        try:
            emb_fn = get_embedding_function(provider=provider_key, openai_api_key=openai_api_key)
            st.session_state.vector_manager.initialize_store(embedding_function=emb_fn)
        except Exception:
            pass

    doc_count = st.session_state.vector_manager.count_documents()
    st.write(f"📊 **Total Chunks in Chroma DB:** `{doc_count}`")

    # Persistent files stored on disk
    files_summary = st.session_state.vector_manager.get_indexed_files_summary()
    if files_summary:
        st.write("📁 **Files Stored on Disk:**")
        for fn, info in list(files_summary.items()):
            f_col1, f_col2 = st.columns([3, 1])
            with f_col1:
                st.caption(f"• **{fn}** ({info['chunks']} chunks)")
            with f_col2:
                if st.button("❌", key=f"del_{fn}", help=f"Delete {fn} from Chroma DB"):
                    st.session_state.vector_manager.delete_file(fn)
                    st.rerun()
    else:
        st.caption("No documents currently stored in Chroma DB.")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Reset Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.thread_id = str(uuid.uuid4())
            st.rerun()

    with col2:
        if st.button("🗑️ Reset All DB", use_container_width=True):
            st.session_state.vector_manager.reset_store()
            st.session_state.processed_files = []
            st.session_state.messages = []
            st.session_state.thread_id = str(uuid.uuid4())
            st.success("Entire database wiped.")
            st.rerun()


# ==========================================
# MAIN CHAT INTERFACE
# ==========================================
st.markdown('<div class="main-header">📚 RAG Document Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">'
    '<span class="badge">Python</span>'
    '<span class="badge">LangChain</span>'
    '<span class="badge">LangGraph</span>'
    '<span class="badge">OpenAI Embeddings</span>'
    '<span class="badge">Chroma</span>'
    '<span class="badge">Groq</span>'
    '<span class="badge">Streamlit</span>'
    '<br><br>Ask questions about your uploaded PDFs with semantic search, persistent vector storage, and conversational context.'
    '</div>',
    unsafe_allow_html=True,
)

# Chroma DB Inspector
with st.expander("🔍 Inspect Chroma DB Records (What is Stored?)", expanded=False):
    stored_chunks = st.session_state.vector_manager.get_all_chunks(limit=50)
    if not stored_chunks:
        st.info("Chroma DB is currently empty. Upload and index PDF files above to see stored chunks.")
    else:
        st.write(f"Showing up to **50** stored chunks across **{len(files_summary)}** document(s):")
        for i, chk in enumerate(stored_chunks):
            meta = chk.get("metadata", {})
            st.markdown(
                f"**#{i+1} | Source:** `{meta.get('source', 'unknown')}` | **Page:** `{meta.get('page', 1)}`\n\n"
                f"> {chk.get('content', '')[:250]}..."
            )
            st.divider()

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("🔍 Retrieved Sources & Citations"):
                seen_snippets = set()
                for src in message["sources"]:
                    if isinstance(src, dict):
                        snippet = src.get("snippet", "").strip()
                        if snippet in seen_snippets:
                            continue
                        seen_snippets.add(snippet)
                        st.markdown(
                            f"**File:** `{src.get('source')}` | **Page:** `{src.get('page')}`\n\n"
                            f"> {snippet}"
                        )
                    else:
                        clean_text = str(src).strip()
                        if clean_text in seen_snippets:
                            continue
                        seen_snippets.add(clean_text)
                        st.markdown(f"> {clean_text}")

# Handle User Input
user_query = st.chat_input("Ask a question about your documents...")

if user_query:
    active_groq_key = groq_api_key or os.getenv("GROQ_API_KEY")
    if not active_groq_key:
        st.warning("⚠️ Please provide a Groq API Key in the sidebar to enable answers.")
        st.stop()

    # Append user question to UI
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Initialize Assistant based on selected Mode
    with st.chat_message("assistant"):
        with st.spinner("Analyzing documents & generating answer..."):
            try:
                # Ensure vector store is initialized
                if not st.session_state.vector_manager.vector_store:
                    emb_fn = get_embedding_function(
                        provider=provider_key,
                        openai_api_key=openai_api_key,
                    )
                    st.session_state.vector_manager.initialize_store(embedding_function=emb_fn)

                retriever = st.session_state.vector_manager.get_retriever(k=DEFAULT_TOP_K)

                if "Agentic RAG" in rag_mode:
                    assistant = AgenticRAGAssistant(
                        retriever=retriever,
                        groq_api_key=active_groq_key,
                        model_name=selected_model,
                    )
                    response_data = assistant.query(
                        user_message=user_query,
                        thread_id=st.session_state.thread_id,
                    )
                    answer = response_data["answer"]
                    sources = response_data.get("sources", [])
                else:
                    # LangGraph 3-stage pipeline
                    assistant = WorkflowRAGAssistant(
                        retriever=retriever,
                        groq_api_key=active_groq_key,
                        model_name=selected_model,
                    )
                    response_data = assistant.query(
                        user_message=user_query,
                        thread_id=st.session_state.thread_id,
                    )
                    answer = response_data["answer"]
                    sources = response_data.get("sources", [])

                st.markdown(answer)

                if sources:
                    with st.expander("🔍 Retrieved Sources & Citations"):
                        seen_snippets = set()
                        for src in sources:
                            if isinstance(src, dict):
                                snippet = src.get("snippet", "").strip()
                                if snippet in seen_snippets:
                                    continue
                                seen_snippets.add(snippet)
                                st.markdown(
                                    f"**File:** `{src.get('source')}` | **Page:** `{src.get('page')}`\n\n"
                                    f"> {snippet}"
                                )
                            else:
                                clean_text = str(src).strip()
                                if clean_text in seen_snippets:
                                    continue
                                seen_snippets.add(clean_text)
                                st.markdown(f"> {clean_text}")

                # Save assistant response to session state
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    }
                )

            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
