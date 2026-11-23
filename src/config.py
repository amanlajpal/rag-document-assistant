import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Project Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CHROMA_DIR = str(BASE_DIR / "chroma_db")

# Default Models
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
FALLBACK_GROQ_MODEL = "llama-3.1-8b-instant"
DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_HF_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Chunking Configuration
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200

# Retrieval Configuration
DEFAULT_TOP_K = 4
