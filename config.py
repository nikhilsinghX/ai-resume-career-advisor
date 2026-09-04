"""
Central configuration for the AI Resume & Career Advisor.
Loads environment variables and exposes shared constants.
"""
import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gemini-3.6-flash")
EMBED_MODEL = os.getenv("EMBED_MODEL", "models/gemini-embedding-001")

# RAG chunking parameters
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
RETRIEVER_TOP_K = 4

# Where the FAISS index is cached on disk between runs
VECTOR_STORE_DIR = "vector_store_cache"
