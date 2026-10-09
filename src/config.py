"""
Central configuration. Everything is read from environment variables
(loaded from a local .env file) so no secret ever lives in code.
"""
import os
from dotenv import load_dotenv

load_dotenv(override=True)

# ---------------- API keys ----------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")        # optional, hosted reranker
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY", "")  # optional, LangSmith tracing/eval

# ---------------- LangSmith (optional, just tracing) ----------------
if LANGCHAIN_API_KEY:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "youtube-rag-advanced")

# ---------------- Model choices ----------------
# Free & fast text-to-text LLM via Groq (generous free tier).
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

# Free, local, multilingual embeddings -> works for Hindi + English transcripts
# and queries without any extra API key.
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)

# Free, local cross-encoder used for the reranking step.
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# ---------------- Pinecone (vector store) ----------------
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "youtube-rag-advanced")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

# ---------------- Chunking ----------------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))

# ---------------- Retrieval ----------------
TOP_K = int(os.getenv("TOP_K", 6))
