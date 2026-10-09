"""
Indexing step 3 - Vector Store (Pinecone) + a BM25 keyword index used
later for hybrid retrieval.

Embeddings run locally and free via a multilingual sentence-transformers
model, so the same pipeline works whether the transcript/question is in
Hindi, English, or mixed.
"""
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.retrievers import BM25Retriever
from pinecone import Pinecone, ServerlessSpec

from . import config

_embeddings = None


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
    return _embeddings


def _ensure_index(pc: Pinecone, dimension: int):
    existing = [i["name"] for i in pc.list_indexes()]
    if config.PINECONE_INDEX_NAME not in existing:
        pc.create_index(
            name=config.PINECONE_INDEX_NAME,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud=config.PINECONE_CLOUD, region=config.PINECONE_REGION),
        )


def build_vector_store(chunks, video_id: str):
    """
    Indexes chunks under a per-video Pinecone namespace, so one index can
    hold many videos without their chunks colliding during search.
    """
    if not config.PINECONE_API_KEY:
        raise RuntimeError("PINECONE_API_KEY .env me set karo (free tier: pinecone.io)")

    embeddings = get_embeddings()
    pc = Pinecone(api_key=config.PINECONE_API_KEY)
    probe_dim = len(embeddings.embed_query("dimension probe"))
    _ensure_index(pc, probe_dim)

    vector_store = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=config.PINECONE_INDEX_NAME,
        namespace=video_id,
    )
    return vector_store


def build_bm25_retriever(chunks):
    """Keyword-based retriever, used as the second leg of hybrid search."""
    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = config.TOP_K
    return bm25
