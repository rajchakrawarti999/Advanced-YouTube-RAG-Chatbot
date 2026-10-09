"""
Step 4 - Retrieval pipeline, matching the roadmap:

  Pre-Retrieval   -> query rewriting (LLM), multi-query generation,
                     domain-aware routing (video / chitchat / offtopic)
  During Retrieval-> MMR (diversity) + Hybrid (dense+BM25) + Reranking
                     (cross-encoder) -- MMR is folded into the hybrid's
                     dense leg so all three live in one retriever.
  Post-Retrieval  -> Contextual compression: trims each retrieved chunk
                     down to only the sentences relevant to the question.
"""
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
# NOTE: EnsembleRetriever / ContextualCompressionRetriever / LLMChainExtractor /
# CrossEncoderReranker were moved out of the `langchain` core package in
# LangChain 1.x -> they now live in the `langchain-classic` compatibility
# package (see requirements.txt).
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor, CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from . import config

# ---------------- Pre-Retrieval ----------------

REWRITE_PROMPT = PromptTemplate.from_template(
    """Rewrite the user's question into a clear, standalone search query
optimised for semantic search over a video transcript. Keep the same
language as the original question. Return ONLY the rewritten query.

Original question: {question}
Rewritten query:"""
)

ROUTER_PROMPT = PromptTemplate.from_template(
    """Classify the user's message into exactly one category:
- video: a question about the content of the video/transcript
- chitchat: greetings, thanks, or small talk unrelated to the video
- offtopic: unrelated to the video and not chitchat either

Respond with just one word: video, chitchat, or offtopic.

Message: {question}
Category:"""
)

MULTI_QUERY_PROMPT = PromptTemplate.from_template(
    """Generate {n} different phrasings of the following question so that
different wordings can surface different relevant chunks from a video
transcript vector database. Keep the same language as the original
question. Return each version on its own line, no numbering, no extra text.

Question: {question}"""
)


def make_query_rewriter(llm):
    return REWRITE_PROMPT | llm | StrOutputParser()


def make_router(llm):
    return ROUTER_PROMPT | llm | StrOutputParser()


def generate_multi_queries(llm, question: str, n: int = 2):
    chain = MULTI_QUERY_PROMPT | llm | StrOutputParser()
    out = chain.invoke({"question": question, "n": n})
    queries = [q.strip(" -\t") for q in out.split("\n") if q.strip()]
    return queries[:n] if queries else []


# ---------------- During Retrieval ----------------

def make_mmr_retriever(vector_store, fetch_k=20, lambda_mult=0.5):
    """MMR gives semantically diverse results instead of near-duplicates."""
    return vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": config.TOP_K, "fetch_k": fetch_k, "lambda_mult": lambda_mult},
    )


def make_hybrid_retriever(vector_store, bm25_retriever, weights=(0.5, 0.5)):
    """Combines the MMR dense retriever with the BM25 keyword retriever."""
    dense = make_mmr_retriever(vector_store)
    return EnsembleRetriever(retrievers=[dense, bm25_retriever], weights=list(weights))


_cross_encoder = None


def get_reranker(top_n=4):
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = HuggingFaceCrossEncoder(model_name=config.RERANKER_MODEL)
    return CrossEncoderReranker(model=_cross_encoder, top_n=top_n)


def make_reranked_retriever(base_retriever, top_n=4):
    reranker = get_reranker(top_n=top_n)
    return ContextualCompressionRetriever(base_compressor=reranker, base_retriever=base_retriever)


# ---------------- Full pipeline ----------------

def retrieve(question, vector_store, bm25_retriever, llm, use_multi_query=True):
    """

    route -> rewrite -> (multi-query) -> hybrid(MMR+BM25) -> rerank
    -> post-retrieval contextual compression.
    Returns (final_docs, route).
    
    """
    router = make_router(llm)
    route = router.invoke({"question": question}).strip().lower()
    if "video" not in route:
        return [], route

    rewriter = make_query_rewriter(llm)
    rewritten = rewriter.invoke({"question": question}).strip() or question

    queries = [rewritten]
    if use_multi_query:
        queries += generate_multi_queries(llm, rewritten, n=2)

    hybrid = make_hybrid_retriever(vector_store, bm25_retriever)
    reranked = make_reranked_retriever(hybrid, top_n=6)

    seen = {}
    for q in queries:
        for d in reranked.invoke(q):
            seen[d.page_content] = d  # de-dupe across the multiple query phrasings
    docs = list(seen.values())

    # Post-Retrieval: contextual compression trims each doc down to the
    # sentences that actually matter for this question.
    extractor = LLMChainExtractor.from_llm(llm)
    try:
        compressed = extractor.compress_documents(docs, rewritten)
        final_docs = [d for d in compressed if d.page_content.strip()] or docs
    except Exception:
        final_docs = docs

    return final_docs[: config.TOP_K], route
