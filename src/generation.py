"""
Step 5 - Augmentation (prompt templating, answer grounding instructions,
context window optimisation) and Step 6 - Generation (citation-aware
answers, grounding-check guardrail).
"""
import tiktoken
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

SYSTEM_PROMPT = """You are a careful video-transcript assistant. Answer the
user's question using ONLY the numbered context chunks below, taken from
the video's transcript. Rules:
1. If the answer is not contained in the context, say clearly that the
   video does not cover it - never invent information.
2. After every factual claim, cite the chunk(s) it came from, like [1], [2].
3. Reply in the same language the user asked in.
4. Be concise and do not repeat the question back.

Context:
{context}
"""

CHAT_PROMPT = ChatPromptTemplate.from_messages(
    [("system", SYSTEM_PROMPT), ("human", "{question}")]
)

GROUNDING_PROMPT = ChatPromptTemplate.from_template(
    """Given the CONTEXT and an ANSWER, reply with just "grounded" if every
factual claim in the answer is supported by the context, or "ungrounded" if
the answer includes claims not present in the context.

CONTEXT:
{context}

ANSWER:
{answer}

Verdict (grounded/ungrounded):"""
)


def count_tokens(text: str) -> int:
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return len(text) // 4  # rough fallback


def optimize_context(docs, max_tokens: int = 3000):
    """Context window optimisation: de-duplicate and trim retrieved chunks
    so the final prompt stays within a token budget. Docs are assumed to
    already be sorted by relevance (best first)."""
    seen, kept, running = set(), [], 0
    for d in docs:
        text = d.page_content.strip()
        if not text or text in seen:
            continue
        t = count_tokens(text)
        if running + t > max_tokens:
            break
        seen.add(text)
        kept.append(d)
        running += t
    return kept


def format_context(docs):
    return "\n\n".join(
        f"[{i}] ({d.metadata.get('timestamp', '')}) {d.page_content}"
        for i, d in enumerate(docs, start=1)
    )


def build_citation_map(docs):
    return {
        str(i): {"timestamp": d.metadata.get("timestamp", ""), "text": d.page_content}
        for i, d in enumerate(docs, start=1)
    }


def generate_answer(llm, question: str, docs):
    """Augmentation + Generation in one call: prompt templating over an
    optimised context, producing a citation-tagged answer."""
    optimized = optimize_context(docs)
    context_text = format_context(optimized)
    chain = CHAT_PROMPT | llm | StrOutputParser()
    answer = chain.invoke({"context": context_text, "question": question})
    return answer, build_citation_map(optimized)


def grounding_check(llm, answer: str, docs) -> bool:
    """Guardrail: flags whether the answer is actually supported by the
    retrieved context (surfaced in the UI, not silently hidden)."""
    context_text = format_context(docs)
    chain = GROUNDING_PROMPT | llm | StrOutputParser()
    verdict = chain.invoke({"context": context_text, "answer": answer}).strip().lower()
    return "ungrounded" not in verdict
