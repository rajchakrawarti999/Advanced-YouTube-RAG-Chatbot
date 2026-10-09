# 🎬 Advanced YouTube RAG Chatbot

Ek YouTube video ID/URL do → transcript automatically fetch, chunk aur index
hota hai → phir uske content ke baare mein chat karo. Poori advanced RAG
pipeline (Indexing → Retrieval → Augmentation → Generation → Evaluation)
implement ki gayi hai, aapke roadmap image ke mutabik.

## Feature map (roadmap → code)

| Roadmap item | Kahan hai |
|---|---|
| Document Ingestion | `src/ingestion.py :: fetch_transcript` |
| Text Splitting | `src/ingestion.py :: build_documents` |
| Vector Store (Pinecone) | `src/vectorstore.py :: build_vector_store` |
| Pre-Retrieval: Query rewriting | `src/retrieval.py :: make_query_rewriter` |
| Pre-Retrieval: Multi-query generation | `src/retrieval.py :: generate_multi_queries` |
| Pre-Retrieval: Domain-aware routing | `src/retrieval.py :: make_router` (video / chitchat / offtopic) |
| During Retrieval: MMR | `src/retrieval.py :: make_mmr_retriever` |
| During Retrieval: Hybrid Retrieval | `src/retrieval.py :: make_hybrid_retriever` (MMR-dense + BM25 ensemble) |
| During Retrieval: Reranking | `src/retrieval.py :: make_reranked_retriever` (cross-encoder) |
| Post-Retrieval: Contextual Compression | `src/retrieval.py :: retrieve` (LLMChainExtractor step) |
| Augmentation: Prompt Templating | `src/generation.py :: CHAT_PROMPT` |
| Augmentation: Answer grounding | `src/generation.py :: grounding_check` |
| Augmentation: Context window optimization | `src/generation.py :: optimize_context` |
| Generation: Answer with Citation | `src/generation.py :: generate_answer` + `build_citation_map` |
| Generation: Guard railing | `src/guardrails.py` |
| Evaluation: Ragas | `src/evaluation.py` |
| Evaluation: LangSmith | `src/config.py` (env vars only — auto-traces every chain call) |
| Auto Hindi/English detection | `src/ingestion.py :: detect_language` (langdetect on transcript text) |
| Streamlit UI: video id → chat | `app.py` |

## Kyun ye choices (free + best)

- **LLM — Groq (`llama-3.3-70b-versatile`)**: free tier bahut generous hai
  aur speed/quality dono kaafi achhe hain text-to-text ke liye. Free key:
  https://console.groq.com
- **Embeddings — `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`**:
  local chalta hai, bilkul free, aur Hindi + English dono ache se handle
  karta hai — isi wajah se language auto-detect ke baad bhi ek hi embedding
  model kaafi hai.
- **Vector Store — Pinecone**: aapke note (image me "Pine Cone") ke mutabik.
  Free tier available hai: https://app.pinecone.io — har video apne khud
  ke namespace me store hota hai, isliye ek hi index me multiple videos
  chal sakte hain.
- **Reranker — local cross-encoder** (`ms-marco-MiniLM-L-6-v2`): free, koi
  extra API key nahi chahiye.

## Setup

```bash
cd youtube_rag_advanced
python3 -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# .env kholo aur GROQ_API_KEY + PINECONE_API_KEY daalo (dono free hain)

streamlit run app.py
```

LangSmith optional hai — agar `LANGCHAIN_API_KEY` `.env` me daal doge, to
har chain call automatically https://smith.langchain.com par trace/eval ho
jayegi, koi extra code nahi likhna padega.

## Ragas evaluation chalane ke liye

`src/evaluation.py :: run_ragas_eval()` ko ek chhota script/notebook se
call karo — apne questions, generated answers, retrieved contexts aur
apne reference (ground-truth) answers do; ye ek pandas DataFrame return
karta hai jisme faithfulness, answer relevancy, context precision aur
context recall scores hote hain.

## ⚠️ Security note (aapke uploaded notebook ke baare mein)

`Copy_of_rag_using_langchain.ipynb` ke Cell 0 me ek **real OpenAI API key
hardcoded** mili thi (`sk-proj-...`). Wo key ab exposed ho chuki hai —
turant https://platform.openai.com/api-keys pe jaake revoke/rotate kar
dijiye. Is naye project me koi bhi key kabhi code me nahi hai — sab kuch
`.env` se load hota hai (jo `.gitignore` me daalna na bhoolein agar Git
use kar rahe hain).

## Structure

```
youtube_rag_advanced/
├── app.py                 # Streamlit UI: video loader + chat
├── src/
│   ├── config.py           # env-driven settings, model choices
│   ├── ingestion.py         # transcript fetch, language detect, chunking+timestamps
│   ├── vectorstore.py       # embeddings + Pinecone + BM25 index
│   ├── retrieval.py         # pre/during/post-retrieval pipeline
│   ├── generation.py        # prompt templating, grounding, citations
│   ├── guardrails.py        # input filter + routing guardrails
│   └── evaluation.py        # Ragas metrics
├── requirements.txt
└── .env.example
```

## Notes / known constraints

- `langchain` v1.x me `EnsembleRetriever`, `ContextualCompressionRetriever`,
  `LLMChainExtractor` aur `CrossEncoderReranker` `langchain-classic` package
  me move ho gaye hain — `requirements.txt` me already included hai, imports
  bhi usi ke mutabik verified hain.
- Reranker aur embeddings pehli baar chalne par model download karenge
  (internet chahiye pehli run par), uske baad local cache se chalte hain.
- Ye pipeline sirf videos par kaam karega jinpe captions/transcript
  available hai (auto-generated bhi chalega).
