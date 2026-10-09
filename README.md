#  Advanced YouTube RAG Chatbot

Paste a YouTube video ID or URL, and chat with the video. The app fetches the transcript, indexes it, and answers your questions using an advanced Retrieval-Augmented Generation (RAG) pipeline. Answers include citations with video timestamps.

Works with **Hindi and English** videos. The language is detected automatically.

##  Features

| Stage | What's implemented |
|---|---|
| **Indexing** | Transcript ingestion, recursive text splitting with timestamps, Pinecone vector store |
| **Pre-Retrieval** | LLM query rewriting, multi-query generation, domain-aware routing (video / chitchat / off-topic) |
| **During Retrieval** | MMR, hybrid search (dense + BM25), cross-encoder reranking |
| **Post-Retrieval** | Contextual compression |
| **Augmentation** | Prompt templating, answer grounding check, context window optimization |
| **Generation** | Answers with citations and timestamps, guardrails |
| **Evaluation** | Ragas metrics, LangSmith tracing |
| **UI** | Streamlit: enter video ID first, then chat |

##  Tech Stack

- **LLM:** Groq (`llama-3.3-70b-versatile`), free tier
- **Embeddings:** `paraphrase-multilingual-mpnet-base-v2` (local, multilingual)
- **Reranker:** `cross-encoder/ms-marco-MiniLM-L-6-v2` (local)
- **Vector DB:** Pinecone (one namespace per video)
- **Keyword index:** BM25 (in-memory)
- **Framework:** LangChain, Streamlit
- **Evaluation:** Ragas, LangSmith

##  How it works

```
YouTube ID/URL
   → transcript fetch + language detection
   → chunking (with timestamps)
   → embeddings → Pinecone

User question
   → guardrail check → router (video / chitchat / off-topic)
   → query rewriting → multi-query
   → hybrid retrieval (MMR + BM25) → reranking
   → contextual compression
   → prompt + context optimization → LLM
   → answer with citations + grounding check
```

##  Setup

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env            # Windows: copy .env.example .env
```

Open `.env` and fill in your keys:

| Key | Required | Where to get it |
|---|---|---|
| `GROQ_API_KEY` | Yes | https://console.groq.com |
| `PINECONE_API_KEY` | Yes | https://app.pinecone.io |
| `LANGCHAIN_API_KEY` | Optional | https://smith.langchain.com (enables tracing) |

Run the app:

```bash
streamlit run app.py
```

##  Project Structure

```
├── app.py                  # Streamlit UI
├── src/
│   ├── config.py           # Settings loaded from .env
│   ├── ingestion.py        # Transcript fetch, language detection, chunking
│   ├── vectorstore.py      # Embeddings, Pinecone, BM25
│   ├── retrieval.py        # Pre / during / post retrieval pipeline
│   ├── generation.py       # Prompting, citations, grounding check
│   ├── guardrails.py       # Input filter and routing responses
│   └── evaluation.py       # Ragas evaluation
├── requirements.txt
└── .env.example
```

##  Evaluation

`src/evaluation.py` provides `run_ragas_eval()`. Pass your questions, generated answers, retrieved contexts, and ground-truth answers. It returns faithfulness, answer relevancy, context precision, and context recall as a DataFrame.

For LangSmith, set `LANGCHAIN_API_KEY` in `.env`. Every chain call is then traced automatically.

##  Notes

- Only videos with captions (manual or auto-generated) are supported.
- Embedding and reranker models download on the first run.
- This project uses `youtube-transcript-api` v1.x and `langchain-classic` (needed for the ensemble retriever, compression retriever, and reranker under LangChain 1.x).
- Never commit your `.env` file. It is listed in `.gitignore`.

##  Future Improvements

- Persistent chat history (SQLite/Postgres)
- Multi-video chat
- Async calls to reduce latency
- Local vector DB fallback (FAISS/Chroma)
