"""
Advanced YouTube RAG Chatbot - Streamlit UI

Flow: paste a YouTube video ID/URL -> transcript is ingested, chunked,
embedded and indexed (Pinecone) -> chat below answers questions using the
full advanced retrieval + generation pipeline in src/.
"""
import streamlit as st
from langchain_groq import ChatGroq

from src import config, ingestion, vectorstore, retrieval, generation, guardrails

st.set_page_config(page_title="YouTube RAG Chatbot", page_icon="🎬", layout="wide")

DEFAULTS = {
    "video_loaded": False,
    "video_id": None,
    "chunks": None,
    "vector_store": None,
    "bm25": None,
    "detected_lang": None,
    "messages": [],
}
for key, default in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default


@st.cache_resource(show_spinner=False)
def get_llm():
    if not config.GROQ_API_KEY:
        st.error("GROQ_API_KEY .env file me set karo (free key: console.groq.com)")
        st.stop()
    return ChatGroq(model=config.LLM_MODEL, api_key=config.GROQ_API_KEY, temperature=0.2)


llm = get_llm()


def reset_video():
    for key, default in DEFAULTS.items():
        st.session_state[key] = default


def render_message(content, citations=None, grounded=None, show_sources=True):
    st.markdown(content)
    if citations and show_sources:
        with st.expander("📎 Sources"):
            for k, v in citations.items():
                st.markdown(f"**[{k}] {v['timestamp']}** — {v['text'][:220]}...")
    if grounded is not None:
        (st.success if grounded else st.warning)(
            "✅ Answer grounded in transcript" if grounded else "⚠️ Answer may include ungrounded claims"
        )


# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("⚙️ Settings")
    use_multi_query = st.toggle("Multi-query generation", value=True)
    show_sources = st.toggle("Show sources / citations", value=True)
    show_grounding = st.toggle("Run grounding check", value=True)
    st.divider()
    if st.session_state.video_loaded:
        st.success(f"Video: {st.session_state.video_id}")
        st.caption(f"Detected language: {st.session_state.detected_lang}")
        if st.button("🔄 Load a different video"):
            reset_video()
            st.rerun()

st.title("🎬 Advanced YouTube RAG Chatbot")

# ---------------- Step 1: load video ----------------
if not st.session_state.video_loaded:
    st.subheader("Step 1 — YouTube video daaliye")
    url_or_id = st.text_input("YouTube video URL ya video ID", placeholder="e.g. Gfr50f6ZBvo")
    if st.button("Load video", type="primary") and url_or_id:
        with st.spinner("Transcript fetch, chunking aur indexing ho raha hai..."):
            try:
                video_id = ingestion.extract_video_id(url_or_id)
                chunks, detected_lang, yt_lang = ingestion.build_documents(video_id)
                vs = vectorstore.build_vector_store(chunks, video_id)
                bm25 = vectorstore.build_bm25_retriever(chunks)

                st.session_state.update(
                    {
                        "video_loaded": True,
                        "video_id": video_id,
                        "chunks": chunks,
                        "vector_store": vs,
                        "bm25": bm25,
                        "detected_lang": detected_lang,
                        "messages": [],
                    }
                )
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")
    st.stop()

# ---------------- Step 2: chat ----------------
st.subheader("Step 2 — Video ke baare mein poochiye")
st.video(f"https://www.youtube.com/watch?v={st.session_state.video_id}")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        render_message(
            msg["content"],
            citations=msg.get("citations"),
            grounded=msg.get("grounded"),
            show_sources=show_sources,
        )

question = st.chat_input("Apna sawaal likhiye...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        guard_msg = guardrails.input_guard(question)
        if guard_msg:
            st.markdown(guard_msg)
            st.session_state.messages.append({"role": "assistant", "content": guard_msg})
        else:
            with st.spinner("Soch raha hoon..."):
                docs, route = retrieval.retrieve(
                    question,
                    st.session_state.vector_store,
                    st.session_state.bm25,
                    llm,
                    use_multi_query=use_multi_query,
                )
                route_msg = guardrails.route_response(route)
                if route_msg:
                    st.markdown(route_msg)
                    st.session_state.messages.append({"role": "assistant", "content": route_msg})
                else:
                    answer, citations = generation.generate_answer(llm, question, docs)
                    grounded = generation.grounding_check(llm, answer, docs) if show_grounding else None
                    render_message(answer, citations=citations, grounded=grounded, show_sources=show_sources)
                    entry = {"role": "assistant", "content": answer, "citations": citations}
                    if grounded is not None:
                        entry["grounded"] = grounded
                    st.session_state.messages.append(entry)
