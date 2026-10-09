"""
Indexing step 1 & 2 - Document Ingestion + Text Splitting.

- Accepts a raw video id OR a full YouTube URL.
- Does NOT assume the caption language: it fetches whatever track YouTube
  has, then runs langdetect on the actual text to confirm the real spoken
  language (Hindi / English / anything else).
- Keeps an approximate timestamp on every chunk so answers can later be
  cited back to a point in the video.
"""
import re
import bisect

from langdetect import detect, DetectorFactory
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    VideoUnavailable,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import config

DetectorFactory.seed = 42  # make langdetect deterministic


def extract_video_id(url_or_id: str) -> str:
    """Accepts a bare video id or a full/short YouTube URL, returns the id."""
    url_or_id = url_or_id.strip()
    patterns = [
        r"(?:v=|/)([0-9A-Za-z_-]{11}).*",
        r"youtu\.be/([0-9A-Za-z_-]{11})",
    ]
    for p in patterns:
        m = re.search(p, url_or_id)
        if m:
            return m.group(1)
    if re.fullmatch(r"[0-9A-Za-z_-]{11}", url_or_id):
        return url_or_id
    raise ValueError("Valid YouTube video ID ya URL dijiye.")


def format_timestamp(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def fetch_transcript(video_id: str):
    """
    Language-agnostic fetch: prefer a manually created track (any language),
    else fall back to the first auto-generated one. This is what lets a
    Hindi video and an English video both work without the caller having
    to specify a language up front.

    NOTE: youtube-transcript-api v1.x replaced the old class-method API
    (YouTubeTranscriptApi.list_transcripts / .get_transcript) with an
    instance-based one (YouTubeTranscriptApi().list / .fetch), and
    transcript.fetch() now returns a FetchedTranscript object of
    FetchedTranscriptSnippet items rather than plain dicts -- .to_raw_data()
    converts it back to the [{"text", "start", "duration"}, ...] shape the
    rest of this module expects.
    """
    ytt_api = YouTubeTranscriptApi()
    try:
        transcript_list = ytt_api.list(video_id)
    except TranscriptsDisabled:
        raise RuntimeError("Is video par captions/transcripts disabled hain.")
    except VideoUnavailable:
        raise RuntimeError("Video available nahi hai ya ID galat hai.")

    chosen = None
    for t in transcript_list:
        if not t.is_generated:
            chosen = t
            break
    if chosen is None:
        for t in transcript_list:
            chosen = t
            break
    if chosen is None:
        raise RuntimeError("Is video ka koi transcript track nahi mila.")

    fetched = chosen.fetch()
    segments = fetched.to_raw_data()
    return segments, chosen.language_code


def detect_language(text: str) -> str:
    """Confirms the real spoken language from the transcript text itself
    (YouTube's own language_code label is sometimes wrong/missing)."""
    sample = text[:2000]
    try:
        return detect(sample)
    except Exception:
        return "unknown"


def build_documents(video_id: str):
    """
    Full ingestion pipeline for one video:
      fetch -> detect language -> split into ~CHUNK_SIZE pieces while
      keeping an accurate start-timestamp on every chunk.
    Returns (chunks, detected_language, youtube_label_language).
    """
    segments, yt_lang_code = fetch_transcript(video_id)

    full_text = ""
    offsets = []  # (char_offset_in_full_text, start_seconds)
    for seg in segments:
        offsets.append((len(full_text), seg["start"]))
        full_text += seg["text"].replace("\n", " ") + " "

    detected_lang = detect_language(full_text)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )
    raw_chunks = splitter.split_text(full_text)
    offset_positions = [o[0] for o in offsets]

    chunks = []
    search_pos = 0
    for i, chunk_text in enumerate(raw_chunks):
        pos = full_text.find(chunk_text[:50], search_pos)
        if pos == -1:
            pos = search_pos
        idx = bisect.bisect_right(offset_positions, pos) - 1
        idx = max(0, min(idx, len(offsets) - 1))
        start_seconds = offsets[idx][1]

        chunks.append(
            Document(
                page_content=chunk_text,
                metadata={
                    "video_id": video_id,
                    "chunk_id": i,
                    "start": start_seconds,
                    "timestamp": format_timestamp(start_seconds),
                    "language": detected_lang,
                },
            )
        )
        search_pos = max(search_pos, pos)

    return chunks, detected_lang, yt_lang_code
