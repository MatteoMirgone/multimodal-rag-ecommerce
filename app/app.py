"""
app.py — IMPROVED VERSION
==========================
Improvement：
1. Past conversations are included in the memory of LLM
2. Chat history displayed on the side bar
3. Reranker 
4. UI : include chat memory

To run：
    python -m streamlit run app/app.py
"""

import os
from pathlib import Path

import streamlit as st
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("CHROMA_DIR", str(PROJECT_ROOT / "chroma_db"))

from rag_chain import MultimodalRAG, RAGResponse, RetrievedProduct, ConversationTurn

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Shop — Improved Multimodal Assistant",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

APPLE_CSS = """
<style>
  :root {
    --apple-bg: #ffffff;
    --apple-bg-soft: #f5f5f7;
    --apple-ink: #1d1d1f;
    --apple-ink-soft: #6e6e73;
    --apple-line: #d2d2d7;
    --apple-blue: #0071e3;
    --apple-blue-hover: #0077ed;
    --apple-green: #29845a;
    --apple-amber: #b25000;
    --apple-red: #b22222;
    --apple-purple: #6e3fad;
    --apple-radius: 18px;
    --apple-radius-sm: 12px;
    --apple-font: -apple-system, BlinkMacSystemFont, "SF Pro Display",
                  "SF Pro Text", "Helvetica Neue", Helvetica, Arial, sans-serif;
  }

  html, body, .stApp,
  .stApp p, .stApp span, .stApp div, .stApp label,
  .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5,
  .stApp button, .stApp textarea, .stApp input {
    font-family: var(--apple-font);
    -webkit-font-smoothing: antialiased;
  }
  [data-testid*="Icon"], [class*="material-symbols"],
  .material-icons, .material-icons-outlined {
    font-family: "Material Symbols Rounded", "Material Icons" !important;
  }

  .stApp { background: var(--apple-bg); color: var(--apple-ink); }
  #MainMenu, footer { display: none; }
  .stDeployButton { display: none; }
  header[data-testid="stHeader"] {
    background: transparent !important;
    border: none;
  }
  [data-testid="stSidebar"] { visibility: visible; }
  [data-testid="stSidebarCollapsedControl"] { visibility: visible !important; opacity: 1 !important; }

  /* Hero */
  .hero {
    text-align: center;
    padding: 2.5rem 1rem 2rem 1rem;
    background:
      radial-gradient(ellipse at top, #eef4ff 0%, transparent 60%),
      linear-gradient(180deg, #ffffff 0%, #fafafa 100%);
    border-bottom: 1px solid var(--apple-line);
    margin: -1rem -1rem 2rem -1rem;
    animation: fadeUp 0.6s ease;
  }
  .hero h1 {
    font-size: clamp(2.4rem, 4.5vw, 3.5rem) !important;
    font-weight: 700; letter-spacing: -0.035em; line-height: 1.05;
    margin: 0 0 0.8rem 0; color: var(--apple-ink) !important;
    background: linear-gradient(135deg, #1d1d1f 0%, #5a5a5f 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
  }
  .hero .subtitle {
    font-size: 1.25rem; color: var(--apple-ink-soft); font-weight: 400;
    max-width: 640px; margin: 0 auto; line-height: 1.4; letter-spacing: -0.01em;
  }
  .hero .accent { color: var(--apple-purple); font-weight: 500; }

  /* Memory badge */
  .memory-badge {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 4px 12px; border-radius: 980px; font-size: 0.78rem;
    font-weight: 500; background: #f0eaff; color: var(--apple-purple);
    border: 1px solid #d9cdf5; margin-top: 4px;
  }
  .memory-badge .dot {
    width: 6px; height: 6px; border-radius: 50%; background: var(--apple-purple);
  }

  @keyframes fadeUp {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
  }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: var(--apple-bg); border-right: 1px solid var(--apple-line); padding-top: 1rem;
  }
  [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2 {
    font-size: 0.78rem !important; font-weight: 600; letter-spacing: 0.06em;
    text-transform: uppercase; color: var(--apple-ink-soft); margin-bottom: 1rem;
  }

  .stButton > button {
    border-radius: 980px !important; background: var(--apple-bg-soft) !important;
    color: var(--apple-ink) !important; border: 1px solid var(--apple-line) !important;
    font-weight: 400 !important; font-size: 0.84rem !important;
    padding: 0.55rem 1rem !important; min-height: 2.4rem !important;
    white-space: normal !important; box-shadow: none !important;
    transition: background 0.2s ease, border-color 0.2s ease, transform 0.15s ease !important;
  }
  .stButton > button:hover {
    background: white !important; border-color: var(--apple-blue) !important;
    color: var(--apple-blue) !important; transform: translateY(-1px);
  }
  .stButton > button[kind="primary"] {
    background: var(--apple-blue) !important; color: white !important;
    border: none !important; font-weight: 500 !important;
  }

  [data-testid="stChatMessage"] {
    background: var(--apple-bg); border: 1px solid var(--apple-line);
    border-radius: var(--apple-radius); padding: 1.1rem 1.4rem;
    box-shadow: 0 1px 2px rgba(0,0,0,0.02); margin-bottom: 0.9rem;
    animation: fadeUp 0.35s ease;
  }

  [data-testid="stChatInput"] {
    background: var(--apple-bg-soft); border-radius: 980px !important;
    border: 1px solid var(--apple-line) !important; padding: 0.25rem 0.5rem !important;
    overflow: hidden !important;
  }
  [data-testid="stChatInput"]:focus-within {
    border-color: var(--apple-blue) !important;
    box-shadow: 0 0 0 3px rgba(0, 113, 227, 0.18) !important;
  }
  [data-testid="stChatInput"] * { background: transparent !important; }
  [data-testid="stChatInput"] *,
  [data-testid="stChatInput"] *:focus,
  [data-testid="stChatInput"] *:focus-within {
    outline: none !important; box-shadow: none !important; border-color: transparent !important;
  }
  [data-testid="stChatInput"] textarea {
    background: transparent !important; font-size: 1rem !important; color: var(--apple-ink) !important;
  }

  .badge {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 4px 12px; border-radius: 980px; font-size: 0.78rem;
    font-weight: 500; margin-top: 8px; border: 1px solid transparent;
  }
  .badge .dot { width: 6px; height: 6px; border-radius: 50%; }
  .badge-high   { background: #ecf8f1; color: var(--apple-green); border-color: #cdebd8; }
  .badge-high .dot   { background: var(--apple-green); }
  .badge-medium { background: #fdf3e7; color: var(--apple-amber); border-color: #f4dbb6; }
  .badge-medium .dot { background: var(--apple-amber); }
  .badge-low    { background: #fbecec; color: var(--apple-red); border-color: #f1c5c5; }
  .badge-low .dot    { background: var(--apple-red); }

  .product-card {
    background: var(--apple-bg-soft); border-radius: var(--apple-radius);
    padding: 1.25rem 1rem 1rem 1rem;
    transition: transform 0.25s ease, box-shadow 0.25s ease; height: 100%;
  }
  .product-card:hover { transform: translateY(-4px); box-shadow: 0 12px 28px rgba(0,0,0,0.06); }
  .product-card img {
    border-radius: var(--apple-radius-sm); width: 100%; aspect-ratio: 1/1;
    object-fit: contain; background: white; margin-bottom: 1rem; padding: 0.5rem;
  }
  .product-name {
    font-weight: 600; font-size: 0.95rem; line-height: 1.3; color: var(--apple-ink);
    margin: 0 0 0.4rem 0; display: -webkit-box; -webkit-line-clamp: 2;
    -webkit-box-orient: vertical; overflow: hidden; min-height: 2.5rem;
  }
  .product-meta { font-size: 0.82rem; color: var(--apple-ink-soft); margin: 3px 0; }
  .product-price { font-weight: 500; color: var(--apple-ink); font-size: 1rem; margin: 8px 0 4px 0; }
  .sim-pill {
    display: inline-flex; align-items: center; gap: 5px; padding: 3px 10px;
    border-radius: 980px; font-size: 0.72rem; font-weight: 500; margin-top: 8px;
    background: white; border: 1px solid var(--apple-line); color: var(--apple-ink-soft);
  }
  .sim-pill .dot { width: 6px; height: 6px; border-radius: 50%; }
  .sim-high .dot   { background: var(--apple-green); }
  .sim-medium .dot { background: var(--apple-amber); }
  .sim-low .dot    { background: var(--apple-red); }

  [data-testid="stExpander"] {
    background: transparent; border: 1px solid var(--apple-line);
    border-radius: var(--apple-radius); margin-top: 0.6rem;
  }
  [data-testid="stExpander"] summary { font-weight: 500; color: var(--apple-blue); font-size: 0.95rem; }

  [data-testid="stSegmentedControl"] button {
    border-radius: 8px !important; font-size: 0.82rem !important;
    padding: 0.4rem 0.7rem !important; background: transparent !important;
    color: var(--apple-ink-soft) !important; border: none !important;
  }
  [data-testid="stSegmentedControl"] button[aria-pressed="true"] {
    background: white !important; color: var(--apple-ink) !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.06) !important;
  }
  [data-testid="stSegmentedControl"] > div {
    background: var(--apple-bg-soft) !important; border-radius: 10px !important; padding: 3px !important;
  }

  .block-container { padding-top: 1rem !important; max-width: 1180px; }
</style>
"""
st.markdown(APPLE_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Load RAG（缓存，只加载一次）
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading CLIP + ChromaDB…")
def load_rag(use_reranker: bool = True):
    chroma_dir = os.environ.get("CHROMA_DIR", "./chroma_db")
    return MultimodalRAG(chroma_dir=chroma_dir, use_reranker=use_reranker)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "history" not in st.session_state:
    st.session_state.history = []          # UI 展示用的消息列表
if "conversation" not in st.session_state:
    st.session_state.conversation = []     # LLM 用的 ConversationTurn 列表


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown('<h2>Settings</h2>', unsafe_allow_html=True)

    top_k = st.slider("Results per query", 1, 10, 5)

    try:
        prompt_style = st.segmented_control(
            "Prompt strategy",
            options=["zero-shot", "few-shot", "multi-shot"],
            default="few-shot",
        )
        if prompt_style is None:
            prompt_style = "few-shot"
    except AttributeError:
        prompt_style = st.radio(
            "Prompt strategy",
            options=["zero-shot", "few-shot", "multi-shot"],
            index=1,
        )

    show_retrieved = st.checkbox("Show retrieved products", value=True)

    # Reranker T/F
    use_reranker = st.checkbox(
        "Use reranker",
        value=True,
        help="Cross-encoder reranking filters out irrelevant results. "
             "Slightly slower but more accurate.",
    )

    st.markdown("---")
    st.markdown('<h2>Conversation Memory</h2>', unsafe_allow_html=True)

    # show current memory badge
    n_turns = len(st.session_state.conversation)
    if n_turns > 0:
        st.markdown(
            f'<div class="memory-badge"><span class="dot"></span>'
            f'Remembering {n_turns} turn{"s" if n_turns > 1 else ""}</div>',
            unsafe_allow_html=True,
        )
        st.caption("The assistant remembers your conversation and can answer follow-up questions.")
    else:
        st.caption("No conversation history yet.")

    st.markdown("---")
    st.markdown('<h2>Engine</h2>', unsafe_allow_html=True)
    provider = os.getenv("LLM_PROVIDER", "groq")
    st.markdown(
        f"<p style='font-size:0.85rem;color:#6e6e73;margin:0;'>"
        f"Provider <span style='float:right;color:#1d1d1f;'>{provider}</span></p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='font-size:0.85rem;color:#6e6e73;margin:4px 0 0 0;'>"
        "Model <span style='float:right;color:#1d1d1f;'>llama-3.1-8b</span></p>",
        unsafe_allow_html=True,
    )
    reranker_status = "✓ on" if use_reranker else "off"
    st.markdown(
        f"<p style='font-size:0.85rem;color:#6e6e73;margin:4px 0 0 0;'>"
        f"Reranker <span style='float:right;color:#1d1d1f;'>{reranker_status}</span></p>",
        unsafe_allow_html=True,
    )

    key_set = bool(os.getenv("GROQ_API_KEY") or os.getenv("TOGETHER_API_KEY"))
    if not key_set:
        st.error("No API key set. Add `GROQ_API_KEY` to `.env`.")

    st.markdown("---")
    if st.button("Clear conversation", type="primary"):
        st.session_state.history = []
        st.session_state.conversation = []
        st.rerun()

    st.markdown("---")
    st.caption("CLIP · ChromaDB · Llama 3.1 · Streamlit  |  Reranker · Dynamic Few-shot · Memory  |  MSADS 2026 (Improved)")


# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div class="hero">
      <h1>Shop smarter.<br/>Now with <span class="accent">memory</span>.</h1>
      <p class="subtitle">
        Ask anything about a product, or upload a photo. 
        Remembers your conversation — ask follow-up questions naturally.
        Powered by CLIP · Reranker · Llama 3.1.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

SUGGESTIONS = [
    ("Captain America Rug",  "Tell me about the Marvel Captain America Shield Rug"),
    ("LEGO Taj Mahal",       "Describe the LEGO Creator Taj Mahal building kit"),
    ("Princess Puzzle",      "Tell me about the Mudpuppy Enchanting Princess Puzzle"),
    ("Pikachu Café",         "What is the Mega Construx Pokemon Detective Pikachu Café?"),
]

_cols = st.columns([2, 1, 1, 1, 1, 2], gap="small")
for i, (label, query) in enumerate(SUGGESTIONS):
    with _cols[i + 1]:
        if st.button(label, key=f"chip_{i}", width="stretch"):
            st.session_state["pending_query"] = query
            st.rerun()


# ---------------------------------------------------------------------------
# Load RAG
# ---------------------------------------------------------------------------

try:
    rag = load_rag(use_reranker=use_reranker)
except Exception as e:
    st.error(f"Failed to load RAG chain: {e}")
    st.info("Run `scripts/rebuild_prepared_data.py` and `scripts/cache_images.py` first.")
    st.stop()


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

CONFIDENCE_LABELS = {
    "high":   ("badge-high",   "High confidence"),
    "medium": ("badge-medium", "Medium confidence"),
    "low":    ("badge-low",    "Closest matches"),
}


def sim_pill_class(sim: float) -> str:
    if sim >= 0.75:
        return "sim-high"
    if sim >= 0.65:
        return "sim-medium"
    return "sim-low"


def render_product_grid(retrieved) -> None:
    if not retrieved:
        return
    cols = st.columns(min(len(retrieved), 5), gap="medium")
    for col, prod in zip(cols, retrieved):
        with col:
            if prod.image_path and Path(prod.image_path).exists():
                try:
                    st.image(prod.image_path, width="stretch")
                except Exception:
                    st.markdown("&nbsp;", unsafe_allow_html=True)
            name = prod.product_name or "(unnamed product)"
            st.markdown(f'<div class="product-name">{name[:80]}</div>', unsafe_allow_html=True)
            if prod.brand:
                st.markdown(f'<div class="product-meta">{prod.brand}</div>', unsafe_allow_html=True)
            if prod.category:
                st.markdown(f'<div class="product-meta">{prod.category}</div>', unsafe_allow_html=True)
            if prod.price:
                st.markdown(f'<div class="product-price">{prod.price}</div>', unsafe_allow_html=True)
            pill = sim_pill_class(prod.similarity)
            st.markdown(
                f'<div class="sim-pill {pill}"><span class="dot"></span>match · {prod.similarity:.2f}</div>',
                unsafe_allow_html=True,
            )


def render_confidence_badge(confidence: str, top_sim: float) -> None:
    if confidence in CONFIDENCE_LABELS:
        klass, label = CONFIDENCE_LABELS[confidence]
        st.markdown(
            f'<div><span class="badge {klass}"><span class="dot"></span>{label} · {top_sim:.2f}</span></div>',
            unsafe_allow_html=True,
        )


def _escape_dollars(text: str) -> str:
    if not text:
        return text
    return text.replace("$", "\\$")


def render_message(msg: dict) -> None:
    with st.chat_message(msg["role"], avatar=None):
        if msg.get("image") is not None:
            st.image(msg["image"], width=200)
        if msg.get("text"):
            st.markdown(_escape_dollars(msg["text"]))
        if msg["role"] == "assistant":
            conf    = msg.get("confidence")
            top_sim = msg.get("top_sim", 0.0)
            if conf:
                render_confidence_badge(conf, top_sim)
            if show_retrieved and msg.get("retrieved"):
                with st.expander(f"View {len(msg['retrieved'])} retrieved products"):
                    render_product_grid(msg["retrieved"])


# ---------------------------------------------------------------------------
# Render conversation history
# ---------------------------------------------------------------------------

for msg in st.session_state.history:
    render_message(msg)


# ---------------------------------------------------------------------------
# Input area
# ---------------------------------------------------------------------------

try:
    chat_value   = st.chat_input(
        "Ask about a product, or attach a photo…",
        accept_file=True,
        file_type=["jpg", "jpeg", "png", "webp"],
    )
    chat_text    = chat_value.text if chat_value else None
    uploaded_files = chat_value.files if chat_value else []
    uploaded     = uploaded_files[0] if uploaded_files else None
except TypeError:
    chat_text = st.chat_input("Ask about a product…")
    uploaded  = st.file_uploader(
        "Optional image", type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )

pending   = st.session_state.pop("pending_query", None)
user_text = pending or chat_text


# ---------------------------------------------------------------------------
# Process input
# ---------------------------------------------------------------------------

if user_text or uploaded:
    pil_image = Image.open(uploaded) if uploaded else None

    user_msg = {
        "role":  "user",
        "text":  user_text or "(image-only query)",
        "image": pil_image,
    }
    st.session_state.history.append(user_msg)
    render_message(user_msg)

    with st.chat_message("assistant", avatar=None):
        with st.spinner("Thinking…"):
            try:
                # 改进1：把 ConversationTurn 历史传给 answer()
                response: RAGResponse = rag.answer(
                    query_text   = user_text or None,
                    query_image  = pil_image,
                    k            = top_k,
                    prompt_style = prompt_style,
                    history      = st.session_state.conversation,
                )
                answer_text = response.answer
                retrieved   = response.retrieved
                confidence  = response.confidence
                top_sim     = response.top_similarity
            except Exception as e:
                answer_text = f"Error: {e}"
                retrieved   = []
                confidence  = None
                top_sim     = 0.0

        st.markdown(_escape_dollars(answer_text))

        if confidence:
            render_confidence_badge(confidence, top_sim)

        # If the past conversations are referenced and activated, it would show purple memory badge
        if st.session_state.conversation:
            st.markdown(
                f'<div class="memory-badge"><span class="dot"></span>'
                f'Using {len(st.session_state.conversation)}-turn memory</div>',
                unsafe_allow_html=True,
            )

        if show_retrieved and retrieved:
            with st.expander(f"View {len(retrieved)} retrieved products", expanded=True):
                render_product_grid(retrieved)

    # 把这轮对话存入 ConversationTurn 历史（供下轮 LLM 使用）
    top_product_name = retrieved[0].product_name if retrieved else ""
    st.session_state.conversation.append(ConversationTurn(
        user_text        = user_text,
        assistant_answer = answer_text,
        top_product_name = top_product_name,
        retrieved        = retrieved,  # 追问时复用
    ))

    st.session_state.history.append({
        "role":      "assistant",
        "text":      answer_text,
        "retrieved": retrieved,
        "confidence": confidence,
        "top_sim":   top_sim,
    })
