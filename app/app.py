"""
app.py — Streamlit chatbot UI for the multimodal RAG e-commerce assistant.

Run with:
    streamlit run app.py

Requires:
    - chroma_db/ built (notebooks/build_index_colab.ipynb)
    - GROQ_API_KEY in environment or .env
"""

import os
from pathlib import Path

import streamlit as st
from PIL import Image

# Resolve chroma_db relative to project root (one level up from this file)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("CHROMA_DIR", str(PROJECT_ROOT / "chroma_db"))

from rag_chain import MultimodalRAG, RAGResponse, RetrievedProduct

# ----------------------------------------------------------------------
# Page config + Apple-style theme
# ----------------------------------------------------------------------

st.set_page_config(
    page_title="Shop — Multimodal Assistant",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Visual language references apple.com/store:
#  - Pure white canvas, near-black ink, single accent (Apple blue #0071e3)
#  - SF Pro typography stack, tight letter-spacing on display sizes
#  - Generous whitespace, very subtle dividers, soft shadows
#  - Buttons are pill-shaped, hover transitions are slow and gentle
#  - Product cards are quiet — image dominates, copy is small and grey
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
    --apple-radius: 18px;
    --apple-radius-sm: 12px;
    --apple-font: -apple-system, BlinkMacSystemFont, "SF Pro Display",
                  "SF Pro Text", "Helvetica Neue", Helvetica, Arial, sans-serif;
  }

  /* IMPORTANT: do NOT use `*` here. Streamlit renders icons via the
     Material Symbols font; an all-elements font override turns those icons
     into literal text ("keyboard_double_arrow_left", "upload", etc). */
  html, body, .stApp,
  .stApp p, .stApp span, .stApp div, .stApp label,
  .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5,
  .stApp button, .stApp textarea, .stApp input {
    font-family: var(--apple-font);
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
  }
  /* Restore Material icon font on anything that needs glyphs */
  [data-testid*="Icon"],
  [class*="material-symbols"],
  [class*="MaterialSymbols"],
  .material-icons, .material-icons-outlined,
  [data-testid="stChatInput"] button svg,
  [data-testid="stFileUploaderDropzone"] svg {
    font-family: "Material Symbols Rounded", "Material Symbols Outlined",
                 "Material Icons" !important;
  }

  .stApp {
    background: var(--apple-bg);
    color: var(--apple-ink);
  }

  /* Hide deploy/menu but KEEP the header (it carries the sidebar toggle
     when collapsed). Header is made transparent so it doesn't show. */
  #MainMenu, footer { display: none; }
  .stDeployButton, [data-testid="stToolbar"] { display: none; }
  header[data-testid="stHeader"] {
    background: transparent !important;
    height: 2.5rem;
    border: none;
  }

  /* Ensure sidebar is reachable */
  [data-testid="stSidebar"] { visibility: visible; }
  [data-testid="stSidebarCollapsedControl"] {
    visibility: visible !important;
    opacity: 1 !important;
  }

  /* Hero — large, calm, Apple-style */
  .hero {
    text-align: center;
    padding: 2.5rem 1rem 2rem 1rem;
    background:
      radial-gradient(ellipse at top, #f0f4ff 0%, transparent 60%),
      linear-gradient(180deg, #ffffff 0%, #fafafa 100%);
    border-bottom: 1px solid var(--apple-line);
    margin: -1rem -1rem 2rem -1rem;
    animation: fadeUp 0.6s ease;
  }
  .hero h1 {
    font-size: clamp(2.4rem, 4.5vw, 3.5rem) !important;
    font-weight: 700;
    letter-spacing: -0.035em;
    line-height: 1.05;
    margin: 0 0 0.8rem 0;
    color: var(--apple-ink) !important;
    background: linear-gradient(135deg, #1d1d1f 0%, #5a5a5f 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }
  .hero .subtitle {
    font-size: 1.25rem;
    color: var(--apple-ink-soft);
    font-weight: 400;
    max-width: 640px;
    margin: 0 auto;
    line-height: 1.4;
    letter-spacing: -0.01em;
  }
  .hero .accent {
    color: var(--apple-blue);
    font-weight: 500;
  }

  /* Subtle fade-up animation */
  @keyframes fadeUp {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
  }
  @keyframes fadeIn {
    from { opacity: 0; }
    to   { opacity: 1; }
  }

  /* Sidebar — clean white with hairline divider */
  [data-testid="stSidebar"] {
    background: var(--apple-bg);
    border-right: 1px solid var(--apple-line);
    padding-top: 1rem;
  }
  [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2 {
    font-size: 0.78rem !important;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--apple-ink-soft);
    margin-bottom: 1rem;
  }
  [data-testid="stSidebar"] label, [data-testid="stSidebar"] p {
    font-size: 0.9rem !important;
    color: var(--apple-ink) !important;
  }
  [data-testid="stSidebar"] hr {
    border-color: var(--apple-line);
    margin: 1.5rem 0;
  }

  /* Slider + radio Apple-blue accent */
  [data-baseweb="slider"] [role="slider"] {
    background: var(--apple-blue) !important;
    border-color: var(--apple-blue) !important;
  }
  [data-baseweb="radio"] input:checked + div {
    border-color: var(--apple-blue) !important;
  }

  /* Default buttons (used for chips) — quiet pill shape, hover lifts.
     min-height ensures labels that wrap still look balanced. */
  .stButton > button {
    border-radius: 980px !important;
    background: var(--apple-bg-soft) !important;
    color: var(--apple-ink) !important;
    border: 1px solid var(--apple-line) !important;
    font-weight: 400 !important;
    font-size: 0.84rem !important;
    padding: 0.55rem 1rem !important;
    line-height: 1.25 !important;
    min-height: 2.4rem !important;
    white-space: normal !important;
    box-shadow: none !important;
    transition: background 0.2s ease, border-color 0.2s ease,
                color 0.2s ease, transform 0.15s ease !important;
  }
  .stButton > button:hover {
    background: white !important;
    border-color: var(--apple-blue) !important;
    color: var(--apple-blue) !important;
    transform: translateY(-1px);
  }
  .stButton > button:active { transform: translateY(0); }
  .stButton > button p { font-size: inherit !important; line-height: inherit !important; }

  /* Primary buttons (e.g. Clear conversation) — filled Apple blue */
  .stButton > button[kind="primary"] {
    background: var(--apple-blue) !important;
    color: white !important;
    border: none !important;
    font-weight: 500 !important;
  }
  .stButton > button[kind="primary"]:hover {
    background: var(--apple-blue-hover) !important;
    color: white !important;
  }

  /* Chat input — make the inline file-attach button discreet but visible */
  [data-testid="stChatInput"] [data-testid="stFileUploaderDropzone"],
  [data-testid="stChatInput"] button {
    background: transparent !important;
    border-radius: 999px !important;
  }

  /* Chat messages — calm white cards */
  [data-testid="stChatMessage"] {
    background: var(--apple-bg);
    border: 1px solid var(--apple-line);
    border-radius: var(--apple-radius);
    padding: 1.1rem 1.4rem;
    box-shadow: 0 1px 2px rgba(0,0,0,0.02);
    margin-bottom: 0.9rem;
    animation: fadeUp 0.35s ease;
  }
  [data-testid="stChatMessage"] p,
  [data-testid="stChatMessage"] li {
    color: var(--apple-ink);
    font-size: 0.98rem;
    line-height: 1.55;
    letter-spacing: -0.005em;
  }

  /* Chat input — pill shape with a Apple-blue focus ring on focus-within
     so the highlight follows the rounded outer shape, not the inner textarea.
     overflow:hidden prevents any inner white wrapper from bleeding past the
     rounded corners and breaking the blue outline at the bottom edge. */
  [data-testid="stChatInput"] {
    background: var(--apple-bg-soft);
    border-radius: 980px !important;
    border: 1px solid var(--apple-line) !important;
    padding: 0.25rem 0.5rem !important;
    overflow: hidden !important;
    transition: border-color 0.18s ease, box-shadow 0.18s ease !important;
  }
  [data-testid="stChatInput"]:focus-within {
    border-color: var(--apple-blue) !important;
    box-shadow: 0 0 0 3px rgba(0, 113, 227, 0.18) !important;
  }
  /* Make every inner wrapper transparent so the pill's background+border
     is the only thing the user sees. */
  [data-testid="stChatInput"] > div,
  [data-testid="stChatInput"] > div > div,
  [data-testid="stChatInput"] [data-testid="stChatInputContainer"],
  [data-testid="stChatInput"] [data-baseweb="textarea"],
  [data-testid="stChatInput"] [data-baseweb="textarea"] > div {
    background: transparent !important;
    border-radius: 0 !important;
  }
  /* Kill the inner rectangular outline on EVERY descendant so only the
     outer pill highlights on focus. (Streamlit nests its chat input in
     several wrapper divs, each of which may carry its own focus border.) */
  [data-testid="stChatInput"] *,
  [data-testid="stChatInput"] *:focus,
  [data-testid="stChatInput"] *:focus-within,
  [data-testid="stChatInput"] *:focus-visible,
  [data-testid="stChatInput"] *:hover,
  [data-testid="stChatInput"] *:active {
    outline: none !important;
    box-shadow: none !important;
    border-color: transparent !important;
  }
  [data-testid="stChatInput"] textarea {
    background: transparent !important;
    font-size: 1rem !important;
    color: var(--apple-ink) !important;
  }
  [data-testid="stChatInput"] textarea::placeholder {
    color: var(--apple-ink-soft) !important;
  }

  /* File uploader — keep Streamlit's default button structure to avoid
     icon/text collisions; just restyle the dropzone shell. */
  [data-testid="stFileUploaderDropzone"] {
    background: var(--apple-bg-soft);
    border: 1px dashed var(--apple-line);
    border-radius: var(--apple-radius-sm);
    transition: border-color 0.2s ease;
  }
  [data-testid="stFileUploaderDropzone"]:hover {
    border-color: var(--apple-blue);
  }
  [data-testid="stFileUploaderDropzoneInstructions"] {
    color: var(--apple-ink-soft);
    font-size: 0.85rem;
  }

  /* Confidence badge */
  .badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 980px;
    font-size: 0.78rem;
    font-weight: 500;
    letter-spacing: -0.005em;
    margin-top: 8px;
    border: 1px solid transparent;
  }
  .badge .dot {
    width: 6px; height: 6px; border-radius: 50%;
  }
  .badge-high   { background: #ecf8f1; color: var(--apple-green); border-color: #cdebd8; }
  .badge-high .dot   { background: var(--apple-green); }
  .badge-medium { background: #fdf3e7; color: var(--apple-amber); border-color: #f4dbb6; }
  .badge-medium .dot { background: var(--apple-amber); }
  .badge-low    { background: #fbecec; color: var(--apple-red); border-color: #f1c5c5; }
  .badge-low .dot    { background: var(--apple-red); }

  /* Product cards — Apple store style */
  .product-card {
    background: var(--apple-bg-soft);
    border-radius: var(--apple-radius);
    padding: 1.25rem 1rem 1rem 1rem;
    transition: transform 0.25s ease, box-shadow 0.25s ease;
    height: 100%;
    animation: fadeUp 0.4s ease;
  }
  .product-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 28px rgba(0,0,0,0.06);
  }
  .product-card img {
    border-radius: var(--apple-radius-sm);
    width: 100%;
    aspect-ratio: 1 / 1;
    object-fit: contain;
    background: white;
    margin-bottom: 1rem;
    padding: 0.5rem;
  }
  .product-name {
    font-weight: 600;
    font-size: 0.95rem;
    line-height: 1.3;
    color: var(--apple-ink);
    margin: 0 0 0.4rem 0;
    letter-spacing: -0.01em;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
    min-height: 2.5rem;
  }
  .product-meta {
    font-size: 0.82rem;
    color: var(--apple-ink-soft);
    margin: 3px 0;
    letter-spacing: -0.005em;
  }
  .product-price {
    font-weight: 500;
    color: var(--apple-ink);
    font-size: 1rem;
    margin: 8px 0 4px 0;
    letter-spacing: -0.01em;
  }
  .sim-pill {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 3px 10px;
    border-radius: 980px;
    font-size: 0.72rem;
    font-weight: 500;
    margin-top: 8px;
    background: white;
    border: 1px solid var(--apple-line);
    color: var(--apple-ink-soft);
  }
  .sim-pill .dot { width: 6px; height: 6px; border-radius: 50%; }
  .sim-high .dot   { background: var(--apple-green); }
  .sim-medium .dot { background: var(--apple-amber); }
  .sim-low .dot    { background: var(--apple-red); }

  /* Expander — flush, quiet */
  [data-testid="stExpander"] {
    background: transparent;
    border: 1px solid var(--apple-line);
    border-radius: var(--apple-radius);
    margin-top: 0.6rem;
  }
  [data-testid="stExpander"] summary {
    font-weight: 500;
    color: var(--apple-blue);
    font-size: 0.95rem;
  }
  [data-testid="stExpander"] summary:hover {
    background: transparent;
  }

  /* Section labels */
  .eyebrow {
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--apple-ink-soft);
    margin-bottom: 0.4rem;
  }

  /* Segmented control (prompt strategy) — Apple-style pill switcher */
  [data-testid="stSegmentedControl"] button {
    border-radius: 8px !important;
    font-size: 0.82rem !important;
    padding: 0.4rem 0.7rem !important;
    background: transparent !important;
    color: var(--apple-ink-soft) !important;
    border: none !important;
  }
  [data-testid="stSegmentedControl"] button[aria-pressed="true"] {
    background: white !important;
    color: var(--apple-ink) !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.06) !important;
  }
  [data-testid="stSegmentedControl"] > div {
    background: var(--apple-bg-soft) !important;
    border-radius: 10px !important;
    padding: 3px !important;
  }

  /* Footer */
  .footer {
    text-align: center;
    color: var(--apple-ink-soft);
    font-size: 0.78rem;
    margin: 3rem 0 1rem 0;
    padding-top: 1.5rem;
    border-top: 1px solid var(--apple-line);
    letter-spacing: -0.005em;
  }
  .footer a { color: var(--apple-blue); text-decoration: none; }

  /* Tighter overall padding */
  .block-container { padding-top: 1rem !important; max-width: 1180px; }
</style>
"""
st.markdown(APPLE_CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------
# Load RAG chain (cached so we only load CLIP once)
# ----------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading…")
def load_rag():
    chroma_dir = os.environ.get("CHROMA_DIR", "./chroma_db")
    return MultimodalRAG(chroma_dir=chroma_dir)


# ----------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------

if "history" not in st.session_state:
    st.session_state.history = []


# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------

with st.sidebar:
    st.markdown('<h2>Settings</h2>', unsafe_allow_html=True)

    top_k = st.slider("Results per query", 1, 10, 5,
                      help="How many products to retrieve.")

    # Segmented control — falls back to radio on Streamlit < 1.32
    try:
        prompt_style = st.segmented_control(
            "Prompt strategy",
            options=["zero-shot", "few-shot", "multi-shot"],
            default="few-shot",
            help="Zero-shot: system prompt only. Few-shot: + 2 examples. "
                 "Multi-shot: + 4 examples.",
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

    st.markdown("---")
    st.markdown('<h2>Engine</h2>', unsafe_allow_html=True)
    provider = os.getenv("LLM_PROVIDER", "groq")
    model = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
    st.markdown(f"<p style='font-size:0.85rem;color:#6e6e73;margin:0;'>Provider <span style='float:right;color:#1d1d1f;'>{provider}</span></p>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size:0.85rem;color:#6e6e73;margin:4px 0 0 0;'>Model <span style='float:right;color:#1d1d1f;'>llama-3.1-8b</span></p>", unsafe_allow_html=True)

    key_set = bool(os.getenv("GROQ_API_KEY") or os.getenv("TOGETHER_API_KEY"))
    if not key_set:
        st.error("No API key set. Add `GROQ_API_KEY` to `.env`.")

    st.markdown("---")
    if st.button("Clear conversation", type="primary"):
        st.session_state.history = []
        st.rerun()

    st.markdown(
        """
        <div style="position:absolute;bottom:1.5rem;left:1rem;right:1rem;
                    font-size:0.72rem;color:#86868b;line-height:1.5;
                    border-top:1px solid #d2d2d7;padding-top:1rem;">
          CLIP · ChromaDB · Llama 3.1 · Streamlit<br/>
          MSADS Final Project · 2026
        </div>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------
# Hero
# ----------------------------------------------------------------------

st.markdown(
    """
    <div class="hero">
      <h1>Shop smarter.<br/>With <span class="accent">vision and language</span>.</h1>
      <p class="subtitle">
        Ask anything about a product, or upload a photo. A CLIP-powered retriever
        finds the closest items in the catalog and Llama 3.1 explains them — grounded,
        never guessed.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Suggestion chips — short labels (full query is sent on click).
SUGGESTIONS = [
    ("Captain America Rug",
     "Tell me about the Marvel Captain America Shield Rug"),
    ("LEGO Taj Mahal",
     "Describe the LEGO Creator Taj Mahal building kit"),
    ("Princess Puzzle",
     "Tell me about the Mudpuppy Enchanting Princess Puzzle"),
    ("Pikachu Café",
     "What is the Mega Construx Pokemon Detective Pikachu Café?"),
]

# Render chips in a single centered row. Each chip auto-sizes; empty side
# columns center the cluster on the page.
_cols = st.columns([2, 1, 1, 1, 1, 2], gap="small")
for i, (label, query) in enumerate(SUGGESTIONS):
    with _cols[i + 1]:
        if st.button(label, key=f"chip_{i}", width="stretch"):
            st.session_state["pending_query"] = query
            st.rerun()


# ----------------------------------------------------------------------
# Load RAG
# ----------------------------------------------------------------------

try:
    rag = load_rag()
except Exception as e:
    st.error(f"Failed to load RAG chain: {e}")
    st.info("Run the indexing notebook + `scripts/cache_images.py` first.")
    st.stop()


# ----------------------------------------------------------------------
# Rendering helpers
# ----------------------------------------------------------------------

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


def render_product_grid(retrieved: list[RetrievedProduct]) -> None:
    """Render retrieved products as a clean Apple-style card grid."""
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
            st.markdown(
                f'<div class="product-name">{name[:80]}</div>',
                unsafe_allow_html=True,
            )

            if prod.brand:
                st.markdown(
                    f'<div class="product-meta">{prod.brand}</div>',
                    unsafe_allow_html=True,
                )
            if prod.category:
                st.markdown(
                    f'<div class="product-meta">{prod.category}</div>',
                    unsafe_allow_html=True,
                )
            if prod.price:
                st.markdown(
                    f'<div class="product-price">{prod.price}</div>',
                    unsafe_allow_html=True,
                )

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
    """Streamlit's markdown renderer treats `$…$` as inline LaTeX (KaTeX),
    so dollar-amounts in LLM responses render as green math instead of prose.
    Escape every `$` to neutralize the math parser."""
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
            conf = msg.get("confidence")
            top_sim = msg.get("top_sim", 0.0)
            if conf:
                render_confidence_badge(conf, top_sim)

            if show_retrieved and msg.get("retrieved"):
                with st.expander(f"View {len(msg['retrieved'])} retrieved products"):
                    render_product_grid(msg["retrieved"])


# ----------------------------------------------------------------------
# Render conversation history
# ----------------------------------------------------------------------

for msg in st.session_state.history:
    render_message(msg)


# ----------------------------------------------------------------------
# Input area — single chat bar at the bottom that accepts text AND images.
# (Streamlit 1.42+ supports accept_file on st.chat_input. Falls back to a
# separate uploader if the parameter isn't supported.)
# ----------------------------------------------------------------------

try:
    chat_value = st.chat_input(
        "Ask about a product, or attach a photo…",
        accept_file=True,
        file_type=["jpg", "jpeg", "png", "webp"],
    )
    chat_text = chat_value.text if chat_value else None
    uploaded_files = chat_value.files if chat_value else []
    uploaded = uploaded_files[0] if uploaded_files else None
except TypeError:
    # Older Streamlit — fall back to separate uploader below the chat input.
    chat_text = st.chat_input("Ask about a product…")
    uploaded = st.file_uploader(
        "Optional image",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )

# A chip click sets pending_query; otherwise fall back to chat input.
pending = st.session_state.pop("pending_query", None)
user_text = pending or chat_text


# ----------------------------------------------------------------------
# Process input
# ----------------------------------------------------------------------

if user_text or uploaded:
    pil_image = Image.open(uploaded) if uploaded else None

    user_msg = {
        "role": "user",
        "text": user_text or "(image-only query)",
        "image": pil_image,
    }
    st.session_state.history.append(user_msg)
    render_message(user_msg)

    with st.chat_message("assistant", avatar=None):
        with st.spinner("Thinking…"):
            try:
                response: RAGResponse = rag.answer(
                    query_text=user_text or None,
                    query_image=pil_image,
                    k=top_k,
                    prompt_style=prompt_style,
                )
                answer_text = response.answer
                retrieved = response.retrieved
                confidence = response.confidence
                top_sim = response.top_similarity
            except Exception as e:
                answer_text = f"Error: {e}"
                retrieved = []
                confidence = None
                top_sim = 0.0

        st.markdown(_escape_dollars(answer_text))

        if confidence:
            render_confidence_badge(confidence, top_sim)

        if show_retrieved and retrieved:
            with st.expander(f"View {len(retrieved)} retrieved products", expanded=True):
                render_product_grid(retrieved)

    st.session_state.history.append({
        "role": "assistant",
        "text": answer_text,
        "retrieved": retrieved,
        "confidence": confidence,
        "top_sim": top_sim,
    })


# Footer credits live in the sidebar (above the chat-input area), to keep
# the main canvas clean and avoid floating elements above the input bar.
