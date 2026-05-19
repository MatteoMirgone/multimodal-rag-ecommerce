"""
rag_chain.py
============
RAG retrieval + LLM grounding for the multimodal e-commerce chatbot.

This module is intentionally framework-agnostic — the Streamlit app
imports from here, and you can also import it for offline eval or scripting.

Pipeline:
1. Encode the user query (text and/or image) with CLIP.
2. Retrieve top-k products from ChromaDB.
3. Build a grounded prompt with retrieved product context.
4. Call the LLM (Llama 3.1 via Groq by default — easy to swap).
5. Return the LLM response + the retrieved products (for rendering).

Configure the LLM via environment variables:
    GROQ_API_KEY     — required for Groq (free tier: https://console.groq.com)
    LLM_PROVIDER     — 'groq' (default) or 'together'
    LLM_MODEL        — defaults to 'llama-3.1-8b-instant' (Groq)
    TOGETHER_API_KEY — required if LLM_PROVIDER='together'
"""

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
import chromadb


# Load a .env at the project root, if present. Kept minimal (no python-dotenv
# dependency) — just KEY=value lines, comments OK.
def _load_dotenv() -> None:
    for candidate in [Path.cwd() / ".env", Path(__file__).resolve().parent.parent / ".env"]:
        if not candidate.exists():
            continue
        for line in candidate.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v
        break


_load_dotenv()

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

DEFAULT_MODEL_NAME = "openai/clip-vit-base-patch32"
DEFAULT_CHROMA_DIR = "./chroma_db"
DEFAULT_TOP_K = 5

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()
LLM_MODEL = os.getenv(
    "LLM_MODEL",
    "llama-3.1-8b-instant" if LLM_PROVIDER == "groq" else "meta-llama/Llama-3.1-8B-Instruct-Turbo",
)


# ----------------------------------------------------------------------
# Data classes
# ----------------------------------------------------------------------

def _clean(v: Any) -> str:
    """Normalize chroma metadata strings — empty 'nan'/'None' values come
    through as the literal string 'nan' from pandas, which would leak into
    the LLM prompt and UI as 'Brand: nan'."""
    if v is None:
        return ""
    s = str(v).strip()
    if s.lower() in {"nan", "none", "null"}:
        return ""
    return s


def _clean_price(v: Any) -> str:
    """Specialized cleaner for price strings.

    The Amazon CSV has ~2% garbage values in Selling Price: 'Total price:',
    'from 7 sellers', 'None', etc. Only keep values that look like a real
    price (start with $ or a digit, contain a digit somewhere)."""
    s = _clean(v)
    if not s:
        return ""
    has_digit = any(ch.isdigit() for ch in s)
    looks_like_price = s.startswith("$") or (s[:1].isdigit() if s else False)
    if not (has_digit and looks_like_price):
        return ""
    return s


@dataclass
class RetrievedProduct:
    product_id: str
    product_name: str
    brand: str
    category: str
    price: str
    about: str
    image_path: str
    image_url: str
    similarity: float

    def as_context_card(self) -> str:
        """Format this product for the LLM prompt, dropping empty fields."""
        lines = [f"Product: {self.product_name}"]
        if self.brand:
            lines.append(f"Brand: {self.brand}")
        if self.category:
            lines.append(f"Category: {self.category}")
        if self.price:
            lines.append(f"Price: {self.price}")
        if self.about:
            lines.append(f"Description: {self.about[:500]}")
        return "\n".join(lines)


@dataclass
class RAGResponse:
    answer: str
    retrieved: List[RetrievedProduct]
    query_text: Optional[str]
    query_image_path: Optional[str]
    confidence: str = "medium"  # "high" | "medium" | "low"
    top_similarity: float = 0.0


# ----------------------------------------------------------------------
# Core RAG chain
# ----------------------------------------------------------------------

class MultimodalRAG:
    """Encapsulates CLIP encoding, ChromaDB retrieval, and LLM grounding."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        chroma_dir: str = DEFAULT_CHROMA_DIR,
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[RAG] Loading CLIP ({model_name}) on {self.device}...")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)

        print(f"[RAG] Connecting to ChromaDB at {chroma_dir}...")
        self.client = chromadb.PersistentClient(path=chroma_dir)
        self.text_col = self.client.get_collection("products_text")
        self.image_col = self.client.get_collection("products_image")
        print(f"[RAG] Ready. {self.text_col.count()} products indexed.")

        # Build an in-memory keyword index over product names+categories. Used
        # for hybrid retrieval — CLIP alone is weak on short generic queries
        # ('skates', 'roller blades') because cosine similarity favors verbose
        # descriptions over precise lexical matches. The keyword index gives
        # us a fast lexical signal we merge with CLIP via Reciprocal Rank
        # Fusion. 1,991 products is small enough to keep entirely in RAM.
        self._build_keyword_index()

    # -- Encoding --------------------------------------------------------

    @staticmethod
    def _l2(x: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(x, axis=-1, keepdims=True)
        return x / np.where(n == 0, 1e-12, n)

    @torch.no_grad()
    def encode_text(self, text: str) -> np.ndarray:
        inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True, max_length=77)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        feats = self.model.get_text_features(**inputs).cpu().numpy()
        return self._l2(feats)[0]

    @torch.no_grad()
    def encode_image(self, image: Image.Image) -> np.ndarray:
        img = image.convert("RGB")
        inputs = self.processor(images=[img], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        feats = self.model.get_image_features(**inputs).cpu().numpy()
        return self._l2(feats)[0]

    # -- Retrieval -------------------------------------------------------

    def _query_collection(self, collection, query_emb: np.ndarray, k: int) -> List[RetrievedProduct]:
        res = collection.query(
            query_embeddings=[query_emb.tolist()],
            n_results=k,
            include=["metadatas", "distances"],
        )
        out = []
        for meta, dist in zip(res["metadatas"][0], res["distances"][0]):
            # Chroma returns 1 - cosine_similarity for cosine space
            similarity = 1.0 - dist
            out.append(RetrievedProduct(
                product_id=_clean(meta.get("product_id")),
                product_name=_clean(meta.get("product_name")),
                brand=_clean(meta.get("brand")),
                category=_clean(meta.get("category")),
                price=_clean_price(meta.get("price")),
                about=_clean(meta.get("about")),
                image_path=self._resolve_image_path(_clean(meta.get("image_path"))),
                image_url=_clean(meta.get("image_url")),
                similarity=similarity,
            ))
        return out

    # -- Keyword / hybrid retrieval --------------------------------------

    _KW_STOPWORDS = {
        "the", "a", "an", "and", "or", "of", "for", "to", "in", "on", "with", "is",
        "are", "this", "that", "what", "show", "tell", "me", "about", "can", "you",
        "please", "do", "have", "any", "from", "by", "at", "as", "it", "be", "give",
        "would", "could", "should", "i", "we", "us", "my", "your", "looking", "find",
        "want", "need", "get", "buy", "purchase", "price", "cost", "available",
    }

    # Common singular↔plural pairs to bridge query/product-name mismatch
    # ('skates' query vs 'skate' in product name, 'puzzles' vs 'puzzle', etc.)
    _PLURAL_MAP = {
        "skates": "skate", "blades": "blade", "puzzles": "puzzle", "games": "game",
        "toys": "toy", "rugs": "rug", "kits": "kit", "figures": "figure",
        "shoes": "shoe", "cards": "card", "wheels": "wheel", "blocks": "block",
    }

    def _build_keyword_index(self) -> None:
        """Cache product metadata for keyword matching. Each entry stores the
        product_id, lowercased name, lowercased category, and (deduped) token
        set so lookups are O(N) with a constant-time set check."""
        all_data = self.text_col.get(include=["metadatas"])
        self._kw_index: List[Dict[str, Any]] = []
        for pid, meta in zip(all_data["ids"], all_data["metadatas"]):
            name = _clean(meta.get("product_name")).lower()
            cat = _clean(meta.get("category")).lower()
            tokens = set(re.findall(r"\w+", name + " " + cat))
            self._kw_index.append({
                "id": pid,
                "name": name,
                "category": cat,
                "tokens": tokens,
                "meta": meta,
            })
        print(f"[RAG] Keyword index built ({len(self._kw_index)} products).")

    @classmethod
    def _query_keywords(cls, query: str) -> List[str]:
        """Extract meaningful tokens from a free-text query."""
        out: List[str] = []
        for w in re.findall(r"\w+", query.lower()):
            if len(w) < 3 or w in cls._KW_STOPWORDS:
                continue
            out.append(w)
            # Also include the singular form for plural-aware matching
            if w in cls._PLURAL_MAP:
                out.append(cls._PLURAL_MAP[w])
        # Dedup while preserving order
        seen = set()
        return [w for w in out if not (w in seen or seen.add(w))]

    def _keyword_retrieve(self, query: str, k: int) -> List[RetrievedProduct]:
        """Score each indexed product by how many query keywords appear in its
        name/category. Returns top-k by score (ties broken by shorter name —
        more precise matches).

        Multi-keyword matches are weighted SUPER-linearly: a 2-of-2 hit beats
        a 1-of-2 hit by far more than 2× (squared scaling). This is important
        because queries like 'cat costume' need to surface a product whose
        name contains BOTH 'cat' and 'costume' (Amscan Miss Meow Cat Costume)
        over one that only contains 'cat' (Cat Accessory Kit)."""
        kws = self._query_keywords(query)
        if not kws:
            return []

        scored = []
        for entry in self._kw_index:
            tokens = entry["tokens"]
            name = entry["name"]
            # Token-set match (whole word) — primary signal
            token_hits = sum(1 for kw in kws if kw in tokens)
            # Substring fallback so 'skate' matches 'rollerskates' etc.
            substr_hits = sum(1 for kw in kws if kw in name)
            raw = max(token_hits, substr_hits * 0.7)
            if raw <= 0:
                continue
            # Squared scaling: 2-of-2 → 1.0, 1-of-2 → 0.25, 3-of-3 → 1.0,
            # 2-of-3 → 0.44, 1-of-3 → 0.11. Disproportionately rewards
            # complete-keyword matches over partial-keyword matches.
            denom = max(1, len(kws))
            norm = (raw / denom) ** 2
            scored.append((entry, norm, len(name)))

        scored.sort(key=lambda x: (-x[1], x[2]))
        top = scored[:k]

        out = []
        for entry, score, _ in top:
            meta = entry["meta"]
            out.append(RetrievedProduct(
                product_id=_clean(meta.get("product_id")),
                product_name=_clean(meta.get("product_name")),
                brand=_clean(meta.get("brand")),
                category=_clean(meta.get("category")),
                price=_clean_price(meta.get("price")),
                about=_clean(meta.get("about")),
                image_path=self._resolve_image_path(_clean(meta.get("image_path"))),
                image_url=_clean(meta.get("image_url")),
                similarity=float(score),  # keyword score in [0, 1]; not a cosine
            ))
        return out

    @staticmethod
    def _rrf_merge(
        rankings: List[List[RetrievedProduct]],
        k: int,
        rrf_c: int = 20,
    ) -> List[RetrievedProduct]:
        """Reciprocal Rank Fusion across multiple rankings. Each rank list
        contributes 1/(rrf_c + rank) to a product's fused score. Standard k=60.
        Cosine similarity from CLIP is preserved on the returned products
        (taken from the highest-cosine occurrence) so downstream confidence
        scoring still has the original signal."""
        fused: Dict[str, Dict[str, Any]] = {}
        for ranking in rankings:
            for rank, prod in enumerate(ranking, start=1):
                slot = fused.setdefault(prod.product_id, {"score": 0.0, "prod": prod})
                slot["score"] += 1.0 / (rrf_c + rank)
                # Keep the version of the product with the higher similarity
                # (so CLIP's cosine isn't overwritten by the keyword score)
                if prod.similarity > slot["prod"].similarity:
                    slot["prod"] = prod

        ranked = sorted(fused.values(), key=lambda r: r["score"], reverse=True)[:k]
        return [r["prod"] for r in ranked]

    @staticmethod
    def _resolve_image_path(path: str) -> str:
        """Image paths stored in chroma are relative to the project root
        (e.g. 'prepared_data/images/prod_000000.jpg'). The app runs from
        the app/ subdir, so resolve against PROJECT_ROOT."""
        if not path:
            return ""
        p = Path(path)
        if p.is_absolute() and p.exists():
            return str(p)
        candidates = [
            Path.cwd() / p,
            Path(__file__).resolve().parent.parent / p,
            Path(__file__).resolve().parent / p,
        ]
        for c in candidates:
            if c.exists():
                return str(c)
        return str(candidates[1])  # default to project-root-relative, even if missing

    def retrieve(
        self,
        query_text: Optional[str] = None,
        query_image: Optional[Image.Image] = None,
        k: int = DEFAULT_TOP_K,
    ) -> List[RetrievedProduct]:
        """Hybrid retrieval — CLIP vector search merged with keyword matching
        via Reciprocal Rank Fusion.

        Strategy:
        - Text query: CLIP-text → top 20 vector candidates, plus keyword
          search over product names → top 20 lexical candidates. RRF merge.
        - Image query: CLIP-image → top-k (vector only; keyword doesn't apply).
        - Combined text + image: hybrid for the text side, image vector on
          top of that, merged by max similarity.
        """
        if query_text is None and query_image is None:
            raise ValueError("Provide query_text or query_image (or both).")

        OVERFETCH = 20  # candidates per side before fusion
        rankings: List[List[RetrievedProduct]] = []

        # Decide whether the text query carries actual information. Queries
        # like 'price of', 'how much', 'what is this' tokenize to ZERO
        # meaningful keywords once stopwords are removed. When that happens
        # AND an image was uploaded, the text query is just grammatical
        # connective tissue — running CLIP-text on it returns noise that
        # outranks the genuine image match. So suppress the text-side.
        text_has_signal = bool(query_text and self._query_keywords(query_text))

        if query_text and (text_has_signal or query_image is None):
            text_emb = self.encode_text(query_text)
            vec_hits = self._query_collection(self.text_col, text_emb, OVERFETCH)
            rankings.append(vec_hits)

            kw_hits = self._keyword_retrieve(query_text, OVERFETCH)
            if kw_hits:
                # Boost CLIP cosine on products that also match lexically —
                # gives downstream confidence scoring a stronger signal when
                # both retrievers agree.
                vec_by_id = {p.product_id: p for p in vec_hits}
                for kw in kw_hits:
                    if kw.product_id in vec_by_id:
                        kw.similarity = max(kw.similarity, vec_by_id[kw.product_id].similarity)
                rankings.append(kw_hits)

        # Image side
        img_top_hit: Optional[RetrievedProduct] = None
        if query_image is not None:
            img_emb = self.encode_image(query_image)
            img_hits = self._query_collection(self.image_col, img_emb, OVERFETCH)
            rankings.append(img_hits)
            # Remember the image's top hit — if it's an exact-or-near-exact
            # match, we'll force it to position 0 after fusion.
            if img_hits:
                img_top_hit = img_hits[0]

        if not rankings:
            return []

        if len(rankings) == 1:
            merged = rankings[0][:k]
        else:
            merged = self._rrf_merge(rankings, k=k)

        # Strong-image-match override: if the image retrieval returned an
        # essentially identical match (cosine ≥ EXACT_MATCH), it IS the
        # product the user uploaded a photo of. Lift it to position 0 so
        # the confidence scorer and the LLM both see it as the top hit.
        if img_top_hit is not None and img_top_hit.similarity >= self.EXACT_MATCH:
            merged = [img_top_hit] + [
                p for p in merged if p.product_id != img_top_hit.product_id
            ]

        # Lexical anchor: if the user's text query has ≥2 meaningful keywords
        # AND one of the retrieved products contains ALL of those keywords as
        # whole-word tokens in its NAME, promote it to position 0. This catches
        # cases like 'cat costume' where the literal "Cat Costume" product
        # should beat a semantically-cat-themed-but-differently-named one.
        if query_text:
            kws = self._query_keywords(query_text)
            if len(kws) >= 2:
                for i, p in enumerate(merged):
                    name_tokens = set(re.findall(r"\w+", p.product_name.lower()))
                    if all(kw in name_tokens for kw in kws):
                        if i != 0:
                            merged = [p] + [q for j, q in enumerate(merged) if j != i]
                        break

        return merged

    # -- Prompting -------------------------------------------------------

    # Exemplars use INVENTED placeholder product names like "{PRODUCT_A}" so
    # the LLM cannot accidentally name them in real responses. (Earlier versions
    # used "Apple AirPods Pro" and "KitchenAid Mixer" as exemplar products;
    # the model kept leaking those exact names into real answers, even when
    # the user had asked about something completely different.)
    _EXEMPLARS_TEXT: List[str] = [
        # 1. HIGH confidence, text feature question
        "EXAMPLE 1 — HIGH confidence text query\n"
        "Retrieval confidence: HIGH (top similarity 0.91).\n"
        "Retrieved: [1] (0.91) Product: {PRODUCT_A} | Brand: BrandA | Price: $99 | "
        "Description: Feature 1, feature 2, feature 3.\n"
        "Question: What are the features of {PRODUCT_A}?\n"
        "Answer: The {PRODUCT_A} offers feature 1, feature 2, and feature 3. At $99 "
        "it's positioned as a mid-tier option in this category.",
        # 2. HIGH confidence, image identification
        "EXAMPLE 2 — HIGH confidence image query\n"
        "Retrieval confidence: HIGH (top similarity 0.97).\n"
        "Retrieved: [1] (0.97) Product: {PRODUCT_B} | Category: Home & Kitchen | "
        "Description: Used for task X, includes attachments Y and Z.\n"
        "Question: Identify the product in the uploaded image.\n"
        "Answer: This is the {PRODUCT_B}. It's used for task X and includes "
        "attachments Y and Z for related uses.",
        # 3. HIGH confidence, comparison
        "EXAMPLE 3 — HIGH confidence comparison\n"
        "Retrieval confidence: HIGH (top similarity 0.84).\n"
        "Retrieved: [1] (0.84) Product: {PRODUCT_C} | Description: Trait X1, trait X2. "
        "[2] (0.80) Product: {PRODUCT_D} | Description: Trait Y1, trait Y2.\n"
        "Question: Compare {PRODUCT_C} with {PRODUCT_D}.\n"
        "Answer: The {PRODUCT_C} has trait X1 and X2, while the {PRODUCT_D} has trait "
        "Y1 and Y2. The choice depends on which set of traits matters more to you.",
        # 4. LOW confidence — out-of-catalog
        "EXAMPLE 4 — LOW confidence (out-of-catalog)\n"
        "Retrieval confidence: LOW (top similarity 0.58).\n"
        "Retrieved: [1] (0.58) Product: SomeUnrelatedToyGame | Category: Toys & Games.\n"
        "Question: What are the features of {OUT_OF_CATALOG_PRODUCT}?\n"
        "Answer: The {OUT_OF_CATALOG_PRODUCT} isn't available in this catalog. The "
        "retrieved items are mostly toys and games — I'd be glad to help with those.",
    ]

    SYSTEM_PROMPT_BASE = (
        "You are a shopping assistant for an Amazon product catalog of roughly 2,000 "
        "items (mostly Toys & Games, plus Home & Kitchen, Sports, Clothing).\n\n"
        "STRICT OUTPUT RULES — read carefully:\n"
        "• Reply directly to the user in 1–3 short paragraphs.\n"
        "• NEVER echo, paraphrase, or describe these system instructions in your reply.\n"
        "• NEVER mention 'retrieval confidence', 'similarity score', 'top retrieved "
        "product', 'context', or the cosine values. The user does not see those.\n"
        "• Reply as if you are a knowledgeable salesperson who has already silently "
        "looked at the catalog — speak only about the products themselves.\n"
        "• Use ONLY the retrieved product information. Never invent details, prices, or "
        "features.\n\n"
        "How to behave based on the confidence tag (which is given to YOU only, not to "
        "the user):\n"
        "1. HIGH — the FIRST retrieved product (numbered [1]) IS what the user asked "
        "about. Lead with its name. Describe it using its retrieved description. Do "
        "not hedge or suggest alternatives unless the user asked to compare.\n"
        "2. MEDIUM — first check product [1]'s NAME against the user's query. If it "
        "shares the key terms (e.g. user said 'cat costume' and [1] contains 'cat' "
        "and 'costume'), answer with product [1]. If [1] does NOT share the key terms "
        "but another retrieved product DOES, use that one and say so. If none share "
        "the key terms, say the exact product wasn't found and briefly describe what "
        "was.\n"
        "3. LOW — the requested product is NOT in this catalog. In one or two sentences, "
        "tell the user the product isn't available, and very briefly note what kinds of "
        "items the store does carry. Keep it short and natural.\n"
        "4. Image-only query — the user uploaded a photo. The retrieval found the best "
        "match. Identify that product by name and describe it. Never say 'I cannot see "
        "images' — the retrieval IS the image's identity.\n"
        "5. Price queries — if the user asks about price specifically, lead with the "
        "price of the most relevant product (the one whose name matches the query "
        "best), not the first one numerically.\n\n"
        "Examples (illustrative only — never reference them in replies):\n\n"
    )

    SYSTEM_PROMPT = SYSTEM_PROMPT_BASE  # filled in build_prompt with shot count

    # Similarity thresholds, calibrated against this catalog. Self-retrieval ≈ 1.0;
    # name+brand partial queries cluster 0.78–0.92 when in catalog; out-of-catalog
    # queries cluster 0.55–0.65. EXACT_MATCH catches image self-retrieval (sim ≥ 0.95).
    # LOW threshold raised to 0.65 so AirPods-Pro-style queries (sim ≈ 0.61 against
    # a toy-heavy catalog) get tagged LOW, not MEDIUM.
    EXACT_MATCH    = 0.95
    HIGH_CONFIDENCE = 0.72
    LOW_CONFIDENCE  = 0.65

    _STOPWORDS = {
        "the", "a", "an", "and", "or", "of", "for", "to", "in", "on", "with", "is",
        "are", "this", "that", "what", "show", "tell", "me", "about", "can", "you",
        "please", "do", "have", "any", "from", "by", "at", "as", "it", "be", "give",
    }

    @classmethod
    def _name_overlap(cls, query: Optional[str], product_name: str) -> int:
        """Count meaningful tokens that appear in both query and product name.

        Used to boost confidence when the user typed words that literally appear in
        the top retrieved product's name — e.g. user 'Captain America Shield Rug'
        vs product 'Gertmenian: Marvel Captain America Shield Rug HD'. That's a
        4-word overlap and should be treated as a high-confidence match even if
        the cosine score is only moderate.
        """
        if not query or not product_name:
            return 0
        q_tokens = {t for t in re.findall(r"\w+", query.lower())
                    if len(t) >= 4 and t not in cls._STOPWORDS}
        n_tokens = {t for t in re.findall(r"\w+", product_name.lower())
                    if len(t) >= 4 and t not in cls._STOPWORDS}
        return len(q_tokens & n_tokens)

    @classmethod
    def _confidence_level(
        cls,
        retrieved: List[RetrievedProduct],
        query_text: Optional[str] = None,
    ) -> str:
        if not retrieved:
            return "low"
        top = retrieved[0]
        sim = top.similarity

        # EXACT_MATCH (≥ 0.95) — typically image self-retrieval. Cannot be
        # anything other than high confidence.
        if sim >= cls.EXACT_MATCH:
            return "high"

        # Text-overlap escalation: if the user's named query words literally
        # appear in the top product name, that's a strong signal that the
        # product is the one they meant, even at lower cosine scores.
        overlap = cls._name_overlap(query_text, top.product_name)
        if overlap >= 3 and sim >= 0.55:
            return "high"
        if overlap >= 2 and sim >= 0.65:
            return "high"

        if sim >= cls.HIGH_CONFIDENCE:
            return "high"
        if sim >= cls.LOW_CONFIDENCE:
            return "medium"
        return "low"

    @classmethod
    def build_prompt(
        cls,
        query_text: Optional[str],
        retrieved: List[RetrievedProduct],
        prompt_style: str = "few-shot",
    ) -> List[Dict[str, str]]:
        """Build chat-format messages for the LLM.

        prompt_style:
          - 'zero-shot'  : system prompt only
          - 'few-shot'   : system prompt + 2 inline exemplars (default)
          - 'multi-shot' : system prompt + 4 inline exemplars

        IMPORTANT: exemplars are embedded inside the system message as plain
        text, NOT as alternating user/assistant turns. Earlier versions used
        chat-role exemplars, but the model would treat them as prior
        conversation history and "remember" exemplar products as things the
        customer had previously asked about. Plain-text examples in the
        system message eliminate that leakage.
        """
        confidence = cls._confidence_level(retrieved, query_text=query_text)
        top_sim = retrieved[0].similarity if retrieved else 0.0
        is_image_only = not query_text

        # ----------------------------------------------------------------
        # System message = base rules + examples + current-turn confidence tag.
        # The confidence is communicated to the LLM here, NOT in the user turn,
        # so the LLM can't echo it back to the customer.
        # ----------------------------------------------------------------
        n_shots = {"zero-shot": 0, "few-shot": 2, "multi-shot": 4}.get(prompt_style, 2)
        if n_shots > 0:
            examples_block = "\n\n".join(cls._EXEMPLARS_TEXT[:n_shots])
        else:
            examples_block = "(No exemplars under zero-shot.)"

        confidence_tag = (
            f"\n\n[CURRENT TURN]\n"
            f"Confidence for this turn: {confidence.upper()} "
            f"(internal similarity {top_sim:.2f}; do not mention either in your reply).\n"
            f"Query type: {'image-only' if is_image_only else 'text'}\n"
        )

        system_content = cls.SYSTEM_PROMPT_BASE + examples_block + confidence_tag

        # ----------------------------------------------------------------
        # User turn = clean data only. No instructions, no confidence tag.
        # Just the products and the question.
        # ----------------------------------------------------------------
        context_parts = ["Retrieved products from the catalog:"]
        for i, p in enumerate(retrieved, 1):
            context_parts.append(f"\n[{i}] {p.as_context_card()}")
        context = "\n".join(context_parts)

        if is_image_only:
            user_question = (
                "I just uploaded a product image. What is this product?"
            )
        else:
            user_question = query_text

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": f"{context}\n\nUser question: {user_question}"},
        ]
        return messages

    # -- LLM -------------------------------------------------------------

    def call_llm(self, messages: List[Dict[str, str]], temperature: float = 0.3) -> str:
        """Call the configured LLM provider."""
        if LLM_PROVIDER == "groq":
            return self._call_groq(messages, temperature)
        elif LLM_PROVIDER == "together":
            return self._call_together(messages, temperature)
        else:
            raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER}")

    @staticmethod
    def _call_groq(messages, temperature):
        from groq import Groq
        key = os.getenv("GROQ_API_KEY")
        if not key:
            raise RuntimeError("Set GROQ_API_KEY env var (https://console.groq.com)")
        client = Groq(api_key=key)
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=512,
        )
        return resp.choices[0].message.content

    @staticmethod
    def _call_together(messages, temperature):
        from together import Together
        key = os.getenv("TOGETHER_API_KEY")
        if not key:
            raise RuntimeError("Set TOGETHER_API_KEY env var (https://api.together.xyz)")
        client = Together(api_key=key)
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=512,
        )
        return resp.choices[0].message.content

    # -- Top-level call --------------------------------------------------

    def answer(
        self,
        query_text: Optional[str] = None,
        query_image: Optional[Image.Image] = None,
        k: int = DEFAULT_TOP_K,
        prompt_style: str = "few-shot",
    ) -> RAGResponse:
        """Full pipeline: retrieve + ground + generate.

        prompt_style: one of 'zero-shot', 'few-shot', 'multi-shot'.
        """
        retrieved = self.retrieve(query_text=query_text, query_image=query_image, k=k)
        messages = self.build_prompt(
            query_text=query_text, retrieved=retrieved, prompt_style=prompt_style
        )
        answer = self.call_llm(messages)
        confidence = self._confidence_level(retrieved, query_text=query_text)
        top_sim = retrieved[0].similarity if retrieved else 0.0
        return RAGResponse(
            answer=answer,
            retrieved=retrieved,
            query_text=query_text,
            query_image_path=None,
            confidence=confidence,
            top_similarity=top_sim,
        )


# ----------------------------------------------------------------------
# CLI quick-test
# ----------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    rag = MultimodalRAG()
    q = " ".join(sys.argv[1:]) or "wireless bluetooth headphones with noise cancellation"
    resp = rag.answer(query_text=q, k=5)
    print("\n=== ANSWER ===\n", resp.answer)
    print("\n=== RETRIEVED ===")
    for i, p in enumerate(resp.retrieved, 1):
        print(f"  {i}. ({p.similarity:.3f}) {p.product_name[:70]}")
