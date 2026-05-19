"""
End-to-end smoke test for the multimodal RAG pipeline (excluding the LLM call).

Exercises:
- CLIP load
- ChromaDB connect, both collections
- Text query retrieval
- Image query retrieval
- Combined text+image retrieval
- Prompt assembly under all three prompt styles
- Resolution of image paths against project root

Skips the actual LLM call so this works without a GROQ_API_KEY. Run after
preparing data + chroma index.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))
os.environ.setdefault("CHROMA_DIR", str(ROOT / "chroma_db"))

from PIL import Image
from rag_chain import MultimodalRAG

PROBES = [
    "wireless bluetooth headphones with noise cancellation",
    "puzzle for kids",
    "stand mixer for baking",
    "running shoes",
]


def main():
    rag = MultimodalRAG(chroma_dir=str(ROOT / "chroma_db"))

    print("\n--- Text-only retrieval ---")
    for q in PROBES:
        rs = rag.retrieve(query_text=q, k=3)
        print(f"\nQ: {q}")
        for i, p in enumerate(rs, 1):
            img_ok = "✓" if Path(p.image_path).exists() else "✗"
            print(f"  {i}. [{p.similarity:.3f}] {img_ok} {p.product_name[:70]}")

    # Pick a real cached image and use it as the image query
    img_dir = ROOT / "prepared_data" / "images"
    sample_imgs = list(img_dir.glob("*.jpg"))[:1]
    if sample_imgs:
        print("\n--- Image-only retrieval ---")
        img = Image.open(sample_imgs[0])
        rs = rag.retrieve(query_image=img, k=3)
        print(f"Q (image): {sample_imgs[0].name}")
        for i, p in enumerate(rs, 1):
            print(f"  {i}. [{p.similarity:.3f}] {p.product_name[:70]}")

        print("\n--- Combined text + image retrieval ---")
        rs = rag.retrieve(query_text="toy", query_image=img, k=3)
        for i, p in enumerate(rs, 1):
            print(f"  {i}. [{p.similarity:.3f}] {p.product_name[:70]}")

    # Prompt assembly under each style
    print("\n--- Prompt assembly ---")
    rs = rag.retrieve(query_text=PROBES[0], k=3)
    for style in ["zero-shot", "few-shot", "multi-shot"]:
        msgs = rag.build_prompt(PROBES[0], rs, prompt_style=style)
        print(f"  {style}: {len(msgs)} messages  ({sum(len(m['content']) for m in msgs)} chars)")

    # Verify clean context cards drop nan brand/category
    print("\n--- Sample context card ---")
    if rs:
        print(rs[0].as_context_card())

    print("\nSmoke test passed.")


if __name__ == "__main__":
    main()
