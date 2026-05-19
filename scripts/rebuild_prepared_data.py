"""
Rebuild prepared_data/products_indexed.parquet from the existing ChromaDB index.

The original indexing happened in Colab, so the local repo doesn't have the
parquet that Phase 1 produces. Phase 3 (eval) and the report need it. This
script reconstructs it from the chroma metadata we already have.
"""

from pathlib import Path
import pandas as pd
import chromadb

ROOT = Path(__file__).resolve().parent.parent
CHROMA_DIR = ROOT / "chroma_db"
OUT_DIR = ROOT / "prepared_data"
OUT_DIR.mkdir(exist_ok=True)


def _clean(v):
    if v is None:
        return ""
    s = str(v).strip()
    return "" if s.lower() == "nan" else s


def main():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    text_col = client.get_collection("products_text")
    n = text_col.count()
    print(f"Reading {n} entries from products_text...")

    data = text_col.get(include=["metadatas", "embeddings"])
    rows = []
    for pid, meta in zip(data["ids"], data["metadatas"]):
        rows.append({
            "product_id": pid,
            "Product Name": _clean(meta.get("product_name")),
            "Brand Name": _clean(meta.get("brand")),
            "top_category": _clean(meta.get("category")) or "Unknown",
            "Selling Price": _clean(meta.get("price")),
            "About Product": _clean(meta.get("about")),
            "desc_standard": _clean(meta.get("desc")),
            "primary_image_url": _clean(meta.get("image_url")),
            "local_image_path": _clean(meta.get("image_path")),
        })

    df = pd.DataFrame(rows).sort_values("product_id").reset_index(drop=True)
    out = OUT_DIR / "products_indexed.parquet"
    df.to_parquet(out, index=False)
    print(f"Wrote {out} ({len(df)} rows)")
    print(f"Categories: {df['top_category'].nunique()}")
    print(df["top_category"].value_counts().head(10))


if __name__ == "__main__":
    main()
