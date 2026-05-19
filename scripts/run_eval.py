"""
Phase 3 — Retrieval Evaluation as a script.

Mirrors notebooks/03_retrieval_eval.ipynb but runs end-to-end without a
notebook server, writing all artifacts the report needs to eval_results/:
- recall_summary.csv
- per_query_results.csv
- recall_by_category.csv
- recall_chart.png

Evaluation regimes (matches the spec's call for Recall@1/5/10):
  1. Text self-retrieval     (sanity: each product's desc → itself)
  2. Image self-retrieval    (sanity: each product's image → itself)
  3. Partial-text retrieval  (realistic: name+brand → text index)
  4. Cross-modal: image → text
  5. Cross-modal: text → image
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor
import chromadb

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "prepared_data"
CHROMA = ROOT / "chroma_db"
OUT = ROOT / "eval_results"
OUT.mkdir(exist_ok=True)

MODEL_NAME = "openai/clip-vit-base-patch32"
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
N_EVAL = 500
K_VALUES = [1, 5, 10]
SEED = 42


def l2(x):
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.where(n == 0, 1e-12, n)


@torch.no_grad()
def encode_text(model, processor, texts, batch=64):
    out = []
    for i in range(0, len(texts), batch):
        b = texts[i:i + batch]
        inputs = processor(text=b, return_tensors="pt", padding=True, truncation=True, max_length=77)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        f = model.get_text_features(**inputs).cpu().numpy()
        out.append(f)
    return l2(np.vstack(out))


@torch.no_grad()
def encode_images(model, processor, paths, batch=32):
    out = []
    for i in tqdm(range(0, len(paths), batch), desc="encode_images"):
        bp = paths[i:i + batch]
        imgs, idxs = [], []
        for j, p in enumerate(bp):
            try:
                imgs.append(Image.open(p).convert("RGB"))
                idxs.append(j)
            except Exception:
                pass
        if not imgs:
            out.append(np.zeros((len(bp), 512), dtype=np.float32))
            continue
        inputs = processor(images=imgs, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        f = model.get_image_features(**inputs).cpu().numpy()
        emb = np.zeros((len(bp), f.shape[1]), dtype=np.float32)
        for k, ix in enumerate(idxs):
            emb[ix] = f[k]
        out.append(emb)
    return l2(np.vstack(out))


def recall_rows(query_embs, gold_ids, collection, max_k=10, label=""):
    rows = []
    batch = 32
    for i in tqdm(range(0, len(query_embs), batch), desc=f"query[{label}]"):
        embs = query_embs[i:i + batch]
        gold = gold_ids[i:i + batch]
        res = collection.query(
            query_embeddings=embs.tolist(),
            n_results=max_k,
            include=["distances"],
        )
        for j, g in enumerate(gold):
            retrieved = res["ids"][j]
            row = {"gold_id": g, "query_type": label}
            for k in K_VALUES:
                row[f"recall@{k}"] = int(g in retrieved[:k])
            rows.append(row)
    return rows


def main():
    print(f"Device: {DEVICE}")
    print(f"Loading {MODEL_NAME}...")
    model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)

    df = pd.read_parquet(DATA / "products_indexed.parquet")
    print(f"Indexed catalog: {len(df)} products, {df['top_category'].nunique()} categories")

    # Keep only rows whose image file actually exists on disk — needed for
    # the image-side eval regimes.
    img_root = DATA
    df["image_exists"] = df["local_image_path"].apply(
        lambda p: (img_root.parent / p if not Path(p).is_absolute() else Path(p)).exists()
        or (ROOT / p).exists()
    )
    df["resolved_image"] = df["local_image_path"].apply(lambda p: str(ROOT / p))
    print(f"Images available locally: {df['image_exists'].sum()}/{len(df)}")

    client = chromadb.PersistentClient(path=str(CHROMA))
    text_col = client.get_collection("products_text")
    image_col = client.get_collection("products_image")

    # Eval sample — stratify on top_category so long-tail categories show up
    eval_df = (
        df[df["image_exists"]]
        .sample(min(N_EVAL, df["image_exists"].sum()), random_state=SEED)
        .reset_index(drop=True)
    )
    print(f"Eval set: {len(eval_df)} products")

    gold_ids = eval_df["product_id"].tolist()
    all_rows = []

    # 1. Text self-retrieval
    print("\n[1/5] Text self-retrieval")
    q = encode_text(model, processor, eval_df["desc_standard"].tolist())
    all_rows += recall_rows(q, gold_ids, text_col, label="text->text (self)")

    # 2. Image self-retrieval
    print("\n[2/5] Image self-retrieval")
    qi = encode_images(model, processor, eval_df["resolved_image"].tolist())
    all_rows += recall_rows(qi, gold_ids, image_col, label="image->image (self)")

    # 3. Partial-text retrieval (realistic user query)
    print("\n[3/5] Partial-text retrieval")
    partial = (eval_df["Product Name"].fillna("") + " " + eval_df["Brand Name"].fillna("")).str.strip().tolist()
    qp = encode_text(model, processor, partial)
    all_rows += recall_rows(qp, gold_ids, text_col, label="partial-text->text")

    # 4. Image → text (cross-modal)
    print("\n[4/5] Image -> text (cross-modal)")
    all_rows += recall_rows(qi, gold_ids, text_col, label="image->text")

    # 5. Text → image (cross-modal)
    print("\n[5/5] Text -> image (cross-modal)")
    all_rows += recall_rows(q, gold_ids, image_col, label="text->image")

    results = pd.DataFrame(all_rows)
    results.to_csv(OUT / "per_query_results.csv", index=False)

    order = [
        "text->text (self)",
        "image->image (self)",
        "partial-text->text",
        "image->text",
        "text->image",
    ]
    summary = (
        results.groupby("query_type")[[f"recall@{k}" for k in K_VALUES]]
        .mean()
        .round(4)
        .reindex(order)
    )
    summary.to_csv(OUT / "recall_summary.csv")
    print("\n=== Recall summary ===")
    print(summary.to_string())

    # Per-category breakdown using the realistic regime
    cat_map = df.set_index("product_id")["top_category"].to_dict()
    results["category"] = results["gold_id"].map(cat_map)
    cat = (
        results[results["query_type"] == "partial-text->text"]
        .groupby("category")[[f"recall@{k}" for k in K_VALUES]]
        .agg(["mean", "count"])
        .round(3)
    )
    cat.to_csv(OUT / "recall_by_category.csv")

    # Chart
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 5))
        summary.plot(kind="bar", ax=ax)
        ax.set_ylabel("Recall")
        ax.set_title("Retrieval performance by query type")
        ax.set_ylim(0, 1.05)
        ax.legend(title="k")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
        plt.tight_layout()
        plt.savefig(OUT / "recall_chart.png", dpi=120)
        print(f"\nChart saved: {OUT / 'recall_chart.png'}")
    except Exception as e:
        print(f"(skipping chart: {e})")


if __name__ == "__main__":
    main()
