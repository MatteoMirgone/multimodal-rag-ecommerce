"""
Download primary product images for every row in products_indexed.parquet.

The Colab Phase 1 pipeline did this once during indexing; the local repo
needs them re-cached so the Streamlit UI can display retrieved product
cards and Phase 3 can run image self-retrieval evaluation.

Idempotent — skips images already on disk.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
import sys

import pandas as pd
import requests
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "prepared_data"
IMAGES = DATA / "images"
IMAGES.mkdir(parents=True, exist_ok=True)

MAX_SIDE = 512
WORKERS = 24
TIMEOUT = 8


def download_one(product_id: str, url: str) -> tuple[str, bool, str]:
    target = IMAGES / f"{product_id}.jpg"
    if target.exists() and target.stat().st_size > 1024:
        return product_id, True, "cached"
    if not url:
        return product_id, False, "no_url"
    try:
        r = requests.get(url, timeout=TIMEOUT)
        if r.status_code != 200 or not r.content:
            return product_id, False, f"http_{r.status_code}"
        img = Image.open(BytesIO(r.content)).convert("RGB")
        img.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
        img.save(target, "JPEG", quality=85)
        return product_id, True, "ok"
    except Exception as e:
        return product_id, False, type(e).__name__


def main():
    df = pd.read_parquet(DATA / "products_indexed.parquet")
    print(f"Loaded {len(df)} products")

    tasks = list(zip(df["product_id"].tolist(), df["primary_image_url"].tolist()))
    success, fail = 0, 0
    reasons: dict[str, int] = {}

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = [ex.submit(download_one, pid, url) for pid, url in tasks]
        for f in tqdm(as_completed(futures), total=len(futures), desc="images"):
            _, ok, reason = f.result()
            if ok:
                success += 1
            else:
                fail += 1
                reasons[reason] = reasons.get(reason, 0) + 1

    print(f"\nSuccess: {success}/{len(tasks)} ({success / len(tasks) * 100:.1f}%)")
    if reasons:
        print(f"Failure reasons: {reasons}")
    print(f"Cached in: {IMAGES}")


if __name__ == "__main__":
    main()
