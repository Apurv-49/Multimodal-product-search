"""Remote product catalog backed by the Hugging Face Dataset Viewer API.

The original prototype expected data/, images/ and faiss_index/ to exist locally.
Those artifacts were not committed, so Streamlit fell back to random vectors.
This module gives the deployed demo a real catalog without committing hundreds
of MB of product images to the repository.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from typing import Iterable

import pandas as pd
import requests
from PIL import Image

DATASET = "mecha2019/fashion-product-images-small"
API_URL = "https://datasets-server.huggingface.co/search"
TIMEOUT = 12


def _search_one(query: str, limit: int = 60) -> list[dict]:
    params = {
        "dataset": DATASET,
        "config": "default",
        "split": "train",
        "query": query,
        "offset": 0,
        "length": min(limit, 100),
    }
    response = requests.get(API_URL, params=params, timeout=TIMEOUT)
    response.raise_for_status()
    return [row.get("row", {}) for row in response.json().get("rows", [])]


def search_catalog(queries: Iterable[str], per_query: int = 40) -> pd.DataFrame:
    """Search several catalog queries and return a de-duplicated product table."""
    cleaned: list[str] = []
    for query in queries:
        query = " ".join(str(query).strip().split())
        if query and query.lower() not in {q.lower() for q in cleaned}:
            cleaned.append(query)

    if not cleaned:
        raise ValueError("At least one catalog query is required.")

    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=min(4, len(cleaned))) as pool:
        futures = [pool.submit(_search_one, query, per_query) for query in cleaned]
        for future in as_completed(futures):
            rows.extend(future.result())

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    keep = [
        "id", "productDisplayName", "articleType", "baseColour",
        "gender", "masterCategory", "subCategory", "usage", "image"
    ]
    df = df[[column for column in keep if column in df.columns]].copy()
    df = df.drop_duplicates(subset=["id"])
    df["id"] = pd.to_numeric(df["id"], errors="coerce")
    df = df.dropna(subset=["id"])
    df["id"] = df["id"].astype(int)
    df["brand"] = df["productDisplayName"].map(_extract_brand)
    df["image_url"] = df["image"].astype(str).str.replace("http://", "https://", regex=False)
    return df.reset_index(drop=True)


def _extract_brand(name: str) -> str:
    """Extract a display brand from the dataset's product title."""
    known = [
        "Nike", "Adidas", "Puma", "Reebok", "Vans", "Fila", "Levis",
        "Levi's", "Skechers", "Asics", "New Balance", "Converse", "Crocs",
        "Timberland", "Under Armour", "Clarks", "Bata", "Woodland"
    ]
    text = str(name)
    lowered = text.lower()
    for brand in known:
        if brand.lower() in lowered:
            return brand
    return text.split()[0] if text.split() else "Unknown"


def download_images(urls: list[str]) -> dict[str, Image.Image]:
    """Download product images concurrently; failed images are skipped."""
    def fetch(url: str):
        try:
            response = requests.get(
                url,
                timeout=TIMEOUT,
                headers={"User-Agent": "ProductLens/1.0"},
            )
            response.raise_for_status()
            return url, Image.open(BytesIO(response.content)).convert("RGB")
        except Exception:
            return url, None

    output: dict[str, Image.Image] = {}
    with ThreadPoolExecutor(max_workers=12) as pool:
        futures = [pool.submit(fetch, url) for url in urls]
        for future in as_completed(futures):
            url, image = future.result()
            if image is not None:
                output[url] = image
    return output
