from __future__ import annotations

from io import BytesIO
from typing import Iterable
import time

import pandas as pd
import requests
from PIL import Image


DATASET = "ashraq/fashion-product-images-small"
API_URL = "https://datasets-server.huggingface.co/search"

# Increased from 15 seconds.
TIMEOUT = 45

# Number of retries for temporary HF failures.
MAX_RETRIES = 3


def _search_one(
    query: str,
    limit: int = 60,
) -> list[dict]:

    params = {
        "dataset": DATASET,
        "config": "default",
        "split": "train",
        "query": query,
        "offset": 0,
        "length": min(limit, 100),
    }

    last_error = None

    for attempt in range(MAX_RETRIES):

        try:

            response = requests.get(
                API_URL,
                params=params,
                timeout=TIMEOUT,
                headers={
                    "User-Agent": "ProductLens/1.0"
                },
            )

            # Hugging Face rate limit.
            if response.status_code == 429:

                wait_time = 2 ** attempt

                time.sleep(
                    wait_time
                )

                last_error = (
                    "Hugging Face rate limit "
                    "(429)"
                )

                continue

            response.raise_for_status()

            data = response.json()

            return [
                row.get("row", {})
                for row in data.get(
                    "rows",
                    [],
                )
            ]

        except requests.RequestException as exc:

            last_error = exc

            if attempt < MAX_RETRIES - 1:

                time.sleep(
                    2 ** attempt
                )

            else:

                break


    if last_error:

        raise RuntimeError(
            f"Catalog request failed for "
            f"'{query}': {last_error}"
        )

    return []


def search_catalog(
    queries: Iterable[str],
    per_query: int = 60,
) -> pd.DataFrame:

    """
    Search the remote fashion catalog.

    We intentionally make requests sequentially instead
    of sending several requests simultaneously.

    This reduces the chance of Hugging Face returning
    HTTP 429 rate-limit errors.
    """

    cleaned: list[str] = []

    seen: set[str] = set()

    for query in queries:

        query = " ".join(
            str(query)
            .strip()
            .split()
        )

        key = query.lower()

        if (
            query
            and key not in seen
        ):

            cleaned.append(
                query
            )

            seen.add(
                key
            )


    if not cleaned:

        raise ValueError(
            "At least one catalog query "
            "is required."
        )


    rows: list[dict] = []


    # IMPORTANT:
    # Do NOT use ThreadPoolExecutor here.
    #
    # The previous implementation could make multiple
    # Hugging Face requests at the same time and trigger
    # HTTP 429 errors.

    for query in cleaned:

        result = _search_one(
            query,
            per_query,
        )

        rows.extend(
            result
        )


    if not rows:

        return pd.DataFrame()


    df = pd.DataFrame(
        rows
    )


    keep = [
        "id",
        "productDisplayName",
        "articleType",
        "baseColour",
        "gender",
        "masterCategory",
        "subCategory",
        "usage",
        "image",
    ]


    df = df[
        [
            column
            for column in keep
            if column in df.columns
        ]
    ].copy()


    if "id" not in df.columns:

        return pd.DataFrame()


    df = df.drop_duplicates(
        subset=["id"]
    )


    df["id"] = pd.to_numeric(
        df["id"],
        errors="coerce",
    )


    df = df.dropna(
        subset=["id"]
    )


    df["id"] = df[
        "id"
    ].astype(int)


    if "productDisplayName" in df.columns:

        df["brand"] = (
            df[
                "productDisplayName"
            ]
            .map(_extract_brand)
        )

    else:

        df["brand"] = "Unknown"


    if "image" in df.columns:

        df["image_url"] = (
            df[
                "image"
            ].map(
                _image_url
            )
        )

    else:

        df["image_url"] = None


    df = df[
        df["image_url"].notna()
    ].reset_index(
        drop=True
    )


    return df


def _image_url(
    value,
) -> str | None:

    """
    Dataset Viewer image cells may be dictionaries
    containing src, url, or path.
    """

    if isinstance(
        value,
        dict,
    ):

        for key in (
            "src",
            "url",
            "path",
        ):

            candidate = value.get(
                key
            )

            if candidate:

                return str(
                    candidate
                ).replace(
                    "http://",
                    "https://",
                )


    if (
        isinstance(
            value,
            str,
        )
        and value.startswith(
            "http"
        )
    ):

        return value.replace(
            "http://",
            "https://",
        )


    return None


def _extract_brand(
    name: str,
) -> str:

    known = [
        "Nike",
        "Adidas",
        "Puma",
        "Reebok",
        "Vans",
        "Fila",
        "Levis",
        "Levi's",
        "Skechers",
        "Asics",
        "New Balance",
        "Converse",
        "Crocs",
        "Timberland",
        "Under Armour",
        "Clarks",
        "Bata",
        "Woodland",
    ]


    text = str(
        name
    )


    lowered = text.lower()


    for brand in known:

        if (
            brand.lower()
            in lowered
        ):

            return brand


    parts = text.split()


    return (
        parts[0]
        if parts
        else "Unknown"
    )


def download_images(
    urls: list[str],
) -> dict[str, Image.Image]:

    """
    Download catalog product images.

    Failed images are skipped rather than crashing
    the complete search.
    """

    output: dict[
        str,
        Image.Image
    ] = {}


    session = requests.Session()


    for url in urls:

        try:

            response = session.get(
                url,
                timeout=TIMEOUT,
                headers={
                    "User-Agent":
                    "ProductLens/1.0"
                },
            )


            response.raise_for_status()


            image = Image.open(
                BytesIO(
                    response.content
                )
            ).convert(
                "RGB"
            )


            output[url] = image


        except Exception:

            continue


    return output
