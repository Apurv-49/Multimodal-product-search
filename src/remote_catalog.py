"""
Remote product catalog for ProductLens.

The catalog is backed by the public
ashraq/fashion-product-images-small dataset on Hugging Face.

Important design principles:

1. Do NOT fire many Hugging Face requests concurrently.
2. Keep catalog retrieval lightweight.
3. Use text retrieval only to create a candidate pool.
4. Let CLIP image similarity perform the actual visual ranking.
5. Never use zero-shot brand prediction as ground truth.
6. Gracefully handle Hugging Face rate limiting.
"""

from __future__ import annotations

import random
import threading
import time
from io import BytesIO
from typing import Iterable

import pandas as pd
import requests
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

DATASET = "ashraq/fashion-product-images-small"

SEARCH_API_URL = (
    "https://datasets-server.huggingface.co/search"
)

TIMEOUT = 20

# Hugging Face allows up to 100 rows per search request.
MAX_SEARCH_LENGTH = 100

# Keep this deliberately small.
#
# The previous implementation could make 4 requests at once
# and the Streamlit app could generate many queries.
#
# We now normally make only ONE request.
MAX_CATALOG_REQUESTS = 1

# Retry configuration for temporary 429 / 5xx responses.
MAX_RETRIES = 4

BASE_BACKOFF = 1.5

# In-memory catalog cache.
# This survives normal Streamlit reruns as long as the process
# remains alive.
_SEARCH_CACHE: dict[
    tuple[str, ...],
    pd.DataFrame,
] = {}

_CACHE_LOCK = threading.Lock()

# Protect the Hugging Face endpoint from bursts.
_REQUEST_LOCK = threading.Lock()

_LAST_REQUEST_TIME = 0.0

# Minimum spacing between HF requests.
MIN_REQUEST_INTERVAL = 1.0


# ============================================================
# HTTP SESSION
# ============================================================

_SESSION = requests.Session()

_SESSION.headers.update(
    {
        "User-Agent": (
            "ProductLens/1.0 "
            "(multimodal-ecommerce-search)"
        ),
        "Accept": "application/json",
    }
)


# ============================================================
# RATE LIMIT CONTROL
# ============================================================

def _wait_before_request() -> None:
    """
    Keep requests to the Hugging Face API spaced apart.

    This is intentionally conservative because the deployed
    Streamlit application can receive multiple user-triggered
    reruns.
    """

    global _LAST_REQUEST_TIME

    with _REQUEST_LOCK:

        now = time.monotonic()

        elapsed = (
            now - _LAST_REQUEST_TIME
        )

        if elapsed < MIN_REQUEST_INTERVAL:

            time.sleep(
                MIN_REQUEST_INTERVAL
                - elapsed
            )

        _LAST_REQUEST_TIME = time.monotonic()


# ============================================================
# RETRY-AFTER PARSER
# ============================================================

def _retry_after_seconds(
    response: requests.Response,
) -> float:

    value = response.headers.get(
        "Retry-After"
    )

    if value:

        try:
            return max(
                1.0,
                float(value),
            )

        except ValueError:
            pass

    return 0.0


# ============================================================
# SINGLE HF SEARCH
# ============================================================

def _search_one(
    query: str,
    limit: int = 80,
) -> list[dict]:

    query = " ".join(
        str(query).strip().split()
    )

    if not query:
        return []

    params = {
        "dataset": DATASET,
        "config": "default",
        "split": "train",
        "query": query,
        "offset": 0,
        "length": min(
            int(limit),
            MAX_SEARCH_LENGTH,
        ),
    }


    last_error: Exception | None = None


    for attempt in range(
        MAX_RETRIES + 1
    ):

        try:

            _wait_before_request()

            response = _SESSION.get(
                SEARCH_API_URL,
                params=params,
                timeout=TIMEOUT,
            )


            # ------------------------------------------------
            # SUCCESS
            # ------------------------------------------------

            if response.ok:

                payload = response.json()

                rows = payload.get(
                    "rows",
                    [],
                )

                return [
                    row.get(
                        "row",
                        {},
                    )
                    for row in rows
                    if isinstance(row, dict)
                ]


            # ------------------------------------------------
            # RATE LIMITED
            # ------------------------------------------------

            if response.status_code == 429:

                retry_after = (
                    _retry_after_seconds(
                        response
                    )
                )

                if retry_after <= 0:

                    retry_after = (
                        BASE_BACKOFF
                        * (2 ** attempt)
                        + random.uniform(
                            0,
                            0.5,
                        )
                    )

                # Don't wait forever.
                retry_after = min(
                    retry_after,
                    12.0,
                )

                if attempt >= MAX_RETRIES:

                    raise RuntimeError(
                        "Hugging Face catalog API "
                        "is temporarily rate-limited. "
                        "Please wait a few seconds "
                        "and try the search again."
                    )

                time.sleep(
                    retry_after
                )

                continue


            # ------------------------------------------------
            # SERVER ERROR
            # ------------------------------------------------

            if response.status_code >= 500:

                last_error = RuntimeError(
                    f"Hugging Face returned "
                    f"HTTP {response.status_code}"
                )

                if attempt >= MAX_RETRIES:

                    raise last_error

                delay = min(
                    BASE_BACKOFF
                    * (2 ** attempt),
                    10.0,
                )

                time.sleep(
                    delay
                )

                continue


            # ------------------------------------------------
            # OTHER HTTP ERROR
            # ------------------------------------------------

            response.raise_for_status()


        except requests.RequestException as exc:

            last_error = exc

            if attempt >= MAX_RETRIES:

                raise RuntimeError(
                    "Unable to reach the Hugging Face "
                    "catalog service."
                ) from exc

            delay = min(
                BASE_BACKOFF
                * (2 ** attempt),
                10.0,
            )

            time.sleep(
                delay
            )


    if last_error:

        raise last_error

    return []


# ============================================================
# QUERY NORMALIZATION
# ============================================================

def _clean_queries(
    queries: Iterable[str],
) -> list[str]:

    cleaned: list[str] = []

    seen: set[str] = set()


    for query in queries:

        query = " ".join(
            str(query).strip().split()
        )

        if not query:
            continue

        key = query.lower()

        if key in seen:
            continue

        seen.add(key)

        cleaned.append(query)


    return cleaned


# ============================================================
# SELECT BEST RETRIEVAL QUERY
# ============================================================

def _select_primary_query(
    queries: list[str],
) -> str:

    """
    Select ONE query for the remote catalog.

    This is important.

    The previous implementation sent many queries such as:

        shoes
        Nike shoes
        Adidas shoes
        Puma shoes
        Reebok shoes
        ...

    That creates two problems:

    1. Hugging Face rate limiting.
    2. Candidate-pool contamination.

    The catalog API should create a broad candidate pool.
    CLIP should perform the actual visual ranking afterward.
    """

    if not queries:

        return "fashion products"


    # Prefer the most descriptive query.
    #
    # In the current application the first query is normally
    # the user's natural-language description.
    #
    # If there is no user text, use the first visual hint.

    longest = max(
        queries,
        key=lambda q: (
            len(q.split()),
            len(q),
        ),
    )

    return longest


# ============================================================
# DATAFRAME NORMALIZATION
# ============================================================

def _normalize_catalog(
    rows: list[dict],
) -> pd.DataFrame:

    if not rows:

        return pd.DataFrame()


    df = pd.DataFrame(
        rows
    )


    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

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


    available = [
        column
        for column in keep
        if column in df.columns
    ]


    df = df[
        available
    ].copy()


    if "id" not in df.columns:

        return pd.DataFrame()


    # --------------------------------------------------------
    # Product ID
    # --------------------------------------------------------

    df["id"] = pd.to_numeric(
        df["id"],
        errors="coerce",
    )

    df = df.dropna(
        subset=["id"]
    )

    df["id"] = (
        df["id"]
        .astype(int)
    )


    # --------------------------------------------------------
    # Remove duplicates
    # --------------------------------------------------------

    df = df.drop_duplicates(
        subset=["id"]
    )


    # --------------------------------------------------------
    # Ensure text columns exist
    # --------------------------------------------------------

    text_columns = [
        "productDisplayName",
        "articleType",
        "baseColour",
        "gender",
        "masterCategory",
        "subCategory",
        "usage",
    ]


    for column in text_columns:

        if column not in df.columns:

            df[column] = ""

        else:

            df[column] = (
                df[column]
                .fillna("")
                .astype(str)
            )


    # --------------------------------------------------------
    # Brand
    # --------------------------------------------------------

    df["brand"] = (
        df["productDisplayName"]
        .map(_extract_brand)
    )


    # --------------------------------------------------------
    # Image URL
    # --------------------------------------------------------

    if "image" in df.columns:

        df["image_url"] = (
            df["image"]
            .map(_image_url)
        )

    else:

        df["image_url"] = None


    df = df[
        df["image_url"].notna()
    ]


    df = df[
        df["image_url"].astype(str).str.len() > 0
    ]


    return (
        df
        .reset_index(drop=True)
    )


# ============================================================
# CATALOG SEARCH
# ============================================================

def search_catalog(
    queries: Iterable[str],
    per_query: int = 80,
) -> pd.DataFrame:

    """
    Retrieve a candidate product pool.

    IMPORTANT:

    Only ONE Hugging Face search request is normally made.

    The application later performs CLIP-based ranking over
    the returned product images.
    """

    cleaned = _clean_queries(
        queries
    )


    if not cleaned:

        raise ValueError(
            "At least one catalog query is required."
        )


    # --------------------------------------------------------
    # ONE PRIMARY QUERY
    # --------------------------------------------------------

    primary_query = (
        _select_primary_query(
            cleaned
        )
    )


    cache_key = (
        primary_query.lower(),
        str(
            min(
                per_query,
                MAX_SEARCH_LENGTH,
            )
        ),
    )


    # --------------------------------------------------------
    # CACHE
    # --------------------------------------------------------

    with _CACHE_LOCK:

        cached = _SEARCH_CACHE.get(
            cache_key
        )

        if cached is not None:

            return cached.copy()


    # --------------------------------------------------------
    # SEARCH
    # --------------------------------------------------------

    rows = _search_one(
        primary_query,
        limit=min(
            per_query,
            MAX_SEARCH_LENGTH,
        ),
    )


    # --------------------------------------------------------
    # FALLBACK QUERY
    #
    # If a very specific query returns nothing,
    # try a broader query ONCE.
    # --------------------------------------------------------

    if not rows:

        words = (
            primary_query
            .lower()
            .split()
        )


        broad_terms = [
            word
            for word in words
            if len(word) > 2
        ]


        if broad_terms:

            fallback_query = (
                " ".join(
                    broad_terms[-3:]
                )
            )


            if (
                fallback_query.lower()
                != primary_query.lower()
            ):

                rows = _search_one(
                    fallback_query,
                    limit=min(
                        per_query,
                        MAX_SEARCH_LENGTH,
                    ),
                )


    # --------------------------------------------------------
    # NORMALIZE
    # --------------------------------------------------------

    df = _normalize_catalog(
        rows
    )


    # --------------------------------------------------------
    # CACHE
    # --------------------------------------------------------

    with _CACHE_LOCK:

        _SEARCH_CACHE[
            cache_key
        ] = df.copy()


    return df


# ============================================================
# IMAGE URL EXTRACTION
# ============================================================

def _image_url(
    value,
) -> str | None:

    """
    Extract an image URL from the Hugging Face
    Dataset Viewer image representation.

    The viewer can return image cells in several forms.
    """

    # --------------------------------------------------------
    # Dictionary representation
    # --------------------------------------------------------

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

                candidate = str(
                    candidate
                )

                if candidate.startswith(
                    "http"
                ):

                    return candidate.replace(
                        "http://",
                        "https://",
                    )


    # --------------------------------------------------------
    # String representation
    # --------------------------------------------------------

    if isinstance(
        value,
        str,
    ):

        value = value.strip()

        if value.startswith(
            "http"
        ):

            return value.replace(
                "http://",
                "https://",
            )


    return None


# ============================================================
# BRAND EXTRACTION
# ============================================================

def _extract_brand(
    name: str,
) -> str:

    """
    Extract a brand from the product title.

    This is metadata extraction only.

    It must NOT be interpreted as an ML prediction.
    """

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
    ).strip()


    lowered = text.lower()


    for brand in known:

        if brand.lower() in lowered:

            return brand


    words = text.split()


    return (
        words[0]
        if words
        else "Unknown"
    )


# ============================================================
# IMAGE DOWNLOADING
# ============================================================

def download_images(
    urls: list[str],
) -> dict[str, Image.Image]:

    """
    Download product images.

    Downloads are kept concurrent because these are normal
    image CDN requests, not Hugging Face Dataset Viewer
    search requests.
    """

    if not urls:

        return {}


    # Deduplicate URLs while preserving order.

    unique_urls = list(
        dict.fromkeys(
            url
            for url in urls
            if url
        )
    )


    def fetch(
        url: str,
    ):

        try:

            response = requests.get(
                url,
                timeout=TIMEOUT,
                headers={
                    "User-Agent": (
                        "ProductLens/1.0"
                    )
                },
            )


            response.raise_for_status()


            content_type = (
                response.headers
                .get(
                    "Content-Type",
                    "",
                )
                .lower()
            )


            # Some servers don't provide a useful
            # content type, so don't reject those.

            if (
                content_type
                and not content_type.startswith(
                    "image/"
                )
            ):

                return url, None


            image = Image.open(
                BytesIO(
                    response.content
                )
            ).convert(
                "RGB"
            )


            return url, image


        except Exception:

            return url, None


    output: dict[
        str,
        Image.Image,
    ] = {}


    # 8 concurrent image downloads is enough.
    # There is no reason to use 12 here for a small
    # candidate set.

    from concurrent.futures import (
        ThreadPoolExecutor,
        as_completed,
    )


    with ThreadPoolExecutor(
        max_workers=min(
            8,
            len(unique_urls),
        )
    ) as pool:

        futures = [
            pool.submit(
                fetch,
                url,
            )
            for url in unique_urls
        ]


        for future in as_completed(
            futures
        ):

            url, image = (
                future.result()
            )


            if image is not None:

                output[
                    url
                ] = image


    return output
