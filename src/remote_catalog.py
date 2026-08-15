from __future__ import annotations

from io import BytesIO
from typing import Iterable
import time

import pandas as pd
import requests
from PIL import Image


# ============================================================
# HUGGING FACE DATASET
# ============================================================

DATASET = "ashraq/fashion-product-images-small"

API_URL = "https://datasets-server.huggingface.co/search"

# Separate connect/read timeouts.
# This is safer than using one very small timeout.
CONNECT_TIMEOUT = 10
READ_TIMEOUT = 60

MAX_RETRIES = 3

# Maximum number of products returned by one HF request.
MAX_RESULTS_PER_QUERY = 100


# ============================================================
# HTTP SESSION
# ============================================================

def _create_session() -> requests.Session:
    """
    Create a reusable HTTP session for Hugging Face requests.
    """

    session = requests.Session()

    session.headers.update(
        {
            "User-Agent": (
                "ProductLens/1.0 "
                "(multimodal product search)"
            ),
            "Accept": "application/json",
        }
    )

    return session


# ============================================================
# SEARCH ONE QUERY
# ============================================================

def _search_one(
    query: str,
    limit: int = 60,
) -> list[dict]:
    """
    Search the Hugging Face Dataset Viewer.

    Retries temporary failures and rate limits.
    """

    query = " ".join(
        str(query).strip().split()
    )

    if not query:
        return []

    limit = max(
        1,
        min(
            int(limit),
            MAX_RESULTS_PER_QUERY,
        ),
    )

    params = {
        "dataset": DATASET,
        "config": "default",
        "split": "train",
        "query": query,
        "offset": 0,
        "length": limit,
    }

    session = _create_session()

    last_error: Exception | None = None

    for attempt in range(MAX_RETRIES):

        try:

            response = session.get(
                API_URL,
                params=params,
                timeout=(
                    CONNECT_TIMEOUT,
                    READ_TIMEOUT,
                ),
            )

            # ------------------------------------------------
            # RATE LIMIT
            # ------------------------------------------------

            if response.status_code == 429:

                retry_after = (
                    response.headers.get(
                        "Retry-After"
                    )
                )

                if retry_after:

                    try:
                        wait_time = float(
                            retry_after
                        )
                    except ValueError:
                        wait_time = 2 ** attempt

                else:

                    wait_time = 2 ** attempt


                if attempt < MAX_RETRIES - 1:

                    time.sleep(
                        min(
                            wait_time,
                            30,
                        )
                    )

                    continue


                raise RuntimeError(
                    "Hugging Face rate limit "
                    "persisted after retries."
                )


            # ------------------------------------------------
            # TEMPORARY SERVER ERRORS
            # ------------------------------------------------

            if response.status_code in {
                500,
                502,
                503,
                504,
            }:

                if attempt < MAX_RETRIES - 1:

                    time.sleep(
                        2 ** attempt
                    )

                    continue


            # ------------------------------------------------
            # OTHER HTTP ERRORS
            # ------------------------------------------------

            response.raise_for_status()


            data = response.json()


            rows = data.get(
                "rows",
                [],
            )


            return [
                row.get(
                    "row",
                    {},
                )
                for row in rows
                if isinstance(
                    row,
                    dict,
                )
            ]


        except (
            requests.Timeout,
            requests.ConnectionError,
        ) as exc:

            last_error = exc

            if attempt < MAX_RETRIES - 1:

                time.sleep(
                    2 ** attempt
                )

                continue


        except requests.RequestException as exc:

            last_error = exc

            if attempt < MAX_RETRIES - 1:

                time.sleep(
                    2 ** attempt
                )

                continue


        except Exception as exc:

            last_error = exc

            if attempt < MAX_RETRIES - 1:

                time.sleep(
                    2 ** attempt
                )

                continue


    if last_error:

        raise RuntimeError(
            f"Hugging Face catalog search failed "
            f"for '{query}': {last_error}"
        )


    return []


# ============================================================
# SEARCH CATALOG
# ============================================================

def search_catalog(
    queries: Iterable[str],
    per_query: int = 60,
) -> pd.DataFrame:
    """
    Search the fashion catalog and return a clean
    product DataFrame.

    Requests are deliberately sequential.

    This prevents multiple simultaneous requests from
    triggering Hugging Face rate limits.
    """

    # --------------------------------------------------------
    # CLEAN QUERIES
    # --------------------------------------------------------

    cleaned: list[str] = []

    seen: set[str] = set()


    for query in queries:

        query = " ".join(
            str(query)
            .strip()
            .split()
        )

        if not query:
            continue


        key = query.lower()


        if key in seen:
            continue


        seen.add(
            key
        )

        cleaned.append(
            query
        )


    if not cleaned:

        raise ValueError(
            "At least one catalog query is required."
        )


    # --------------------------------------------------------
    # FETCH ROWS
    # --------------------------------------------------------

    rows: list[dict] = []


    for query in cleaned:

        query_rows = _search_one(
            query=query,
            limit=per_query,
        )

        rows.extend(
            query_rows
        )


    # --------------------------------------------------------
    # EMPTY RESULT
    # --------------------------------------------------------

    if not rows:

        return pd.DataFrame()


    # --------------------------------------------------------
    # DATAFRAME
    # --------------------------------------------------------

    df = pd.DataFrame(
        rows
    )


    # --------------------------------------------------------
    # KEEP ONLY REQUIRED COLUMNS
    # --------------------------------------------------------

    required_columns = [
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


    available_columns = [
        column
        for column in required_columns
        if column in df.columns
    ]


    df = df[
        available_columns
    ].copy()


    if "id" not in df.columns:

        return pd.DataFrame()


    # --------------------------------------------------------
    # CLEAN IDS
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # BRAND
    # --------------------------------------------------------

    if (
        "productDisplayName"
        in df.columns
    ):

        df["brand"] = (
            df[
                "productDisplayName"
            ]
            .map(
                _extract_brand
            )
        )

    else:

        df["brand"] = "Unknown"


    # --------------------------------------------------------
    # IMAGE URL
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # REMOVE PRODUCTS WITHOUT IMAGES
    # --------------------------------------------------------

    df = df[
        df["image_url"].notna()
    ].copy()


    df = df[
        df["image_url"].astype(str).str.len() > 0
    ].copy()


    # --------------------------------------------------------
    # FINAL CLEANUP
    # --------------------------------------------------------

    df = (
        df
        .drop_duplicates(
            subset=["id"]
        )
        .reset_index(
            drop=True
        )
    )


    return df


# ============================================================
# EXTRACT IMAGE URL
# ============================================================

def _image_url(
    value,
) -> str | None:
    """
    Convert the Dataset Viewer image field into
    a usable HTTPS image URL.

    The dataset's image feature can be represented
    as a dictionary containing src/url/path.
    """

    # --------------------------------------------------------
    # DICTIONARY IMAGE
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
                    "http://"
                ):

                    candidate = candidate.replace(
                        "http://",
                        "https://",
                        1,
                    )


                return candidate


    # --------------------------------------------------------
    # STRING IMAGE URL
    # --------------------------------------------------------

    if isinstance(
        value,
        str,
    ):

        value = value.strip()


        if value.startswith(
            "http://"
        ):

            value = value.replace(
                "http://",
                "https://",
                1,
            )


        if value.startswith(
            "https://"
        ):

            return value


    return None


# ============================================================
# BRAND EXTRACTION
# ============================================================

def _extract_brand(
    name: str,
) -> str:
    """
    Extract a brand name from the catalog product name.

    IMPORTANT:
    This is metadata extraction only.

    It is NOT used as a visual prediction or ranking signal.
    """

    known_brands = [
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


    if not text:

        return "Unknown"


    lowered = text.lower()


    for brand in known_brands:

        if brand.lower() in lowered:

            return brand


    parts = text.split()


    if parts:

        return parts[0]


    return "Unknown"


# ============================================================
# DOWNLOAD PRODUCT IMAGES
# ============================================================

def download_images(
    urls: list[str],
) -> dict[str, Image.Image]:
    """
    Download product images from the catalog.

    This uses a separate session because product image
    downloads are different from the Hugging Face search API.
    """

    output: dict[
        str,
        Image.Image,
    ] = {}


    if not urls:

        return output


    session = requests.Session()


    session.headers.update(
        {
            "User-Agent":
                "ProductLens/1.0",
            "Accept":
                "image/avif,image/webp,image/apng,"
                "image/svg+xml,image/*,*/*;q=0.8",
        }
    )


    for url in urls:

        if not url:
            continue


        try:

            response = session.get(
                url,
                timeout=(
                    CONNECT_TIMEOUT,
                    READ_TIMEOUT,
                ),
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

            # One broken product image should not
            # destroy the entire search.
            continue


    session.close()


    return output
