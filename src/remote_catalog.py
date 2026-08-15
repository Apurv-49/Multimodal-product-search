from __future__ import annotations

import re
from functools import lru_cache
from typing import Iterable

import pandas as pd
from PIL import Image

try:
    from datasets import load_dataset
except ImportError as exc:
    raise ImportError(
        "ProductLens requires the 'datasets' package. "
        "Add 'datasets' to requirements.txt."
    ) from exc


# ============================================================
# DATASET
# ============================================================

DATASET = "ashraq/fashion-product-images-small"
SPLIT = "train"

# Internal identifiers used to connect catalog metadata
# with the actual PIL image stored in the Hugging Face dataset.
IMAGE_KEY_PREFIX = "product://"


# ============================================================
# IMAGE CACHE
# ============================================================

# Maps:
#
#     product://12345
#
# to:
#
#     PIL.Image.Image
#
# Only images actually selected as candidates are decoded.
_LOCAL_IMAGE_CACHE: dict[str, Image.Image] = {}


# ============================================================
# TEXT NORMALIZATION
# ============================================================

STOPWORDS = {
    "a",
    "an",
    "the",
    "for",
    "with",
    "and",
    "or",
    "of",
    "to",
    "in",
    "on",
    "from",
    "is",
    "are",
    "looking",
    "like",
    "find",
    "show",
    "me",
}


SYNONYMS = {
    "shoe": "shoes",
    "sneaker": "sneakers",
    "sneaker": "shoes",
    "trainer": "shoes",
    "trainers": "shoes",

    "pant": "pants",
    "trouser": "trousers",

    "tee": "tshirt",
    "tees": "tshirt",
    "tshirt": "tshirt",
    "tshirts": "tshirt",
    "t-shirt": "tshirt",
    "t-shirts": "tshirt",

    "woman": "women",
    "womens": "women",
    "women's": "women",

    "man": "men",
    "mens": "men",
    "men's": "men",

    "bag": "bags",
    "watch": "watches",
}


def _normalize_text(value: object) -> str:
    """
    Normalize catalog text and search queries.

    This creates a simple, robust local text representation
    without relying on the Hugging Face search API.
    """

    text = str(value or "").lower()

    text = text.replace(
        "&",
        " and ",
    )

    text = re.sub(
        r"[^a-z0-9\s-]",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    words: list[str] = []

    for word in text.split():

        normalized = SYNONYMS.get(
            word,
            word,
        )

        if normalized in STOPWORDS:
            continue

        words.append(
            normalized
        )

    return " ".join(words)


def _query_tokens(
    query: str,
) -> list[str]:
    """
    Convert a user query into normalized search tokens.
    """

    normalized = _normalize_text(
        query
    )

    return [
        token
        for token in normalized.split()
        if token
    ]


# ============================================================
# DATASET LOADING
# ============================================================

@lru_cache(maxsize=1)
def _load_dataset():
    """
    Load the fashion dataset once.

    Hugging Face caches the downloaded Parquet data locally.
    Subsequent Streamlit reruns reuse the cached dataset.

    The dataset contains:
        - 44,072 products
        - structured metadata
        - product images
    """

    return load_dataset(
        DATASET,
        split=SPLIT,
    )


@lru_cache(maxsize=1)
def _load_metadata() -> pd.DataFrame:
    """
    Build a metadata-only DataFrame.

    The image column is removed before converting to Pandas,
    preventing unnecessary image decoding into the DataFrame.
    """

    dataset = _load_dataset()

    metadata_dataset = dataset.remove_columns(
        "image"
    )

    df = metadata_dataset.to_pandas()

    # Preserve the original dataset row index.
    df["_dataset_index"] = range(
        len(df)
    )

    return df


# ============================================================
# BRAND EXTRACTION
# ============================================================

def _extract_brand(
    name: object,
) -> str:

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
        name or ""
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
# PREPARE METADATA
# ============================================================

@lru_cache(maxsize=1)
def _prepare_metadata() -> pd.DataFrame:
    """
    Prepare normalized text fields used by local catalog search.
    """

    df = _load_metadata().copy()

    text_columns = [
        "productDisplayName",
        "articleType",
        "baseColour",
        "gender",
        "masterCategory",
        "subCategory",
        "usage",
        "season",
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

    df["brand"] = (
        df["productDisplayName"]
        .map(
            _extract_brand
        )
    )

    # --------------------------------------------------------
    # Normalized search fields
    # --------------------------------------------------------

    df["_name_search"] = (
        df["productDisplayName"]
        .map(_normalize_text)
    )

    df["_article_search"] = (
        df["articleType"]
        .map(_normalize_text)
    )

    df["_colour_search"] = (
        df["baseColour"]
        .map(_normalize_text)
    )

    df["_gender_search"] = (
        df["gender"]
        .map(_normalize_text)
    )

    df["_master_search"] = (
        df["masterCategory"]
        .map(_normalize_text)
    )

    df["_sub_search"] = (
        df["subCategory"]
        .map(_normalize_text)
    )

    df["_usage_search"] = (
        df["usage"]
        .map(_normalize_text)
    )

    df["_full_search"] = (
        df["_name_search"]
        + " | "
        + df["_article_search"]
        + " | "
        + df["_colour_search"]
        + " | "
        + df["_gender_search"]
        + " | "
        + df["_master_search"]
        + " | "
        + df["_sub_search"]
        + " | "
        + df["_usage_search"]
    )

    return df


# ============================================================
# LOCAL CATALOG SEARCH
# ============================================================

def _score_catalog_rows(
    df: pd.DataFrame,
    query: str,
) -> pd.DataFrame:
    """
    Score products using local metadata.

    This is only candidate generation.

    CLIP image similarity remains responsible for actual
    visual ranking later in streamlit_app.py.
    """

    tokens = _query_tokens(
        query
    )

    if not tokens:
        return pd.DataFrame(
            columns=df.columns
            .tolist()
            + ["_catalog_score"]
        )

    scores = pd.Series(
        0.0,
        index=df.index,
        dtype="float32",
    )

    # --------------------------------------------------------
    # Field weights
    #
    # Product name is strongest.
    # Category/color/gender provide additional signals.
    # --------------------------------------------------------

    fields = [
        ("_name_search", 6.0),
        ("_article_search", 5.0),
        ("_colour_search", 4.0),
        ("_gender_search", 3.0),
        ("_sub_search", 3.0),
        ("_master_search", 2.0),
        ("_usage_search", 1.0),
    ]

    normalized_query = _normalize_text(
        query
    )

    # Exact phrase bonus.
    if normalized_query:

        phrase_mask = df[
            "_full_search"
        ].str.contains(
            re.escape(
                normalized_query
            ),
            regex=True,
            na=False,
        )

        scores.loc[phrase_mask] += 10.0


    # Token-level matching.
    for token in tokens:

        escaped_token = re.escape(
            token
        )

        for field, weight in fields:

            mask = df[
                field
            ].str.contains(
                escaped_token,
                regex=True,
                na=False,
            )

            scores.loc[
                mask
            ] += weight


    result = df.copy()

    result["_catalog_score"] = (
        scores
    )

    result = (
        result[
            result[
                "_catalog_score"
            ] > 0
        ]
        .sort_values(
            [
                "_catalog_score",
                "id",
            ],
            ascending=[
                False,
                True,
            ],
        )
    )

    return result


# ============================================================
# BUILD OUTPUT
# ============================================================

def _to_product_dataframe(
    candidates: pd.DataFrame,
) -> pd.DataFrame:

    if candidates.empty:
        return pd.DataFrame()


    output = candidates.copy()

    # --------------------------------------------------------
    # Create local image identifiers.
    # --------------------------------------------------------

    image_keys: list[str] = []

    dataset = _load_dataset()

    for _, row in output.iterrows():

        dataset_index = int(
            row["_dataset_index"]
        )

        product_id = int(
            row["id"]
        )

        image_key = (
            f"{IMAGE_KEY_PREFIX}"
            f"{product_id}"
        )

        try:

            image = dataset[
                dataset_index
            ][
                "image"
            ]

            if image is not None:

                if not isinstance(
                    image,
                    Image.Image,
                ):

                    image = Image.fromarray(
                        np.asarray(image)
                    )

                image = image.convert(
                    "RGB"
                )

                _LOCAL_IMAGE_CACHE[
                    image_key
                ] = image

                image_keys.append(
                    image_key
                )

            else:

                image_keys.append(
                    ""
                )

        except Exception:

            image_keys.append(
                ""
            )


    output["image_url"] = (
        image_keys
    )


    output = output[
        output[
            "image_url"
        ].astype(str).str.startswith(
            IMAGE_KEY_PREFIX
        )
    ]


    # --------------------------------------------------------
    # Keep the columns expected by Streamlit.
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
        "brand",
        "image_url",
    ]

    for column in keep:

        if column not in output.columns:

            output[column] = ""


    return (
        output[
            keep
        ]
        .drop_duplicates(
            subset=["id"]
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# PUBLIC SEARCH FUNCTION
# ============================================================

def search_catalog(
    queries: Iterable[str],
    per_query: int = 60,
) -> pd.DataFrame:
    """
    Search the local cached catalog.

    IMPORTANT:

    No request is made to:

        datasets-server.huggingface.co/search

    Therefore the Streamlit application no longer depends
    on that endpoint being available for every search.
    """

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
    # Load metadata once.
    # --------------------------------------------------------

    catalog = _prepare_metadata()


    # --------------------------------------------------------
    # Search every supplied query locally and merge results.
    # --------------------------------------------------------

    scored_parts: list[
        pd.DataFrame
    ] = []


    for query in cleaned:

        scored = _score_catalog_rows(
            catalog,
            query,
        )

        if not scored.empty:

            scored_parts.append(
                scored
            )


    # --------------------------------------------------------
    # No matches.
    # --------------------------------------------------------

    if not scored_parts:

        return pd.DataFrame()


    candidates = pd.concat(
        scored_parts,
        ignore_index=False,
    )


    # --------------------------------------------------------
    # Keep strongest result per product.
    # --------------------------------------------------------

    candidates = (
        candidates
        .sort_values(
            "_catalog_score",
            ascending=False,
        )
        .drop_duplicates(
            subset=["id"],
        )
        .head(
            max(
                20,
                min(
                    int(per_query),
                    100,
                ),
            )
        )
    )


    # --------------------------------------------------------
    # Convert candidate rows to actual product records.
    # --------------------------------------------------------

    return _to_product_dataframe(
        candidates
    )


# ============================================================
# IMAGE DOWNLOAD INTERFACE
# ============================================================

def download_images(
    urls: list[str],
) -> dict[str, Image.Image]:
    """
    Compatibility layer for streamlit_app.py.

    ProductLens now uses local cached dataset images.

    Instead of downloading each product image from the
    internet, this function returns the PIL image already
    cached by search_catalog().
    """

    output: dict[
        str,
        Image.Image,
    ] = {}


    for url in urls:

        if not url:
            continue


        # ----------------------------------------------------
        # Local ProductLens image
        # ----------------------------------------------------

        if url.startswith(
            IMAGE_KEY_PREFIX
        ):

            image = (
                _LOCAL_IMAGE_CACHE.get(
                    url
                )
            )

            if image is not None:

                output[url] = image


    return output


# ============================================================
# OPTIONAL UTILITY
# ============================================================

def clear_catalog_cache() -> None:
    """
    Clear cached dataset and metadata.

    Useful during local development if you need to force
    the dataset to be reloaded.
    """

    global _LOCAL_IMAGE_CACHE

    _LOCAL_IMAGE_CACHE.clear()

    _prepare_metadata.cache_clear()
    _load_metadata.cache_clear()
    _load_dataset.cache_clear()
