from __future__ import annotations

import html
import os
import sys
from typing import Iterable

import numpy as np
import streamlit as st
from PIL import Image


# ============================================================
# PATH
# ============================================================

ROOT_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


from src.embeddings import CLIPEmbedder
from src.remote_catalog import (
    download_images,
    search_catalog,
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="ProductLens — Product Search",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# UI STYLE
# ============================================================

st.html(
    """
    <style>

    :root {
        --bg: #F7F8FA;
        --surface: #FFFFFF;
        --border: #DEE3E8;

        --text: #20252B;
        --muted: #68727D;
        --light: #8B949D;

        --accent: #24466F;
        --accent-hover: #193958;

        --success: #55755E;
        --danger: #B84A4A;
    }


    /* =====================================================
       APP
    ===================================================== */

    .stApp {
        background: var(--bg);
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    .block-container {
        max-width: 1450px;
        padding: 1.2rem 2.5rem 3rem;
    }


    /* =====================================================
       SIDEBAR
    ===================================================== */

    section[data-testid="stSidebar"] {
        background: #FCFCFB;
        border-right: 1px solid var(--border);
    }

    section[data-testid="stSidebar"] > div {
        padding: 1.35rem 1.15rem;
    }

    .sidebar-brand {
        font-size: 1.3rem;
        font-weight: 800;
        letter-spacing: -0.04em;
        color: var(--text);
        margin-bottom: 0.45rem;
    }

    .sidebar-brand span {
        color: var(--accent);
    }

    .sidebar-description {
        color: var(--muted);
        font-size: 0.73rem;
        line-height: 1.5;
        margin-bottom: 1.4rem;
    }

    .sidebar-title {
        color: #737D87;
        font-size: 0.63rem;
        font-weight: 800;
        letter-spacing: 0.11em;
        text-transform: uppercase;
        margin-top: 1.25rem;
        margin-bottom: 0.5rem;
    }


    /* =====================================================
       INPUTS
    ===================================================== */

    .stTextInput input {
        background: #FFFFFF !important;
        color: var(--text) !important;

        border: 1px solid #D7DDE3 !important;
        border-radius: 6px !important;

        min-height: 44px;
    }

    .stTextInput input:focus {
        border-color: var(--accent) !important;

        box-shadow:
            0 0 0 1px var(--accent) !important;
    }

    .stTextInput input::placeholder {
        color: #9AA2AA !important;
    }


    /* =====================================================
       BUTTON
    ===================================================== */

    .stButton > button {
        border-radius: 6px !important;
        min-height: 42px;

        font-weight: 700 !important;
        font-size: 0.78rem !important;
    }

    .stButton > button[kind="primary"] {
        background: var(--accent) !important;
        border-color: var(--accent) !important;
        color: #FFFFFF !important;
    }

    .stButton > button[kind="primary"]:hover {
        background: var(--accent-hover) !important;
        border-color: var(--accent-hover) !important;
    }


    /* =====================================================
       FILE UPLOADER
    ===================================================== */

    [data-testid="stFileUploader"] {
        background: #FFFFFF;
        border-radius: 7px;
    }


    /* =====================================================
       TOP BAR
    ===================================================== */

    .topbar {
        display: flex;
        align-items: center;
        justify-content: space-between;

        border-bottom: 1px solid var(--border);

        padding-bottom: 0.9rem;
        margin-bottom: 2rem;
    }

    .topbar-brand {
        font-size: 1rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        color: var(--text);
    }

    .topbar-brand span {
        color: var(--accent);
    }

    .topbar-right {
        display: flex;
        align-items: center;
        gap: 1.3rem;

        color: var(--muted);
        font-size: 0.7rem;
    }

    .catalog-status {
        display: inline-flex;
        align-items: center;
        gap: 6px;

        color: #53616F;

        background: #FFFFFF;
        border: 1px solid var(--border);

        padding: 5px 9px;
        border-radius: 5px;
    }

    .status-dot {
        width: 6px;
        height: 6px;

        border-radius: 50%;

        background: #55755E;
    }


    /* =====================================================
       PAGE TITLE
    ===================================================== */

    .page-title {
        color: var(--text);

        font-size: 2rem;
        font-weight: 760;

        letter-spacing: -0.045em;

        margin-bottom: 0.3rem;
    }

    .page-description {
        color: var(--muted);

        font-size: 0.84rem;
        line-height: 1.55;

        max-width: 760px;

        margin-bottom: 1.5rem;
    }


    /* =====================================================
       SEARCH LABEL
    ===================================================== */

    .field-label {
        color: #4D5862;

        font-size: 0.68rem;
        font-weight: 750;

        margin-bottom: 0.4rem;
    }


    /* =====================================================
       POPULAR SEARCH
    ===================================================== */

    .popular {
        margin-top: 0.6rem;
        margin-bottom: 1.6rem;
    }

    .popular-label {
        color: var(--light);

        font-size: 0.65rem;

        margin-bottom: 0.4rem;
    }

    .chip {
        display: inline-block;

        color: #5D6872;

        background: #FFFFFF;
        border: 1px solid #E0E4E8;

        padding: 4px 8px;

        border-radius: 4px;

        font-size: 0.62rem;

        margin-right: 4px;
    }


    /* =====================================================
       SEARCH INFO
    ===================================================== */

    .search-info {
        display: flex;
        align-items: center;
        gap: 8px;

        color: var(--muted);

        font-size: 0.68rem;

        margin-top: 0.5rem;
    }

    .info-dot {
        width: 6px;
        height: 6px;

        border-radius: 50%;

        background: var(--accent);
    }


    /* =====================================================
       EMPTY STATE
    ===================================================== */

    .empty-state {
        background: #FFFFFF;

        border: 1px solid var(--border);
        border-radius: 8px;

        padding: 3rem 2rem;

        text-align: center;

        margin-top: 1rem;
    }

    .empty-icon {
        width: 42px;
        height: 42px;

        display: flex;
        align-items: center;
        justify-content: center;

        margin: 0 auto 0.8rem;

        background: #F2F4F6;
        border: 1px solid #E1E5E9;

        border-radius: 7px;

        font-size: 1rem;
    }

    .empty-title {
        color: var(--text);

        font-size: 1rem;
        font-weight: 750;

        margin-bottom: 0.35rem;
    }

    .empty-description {
        color: var(--muted);

        font-size: 0.72rem;
        line-height: 1.5;

        max-width: 560px;

        margin: 0 auto;
    }


    /* =====================================================
       RESULTS HEADER
    ===================================================== */

    .results-header {
        display: flex;
        align-items: end;
        justify-content: space-between;

        border-bottom: 1px solid var(--border);

        padding-bottom: 0.75rem;

        margin-top: 1.6rem;
        margin-bottom: 1rem;
    }

    .results-title {
        color: var(--text);

        font-size: 1.12rem;
        font-weight: 750;
    }

    .results-subtitle {
        color: var(--muted);

        font-size: 0.68rem;

        margin-top: 0.2rem;
    }


    /* =====================================================
       PRODUCT CARD CONTENT
    ===================================================== */

    .product-rank {
        display: flex;
        justify-content: space-between;
        align-items: center;

        margin-bottom: 8px;
    }

    .rank-number {
        color: var(--light);

        font-size: 0.62rem;
        font-weight: 800;
    }

    .score {
        color: var(--accent);

        background: #F2F5F8;
        border: 1px solid #DCE3EA;

        padding: 3px 6px;

        border-radius: 4px;

        font-size: 0.6rem;
        font-weight: 750;
    }

    .product-name {
        color: var(--text);

        font-size: 0.82rem;
        font-weight: 750;

        line-height: 1.35;

        margin-top: 9px;

        min-height: 34px;
    }

    .product-sub {
        color: var(--muted);

        font-size: 0.66rem;

        margin-top: 3px;
    }

    .meta {
        display: flex;
        flex-wrap: wrap;

        gap: 4px;

        margin-top: 8px;
    }

    .meta span {
        color: #69737D;

        background: #F5F6F7;
        border: 1px solid #E4E7EA;

        padding: 3px 5px;

        border-radius: 3px;

        font-size: 0.57rem;
    }

    .metrics {
        display: flex;
        gap: 9px;

        border-top: 1px solid var(--border);

        margin-top: 10px;
        padding-top: 9px;
    }

    .metric {
        color: var(--light);

        font-size: 0.59rem;
    }

    .metric b {
        color: var(--text);
    }

    .match-note {
        color: var(--success);

        font-size: 0.59rem;
        font-weight: 700;

        margin-top: 8px;
    }


    /* =====================================================
       ERROR
    ===================================================== */

    .error-box {
        background: #FFF5F5;
        border: 1px solid #F0CACA;
        border-radius: 7px;

        padding: 0.8rem 1rem;

        color: var(--danger);

        font-size: 0.72rem;
        line-height: 1.5;
    }


    /* =====================================================
       TECHNICAL NOTE
    ===================================================== */

    .technical-note {
        border-top: 1px solid var(--border);

        margin-top: 1.5rem;
        padding-top: 0.8rem;

        color: var(--light);

        font-size: 0.61rem;
        line-height: 1.5;
    }


    /* =====================================================
       STREAMLIT CARD CONTAINER
    ===================================================== */

    [data-testid="stVerticalBlockBorderWrapper"] {
        background: #FFFFFF !important;
        border-color: var(--border) !important;
        border-radius: 8px !important;
        padding: 0.55rem !important;
    }


    /* =====================================================
       PRODUCT IMAGE
    ===================================================== */

    [data-testid="stImage"] {
        margin-bottom: 0.2rem;
    }


    /* =====================================================
       MOBILE
    ===================================================== */

    @media (max-width: 900px) {

        .block-container {
            padding: 1rem;
        }

        .topbar-right {
            display: none;
        }

        .page-title {
            font-size: 1.6rem;
        }
    }

    </style>
    """
)


# ============================================================
# MODEL
# ============================================================

@st.cache_resource(
    show_spinner="Loading ProductLens model..."
)
def load_embedder() -> CLIPEmbedder:
    return CLIPEmbedder()


embedder = load_embedder()


# ============================================================
# CATALOG CACHE
# ============================================================

@st.cache_data(
    ttl=1800,
    show_spinner=False,
)
def fetch_catalog(
    queries: tuple[str, ...],
):
    """
    Get a local candidate pool from the cached catalog.

    No live Dataset Viewer /search API call should be made here.
    """

    return search_catalog(
        queries,
        per_query=60,
    )


# ============================================================
# IMAGE CACHE
# ============================================================

@st.cache_data(
    ttl=1800,
    show_spinner=False,
)
def fetch_candidate_images(
    urls: tuple[str, ...],
):
    return download_images(
        list(urls)
    )


# ============================================================
# LABELS
# ============================================================

CATEGORY_LABELS = [
    "shoes",
    "sports shoes",
    "running shoes",
    "sandals",
    "shirts",
    "t-shirts",
    "jeans",
    "trousers",
    "jackets",
    "dresses",
    "bags",
    "watches",
    "sunglasses",
    "accessories",
    "skirts",
    "tops",
    "hoodies",
    "sweaters",
]


# ============================================================
# HELPERS
# ============================================================

def unique_queries(
    values: Iterable[str],
) -> tuple[str, ...]:

    output: list[str] = []
    seen: set[str] = set()

    for value in values:

        cleaned = " ".join(
            str(value).split()
        )

        if (
            cleaned
            and cleaned.lower() not in seen
        ):

            output.append(
                cleaned
            )

            seen.add(
                cleaned.lower()
            )

    return tuple(output)


def safe_text(
    value,
    fallback: str = "N/A",
) -> str:

    if value is None:
        return fallback

    text = str(value).strip()

    if not text:
        return fallback

    return html.escape(
        text
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="sidebar-brand">
            Product<span>Lens</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="sidebar-description">
            Search fashion products using an image,
            a description, or both.
        </div>
        """,
        unsafe_allow_html=True,
    )


    # --------------------------------------------------------
    # REFERENCE IMAGE
    # --------------------------------------------------------

    st.markdown(
        '<div class="sidebar-title">Reference image</div>',
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload product image",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp",
        ],
        label_visibility="collapsed",
    )


    if uploaded_file:

        st.image(
            uploaded_file,
            width="stretch",
        )


    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    st.markdown(
        '<div class="sidebar-title">Results</div>',
        unsafe_allow_html=True,
    )

    top_k = st.selectbox(
        "Number of results",
        [4, 6, 8],
        index=0,
        label_visibility="collapsed",
    )


    # --------------------------------------------------------
    # SEARCH BUTTON
    # --------------------------------------------------------

    run_search = st.button(
        "Search products",
        type="primary",
        use_container_width=True,
    )


    # --------------------------------------------------------
    # ABOUT
    # --------------------------------------------------------

    st.markdown(
        '<div class="sidebar-title">About</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="sidebar-description">
            ProductLens combines image and language
            understanding to find relevant catalog products.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# TOP NAV
# ============================================================

st.html(
    """
    <div class="topbar">

        <div class="topbar-brand">
            Product<span>Lens</span>
        </div>

        <div class="topbar-right">

            <div class="catalog-status">
                <span class="status-dot"></span>
                Live catalog
            </div>

            <span>
                Product Search
            </span>

            <span>
                How it works
            </span>

        </div>

    </div>
    """
)


# ============================================================
# PAGE INTRO
# ============================================================

st.html(
    """
    <div class="page-title">
        Search products
    </div>

    <div class="page-description">
        Find visually and semantically similar fashion
        products from the catalog using a product image,
        natural language, or both.
    </div>
    """
)


# ============================================================
# PRODUCT DESCRIPTION
# ============================================================

st.html(
    """
    <div class="field-label">
        Describe the product you're looking for
    </div>
    """
)


text_query = st.text_input(
    "Product description",
    placeholder="e.g. black running shoes for men",
    label_visibility="collapsed",
)


# ============================================================
# POPULAR SEARCHES
# ============================================================

st.html(
    """
    <div class="popular">

        <div class="popular-label">
            Popular searches
        </div>

        <span class="chip">
            running shoes
        </span>

        <span class="chip">
            black backpack
        </span>

        <span class="chip">
            men's watch
        </span>

        <span class="chip">
            white sneakers
        </span>

    </div>
    """
)


# ============================================================
# SEARCH INFORMATION
# ============================================================

if (
    uploaded_file
    and text_query.strip()
):

    st.html(
        """
        <div class="search-info">
            <span class="info-dot"></span>
            Searching using product image + description
        </div>
        """
    )

elif uploaded_file:

    st.html(
        """
        <div class="search-info">
            <span class="info-dot"></span>
            Searching by visual similarity
        </div>
        """
    )

elif text_query.strip():

    st.html(
        """
        <div class="search-info">
            <span class="info-dot"></span>
            Searching by product description
        </div>
        """
    )


# ============================================================
# EMPTY STATE
# ============================================================

if not run_search:

    st.html(
        """
        <div class="empty-state">

            <div class="empty-icon">
                🔎
            </div>

            <div class="empty-title">
                Start your product search
            </div>

            <div class="empty-description">
                Describe the product you're looking for,
                upload a reference image, or use both.
                ProductLens will rank the most relevant
                products from the catalog.
            </div>

        </div>
        """
    )

    st.stop()


# ============================================================
# VALIDATION
# ============================================================

if (
    not text_query.strip()
    and not uploaded_file
):

    st.warning(
        "Describe a product or upload a reference image."
    )

    st.stop()


# ============================================================
# LOAD REFERENCE IMAGE
# ============================================================

reference_image: Image.Image | None = None


if uploaded_file:

    try:

        reference_image = Image.open(
            uploaded_file
        ).convert(
            "RGB"
        )

    except Exception:

        st.error(
            "The uploaded image could not be read."
        )

        st.stop()


# ============================================================
# BUILD SEARCH QUERY
# ============================================================

queries: list[str] = []


if text_query.strip():

    # For text and hybrid search, use the user's
    # actual description.
    queries.append(
        text_query.strip()
    )


elif reference_image:

    # Image-only search:
    # use a broad category hint only.
    #
    # Brand recognition is deliberately NOT used
    # for candidate generation.

    category_candidates = (
        embedder.top_image_labels(
            reference_image,
            CATEGORY_LABELS,
            "a studio product photo of {label}",
            top_k=1,
        )
    )

    if category_candidates:

        visual_category = (
            category_candidates[0][0]
        )

        queries.append(
            visual_category
        )

    else:

        queries.append(
            "fashion products"
        )


if not queries:

    queries = [
        "fashion products"
    ]


queries = list(
    unique_queries(
        queries
    )
)


# ============================================================
# FETCH CATALOG
# ============================================================

try:

    catalog = fetch_catalog(
        tuple(queries)
    )

except Exception as exc:

    error_message = safe_text(
        exc,
        "Unknown catalog error",
    )

    st.html(
        f"""
        <div class="error-box">

            <strong>
                Catalog service unavailable.
            </strong>

            <br><br>

            {error_message}

            <br><br>

            Please try again after the catalog
            has finished loading.

        </div>
        """
    )

    st.stop()


# ============================================================
# EMPTY CATALOG
# ============================================================

if catalog.empty:

    st.warning(
        "No products were found. Try a broader search "
        "such as 'running shoes', 'jacket', or 'black bag'."
    )

    st.stop()


# ============================================================
# LIMIT CANDIDATE POOL
# ============================================================

catalog = (
    catalog
    .head(60)
    .copy()
)


# ============================================================
# LOAD PRODUCT IMAGES
# ============================================================

image_urls = tuple(
    catalog[
        "image_url"
    ]
    .dropna()
    .astype(str)
    .tolist()
)


image_map = fetch_candidate_images(
    image_urls
)


# ============================================================
# KEEP ONLY PRODUCTS WITH IMAGES
# ============================================================

valid = (
    catalog[
        catalog[
            "image_url"
        ].isin(
            image_map.keys()
        )
    ]
    .copy()
    .reset_index(
        drop=True
    )
)


if valid.empty:

    st.error(
        "Products were found, but their images could not be loaded."
    )

    st.stop()


# ============================================================
# IMAGE EMBEDDINGS
# ============================================================

images = [
    image_map[
        url
    ]
    for url in valid[
        "image_url"
    ]
]


image_matrix = (
    embedder.get_image_embeddings_from_pil(
        images,
        batch_size=24,
    )
)


# Explicitly normalize candidate image embeddings.
image_norms = np.linalg.norm(
    image_matrix,
    axis=1,
    keepdims=True,
)


image_matrix = (
    image_matrix
    / np.clip(
        image_norms,
        1e-12,
        None,
    )
)


# ============================================================
# INITIALIZE SCORES
# ============================================================

visual_scores = np.zeros(
    len(valid),
    dtype="float32",
)


text_scores = np.zeros(
    len(valid),
    dtype="float32",
)


# ============================================================
# VISUAL SIMILARITY
# ============================================================

if reference_image:

    query_image = (
        embedder.get_image_embedding_from_pil(
            reference_image
        )
    )


    query_image = (
        query_image
        / max(
            np.linalg.norm(
                query_image
            ),
            1e-12,
        )
    )


    visual_scores = (
        image_matrix
        @ query_image
    )


# ============================================================
# TEXT SIMILARITY
# ============================================================

if text_query.strip():

    product_text = (
        valid[
            "productDisplayName"
        ]
        .fillna("")
        .astype(str)

        + " | "

        + valid[
            "articleType"
        ]
        .fillna("")
        .astype(str)

        + " | "

        + valid[
            "baseColour"
        ]
        .fillna("")
        .astype(str)

        + " | "

        + valid[
            "gender"
        ]
        .fillna("")
        .astype(str)

        + " | "

        + valid[
            "masterCategory"
        ]
        .fillna("")
        .astype(str)

        + " | "

        + valid[
            "subCategory"
        ]
        .fillna("")
        .astype(str)

    ).tolist()


    text_matrix = (
        embedder.get_text_embeddings(
            product_text,
            batch_size=64,
        )
    )


    query_text_vector = (
        embedder.get_text_embedding(
            text_query.strip()
        )
    )


    # Normalize product text embeddings.
    text_matrix = (
        text_matrix
        / np.clip(
            np.linalg.norm(
                text_matrix,
                axis=1,
                keepdims=True,
            ),
            1e-12,
            None,
        )
    )


    # Normalize query text embedding.
    query_text_vector = (
        query_text_vector
        / max(
            np.linalg.norm(
                query_text_vector
            ),
            1e-12,
        )
    )


    text_scores = (
        text_matrix
        @ query_text_vector
    )


# ============================================================
# FINAL RANKING
# ============================================================

if (
    reference_image
    and text_query.strip()
):

    # Hybrid search.
    #
    # The reference image gets more influence because
    # this application is primarily a visual product
    # retrieval system.

    VISUAL_WEIGHT = 0.75
    TEXT_WEIGHT = 0.25

    combined = (
        VISUAL_WEIGHT
        * visual_scores
        + TEXT_WEIGHT
        * text_scores
    )


elif reference_image:

    # Image search.
    combined = (
        visual_scores.copy()
    )


else:

    # Text search.
    combined = (
        text_scores.copy()
    )


# ============================================================
# SAVE SCORES
# ============================================================

valid["visual_score"] = (
    visual_scores
)

valid["text_score"] = (
    text_scores
)

valid["score"] = (
    combined
)


# ============================================================
# RANK
# ============================================================

valid = (
    valid
    .sort_values(
        "score",
        ascending=False,
    )
    .head(
        top_k
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# SEARCH MODE
# ============================================================

if (
    reference_image
    and text_query.strip()
):

    mode = "Image + text"

elif reference_image:

    mode = "Image"

else:

    mode = "Text"


# ============================================================
# RESULTS HEADER
# ============================================================

st.html(
    f"""
    <div class="results-header">

        <div>

            <div class="results-title">
                Results
            </div>

            <div class="results-subtitle">
                {len(valid)} products found
                · {mode}
                · {len(catalog)} candidates evaluated
            </div>

        </div>

    </div>
    """
)


# ============================================================
# PRODUCT GRID
# ============================================================

columns = st.columns(
    min(
        4,
        len(valid),
    )
)


for index, (_, item) in enumerate(
    valid.iterrows()
):

    with columns[
        index % len(columns)
    ]:

        # ----------------------------------------------------
        # ONE STREAMLIT CARD
        #
        # The image is now inside the same container as the
        # rank, score and product information.
        # This removes the large empty area from the old UI.
        # ----------------------------------------------------

        with st.container(
            border=True
        ):

            # ------------------------------------------------
            # RANK
            # ------------------------------------------------

            st.html(
                f"""
                <div class="product-rank">

                    <span class="rank-number">
                        #{index + 1}
                    </span>

                    <span class="score">
                        {item["score"]:.3f}
                    </span>

                </div>
                """
            )


            # ------------------------------------------------
            # PRODUCT IMAGE
            # ------------------------------------------------

            product_url = str(
                item["image_url"]
            )

            product_image = (
                image_map.get(
                    product_url
                )
            )


            if product_image is not None:

                st.image(
                    product_image,
                    width="stretch",
                )

            else:

                st.markdown(
                    """
                    <div style="
                        height: 250px;
                        display: flex;
                        align-items: center;
                        justify-content: center;

                        background: #F5F6F7;
                        border: 1px solid #E4E7EA;
                        border-radius: 5px;

                        color: #8B949D;
                        font-size: 0.7rem;
                    ">
                        Image unavailable
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


            # ------------------------------------------------
            # PRODUCT INFORMATION
            # ------------------------------------------------

            name = safe_text(
                item.get(
                    "productDisplayName",
                    "Product",
                ),
                "Product",
            )


            category = safe_text(
                item.get(
                    "articleType",
                    "Fashion",
                ),
                "Fashion",
            )


            brand = safe_text(
                item.get(
                    "brand",
                    "Unknown",
                ),
                "Unknown",
            )


            colour = safe_text(
                item.get(
                    "baseColour",
                    "N/A",
                ),
                "N/A",
            )


            gender = safe_text(
                item.get(
                    "gender",
                    "Unisex",
                ),
                "Unisex",
            )


            product_id = safe_text(
                item.get(
                    "id",
                    "N/A",
                ),
                "N/A",
            )


            st.html(
                f"""
                <div class="product-name">
                    {name}
                </div>

                <div class="product-sub">
                    {category}
                </div>

                <div class="meta">

                    <span>{brand}</span>
                    <span>{colour}</span>
                    <span>{gender}</span>
                    <span>ID {product_id}</span>

                </div>
                """
            )


            # ------------------------------------------------
            # SIMILARITY METRICS
            # ------------------------------------------------

            if (
                reference_image
                and text_query.strip()
            ):

                st.html(
                    f"""
                    <div class="metrics">

                        <span class="metric">
                            Visual
                            <b>
                                {item["visual_score"]:.3f}
                            </b>
                        </span>

                        <span class="metric">
                            Text
                            <b>
                                {item["text_score"]:.3f}
                            </b>
                        </span>

                    </div>
                    """
                )


            elif reference_image:

                st.html(
                    f"""
                    <div class="metrics">

                        <span class="metric">
                            Image similarity
                            <b>
                                {item["visual_score"]:.3f}
                            </b>
                        </span>

                    </div>
                    """
                )


            else:

                st.html(
                    f"""
                    <div class="metrics">

                        <span class="metric">
                            Semantic similarity
                            <b>
                                {item["text_score"]:.3f}
                            </b>
                        </span>

                    </div>
                    """
                )


            # ------------------------------------------------
            # MATCH NOTE
            # ------------------------------------------------

            if reference_image:

                st.html(
                    """
                    <div class="match-note">
                        ✓ Ranked using visual similarity
                    </div>
                    """
                )


# ============================================================
# TECHNICAL NOTE
# ============================================================

st.html(
    """
    <div class="technical-note">

        ProductLens uses OpenCLIP shared image and text
        embeddings to rank catalog products.

        Image searches use image-to-image cosine similarity.
        Text searches use semantic similarity between the
        query and product metadata.

        Hybrid searches give the reference image greater
        weight than text.

        Brand recognition is not presented as a verified
        prediction because zero-shot CLIP classification can
        confuse visually similar products.

    </div>
    """
)
