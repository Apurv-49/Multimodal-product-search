from __future__ import annotations

import os
import sys
from typing import Iterable

import numpy as np
import streamlit as st
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.embeddings import CLIPEmbedder
from src.remote_catalog import download_images, search_catalog


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
# PROFESSIONAL E-COMMERCE UI
# ============================================================

st.markdown(
    """
<style>

:root {
    --background: #F7F8FA;
    --surface: #FFFFFF;
    --surface-soft: #F2F4F7;
    --border: #E1E5EA;

    --text: #20242A;
    --text-secondary: #66707A;
    --text-light: #8A929B;

    --accent: #24466F;
    --accent-hover: #1B385B;

    --success: #4F7359;
    --warning: #A36A32;
}


/* ----------------------------------------------------------
   GLOBAL
---------------------------------------------------------- */

.stApp {
    background: var(--background);
    color: var(--text);
}

[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    max-width: 1500px;
    padding: 1.5rem 2.8rem 4rem;
}


/* ----------------------------------------------------------
   SIDEBAR
---------------------------------------------------------- */

section[data-testid="stSidebar"] {
    background: #FBFBFA;
    border-right: 1px solid var(--border);
}

section[data-testid="stSidebar"] > div {
    padding: 1.5rem 1.25rem;
}


/* Brand */

.sidebar-brand {
    font-size: 1.35rem;
    font-weight: 800;
    letter-spacing: -0.04em;
    color: var(--text);
    margin-bottom: 2rem;
}

.sidebar-brand span {
    color: var(--accent);
}


/* Section labels */

.sidebar-title {
    color: #747C85;
    font-size: 0.66rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin: 1.5rem 0 0.55rem;
}


/* Small description */

.sidebar-description {
    color: var(--text-secondary);
    font-size: 0.76rem;
    line-height: 1.5;
    margin-bottom: 1rem;
}


/* ----------------------------------------------------------
   MAIN HEADER
---------------------------------------------------------- */

.topbar {
    display: flex;
    justify-content: space-between;
    align-items: center;

    padding-bottom: 1rem;
    border-bottom: 1px solid var(--border);

    margin-bottom: 2rem;
}

.topbar-brand {
    font-size: 1rem;
    font-weight: 800;
    color: var(--text);
}

.topbar-brand span {
    color: var(--accent);
}

.topbar-links {
    display: flex;
    gap: 1.5rem;
    color: var(--text-secondary);
    font-size: 0.78rem;
}


/* ----------------------------------------------------------
   PAGE INTRO
---------------------------------------------------------- */

.page-title {
    font-size: 2rem;
    font-weight: 750;
    letter-spacing: -0.04em;
    color: var(--text);
    margin-bottom: 0.35rem;
}

.page-description {
    color: var(--text-secondary);
    font-size: 0.9rem;
    line-height: 1.55;
    max-width: 720px;
    margin-bottom: 1.5rem;
}


/* ----------------------------------------------------------
   SEARCH AREA
---------------------------------------------------------- */

.search-panel {
    background: var(--surface);
    border: 1px solid var(--border);
    padding: 1.1rem 1.2rem;

    border-radius: 10px;

    margin-bottom: 1.2rem;
}

.search-label {
    color: var(--text-secondary);
    font-size: 0.72rem;
    font-weight: 700;

    margin-bottom: 0.4rem;
}


/* Popular searches */

.popular-label {
    color: var(--text-light);
    font-size: 0.7rem;
    margin-top: 0.8rem;
    margin-bottom: 0.45rem;
}

.search-chip {
    display: inline-block;

    background: #F4F5F6;
    border: 1px solid #E2E5E8;

    color: #505861;

    padding: 5px 9px;
    margin-right: 5px;

    border-radius: 5px;

    font-size: 0.7rem;
}


/* ----------------------------------------------------------
   REFERENCE IMAGE
---------------------------------------------------------- */

.reference-panel {
    background: var(--surface);

    border: 1px solid var(--border);

    border-radius: 10px;

    padding: 1rem;

    margin-bottom: 1.2rem;
}

.reference-title {
    font-size: 0.72rem;
    font-weight: 750;

    color: var(--text);

    margin-bottom: 0.7rem;
}

.reference-caption {
    color: var(--text-light);
    font-size: 0.68rem;

    margin-top: 0.45rem;
}


/* ----------------------------------------------------------
   SEARCH SUMMARY
---------------------------------------------------------- */

.results-header {
    display: flex;
    justify-content: space-between;
    align-items: end;

    border-bottom: 1px solid var(--border);

    padding-bottom: 0.8rem;

    margin-top: 1.8rem;
    margin-bottom: 1rem;
}

.results-title {
    font-size: 1.15rem;
    font-weight: 750;

    color: var(--text);
}

.results-subtitle {
    color: var(--text-secondary);
    font-size: 0.72rem;

    margin-top: 0.2rem;
}


/* ----------------------------------------------------------
   INSIGHTS
---------------------------------------------------------- */

.insights {
    display: grid;
    grid-template-columns: repeat(3, 1fr);

    gap: 10px;

    margin-bottom: 1.4rem;
}

.insight {
    background: var(--surface);

    border: 1px solid var(--border);

    border-radius: 8px;

    padding: 0.8rem 0.9rem;
}

.insight-title {
    color: var(--text-light);

    font-size: 0.63rem;
    font-weight: 750;

    text-transform: uppercase;
    letter-spacing: 0.08em;
}

.insight-value {
    color: var(--text);

    font-size: 0.82rem;
    font-weight: 700;

    margin-top: 0.3rem;
}

.insight-note {
    color: var(--text-light);

    font-size: 0.65rem;

    margin-top: 0.2rem;
}


/* ----------------------------------------------------------
   PRODUCT CARDS
---------------------------------------------------------- */

.product-card {
    background: var(--surface);

    border: 1px solid var(--border);

    border-radius: 8px;

    padding: 10px;

    min-height: 510px;

    transition:
        border-color 0.15s ease,
        box-shadow 0.15s ease;
}

.product-card:hover {
    border-color: #C5CCD4;

    box-shadow:
        0 5px 18px rgba(30, 40, 50, 0.07);
}


/* Ranking */

.product-rank {
    display: flex;
    justify-content: space-between;
    align-items: center;

    margin-bottom: 8px;
}

.rank-number {
    color: var(--text-light);

    font-size: 0.65rem;
    font-weight: 800;
}

.score {
    color: var(--accent);

    background: #F1F4F8;

    border: 1px solid #DCE3EA;

    border-radius: 5px;

    padding: 4px 7px;

    font-size: 0.64rem;
    font-weight: 750;
}


/* Product name */

.product-name {
    color: var(--text);

    font-size: 0.88rem;
    font-weight: 750;

    line-height: 1.3;

    margin-top: 10px;
}

.product-sub {
    color: var(--text-secondary);

    font-size: 0.69rem;

    margin-top: 3px;
}


/* Metadata */

.meta {
    display: flex;
    flex-wrap: wrap;

    gap: 5px;

    margin-top: 9px;
}

.meta span {
    color: #646D76;

    background: #F5F6F7;

    border: 1px solid #E5E7EA;

    border-radius: 4px;

    padding: 3px 6px;

    font-size: 0.61rem;
}


/* Metrics */

.metrics {
    display: flex;

    gap: 10px;

    margin-top: 12px;
    padding-top: 10px;

    border-top: 1px solid var(--border);
}

.metric {
    color: var(--text-light);

    font-size: 0.63rem;
}

.metric b {
    color: var(--text);
}


/* Match note */

.match-note {
    margin-top: 9px;

    color: var(--success);

    font-size: 0.62rem;
    font-weight: 700;
}

.match-note.warn {
    color: var(--warning);
}


/* ----------------------------------------------------------
   EMPTY STATE
---------------------------------------------------------- */

.empty-state {
    background: var(--surface);

    border: 1px solid var(--border);

    border-radius: 10px;

    padding: 4rem 2rem;

    text-align: center;

    margin-top: 1rem;
}

.empty-title {
    color: var(--text);

    font-size: 1rem;
    font-weight: 750;

    margin-bottom: 0.4rem;
}

.empty-description {
    color: var(--text-secondary);

    font-size: 0.78rem;

    max-width: 580px;

    margin: auto;

    line-height: 1.5;
}


/* ----------------------------------------------------------
   BUTTONS
---------------------------------------------------------- */

.stButton > button {
    border-radius: 6px;

    font-weight: 700;

    min-height: 42px;
}


/* Primary button */

.stButton > button[kind="primary"] {
    background: var(--accent);
    border-color: var(--accent);

    color: white;
}

.stButton > button[kind="primary"]:hover {
    background: var(--accent-hover);
    border-color: var(--accent-hover);
}


/* ----------------------------------------------------------
   INPUTS
---------------------------------------------------------- */

.stTextInput input {
    border-radius: 6px;

    border: 1px solid #D8DDE2;

    background: white;

    color: var(--text);

    min-height: 42px;
}

.stTextInput input:focus {
    border-color: var(--accent);

    box-shadow:
        0 0 0 1px var(--accent);
}


/* File uploader */

[data-testid="stFileUploader"] {
    border-radius: 7px;
}


/* ----------------------------------------------------------
   SLIDER
---------------------------------------------------------- */

.stSlider {
    padding-top: 0.2rem;
}


/* ----------------------------------------------------------
   FOOTER
---------------------------------------------------------- */

footer {
    visibility: hidden;
}


/* ----------------------------------------------------------
   RESPONSIVE
---------------------------------------------------------- */

@media (max-width: 900px) {

    .block-container {
        padding: 1rem;
    }

    .topbar-links {
        display: none;
    }

    .page-title {
        font-size: 1.6rem;
    }

    .insights {
        grid-template-columns: 1fr;
    }
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# MODEL / DATA
# ============================================================

@st.cache_resource(show_spinner="Loading product search model...")
def load_embedder() -> CLIPEmbedder:
    return CLIPEmbedder()


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_catalog(queries: tuple[str, ...]):
    return search_catalog(queries, per_query=45)


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_candidate_images(urls: tuple[str, ...]):
    return download_images(list(urls))


embedder = load_embedder()


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

BRAND_LABELS = [
    "Nike",
    "Adidas",
    "Puma",
    "Reebok",
    "New Balance",
    "Skechers",
    "Asics",
    "Converse",
    "Vans",
    "Fila",
    "Under Armour",
    "Crocs",
    "Levis",
    "Bata",
    "Woodland",
]

COMMON_BRANDS = [
    "Nike",
    "Adidas",
    "Puma",
    "Reebok",
    "Skechers",
    "New Balance",
]


# ============================================================
# HELPERS
# ============================================================

def unique_queries(values: Iterable[str]) -> tuple[str, ...]:

    output: list[str] = []
    seen: set[str] = set()

    for value in values:

        cleaned = " ".join(str(value).split())

        if cleaned and cleaned.lower() not in seen:
            output.append(cleaned)
            seen.add(cleaned.lower())

    return tuple(output)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        '<div class="sidebar-brand">Product<span>Lens</span></div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sidebar-description">'
        'Search fashion products using text, images, or both.'
        '</div>',
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # SEARCH MODE
    # --------------------------------------------------------

    st.markdown(
        '<div class="sidebar-title">Search mode</div>',
        unsafe_allow_html=True,
    )

    search_mode = st.radio(
        "Search mode",
        ["Text", "Image", "Hybrid"],
        horizontal=True,
        label_visibility="collapsed",
    )

    # --------------------------------------------------------
    # IMAGE SEARCH
    # --------------------------------------------------------

    uploaded_file = None

    if search_mode in ["Image", "Hybrid"]:

        st.markdown(
            '<div class="sidebar-title">Reference image</div>',
            unsafe_allow_html=True,
        )

        uploaded_file = st.file_uploader(
            "Upload product image",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed",
        )

        if uploaded_file:

            st.image(
                uploaded_file,
                width="stretch",
            )

            st.markdown(
                '<div class="reference-caption">'
                'Reference product'
                '</div>',
                unsafe_allow_html=True,
            )

    # --------------------------------------------------------
    # RANKING
    # --------------------------------------------------------

    st.markdown(
        '<div class="sidebar-title">Ranking</div>',
        unsafe_allow_html=True,
    )

    alpha = st.slider(
        "Visual weight",
        0.0,
        1.0,
        0.80,
        0.05,
        help=(
            "Controls the contribution of image similarity "
            "when both image and text are provided."
        ),
    )

    top_k = st.selectbox(
        "Number of results",
        [4, 6, 8],
        index=0,
    )

    # --------------------------------------------------------
    # SEARCH BUTTON
    # --------------------------------------------------------

    run_search = st.button(
        "Search products",
        type="primary",
        use_container_width=True,
    )

    st.markdown(
        '<div class="sidebar-title">About</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sidebar-description">'
        'ProductLens combines visual and semantic similarity '
        'to retrieve relevant catalog products.'
        '</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    """
<div class="topbar">

    <div class="topbar-brand">
        Product<span>Lens</span>
    </div>

    <div class="topbar-links">
        <span>Product Search</span>
        <span>How it works</span>
    </div>

</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# PAGE INTRO
# ============================================================

st.markdown(
    '<div class="page-title">Search products</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="page-description">'
    'Find visually and semantically similar fashion products '
    'from the catalog using a product image, natural language, '
    'or both.'
    '</div>',
    unsafe_allow_html=True,
)


# ============================================================
# MAIN TEXT SEARCH
# ============================================================

text_query = ""

if search_mode in ["Text", "Hybrid"]:

    st.markdown(
        '<div class="search-label">What are you looking for?</div>',
        unsafe_allow_html=True,
    )

    text_query = st.text_input(
        "Product search",
        placeholder="e.g. blue running shoes for men",
        label_visibility="collapsed",
    )

    st.markdown(
        '<div class="popular-label">Popular searches</div>'
        '<span class="search-chip">running shoes</span>'
        '<span class="search-chip">black backpack</span>'
        '<span class="search-chip">men watch</span>'
        '<span class="search-chip">white sneakers</span>',
        unsafe_allow_html=True,
    )


# ============================================================
# EMPTY STATE
# ============================================================

if not run_search:

    st.markdown(
        """
<div class="empty-state">

    <div class="empty-title">
        Start your product search
    </div>

    <div class="empty-description">
        Enter a product description or upload a reference image.
        ProductLens will compare your query against the fashion
        catalog and rank the most relevant products.
    </div>

</div>
""",
        unsafe_allow_html=True,
    )

    st.stop()


# ============================================================
# VALIDATION
# ============================================================

if not text_query.strip() and not uploaded_file:

    st.warning(
        "Enter a product description or upload a reference image."
    )

    st.stop()


# ============================================================
# REFERENCE IMAGE
# ============================================================

reference_image: Image.Image | None = None

category_candidates: list[tuple[str, float]] = []
brand_candidates: list[tuple[str, float]] = []

visual_hint = ""
brand_hint = ""


if uploaded_file:

    reference_image = Image.open(uploaded_file).convert("RGB")

    category_candidates = embedder.top_image_labels(
        reference_image,
        CATEGORY_LABELS,
        "a studio product photo of {label}",
        top_k=3,
    )

    brand_candidates = embedder.top_image_labels(
        reference_image,
        BRAND_LABELS,
        "a product photo from the brand {label}",
        top_k=5,
    )

    visual_hint = (
        category_candidates[0][0]
        if category_candidates
        else "fashion product"
    )

    brand_hint = (
        brand_candidates[0][0]
        if brand_candidates
        else ""
    )


# ============================================================
# BUILD CATALOG QUERIES
# ============================================================

queries: list[str] = []


if text_query.strip():
    queries.append(text_query.strip())


if reference_image:

    # Use several category hypotheses.
    for label, confidence in category_candidates:

        queries.append(label)

        if confidence > 0.02:

            for brand in COMMON_BRANDS:

                queries.append(
                    f"{brand} {label}"
                )

    # Search likely brands independently.
    for brand, _ in brand_candidates[:3]:

        queries.append(
            f"{brand} {visual_hint}"
        )


queries = list(unique_queries(queries))


if not queries:

    queries = ["fashion products"]


# ============================================================
# RETRIEVE CATALOG
# ============================================================

try:

    catalog = fetch_catalog(
        tuple(queries)
    )

except Exception as exc:

    st.error(
        f"Catalog service could not be reached: {exc}"
    )

    st.stop()


if catalog.empty:

    st.warning(
        "No products were found. "
        "Try a broader search such as "
        "'shoes', 'jacket', or 'black bag'."
    )

    st.stop()


# ============================================================
# CANDIDATE POOL
# ============================================================

catalog = catalog.head(260).copy()

image_map = fetch_candidate_images(
    tuple(catalog["image_url"].tolist())
)

valid = (
    catalog[
        catalog["image_url"].isin(image_map.keys())
    ]
    .copy()
    .reset_index(drop=True)
)


if valid.empty:

    st.error(
        "Product metadata was found, but the product images "
        "could not be loaded."
    )

    st.stop()


# ============================================================
# IMAGE EMBEDDINGS
# ============================================================

images = [
    image_map[url]
    for url in valid["image_url"]
]

image_matrix = embedder.get_image_embeddings_from_pil(
    images,
    batch_size=24,
)


visual_scores = np.zeros(
    len(valid),
    dtype="float32",
)

text_scores = np.zeros(
    len(valid),
    dtype="float32",
)


# ============================================================
# IMAGE SIMILARITY
# ============================================================

if reference_image:

    query_image = (
        embedder.get_image_embedding_from_pil(
            reference_image
        )
    )

    visual_scores = (
        image_matrix @ query_image
    )


# ============================================================
# TEXT SIMILARITY
# ============================================================

if text_query.strip():

    product_text = (
        valid["productDisplayName"].fillna("")
        + " | "
        + valid["articleType"].fillna("")
        + " | "
        + valid["baseColour"].fillna("")
        + " | "
        + valid["gender"].fillna("")
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

    text_scores = (
        text_matrix @ query_text_vector
    )


# ============================================================
# HYBRID SCORE
# ============================================================

if reference_image and text_query.strip():

    combined = (
        alpha * visual_scores
        + (1.0 - alpha) * text_scores
    )

elif reference_image:

    combined = visual_scores.copy()

else:

    combined = text_scores.copy()


# ============================================================
# BRAND RERANKING
# ============================================================

if reference_image and brand_candidates:

    brand_scores = {
        brand.lower(): score
        for brand, score in brand_candidates
    }

    metadata_brand = (
        valid["brand"]
        .fillna("")
        .astype(str)
        .str.lower()
    )

    brand_bonus = np.array(
        [
            brand_scores.get(brand, 0.0)
            for brand in metadata_brand
        ],
        dtype="float32",
    )

    combined = (
        combined
        + 0.05 * brand_bonus
    )


# ============================================================
# FINAL RANKING
# ============================================================

valid["visual_score"] = visual_scores
valid["text_score"] = text_scores
valid["score"] = combined

valid = (
    valid
    .sort_values(
        "score",
        ascending=False,
    )
    .head(top_k)
    .reset_index(drop=True)
)


# ============================================================
# SEARCH MODE
# ============================================================

mode = (
    "Image + text"
    if reference_image and text_query.strip()
    else (
        "Image only"
        if reference_image
        else "Text only"
    )
)


# ============================================================
# RESULTS HEADER
# ============================================================

st.markdown(
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
""",
    unsafe_allow_html=True,
)


# ============================================================
# RETRIEVAL INSIGHTS
# ============================================================

if reference_image:

    category_text = (
        ", ".join(
            label.title()
            for label, _ in category_candidates[:3]
        )
        or "Unknown"
    )

    brand_text = (
        ", ".join(
            brand
            for brand, _ in brand_candidates[:3]
        )
        or "Unknown"
    )

    st.markdown(
        f"""
<div class="insights">

    <div class="insight">

        <div class="insight-title">
            Product category
        </div>

        <div class="insight-value">
            {category_text}
        </div>

        <div class="insight-note">
            Visual category candidates
        </div>

    </div>


    <div class="insight">

        <div class="insight-title">
            Brand candidates
        </div>

        <div class="insight-value">
            {brand_text}
        </div>

        <div class="insight-note">
            Used as a ranking signal
        </div>

    </div>


    <div class="insight">

        <div class="insight-title">
            Visual weight
        </div>

        <div class="insight-value">
            {alpha:.0%}
        </div>

        <div class="insight-note">
            Image contribution to ranking
        </div>

    </div>

</div>
""",
        unsafe_allow_html=True,
    )


# ============================================================
# PRODUCT GRID
# ============================================================

cols = st.columns(
    min(4, len(valid))
)


for index, (_, item) in enumerate(
    valid.iterrows()
):

    with cols[index % len(cols)]:

        st.markdown(
            '<div class="product-card">',
            unsafe_allow_html=True,
        )

        # Ranking
        st.markdown(
            f"""
<div class="product-rank">

    <span class="rank-number">
        #{index + 1}
    </span>

    <span class="score">
        {item["score"]:.3f}
    </span>

</div>
""",
            unsafe_allow_html=True,
        )


        # Product image
        st.image(
            item["image_url"],
            width="stretch",
        )


        # Product information
        name = str(
            item.get(
                "productDisplayName",
                "Product",
            )
        )

        category = str(
            item.get(
                "articleType",
                "Fashion",
            )
        )

        brand = str(
            item.get(
                "brand",
                "Unknown",
            )
        )

        colour = str(
            item.get(
                "baseColour",
                "N/A",
            )
        )

        gender = str(
            item.get(
                "gender",
                "Unisex",
            )
        )


        st.markdown(
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

</div>
""",
            unsafe_allow_html=True,
        )


        # Similarity metrics
        if reference_image and text_query.strip():

            st.markdown(
                f"""
<div class="metrics">

    <span class="metric">
        Visual
        <b>{item["visual_score"]:.3f}</b>
    </span>

    <span class="metric">
        Text
        <b>{item["text_score"]:.3f}</b>
    </span>

</div>
""",
                unsafe_allow_html=True,
            )


        elif reference_image:

            st.markdown(
                f"""
<div class="metrics">

    <span class="metric">
        Image similarity
        <b>{item["visual_score"]:.3f}</b>
    </span>

</div>
""",
                unsafe_allow_html=True,
            )


        else:

            st.markdown(
                f"""
<div class="metrics">

    <span class="metric">
        Semantic similarity
        <b>{item["text_score"]:.3f}</b>
    </span>

</div>
""",
                unsafe_allow_html=True,
            )


        # Brand note
        if (
            reference_image
            and brand_hint
            and brand.lower() == brand_hint.lower()
        ):

            st.markdown(
                '<div class="match-note">'
                '✓ Brand matches visual candidate'
                '</div>',
                unsafe_allow_html=True,
            )

        elif reference_image:

            st.markdown(
                '<div class="match-note warn">'
                'Visual match · brand is not ground truth'
                '</div>',
                unsafe_allow_html=True,
            )


        st.markdown(
            '</div>',
            unsafe_allow_html=True,
        )


# ============================================================
# TECHNICAL NOTE
# ============================================================

st.markdown(
    """
<div style="
    margin-top: 1.8rem;
    padding-top: 1rem;
    border-top: 1px solid #E1E5EA;
    color: #8A929B;
    font-size: 0.65rem;
    line-height: 1.5;
">
    ProductLens uses image and text embeddings to rank catalog
    products. Zero-shot brand and category predictions are used
    as retrieval hints rather than verified product labels.
</div>
""",
    unsafe_allow_html=True,
)
