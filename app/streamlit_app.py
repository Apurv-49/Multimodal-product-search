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


st.set_page_config(
    page_title="ProductLens — Multimodal Product Search",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
:root{--bg:#070B12;--panel:#0E1521;--panel2:#111B2A;--line:#202C3D;--text:#F5F7FB;--muted:#91A0B5;--blue:#5B8CFF;--green:#42D392;--danger:#FF7B7B}
.stApp{background:var(--bg);color:var(--text)}
[data-testid="stHeader"]{background:transparent}
.block-container{max-width:1450px;padding:2rem 3rem 4rem}
section[data-testid="stSidebar"]{background:#09101A;border-right:1px solid var(--line)}
section[data-testid="stSidebar"]>div{padding:1.25rem 1.1rem}
.sidebar-brand{font-size:1.35rem;font-weight:900;letter-spacing:-.04em;margin-bottom:1.8rem}.sidebar-brand span{color:var(--blue)}
.sidebar-title{color:#A9B6C8;font-size:.68rem;font-weight:800;text-transform:uppercase;letter-spacing:.13em;margin:18px 0 8px}
.hero{display:flex;justify-content:space-between;align-items:flex-start;gap:24px;margin-bottom:26px}
.eyebrow{color:var(--blue);font-size:.68rem;font-weight:900;letter-spacing:.15em;text-transform:uppercase;margin-bottom:9px}
.hero h1{font-size:2.5rem;line-height:1.02;margin:0;letter-spacing:-.055em}.hero p{color:var(--muted);margin:.7rem 0 0;max-width:720px;line-height:1.55}
.badge{background:#101B2E;border:1px solid #294775;color:#8FB0FF;border-radius:999px;padding:8px 13px;font-size:.7rem;font-weight:800;white-space:nowrap}
.summary{display:flex;justify-content:space-between;align-items:center;gap:16px;background:linear-gradient(135deg,#0D1725,#0A111B);border:1px solid var(--line);border-radius:16px;padding:15px 18px;margin-bottom:18px}
.summary-main{font-weight:800;font-size:1rem}.summary-sub{color:var(--muted);font-size:.78rem;margin-top:3px}.summary-score{color:#9AB7FF;font-size:.75rem;font-weight:800}
.insights{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:20px}.insight{background:#0D1622;border:1px solid var(--line);border-radius:13px;padding:12px 14px}.insight-title{color:#718096;font-size:.65rem;text-transform:uppercase;letter-spacing:.1em}.insight-value{font-size:.9rem;font-weight:800;margin-top:4px}.insight-note{color:#66758A;font-size:.68rem;margin-top:3px}
.result-card{background:var(--panel);border:1px solid var(--line);border-radius:18px;padding:12px;min-height:530px;box-shadow:0 10px 32px rgba(0,0,0,.18);overflow:hidden}.result-card:hover{border-color:#355C9E}
.rank{display:flex;justify-content:space-between;align-items:center;margin:1px 2px 10px}.rank-number{color:#748399;font-size:.7rem;font-weight:900}.score{color:#A0BCFF;background:#101D33;border:1px solid #2B4C83;border-radius:999px;padding:5px 8px;font-size:.68rem;font-weight:900}
.product-name{font-size:.96rem;font-weight:850;color:#fff;margin-top:12px;line-height:1.3}.product-sub{color:#7F8DA2;font-size:.72rem;margin-top:4px}
.meta{display:flex;flex-wrap:wrap;gap:5px;margin-top:10px}.meta span{background:#111B2A;border:1px solid #202E42;color:#AAB7C8;border-radius:7px;padding:4px 6px;font-size:.65rem}
.metrics{display:flex;gap:6px;margin-top:12px;padding-top:11px;border-top:1px solid var(--line);flex-wrap:wrap}.metric{color:#77879C;font-size:.65rem}.metric b{color:#B7C3D3}
.match-note{margin-top:9px;color:#55D99A;font-size:.66rem;font-weight:800}.match-note.warn{color:#FFB86B}
.empty{border:1px dashed #2A384B;background:#0C141F;border-radius:18px;padding:60px 30px;text-align:center;color:#738196}.empty strong{display:block;color:#E8EDF5;font-size:1.05rem;margin-bottom:7px}.small-note{color:#68778C;font-size:.69rem;line-height:1.5;margin-top:9px}
.stButton>button{border-radius:10px;font-weight:800}.stFileUploader{border-radius:10px}
footer{visibility:hidden}
@media(max-width:900px){.block-container{padding:1.2rem}.hero{display:block}.badge{display:inline-block;margin-top:14px}.insights{grid-template-columns:1fr}}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading OpenCLIP model…")
def load_embedder() -> CLIPEmbedder:
    return CLIPEmbedder()


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_catalog(queries: tuple[str, ...]):
    return search_catalog(queries, per_query=45)


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_candidate_images(urls: tuple[str, ...]):
    return download_images(list(urls))


embedder = load_embedder()

CATEGORY_LABELS = [
    "shoes", "sports shoes", "running shoes", "sandals", "shirts", "t-shirts",
    "jeans", "trousers", "jackets", "dresses", "bags", "watches", "sunglasses",
    "accessories", "skirts", "tops", "hoodies", "sweaters"
]
BRAND_LABELS = [
    "Nike", "Adidas", "Puma", "Reebok", "New Balance", "Skechers", "Asics",
    "Converse", "Vans", "Fila", "Under Armour", "Crocs", "Levis", "Bata", "Woodland"
]
COMMON_BRANDS = ["Nike", "Adidas", "Puma", "Reebok", "Skechers", "New Balance"]


def unique_queries(values: Iterable[str]) -> tuple[str, ...]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = " ".join(str(value).split())
        if cleaned and cleaned.lower() not in seen:
            output.append(cleaned)
            seen.add(cleaned.lower())
    return tuple(output)


with st.sidebar:
    st.markdown('<div class="sidebar-brand">Product<span>Lens</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">Text search</div>', unsafe_allow_html=True)
    text_query = st.text_input(
        "Text query", placeholder="blue running shoes for men", label_visibility="collapsed"
    )

    st.markdown('<div class="sidebar-title">Visual search</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload product image", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed"
    )
    if uploaded_file:
        st.image(uploaded_file, caption="Reference image", width="stretch")

    st.markdown('<div class="sidebar-title">Ranking controls</div>', unsafe_allow_html=True)
    alpha = st.slider(
        "Visual weight", 0.0, 1.0, 0.80, 0.05,
        help="When both inputs are present, this controls the image-vs-text contribution."
    )
    top_k = st.selectbox("Results", [4, 6, 8], index=0)
    run_search = st.button("Search products", type="primary", use_container_width=True)
    st.markdown(
        '<div class="small-note">OpenCLIP embeddings · live catalog images · visual/semantic reranking.</div>',
        unsafe_allow_html=True,
    )

st.markdown(
    """
<div class="hero">
  <div>
    <div class="eyebrow">Multimodal commerce intelligence</div>
    <h1>Find products by meaning, not keywords.</h1>
    <p>Search with a product photo, natural language, or both. ProductLens ranks real catalog items using shared image-text embeddings and transparent similarity signals.</p>
  </div>
  <div class="badge">OPENCLIP · 512-D · LIVE CATALOG</div>
</div>
""",
    unsafe_allow_html=True,
)

if not run_search:
    st.markdown(
        '<div class="empty"><strong>Start a product search</strong>Upload a product image or enter a query. Results will show real product imagery, metadata, and separate visual/semantic scores.</div>',
        unsafe_allow_html=True,
    )
    st.stop()

if not text_query.strip() and not uploaded_file:
    st.warning("Add a text query or upload a reference image first.")
    st.stop()

reference_image: Image.Image | None = None
category_candidates: list[tuple[str, float]] = []
brand_candidates: list[tuple[str, float]] = []
visual_hint = ""
brand_hint = ""

if uploaded_file:
    reference_image = Image.open(uploaded_file).convert("RGB")
    category_candidates = embedder.top_image_labels(
        reference_image, CATEGORY_LABELS, "a studio product photo of {label}", top_k=3
    )
    brand_candidates = embedder.top_image_labels(
        reference_image, BRAND_LABELS, "a product photo from the brand {label}", top_k=5
    )
    visual_hint = category_candidates[0][0] if category_candidates else "fashion product"
    brand_hint = brand_candidates[0][0] if brand_candidates else ""

queries: list[str] = []
if text_query.strip():
    queries.append(text_query.strip())

if reference_image:
    # Use several category candidates rather than trusting one zero-shot label.
    for label, confidence in category_candidates:
        queries.append(label)
        if confidence > 0.02:
            for brand in COMMON_BRANDS:
                queries.append(f"{brand} {label}")
    # Also search the most likely brands independently. This makes brand retrieval
    # robust when the top CLIP brand label is imperfect.
    for brand, _ in brand_candidates[:3]:
        queries.append(f"{brand} {visual_hint}")

queries = list(unique_queries(queries))
if not queries:
    queries = ["fashion products"]

try:
    catalog = fetch_catalog(tuple(queries))
except Exception as exc:
    st.error(f"Catalog service could not be reached: {exc}")
    st.stop()

if catalog.empty:
    st.warning("No products were found. Try a broader query such as 'shoes', 'jacket', or 'black bag'.")
    st.stop()

# Keep the candidate pool bounded before downloading/encoding images.
catalog = catalog.head(260).copy()
image_map = fetch_candidate_images(tuple(catalog["image_url"].tolist()))
valid = catalog[catalog["image_url"].isin(image_map.keys())].copy().reset_index(drop=True)

if valid.empty:
    st.error("Product metadata was found, but the product images could not be loaded.")
    st.stop()

images = [image_map[url] for url in valid["image_url"]]
image_matrix = embedder.get_image_embeddings_from_pil(images, batch_size=24)

visual_scores = np.zeros(len(valid), dtype="float32")
text_scores = np.zeros(len(valid), dtype="float32")

if reference_image:
    query_image = embedder.get_image_embedding_from_pil(reference_image)
    visual_scores = image_matrix @ query_image

if text_query.strip():
    product_text = (
        valid["productDisplayName"].fillna("") + " | "
        + valid["articleType"].fillna("") + " | "
        + valid["baseColour"].fillna("") + " | "
        + valid["gender"].fillna("")
    ).tolist()
    text_matrix = embedder.get_text_embeddings(product_text, batch_size=64)
    query_text_vector = embedder.get_text_embedding(text_query.strip())
    text_scores = text_matrix @ query_text_vector

if reference_image and text_query.strip():
    combined = alpha * visual_scores + (1.0 - alpha) * text_scores
elif reference_image:
    combined = visual_scores.copy()
else:
    combined = text_scores.copy()

# Metadata is used only for a cautious reranking signal. It never replaces
# image-to-image similarity, so a visually close item cannot be relabeled solely
# because a zero-shot brand guess was wrong.
if reference_image and brand_candidates:
    brand_scores = {brand.lower(): score for brand, score in brand_candidates}
    metadata_brand = valid["brand"].fillna("").astype(str).str.lower()
    brand_bonus = np.array([brand_scores.get(brand, 0.0) for brand in metadata_brand], dtype="float32")
    combined = combined + 0.05 * brand_bonus

valid["visual_score"] = visual_scores
valid["text_score"] = text_scores
valid["score"] = combined
valid = valid.sort_values("score", ascending=False).head(top_k).reset_index(drop=True)

mode = "Image + text" if reference_image and text_query.strip() else ("Image only" if reference_image else "Text only")

st.markdown(
    f'<div class="summary"><div><div class="summary-main">{len(valid)} products ranked</div><div class="summary-sub">{mode} · {len(catalog)} candidates evaluated</div></div><div class="summary-score">Visual weight α = {alpha:.2f}</div></div>',
    unsafe_allow_html=True,
)

if reference_image:
    category_text = ", ".join(label.title() for label, _ in category_candidates[:3]) or "Unknown"
    brand_text = ", ".join(brand for brand, _ in brand_candidates[:3]) or "Unknown"
    st.markdown(
        f'<div class="insights">'
        f'<div class="insight"><div class="insight-title">Visual candidates</div><div class="insight-value">{category_text}</div><div class="insight-note">Top category hypotheses from CLIP</div></div>'
        f'<div class="insight"><div class="insight-title">Brand hypotheses</div><div class="insight-value">{brand_text}</div><div class="insight-note">Used only as a small reranking signal</div></div>'
        f'<div class="insight"><div class="insight-title">Retrieval mode</div><div class="insight-value">{mode}</div><div class="insight-note">Image similarity remains the primary signal</div></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

st.markdown("### Results")
cols = st.columns(min(4, len(valid)))

for index, (_, item) in enumerate(valid.iterrows()):
    with cols[index % len(cols)]:
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown(
            f'<div class="rank"><span class="rank-number">RANK #{index + 1:02d}</span><span class="score">{item["score"]:.3f} match</span></div>',
            unsafe_allow_html=True,
        )
        st.image(item["image_url"], width="stretch")

        name = str(item.get("productDisplayName", "Product"))
        category = str(item.get("articleType", "Fashion"))
        brand = str(item.get("brand", "Unknown"))
        colour = str(item.get("baseColour", "N/A"))
        gender = str(item.get("gender", "Unisex"))

        st.markdown(
            f'<div class="product-name">{name}</div><div class="product-sub">{category}</div>'
            f'<div class="meta"><span>{brand}</span><span>{colour}</span><span>{gender}</span><span>ID {int(item["id"])}</span></div>',
            unsafe_allow_html=True,
        )

        if reference_image and text_query.strip():
            st.markdown(
                f'<div class="metrics"><span class="metric">Visual <b>{item["visual_score"]:.3f}</b></span><span class="metric">Text <b>{item["text_score"]:.3f}</b></span></div>',
                unsafe_allow_html=True,
            )
        elif reference_image:
            st.markdown(
                f'<div class="metrics"><span class="metric">Image similarity <b>{item["visual_score"]:.3f}</b></span></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="metrics"><span class="metric">Semantic similarity <b>{item["text_score"]:.3f}</b></span></div>',
                unsafe_allow_html=True,
            )

        if reference_image and brand_hint and brand.lower() == brand_hint.lower():
            st.markdown('<div class="match-note">✓ Matches top brand hypothesis</div>', unsafe_allow_html=True)
        elif reference_image:
            st.markdown('<div class="match-note warn">Visual match · brand not treated as ground truth</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="small-note" style="margin-top:18px">Note: CLIP zero-shot brand/category predictions are retrieval hints, not verified labels. Final ranking is driven by image/text embedding similarity.</div>',
    unsafe_allow_html=True,
)
