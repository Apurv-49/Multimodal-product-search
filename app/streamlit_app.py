import os
import sys

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
:root { --bg:#070B12; --panel:#0E1420; --panel2:#111927; --line:#202B3B; --text:#F4F7FB; --muted:#91A0B5; --blue:#5B8CFF; --green:#45D483; }
.stApp { background:var(--bg); color:var(--text); }
[data-testid="stHeader"] { background:transparent; }
.block-container { max-width:1400px; padding:2.2rem 3rem 4rem; }
section[data-testid="stSidebar"] { background:#0A101A; border-right:1px solid var(--line); }
section[data-testid="stSidebar"] > div { padding:1.4rem 1.1rem; }
.hero { display:flex; justify-content:space-between; align-items:flex-start; gap:24px; margin-bottom:28px; }
.eyebrow { color:var(--blue); font-size:.72rem; font-weight:800; letter-spacing:.14em; text-transform:uppercase; margin-bottom:8px; }
.hero h1 { font-size:2.45rem; line-height:1.05; margin:0; letter-spacing:-.045em; }
.hero p { color:var(--muted); margin:.7rem 0 0; max-width:680px; font-size:1rem; }
.badge { background:#101B2D; border:1px solid #29446F; color:#83A8FF; border-radius:999px; padding:8px 13px; font-size:.74rem; font-weight:700; white-space:nowrap; }
.search-summary { background:linear-gradient(135deg,#0D1625,#0B111C); border:1px solid var(--line); border-radius:18px; padding:18px 20px; margin-bottom:22px; }
.search-summary .label { color:var(--muted); font-size:.72rem; text-transform:uppercase; letter-spacing:.1em; }
.search-summary .value { color:#fff; font-size:1.1rem; font-weight:700; margin-top:5px; }
.insight { background:#0D1622; border:1px solid var(--line); border-radius:14px; padding:13px 15px; margin-top:10px; }
.insight-title { font-size:.72rem; color:var(--muted); text-transform:uppercase; letter-spacing:.09em; }
.insight-value { font-size:.95rem; font-weight:700; margin-top:4px; }
.result-card { background:var(--panel); border:1px solid var(--line); border-radius:18px; padding:12px; min-height:520px; box-shadow:0 8px 30px rgba(0,0,0,.16); }
.result-card:hover { border-color:#345A9E; }
.rank { display:flex; justify-content:space-between; align-items:center; margin:2px 2px 10px; }
.rank-number { color:#7D8BA0; font-size:.72rem; font-weight:800; }
.score { color:#8DB0FF; background:#101C31; border:1px solid #28477A; border-radius:999px; padding:5px 9px; font-size:.72rem; font-weight:800; }
.product-name { font-size:1rem; font-weight:800; color:#fff; margin-top:12px; line-height:1.25; }
.product-sub { color:#7F8DA3; font-size:.78rem; margin-top:4px; }
.meta { display:flex; flex-wrap:wrap; gap:6px; margin-top:11px; }
.meta span { background:#111B2A; border:1px solid #202D40; color:#AAB6C7; border-radius:7px; padding:4px 7px; font-size:.69rem; }
.confidence { margin-top:15px; padding-top:13px; border-top:1px solid var(--line); color:#77869B; font-size:.72rem; }
.empty { border:1px dashed #2A3749; background:#0D141F; border-radius:18px; padding:55px 30px; text-align:center; color:#738196; }
.empty strong { display:block; color:#E8EDF5; font-size:1.05rem; margin-bottom:7px; }
.sidebar-title { color:#AAB7C9; font-size:.72rem; font-weight:800; text-transform:uppercase; letter-spacing:.12em; margin:18px 0 7px; }
.sidebar-brand { font-size:1.2rem; font-weight:900; letter-spacing:-.03em; margin-bottom:2rem; }
.sidebar-brand span { color:var(--blue); }
.small-note { color:#6F7D91; font-size:.72rem; line-height:1.5; margin-top:9px; }
footer { visibility:hidden; }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading OpenCLIP model…")
def load_embedder():
    return CLIPEmbedder()


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_catalog(queries: tuple[str, ...]):
    return search_catalog(queries, per_query=70)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_candidate_images(urls: tuple[str, ...]):
    return download_images(list(urls))


embedder = load_embedder()

CATEGORY_LABELS = [
    "shoes", "sports shoes", "sandals", "shirts", "t-shirts", "jeans",
    "trousers", "jackets", "dresses", "bags", "watches", "sunglasses",
    "accessories", "skirts", "tops", "hoodies"
]
BRAND_LABELS = [
    "Nike", "Adidas", "Puma", "Reebok", "New Balance", "Skechers",
    "Asics", "Converse", "Vans", "Fila", "Under Armour", "Crocs",
    "Levis", "Bata", "Woodland"
]

with st.sidebar:
    st.markdown('<div class="sidebar-brand">Product<span>Lens</span></div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-title">Text search</div>', unsafe_allow_html=True)
    text_query = st.text_input(
        "Text query",
        placeholder="black running shoes for men",
        label_visibility="collapsed",
    )

    st.markdown('<div class="sidebar-title">Visual search</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload a product image",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )

    if uploaded_file:
        st.image(uploaded_file, caption="Reference image", width="stretch")

    st.markdown('<div class="sidebar-title">Ranking</div>', unsafe_allow_html=True)
    alpha = st.slider(
        "Visual weight",
        min_value=0.0,
        max_value=1.0,
        value=0.75,
        step=0.05,
        help="Higher values give more weight to image similarity when both text and image are provided.",
    )
    top_k = st.selectbox("Results", [4, 6, 8], index=0)
    run_search = st.button("Search products", type="primary", use_container_width=True)

    st.markdown(
        '<div class="small-note">Powered by OpenCLIP + visual reranking + Hugging Face product catalog.</div>',
        unsafe_allow_html=True,
    )

st.markdown(
    """
<div class="hero">
  <div>
    <div class="eyebrow">Multimodal commerce intelligence</div>
    <h1>Find products by meaning, not keywords.</h1>
    <p>Combine natural-language intent with a reference image to retrieve visually and semantically similar fashion products.</p>
  </div>
  <div class="badge">OPENCLIP · 512-D EMBEDDINGS</div>
</div>
""",
    unsafe_allow_html=True,
)

if not run_search:
    st.markdown(
        '<div class="empty"><strong>Start a product search</strong>Enter a query, upload an image, or use both. Your results will appear here with live product imagery and similarity scores.</div>',
        unsafe_allow_html=True,
    )
    st.stop()

if not text_query.strip() and not uploaded_file:
    st.warning("Add a text query or upload a reference image first.")
    st.stop()

reference_image = None
visual_hint = None
visual_hint_conf = 0.0
brand_hint = None
brand_hint_conf = 0.0

if uploaded_file:
    reference_image = Image.open(uploaded_file).convert("RGB")
    visual_hint, visual_hint_conf, _ = embedder.classify_image(
        reference_image,
        CATEGORY_LABELS,
        "a studio product photo of {label}",
    )
    brand_hint, brand_hint_conf, _ = embedder.classify_image(
        reference_image,
        BRAND_LABELS,
        "a product photo from the brand {label}",
    )

queries: list[str] = []
if text_query.strip():
    queries.append(text_query.strip())
if visual_hint:
    if brand_hint and brand_hint_conf >= 0.40:
        queries.append(f"{brand_hint} {visual_hint}")
    queries.append(visual_hint)
if not queries:
    queries = ["fashion products"]

try:
    catalog = fetch_catalog(tuple(queries))
except Exception as exc:
    st.error(f"Catalog service could not be reached: {exc}")
    st.stop()

if catalog.empty:
    st.warning("No products were found for this search. Try a broader query such as 'shoes' or 'black jacket'.")
    st.stop()

# Download candidate images and compute real visual similarity. The old app used
# random fallback vectors when its local FAISS files were absent; this path does not.
image_map = fetch_candidate_images(tuple(catalog["image_url"].tolist()))
valid = catalog[catalog["image_url"].isin(image_map.keys())].copy()

if valid.empty:
    st.error("Product metadata was found, but product images could not be loaded.")
    st.stop()

images = [image_map[url] for url in valid["image_url"]]
image_matrix = embedder.get_image_embeddings_from_pil(images)

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
    text_matrix = embedder.get_text_embeddings(product_text)
    query_text_vector = embedder.get_text_embedding(text_query.strip())
    text_scores = text_matrix @ query_text_vector

if reference_image and text_query.strip():
    combined = alpha * visual_scores + (1.0 - alpha) * text_scores
elif reference_image:
    combined = visual_scores.copy()
else:
    combined = text_scores.copy()

# A high-confidence zero-shot brand hint is used as a small reranking signal,
# never as the sole prediction. This prevents an image match from being silently
# relabeled as another brand when the catalog contains many visually similar items.
if reference_image and brand_hint and brand_hint_conf >= 0.40:
    brand_match = valid["brand"].fillna("").str.lower().eq(brand_hint.lower()).to_numpy()
    combined = combined + np.where(brand_match, 0.08, 0.0).astype("float32")

valid["visual_score"] = visual_scores
valid["text_score"] = text_scores
valid["score"] = combined
valid = valid.sort_values("score", ascending=False).head(top_k).reset_index(drop=True)

st.markdown(
    f'<div class="search-summary"><div class="label">Search completed</div><div class="value">{len(valid)} ranked products</div></div>',
    unsafe_allow_html=True,
)

if reference_image:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f'<div class="insight"><div class="insight-title">Visual category</div><div class="insight-value">{visual_hint.title()} · {visual_hint_conf:.0%}</div></div>', unsafe_allow_html=True)
    with c2:
        brand_display = brand_hint if brand_hint_conf >= 0.40 else "Low confidence"
        st.markdown(f'<div class="insight"><div class="insight-title">Brand hint</div><div class="insight-value">{brand_display} · {brand_hint_conf:.0%}</div></div>', unsafe_allow_html=True)
    with c3:
        mode = "Image + text" if text_query.strip() else "Image only"
        st.markdown(f'<div class="insight"><div class="insight-title">Retrieval mode</div><div class="insight-value">{mode}</div></div>', unsafe_allow_html=True)

st.markdown("### Results")
cols = st.columns(min(4, len(valid)))

for index, (_, item) in enumerate(valid.iterrows()):
    with cols[index % len(cols)]:
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown(
            f'<div class="rank"><span class="rank-number">#{index + 1:02d}</span><span class="score">{item["score"]:.3f} match</span></div>',
            unsafe_allow_html=True,
        )
        st.image(item["image_url"], width="stretch")
        name = str(item.get("productDisplayName", "Product"))
        category = str(item.get("articleType", "Fashion"))
        st.markdown(f'<div class="product-name">{name}</div><div class="product-sub">{category}</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="meta"><span>{item.get("brand", "Unknown")}</span><span>{item.get("baseColour", "N/A")}</span><span>{item.get("gender", "Unisex")}</span></div>',
            unsafe_allow_html=True,
        )
        if reference_image:
            st.markdown(f'<div class="confidence">Visual similarity · {item["visual_score"]:.3f}</div>', unsafe_allow_html=True)
        elif text_query.strip():
            st.markdown(f'<div class="confidence">Semantic similarity · {item["text_score"]:.3f}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
