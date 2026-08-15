# ProductLens — Multimodal E-Commerce Product Search

ProductLens is a multimodal fashion-search application that combines **natural-language queries and reference images** to retrieve visually and semantically similar products from a retail catalog.

**Live demo:** https://multimodal-appuct-search-f4rko3skzqndappcseyhan.streamlit.app/

## What the system does

- **Text search:** understands queries such as `black running shoes for men`.
- **Image search:** accepts a product photo and retrieves visually similar catalog items.
- **Hybrid search:** combines image and text similarity with a configurable visual-weight parameter.
- **Zero-shot hints:** OpenCLIP provides category and brand hints that improve candidate retrieval and reranking.
- **Live product imagery:** result cards render real catalog images instead of local placeholder assets.
- **Responsive Streamlit UI:** focused search workspace, ranking controls, confidence signals, and product metadata.

## Architecture

```text
User Query / Reference Image
            |
            v
      OpenCLIP ViT-B/32
       /             \
  image 512-D      text 512-D
       \             /
        \           /
         v         v
       Candidate Retrieval
       Hugging Face Dataset
               |
               v
      Visual / Semantic Reranking
               |
               v
       Top-K Product Results
               |
               v
       Streamlit Product UI
```

### Retrieval pipeline

1. A reference image is encoded into a normalized 512-dimensional CLIP embedding.
2. A text query is encoded into the same shared embedding space.
3. The application uses the Hugging Face Dataset Viewer `/search` API to obtain relevant catalog candidates from the Fashion Product Images dataset.
4. Candidate product images are encoded with the same OpenCLIP model.
5. Image-to-image cosine similarity and text-to-product semantic similarity are combined for hybrid ranking.
6. A high-confidence zero-shot brand/category hint is used only as a small reranking signal, rather than replacing visual similarity.
7. The top-ranked products are rendered with live catalog imagery and metadata.

## Tech Stack

- Python
- PyTorch
- OpenCLIP ViT-B/32
- FAISS utilities for the local retrieval prototype
- Pandas / NumPy
- Pillow
- Streamlit
- Hugging Face Dataset Viewer API

## Dataset

The application uses the public `ashraq/fashion-product-images-small` dataset, containing roughly 44k fashion-product rows with structured metadata and product images.

## Project Structure

```text
Multimodal-product-search/
├── app/
│   └── streamlit_app.py
├── src/
│   ├── embeddings.py
│   ├── llm_assistant.py
│   ├── preprocessing.py
│   ├── recommendation.py
│   ├── remote_catalog.py
│   └── retrieval.py
├── requirements.txt
├── run_ingestion.py
└── README.md
```

## Running locally

```bash
git clone https://github.com/Apurv-49/Multimodal-product-search.git
cd Multimodal-product-search
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

The first run downloads the OpenCLIP weights. Streamlit caches the model resource so it is not recreated on every interaction.

## Resume-ready description

**ProductLens — Multimodal E-Commerce Product Search**  
Built an OpenCLIP-based multimodal retrieval system that maps product images and natural-language queries into a shared 512-D embedding space; implemented hybrid visual/semantic reranking, zero-shot category/brand hints, remote catalog retrieval, and a deployed Streamlit interface for interactive product discovery.

## Important engineering note

The original prototype depended on local `data/`, `images/`, and `faiss_index/` artifacts that were not committed to GitHub. When those files were unavailable on Streamlit Cloud, the old application generated random fallback vectors, which made its predictions unreliable. The current deployment removes that fallback path and retrieves real catalog candidates and images instead.
