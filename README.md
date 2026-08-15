# ProductLens — Multimodal E-Commerce Product Search

ProductLens is a multimodal fashion-search application that combines **natural-language queries and reference images** to retrieve visually and semantically similar products from a real fashion catalog.

**Live demo:** https://multimodal-appuct-search-f4rko3skzqndappcseyhan.streamlit.app/

## What the system does

- **Image search:** accepts a product photo and retrieves visually similar catalog items.
- **Text search:** understands natural-language product intent such as `blue running shoes for men`.
- **Hybrid search:** combines image and text similarity with a configurable visual-weight parameter.
- **Candidate expansion:** uses multiple CLIP category and brand hypotheses instead of trusting one zero-shot prediction.
- **Brand-aware reranking:** applies brand metadata only as a small ranking signal; it never replaces visual similarity.
- **Live product imagery:** result cards render real catalog images from the remote product dataset.
- **Transparent ranking:** the UI exposes visual and semantic similarity separately.
- **Responsive interface:** redesigned Streamlit search workspace with product cards, metadata, confidence signals, and retrieval diagnostics.

## Architecture

```text
User Query / Reference Image
            |
            v
      OpenCLIP ViT-B/32
       /             \
  image embedding   text embedding
       |                  |
       v                  v
   Zero-shot hints     Query vector
       |                  |
       +--------+---------+
                v
       Candidate Expansion
     category + common brands
                |
                v
        Remote Fashion Catalog
                |
                v
       Candidate Image Encoding
                |
                v
         Similarity Reranking
       image + text + small
          metadata bonus
                |
                v
            Top-K Results
                |
                v
         Streamlit Product UI
```

## Retrieval pipeline

1. A reference image is encoded into a normalized CLIP embedding.
2. The image is passed through zero-shot category and brand prompts to generate **multiple hypotheses**.
3. Those hypotheses expand catalog queries, including common brand/category combinations, reducing the chance that one incorrect zero-shot label removes the correct brand from the candidate pool.
4. Candidate product metadata and real product image URLs are retrieved from the Fashion Product Images dataset.
5. Candidate images are encoded with the same OpenCLIP model.
6. Image-to-image cosine similarity is computed directly between the uploaded image and candidate product images.
7. If text is supplied, text-to-product semantic similarity is combined with visual similarity using `alpha`.
8. A small brand-aware metadata bonus is applied only when the catalog brand agrees with a high-ranked CLIP brand hypothesis.
9. The final top-K products are rendered with their actual catalog images and independent visual/text scores.

### Why the previous deployment failed

The original prototype expected `data/`, `images/`, and `faiss_index/` to exist locally. Those artifacts were not committed to GitHub. When the deployment could not find the FAISS index, the old application generated **random fallback vectors**. That meant an uploaded Nike image could return an unrelated Adidas product.

The deployed path no longer uses random vectors. It retrieves real catalog candidates, downloads their real images, encodes those images with OpenCLIP, and ranks them against the uploaded image.

## Tech Stack

- Python
- PyTorch
- OpenCLIP ViT-B/32
- FAISS utilities for the local ingestion/retrieval prototype
- Pandas / NumPy
- Pillow
- Streamlit
- Hugging Face Dataset Viewer API

## Dataset

ProductLens uses the public `mecha2019/fashion-product-images-small` dataset. The current dataset viewer reports **42,426 training rows** with product IDs, article types, colours, gender, product names, and product image URLs. citeturn0search0

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

The first run downloads the OpenCLIP weights. Streamlit caches the model so it is reused across interactions.

## Evaluation status

The repository intentionally does **not** claim Recall@K, mAP, latency, or brand-accuracy numbers without a reproducible benchmark. The next evaluation step should use a fixed held-out query set and report:

- Recall@K
- Precision@K
- Brand top-K hit rate
- Category top-K hit rate
- Median and p95 end-to-end query latency

This keeps the project resume-ready without presenting unsupported performance numbers.

## Resume-ready description

**ProductLens — Multimodal E-Commerce Product Search**  
Built an OpenCLIP-based multimodal retrieval system that maps product images and natural-language queries into a shared embedding space; implemented multi-hypothesis candidate expansion, visual/semantic reranking, brand-aware retrieval signals, remote catalog image retrieval, and a deployed Streamlit interface for interactive product discovery.
