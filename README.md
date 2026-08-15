# ProductLens — Multimodal E-Commerce Product Search

ProductLens is a multimodal product search application for fashion catalogs. It lets users search for products using a natural-language description, a reference image, or both.

The system uses OpenCLIP to compare product images and text in a shared embedding space and ranks the most relevant catalog items.

**Live Demo:**  
https://multimodal-appuct-search-f4rk8r3skzqndappcseyhan.streamlit.app/

## What it does

- **Image search** — upload a product image and find visually similar products.
- **Text search** — describe a product such as `blue running shoes for men`.
- **Hybrid search** — combine a reference image with a text description.
- **Visual ranking** — candidate product images are compared directly with the uploaded image using CLIP embeddings.
- **Semantic ranking** — product metadata is encoded as text and compared with the user's query.
- **Real catalog images** — search results use the actual product images stored in the dataset.
- **Product metadata** — results show product name, category, brand, colour, gender, and product ID.
- **Transparent scores** — visual and text similarity are shown separately for hybrid searches.

## How it works

```text
User
 ├── Product image
 └── Text description
          |
          v
     OpenCLIP ViT-B/32
      /            \
 Image embedding   Text embedding
      |                  |
      v                  v
Local catalog       Text query
candidate search    similarity
      |
      v
Candidate product images
      |
      v
OpenCLIP image embeddings
      |
      v
Image-to-image similarity
      |
      +---------+
                |
                v
        Final ranking
                |
                v
        Top-K products
                |
                v
        Streamlit UI

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
