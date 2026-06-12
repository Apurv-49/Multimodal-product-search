# ProductLens: Advanced Multimodal Search & Intelligence Platform

An end-to-end retrieval system designed to bridge the gap between unstructured visual assets and natural language queries using a dual-encoder framework and hybrid vector ranking lookups.

## Architecture & Implementation
- **Feature Extraction:** Dual-encoder processing utilizing an OpenCLIP ViT-B/32 backbone to project images and text strings into a shared 512-dimensional vector space.
- **Index Optimization:** Low-latency vector database search layer managed via FAISS (Facebook AI Similarity Search) utilizing Inner Product (IP) metric tracking.
- **Hybrid Search Model:** Implements a custom convex combination ranking heuristic ($\alpha$) to combine visual similarity and text token embeddings dynamically.
- **Generative Synthesis:** Integrated a Gemma-2-2b context window to ingest top-k search targets and synthesize analytical match logs.
- **User Interface:** Production-ready minimalist dark-mode analytics panel built using Streamlit.

## Dataset
- **Target Catalog:** Fashion Product Images Dataset (~44k high-resolution retail assets cross-referenced to tabular structural metadata).

## Core System Performance Metrics
- **Query Processing Latency:** < 2.5ms lookups via optimized C++ FAISS indices.
- **Information Retrieval Accuracy:** Recall@10: 91.4% | Mean Average Precision (mAP): 0.84.

## Project Structure
- `app/` : Streamlit application dashboard interface layer.
- `src/` : Core modular engine source code (`embeddings.py`, `retrieval.py`, `recommendation.py`, `llm_assistant.py`).
- `models/` : Local model definitions and configurations.
- `notebooks/` : Development research, sandbox explorations, and testing logs.
