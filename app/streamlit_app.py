import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
from PIL import Image

# Setup sys path alignment for module resolution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.embeddings import CLIPEmbedder
from src.retrieval import ProductRetriever
from src.recommendation import RecommendationEngine
from src.llm_assistant import ProductAssistant

# Page Layout Configuration for a modern, dark SaaS UI
st.set_page_config(page_title="ProductLens Platform", layout="wide", initial_sidebar_state="expanded")

# Custom enterprise dark theme CSS injection matching your premium layout
st.markdown("""
    <style>
        /* Base page canvas configurations */
        .main { background-color: #0B0F19; color: #E2E8F0; font-family: -apple-system, BlinkMacSystemFont, sans-serif; }
        header[data-testid="stHeader"] { background-color: rgba(11, 15, 25, 0.0) !important; }
        
        /* Modernized Header Navigation layout */
        .brand-container { display: flex; justify-content: space-between; align-items: center; padding-bottom: 1rem; border-bottom: 1px solid #1E293B; margin-bottom: 2rem; }
        .brand-title { font-size: 1.8rem; font-weight: 700; color: #FFFFFF; letter-spacing: -0.02em; }
        .brand-accent { color: #3B82F6; }
        .badge-pill { background-color: #1E293B; color: #3B82F6; font-size: 0.75rem; font-weight: 600; padding: 4px 12px; border-radius: 12px; text-transform: uppercase; letter-spacing: 0.05em; border: 1px solid #334155; }
        
        /* Clean Sidebar structural overhauls */
        section[data-testid="stSidebar"] { background-color: #0F172A !important; border-right: 1px solid #1E293B; }
        section[data-testid="stSidebar"] .stMarkdown h3 { color: #94A3B8; font-size: 0.85rem !important; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem !important; }
        
        /* Custom styled search container elements */
        .results-header { font-size: 1.1rem; color: #94A3B8; font-weight: 500; margin-bottom: 1.5rem; }
        .results-count { color: #FFFFFF; font-weight: 700; font-size: 1.3rem; }
        
        /* Premium deep-slate micro card layout matrix */
        .product-card {
            background-color: #111827;
            border: 1px solid #1F2937;
            border-radius: 12px;
            padding: 16px;
            transition: border-color 0.2s ease, box-shadow 0.2s ease;
            margin-bottom: 1rem;
        }
        .product-card:hover { border-color: #3B82F6; box-shadow: 0 0 15px rgba(59, 130, 246, 0.1); }
        
        /* Cosine similarity tracking chips */
        .match-badge {
            background-color: #1E293B;
            color: #60A5FA;
            font-size: 0.8rem;
            font-weight: 600;
            padding: 6px 12px;
            border-radius: 20px;
            display: inline-block;
            margin-bottom: 14px;
            border: 1px solid #2563EB;
        }
        
        /* Clean typography grids inside result blocks */
        .product-title { font-weight: 600; color: #FFFFFF; font-size: 1rem; margin-bottom: 8px; line-height: 1.4; }
        .meta-row { display: flex; margin-bottom: 4px; font-size: 0.85rem; }
        .meta-label { color: #64748B; width: 70px; flex-shrink: 0; }
        .meta-value { color: #94A3B8; }
        
        /* Graph recommendation sub-elements */
        .rec-section { margin-top: 14px; padding-top: 10px; border-top: 1px solid #1F2937; }
        .rec-label { font-size: 0.75rem; color: #4B5563; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px; }
        .rec-chip { display: inline-block; background: #1F2937; color: #3B82F6; font-size: 0.75rem; padding: 2px 8px; border-radius: 4px; margin-right: 4px; border: 1px solid #334155; }
        
        /* AI inference text panel */
        .assistant-panel { background-color: #0F172A; border: 1px solid #1E293B; border-radius: 8px; padding: 20px; margin-top: 2rem; }
        .assistant-heading { font-size: 0.8rem; font-weight: 600; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 10px; }
        .assistant-body { color: #E2E8F0; font-size: 0.95rem; line-height: 1.6; }
        
        /* Status blocks fallback overrides */
        .status-container { background: #111827; border: 1px dashed #1F2937; border-radius: 8px; padding: 40px; text-align: center; color: #64748B; font-size: 0.95rem; }
    </style>
""", unsafe_allow_html=True)

# Top Bar Header Setup
st.markdown("""
    <div class="brand-container">
        <div class="brand-title">Product<span class="brand-accent">Lens</span></div>
        <div class="badge-pill">Multimodal Search Engine</div>
    </div>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_pipeline():
    embedder = CLIPEmbedder()
    retriever = ProductRetriever()
    
    if os.path.exists("faiss_index/vector.index"):
        retriever.load_index("faiss_index/vector.index", "faiss_index/ids.pkl")
    else:
        mock_embs = np.random.randn(10, 512).astype('float32')
        mock_ids = [15970, 17891, 12043, 19320, 1005, 1006, 1007, 1008, 1009, 1010]
        retriever.add_products(mock_embs, mock_ids)
        
    if os.path.exists("data/cleaned_products.csv"):
        df = pd.read_csv("data/cleaned_products.csv")
    else:
        df = pd.DataFrame({
            'id': [15970, 17891, 12043, 19320, 1005, 1006, 1007, 1008, 1009, 1010],
            'brand': ['Nike', 'Adidas', 'Puma', 'Reebok', 'Under Armour', 'Adidas', 'Reebok', 'Asics', 'Nike', 'Puma'],
            'category': ['Running Shoes', 'Sports Shoes', 'Running Shoes', 'Casual Shoes', 'Jackets', 'Pants', 'Shoes', 'Shoes', 'Shirts', 'Bags'],
            'color': ['Black', 'Black', 'Grey', 'Black', 'Blue', 'Black', 'Grey', 'Blue', 'White', 'Black'],
            'gender': ['Men', 'Men', 'Men', 'Men', 'Men', 'Women', 'Women', 'Men', 'Women', 'Unisex']
        })
        
    rec_engine = RecommendationEngine(retriever, df)
    assistant = ProductAssistant()
    return embedder, retriever, rec_engine, df, assistant

embedder, retriever, rec_engine, df, assistant = initialize_pipeline()

# Sidebar Query Interface Panel
with st.sidebar:
    st.markdown("<h3>Search Query</h3>", unsafe_allow_html=True)
    text_query = st.text_input("Text parameters:", placeholder="e.g., black running shoes for men", label_visibility="collapsed")
    
    st.markdown("<h3>Reference Image</h3>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Image matching canvas:", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
    
    st.markdown("<h3>Visual vs Text Weight</h3>", unsafe_allow_html=True)
    alpha = st.slider("Alpha Value Balance", 0.0, 1.0, 0.7, label_visibility="collapsed")
    
    st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)
    search_triggered = st.button("Run Search", use_container_width=True)

# Main Output Context Engine logic
if search_triggered:
    img_vector, txt_vector = None, None
    
    if uploaded_file:
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        img_vector = embedder.get_image_embedding(temp_path)
        os.remove(temp_path)
        
        with st.sidebar:
            st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
            st.markdown("<h3>Active Visual Reference</h3>", unsafe_allow_html=True)
            st.image(uploaded_file, use_container_width=True)
            
    if text_query:
        txt_vector = embedder.get_text_embedding(text_query)
        
    if img_vector is not None or txt_vector is not None:
        results = retriever.hybrid_search(image_vector=img_vector, text_vector=txt_vector, k=4, alpha=alpha)
        
        # Dashboard parameters info sub-row matching your preview reference layout
        st.markdown(f"""
            <div class="results-header">
                <span class="results-count">{len(results)} results</span> &nbsp;&middot;&nbsp; 
                HYBRID (IMAGE + TEXT) &nbsp;&middot;&nbsp; A = {alpha:.2f}
            </div>
        """, unsafe_allow_html=True)
        
        cols = st.columns(len(results))
        retrieved_list = []
        
        for i, (pid, score) in enumerate(results):
            match = df[df['id'] == pid]
            if not match.empty:
                item = match.iloc[0].to_dict()
                retrieved_list.append(item)
                
                with cols[i]:
                    # Open engineered CSS dark micro-container frame
                    st.markdown(f"""
                        <div class="product-card">
                            <div class="match-badge">{score:.3f} match</div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Nest matching graphics context safely within the layout grids
                    item_img_path = f"images/{int(pid)}.jpg"
                    if os.path.exists(item_img_path):
                        st.image(item_img_path, use_container_width=True)
                    else:
                        st.markdown("<div style='height:160px; background:#1F2937; display:flex; align-items:center; justify-content:center; color:#4B5563; font-size:0.85rem; border-radius:6px; margin-bottom:14px; border:1px dashed #374151;'>Image Offline</div>", unsafe_allow_html=True)
                    
                    display_brand = item.get('brand', item.get('brandName', 'Generic'))
                    display_type = item.get('category', item.get('articleType', 'Item'))
                    display_color = item.get('color', item.get('baseColour', 'N/A'))
                    display_gender = item.get('gender', 'Unisex')
                    
                    st.markdown(f"""
                        <div class="product-title">{display_brand} &mdash; {display_type}</div>
                        <div class="meta-row"><span class="meta-label">Colour</span><span class="meta-value">{display_color}</span></div>
                        <div class="meta-row"><span class="meta-label">Gender</span><span class="meta-value">{display_gender}</span></div>
                        <div class="meta-row"><span class="meta-label">ID</span><span class="meta-value">{item.get('id', pid)}</span></div>
                    """, unsafe_allow_html=True)
                    
                    # Fetching item metadata recommendations mapping
                    recs = rec_engine.recommend_alternatives(pid, k=2)
                    if recs:
                        chips_html = "".join([f'<span class="rec-chip">#{r}</span>' for r in recs])
                        st.markdown(f"""
                            <div class="rec-section">
                                <div class="rec-label">You May Also Like</div>
                                {chips_html}
                            </div>
                        """, unsafe_allow_html=True)
        
        # Generation Language Evaluation Window block
        if text_query:
            st.markdown("""
                <div class="assistant-panel">
                    <div class="assistant-heading">Product Assistant</div>
                </div>
            """, unsafe_allow_html=True)
            with st.spinner("Synthesizing context vector matching properties..."):
                llm_output = assistant.generate_response(text_query, retrieved_list)
                st.markdown(f'<div class="assistant-body">{llm_output}</div>', unsafe_allow_html=True)
    else:
        st.markdown("<div class='status-container'>Execution stopped: Text descriptors or target images are required.</div>", unsafe_allow_html=True)
else:
    st.markdown("<div class='status-container'>Configure context query configurations on the left sidebar engine layout panel to map spatial matrix points.</div>", unsafe_allow_html=True)