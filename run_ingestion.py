import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from src.embeddings import CLIPEmbedder
from src.retrieval import ProductRetriever

def main():
    # 1. Read and fix Kaggle Metadata Columns
    print("Processing metadata file...")
    csv_path = "data/styles.csv"
    cleaned_path = "data/cleaned_products.csv"
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find {csv_path}. Please place your Kaggle styles.csv inside the data/ folder.")
        
    # Read CSV, skip broken formatting rows automatically
    df = pd.read_csv(csv_path, on_bad_lines='skip')
    
    # Standardize column naming convention to match our schema
    df = df.rename(columns={'filename': 'image_path', 'articleType': 'category', 'baseColour': 'color'})
    df = df.dropna(subset=['id', 'category', 'gender'])
    df['id'] = df['id'].astype(int)
    
    # Save the cleaned metadata frame
    df.to_csv(cleaned_path, index=False)
    print(f"Metadata standardized. Found {len(df)} total entries in raw catalog.")

    # 2. Initialize Core Models
    print("Initializing OpenCLIP Model...")
    embedder = CLIPEmbedder()
    retriever = ProductRetriever(dimension=512)

    # 3. Batch Image Vector Extraction (Optimized Prototype Mode: Top 2,000 items)
    print("Extracting high-dimensional embeddings for a 2,000-product prototype...")
    embeddings = []
    valid_ids = []
    
    # Using df.head(2000) for a rapid, flawless showcase build
    for idx, row in tqdm(df.head(2000).iterrows(), total=min(2000, len(df))):
        img_path = f"images/{int(row['id'])}.jpg"
        
        if os.path.exists(img_path):
            try:
                # Convert image to 512-D vector
                emb = embedder.get_image_embedding(img_path)
                embeddings.append(emb)
                valid_ids.append(int(row['id']))
            except Exception:
                continue # Skip if file is unreadable or corrupted
                
    # 4. Compile and Write out Vector Database
    if embeddings:
        print(f"Writing {len(embeddings)} vectors into FAISS DB...")
        embeddings_arr = np.array(embeddings).astype('float32')
        retriever.add_products(embeddings_arr, valid_ids)
        
        os.makedirs("faiss_index", exist_ok=True)
        retriever.save_index("faiss_index/vector.index", "faiss_index/ids.pkl")
        print("\n Successfully generated prototype index! Your database is ready.")
        print(" You can now restart or refresh your Streamlit app to test real multimodal search!")
    else:
        print("\n Error: No matching images were found inside your 'images/' directory.")
        print(" Double-check that your image files (e.g., 1525.jpg) are placed directly inside the 'images/' folder at the root of your project.")

if __name__ == "__main__":
    main()