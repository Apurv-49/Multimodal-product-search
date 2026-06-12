import faiss
import numpy as np
import pickle
import os

class ProductRetriever:
    def __init__(self, dimension: int = 512):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension) # Inner Product for Normalized Cosine Similarity
        self.product_ids = []

    def add_products(self, embeddings: np.ndarray, ids: list):
        if len(embeddings) != len(ids):
            raise ValueError("Size mismatch between embeddings and IDs")
        # Ensure vectors are explicitly float32 for FAISS natively
        embeddings = embeddings.astype('float32')
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.product_ids.extend(ids)

    def search(self, query_vector: np.ndarray, k: int = 10):
        query_vector = query_vector.reshape(1, -1).astype('float32')
        faiss.normalize_L2(query_vector)
        distances, indices = self.index.search(query_vector, k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1:
                results.append((self.product_ids[idx], float(dist)))
        return results

    def hybrid_search(self, image_vector: np.ndarray = None, text_vector: np.ndarray = None, k: int = 10, alpha: float = 0.7):
        if image_vector is None and text_vector is None:
            raise ValueError("Both search vectors cannot be None")
            
        if image_vector is None:
            return self.search(text_vector, k)
        if text_vector is None:
            return self.search(image_vector, k)
            
        # Standardize arrays
        image_vector = image_vector.flatten()
        text_vector = text_vector.flatten()
        
        # Pull candidate items pool to filter down and combine scores
        candidate_k = min(200, self.index.ntotal)
        if candidate_k == 0: return []
        
        img_res = {pid: score for pid, score in self.search(image_vector, candidate_k)}
        txt_res = {pid: score for pid, score in self.search(text_vector, candidate_k)}
        
        combined_scores = {}
        all_pids = set(img_res.keys()).union(set(txt_res.keys()))
        
        for pid in all_pids:
            img_score = img_res.get(pid, 0.0)
            txt_score = txt_res.get(pid, 0.0)
            combined_scores[pid] = (alpha * img_score) + ((1.0 - alpha) * txt_score)
            
        sorted_res = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        return sorted_res

    def save_index(self, index_path: str, ids_path: str):
        faiss.write_index(self.index, index_path)
        with open(ids_path, 'wb') as f:
            pickle.dump(self.product_ids, f)

    def load_index(self, index_path: str, ids_path: str):
        if os.path.exists(index_path) and os.path.exists(ids_path):
            self.index = faiss.read_index(index_path)
            with open(ids_path, 'rb') as f:
                self.product_ids = pickle.load(f)
