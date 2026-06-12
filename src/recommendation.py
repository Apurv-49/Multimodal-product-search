import numpy as np

class RecommendationEngine:
    def __init__(self, retriever_instance, metadata_df):
        self.retriever = retriever_instance
        self.df = metadata_df.set_index('id')

    def recommend_alternatives(self, product_id: int, k: int = 5) -> list:
        """
        Finds products within the index that match the items brand context or category vectors
        """
        if product_id not in self.retriever.product_ids:
            return []
            
        idx_pos = self.retriever.product_ids.index(product_id)
        # Reconstruct the vector directly from FAISS database
        target_vector = self.retriever.index.reconstruct(idx_pos)
        
        raw_hits = self.retriever.search(target_vector, k = k + 1)
        
        # Filter out the current target item itself from recommendations
        recommendations = [pid for pid, score in raw_hits if pid != product_id][:k]
        return recommendations
