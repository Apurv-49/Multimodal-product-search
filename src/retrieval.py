from __future__ import annotations

import os
import pickle
from typing import Optional

import faiss
import numpy as np


class ProductRetriever:
    """
    FAISS-based product retriever.

    Uses normalized CLIP embeddings with an Inner Product
    index. For normalized vectors:

        inner_product == cosine_similarity

    This class is kept for the local retrieval prototype.

    The deployed remote-catalog Streamlit pipeline performs
    its final ranking directly with CLIP embeddings.
    """

    def __init__(
        self,
        dimension: int = 512,
    ):
        self.dimension = int(dimension)

        self.index = faiss.IndexFlatIP(
            self.dimension
        )

        self.product_ids: list = []


    # ========================================================
    # VALIDATION
    # ========================================================

    def _validate_embeddings(
        self,
        embeddings: np.ndarray,
    ) -> np.ndarray:

        embeddings = np.asarray(
            embeddings,
            dtype="float32",
        )


        if embeddings.ndim == 1:

            embeddings = embeddings.reshape(
                1,
                -1,
            )


        if embeddings.ndim != 2:

            raise ValueError(
                "Embeddings must have shape "
                "(n_products, dimension)."
            )


        if embeddings.shape[1] != self.dimension:

            raise ValueError(
                f"Embedding dimension mismatch: "
                f"expected {self.dimension}, "
                f"got {embeddings.shape[1]}."
            )


        if not np.isfinite(
            embeddings
        ).all():

            raise ValueError(
                "Embeddings contain NaN or infinite values."
            )


        # Copy before normalization so the caller's
        # original array is not modified.
        return embeddings.copy()


    def _validate_query(
        self,
        query_vector: np.ndarray,
    ) -> np.ndarray:

        query_vector = np.asarray(
            query_vector,
            dtype="float32",
        ).reshape(1, -1)


        if query_vector.shape[1] != self.dimension:

            raise ValueError(
                f"Query dimension mismatch: "
                f"expected {self.dimension}, "
                f"got {query_vector.shape[1]}."
            )


        if not np.isfinite(
            query_vector
        ).all():

            raise ValueError(
                "Query vector contains "
                "NaN or infinite values."
            )


        return query_vector.copy()


    # ========================================================
    # ADD PRODUCTS
    # ========================================================

    def add_products(
        self,
        embeddings: np.ndarray,
        ids: list,
    ) -> None:

        if len(embeddings) != len(ids):

            raise ValueError(
                "Size mismatch between embeddings and IDs."
            )


        if len(ids) == 0:

            return


        embeddings = (
            self._validate_embeddings(
                embeddings
            )
        )


        # L2 normalization makes Inner Product equivalent
        # to cosine similarity.
        faiss.normalize_L2(
            embeddings
        )


        self.index.add(
            embeddings
        )


        self.product_ids.extend(
            ids
        )


    # ========================================================
    # BASIC SEARCH
    # ========================================================

    def search(
        self,
        query_vector: np.ndarray,
        k: int = 10,
    ) -> list[tuple]:

        if self.index.ntotal == 0:

            return []


        query_vector = (
            self._validate_query(
                query_vector
            )
        )


        faiss.normalize_L2(
            query_vector
        )


        k = max(
            1,
            min(
                int(k),
                self.index.ntotal,
            ),
        )


        distances, indices = (
            self.index.search(
                query_vector,
                k,
            )
        )


        results: list[tuple] = []


        for distance, index in zip(
            distances[0],
            indices[0],
        ):

            if index < 0:

                continue


            product_id = (
                self.product_ids[
                    int(index)
                ]
            )


            results.append(
                (
                    product_id,
                    float(distance),
                )
            )


        return results


    # ========================================================
    # HYBRID SEARCH
    # ========================================================

    def hybrid_search(
        self,
        image_vector: Optional[
            np.ndarray
        ] = None,
        text_vector: Optional[
            np.ndarray
        ] = None,
        k: int = 10,
        alpha: float = 0.7,
    ) -> list[tuple]:

        """
        Hybrid image + text retrieval.

        alpha:
            image contribution

        1 - alpha:
            text contribution

        Examples:

            alpha = 1.0
                image only

            alpha = 0.0
                text only

            alpha = 0.7
                70% image
                30% text
        """

        if (
            image_vector is None
            and text_vector is None
        ):

            raise ValueError(
                "At least one search vector "
                "must be provided."
            )


        alpha = float(
            np.clip(
                alpha,
                0.0,
                1.0,
            )
        )


        # ----------------------------------------------------
        # Single modality
        # ----------------------------------------------------

        if image_vector is None:

            return self.search(
                text_vector,
                k,
            )


        if text_vector is None:

            return self.search(
                image_vector,
                k,
            )


        # ----------------------------------------------------
        # Empty index
        # ----------------------------------------------------

        if self.index.ntotal == 0:

            return []


        # ----------------------------------------------------
        # Normalize query vectors
        # ----------------------------------------------------

        image_vector = (
            self._validate_query(
                image_vector
            )
        )


        text_vector = (
            self._validate_query(
                text_vector
            )
        )


        faiss.normalize_L2(
            image_vector
        )


        faiss.normalize_L2(
            text_vector
        )


        # ----------------------------------------------------
        # Candidate pool
        # ----------------------------------------------------

        candidate_k = min(
            max(
                int(k) * 10,
                50,
            ),
            self.index.ntotal,
        )


        image_distances, image_indices = (
            self.index.search(
                image_vector,
                candidate_k,
            )
        )


        text_distances, text_indices = (
            self.index.search(
                text_vector,
                candidate_k,
            )
        )


        # ----------------------------------------------------
        # Convert FAISS results into dictionaries
        # ----------------------------------------------------

        image_scores = {}


        for score, index in zip(
            image_distances[0],
            image_indices[0],
        ):

            if index < 0:

                continue


            product_id = (
                self.product_ids[
                    int(index)
                ]
            )


            image_scores[
                product_id
            ] = float(score)


        text_scores = {}


        for score, index in zip(
            text_distances[0],
            text_indices[0],
        ):

            if index < 0:

                continue


            product_id = (
                self.product_ids[
                    int(index)
                ]
            )


            text_scores[
                product_id
            ] = float(score)


        # ----------------------------------------------------
        # Candidate union
        # ----------------------------------------------------

        candidate_ids = (
            set(image_scores)
            | set(text_scores)
        )


        # ----------------------------------------------------
        # Hybrid scoring
        # ----------------------------------------------------

        combined_scores = {}


        for product_id in candidate_ids:

            image_score = (
                image_scores.get(
                    product_id,
                    0.0,
                )
            )


            text_score = (
                text_scores.get(
                    product_id,
                    0.0,
                )
            )


            combined_score = (
                alpha * image_score
                + (1.0 - alpha)
                * text_score
            )


            combined_scores[
                product_id
            ] = float(
                combined_score
            )


        # ----------------------------------------------------
        # Sort
        # ----------------------------------------------------

        ranked = sorted(
            combined_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )


        return ranked[
            :max(1, int(k))
        ]


    # ========================================================
    # INDEX INFORMATION
    # ========================================================

    @property
    def size(self) -> int:

        return int(
            self.index.ntotal
        )


    def is_empty(self) -> bool:

        return (
            self.index.ntotal == 0
        )


    # ========================================================
    # SAVE
    # ========================================================

    def save_index(
        self,
        index_path: str,
        ids_path: str,
    ) -> None:

        directory = os.path.dirname(
            index_path
        )

        if directory:

            os.makedirs(
                directory,
                exist_ok=True,
            )


        directory = os.path.dirname(
            ids_path
        )

        if directory:

            os.makedirs(
                directory,
                exist_ok=True,
            )


        faiss.write_index(
            self.index,
            index_path,
        )


        with open(
            ids_path,
            "wb",
        ) as file:

            pickle.dump(
                self.product_ids,
                file,
            )


    # ========================================================
    # LOAD
    # ========================================================

    def load_index(
        self,
        index_path: str,
        ids_path: str,
    ) -> bool:

        if not (
            os.path.exists(
                index_path
            )
            and os.path.exists(
                ids_path
            )
        ):

            return False


        loaded_index = (
            faiss.read_index(
                index_path
            )
        )


        with open(
            ids_path,
            "rb",
        ) as file:

            product_ids = (
                pickle.load(
                    file
                )
            )


        # ----------------------------------------------------
        # Validate loaded index
        # ----------------------------------------------------

        if (
            loaded_index.d
            != self.dimension
        ):

            raise ValueError(
                f"Loaded FAISS index has dimension "
                f"{loaded_index.d}, expected "
                f"{self.dimension}."
            )


        if (
            loaded_index.ntotal
            != len(product_ids)
        ):

            raise ValueError(
                "FAISS index size does not match "
                "the number of stored product IDs."
            )


        self.index = loaded_index

        self.product_ids = list(
            product_ids
        )


        return True
