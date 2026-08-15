from __future__ import annotations

import numpy as np
import open_clip
import torch
from PIL import Image, ImageOps


class CLIPEmbedder:
    """
    OpenCLIP encoder used by ProductLens.

    The important retrieval rule is:

        image -> image embedding
        text  -> text embedding

    Both embeddings are L2-normalized, which makes their dot
    product equivalent to cosine similarity.

    Zero-shot classification is used only as a weak hint.
    It should NOT be treated as ground-truth brand/category
    prediction.
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
    ):
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.model, _, self.preprocess = (
            open_clip.create_model_and_transforms(
                model_name,
                pretrained=pretrained,
                device=self.device,
            )
        )

        self.tokenizer = (
            open_clip.get_tokenizer(
                model_name
            )
        )

        self.model.eval()

        self.dimension = int(
            self.model.visual.output_dim
        )


    # ========================================================
    # NORMALIZATION
    # ========================================================

    @staticmethod
    def _normalize(
        features: torch.Tensor,
    ) -> torch.Tensor:

        """
        L2-normalize feature vectors.

        After normalization:

            cosine_similarity(a, b)
            =
            a @ b

        """

        norm = features.norm(
            dim=-1,
            keepdim=True,
        )

        return features / norm.clamp_min(
            1e-12
        )


    @staticmethod
    def _safe_image(
        image: Image.Image,
    ) -> Image.Image:

        """
        Normalize uploaded images before CLIP preprocessing.

        Handles:

        - EXIF rotation
        - RGBA images
        - grayscale images
        - palette images
        """

        image = ImageOps.exif_transpose(
            image
        )

        if image.mode != "RGB":

            image = image.convert(
                "RGB"
            )

        return image


    # ========================================================
    # IMAGE EMBEDDINGS
    # ========================================================

    def get_image_embedding(
        self,
        image_path: str,
    ) -> np.ndarray:

        with Image.open(
            image_path
        ) as image:

            return (
                self.get_image_embedding_from_pil(
                    image
                )
            )


    def get_image_embedding_from_pil(
        self,
        image: Image.Image,
    ) -> np.ndarray:

        return (
            self.get_image_embeddings_from_pil(
                [image]
            )[0]
        )


    def get_image_embeddings_from_pil(
        self,
        images: list[Image.Image],
        batch_size: int = 32,
    ) -> np.ndarray:

        if not images:

            return np.empty(
                (
                    0,
                    self.dimension,
                ),
                dtype="float32",
            )


        outputs: list[np.ndarray] = []


        for start in range(
            0,
            len(images),
            batch_size,
        ):

            batch_images = images[
                start : start + batch_size
            ]


            processed = [
                self.preprocess(
                    self._safe_image(
                        image
                    )
                )
                for image in batch_images
            ]


            batch = torch.stack(
                processed
            ).to(
                self.device
            )


            with torch.inference_mode():

                features = (
                    self.model.encode_image(
                        batch
                    )
                )

                features = self._normalize(
                    features
                )


            outputs.append(
                features
                .cpu()
                .numpy()
                .astype("float32")
            )


        return np.concatenate(
            outputs,
            axis=0,
        )


    # ========================================================
    # TEXT EMBEDDINGS
    # ========================================================

    def get_text_embedding(
        self,
        text: str,
    ) -> np.ndarray:

        return (
            self.get_text_embeddings(
                [text]
            )[0]
        )


    def get_text_embeddings(
        self,
        texts: list[str],
        batch_size: int = 64,
    ) -> np.ndarray:

        if not texts:

            return np.empty(
                (
                    0,
                    self.dimension,
                ),
                dtype="float32",
            )


        outputs: list[np.ndarray] = []


        for start in range(
            0,
            len(texts),
            batch_size,
        ):

            batch_texts = texts[
                start : start + batch_size
            ]


            text_input = (
                self.tokenizer(
                    batch_texts
                ).to(
                    self.device
                )
            )


            with torch.inference_mode():

                features = (
                    self.model.encode_text(
                        text_input
                    )
                )

                features = self._normalize(
                    features
                )


            outputs.append(
                features
                .cpu()
                .numpy()
                .astype("float32")
            )


        return np.concatenate(
            outputs,
            axis=0,
        )


    # ========================================================
    # ZERO-SHOT CLASSIFICATION
    # ========================================================

    def classify_image(
        self,
        image: Image.Image,
        labels: list[str],
        template: str,
    ) -> tuple[
        str,
        float,
        dict[str, float],
    ]:

        """
        Zero-shot classification.

        IMPORTANT:

        This is only a hint.

        CLIP is not a reliable product-brand verifier.
        Therefore this result must not be given a large
        ranking bonus in the retrieval pipeline.
        """

        image = self._safe_image(
            image
        )


        image_input = (
            self.preprocess(
                image
            )
            .unsqueeze(0)
            .to(
                self.device
            )
        )


        texts = [
            template.format(
                label=label
            )
            for label in labels
        ]


        text_input = (
            self.tokenizer(
                texts
            ).to(
                self.device
            )
        )


        with torch.inference_mode():

            image_features = (
                self.model.encode_image(
                    image_input
                )
            )

            image_features = (
                self._normalize(
                    image_features
                )
            )


            text_features = (
                self.model.encode_text(
                    text_input
                )
            )

            text_features = (
                self._normalize(
                    text_features
                )
            )


            similarity = (
                100.0
                * image_features
                @ text_features.T
            )


            probabilities = (
                similarity.softmax(
                    dim=-1
                )[0]
            )


        probabilities = (
            probabilities
            .cpu()
            .numpy()
        )


        scores = {
            label: float(
                probability
            )
            for label, probability
            in zip(
                labels,
                probabilities,
            )
        }


        best_index = int(
            np.argmax(
                probabilities
            )
        )


        return (
            labels[best_index],
            float(
                probabilities[
                    best_index
                ]
            ),
            scores,
        )


    # ========================================================
    # TOP ZERO-SHOT LABELS
    # ========================================================

    def top_image_labels(
        self,
        image: Image.Image,
        labels: list[str],
        template: str,
        top_k: int = 3,
    ) -> list[
        tuple[str, float]
    ]:

        """
        Return the strongest zero-shot candidates.

        These are retrieval hints, not verified labels.
        """

        _, _, scores = (
            self.classify_image(
                image,
                labels,
                template,
            )
        )


        ranked = sorted(
            scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )


        return ranked[
            :max(1, top_k)
        ]


    # ========================================================
    # PROMPT ENSEMBLE
    # ========================================================

    def classify_image_with_prompts(
        self,
        image: Image.Image,
        labels: list[str],
        templates: list[str],
    ) -> list[
        tuple[str, float]
    ]:

        """
        More robust zero-shot classification.

        Instead of relying on one prompt:

            "a product photo of Nike"

        we compare several descriptions and average their
        normalized text embeddings.

        This reduces sensitivity to prompt wording.
        """

        if not labels:

            return []


        if not templates:

            templates = [
                "a photo of {label}",
            ]


        image = self._safe_image(
            image
        )


        image_input = (
            self.preprocess(
                image
            )
            .unsqueeze(0)
            .to(
                self.device
            )
        )


        with torch.inference_mode():

            image_features = (
                self.model.encode_image(
                    image_input
                )
            )

            image_features = (
                self._normalize(
                    image_features
                )
            )


            label_features = []


            for label in labels:

                prompts = [
                    template.format(
                        label=label
                    )
                    for template in templates
                ]


                text_input = (
                    self.tokenizer(
                        prompts
                    ).to(
                        self.device
                    )
                )


                features = (
                    self.model.encode_text(
                        text_input
                    )
                )


                features = (
                    self._normalize(
                        features
                    )
                )


                # Average the prompt embeddings.
                feature = features.mean(
                    dim=0,
                    keepdim=True,
                )


                feature = (
                    self._normalize(
                        feature
                    )
                )


                label_features.append(
                    feature
                )


            label_features = torch.cat(
                label_features,
                dim=0,
            )


            similarity = (
                image_features
                @ label_features.T
            )[0]


        scores = similarity.cpu().numpy()


        ranked_indices = np.argsort(
            scores
        )[::-1]


        return [
            (
                labels[int(index)],
                float(
                    scores[int(index)]
                ),
            )
            for index in ranked_indices[
                :max(1, len(labels))
            ]
        ]


    # ========================================================
    # FUSED EMBEDDING
    # ========================================================

    def get_fused_embedding(
        self,
        image_path: str,
        metadata_text: str,
        alpha: float = 0.5,
    ) -> np.ndarray:

        """
        Combine image and text embeddings.

        This method is retained for compatibility with the
        original prototype.

        alpha = image contribution
        1-alpha = text contribution
        """

        image_embedding = (
            self.get_image_embedding(
                image_path
            )
        )


        text_embedding = (
            self.get_text_embedding(
                metadata_text
            )
        )


        alpha = float(
            np.clip(
                alpha,
                0.0,
                1.0,
            )
        )


        fused = (
            alpha * image_embedding
            + (1.0 - alpha)
            * text_embedding
        )


        norm = np.linalg.norm(
            fused
        )


        if norm < 1e-12:

            return fused.astype(
                "float32"
            )


        fused = (
            fused / norm
        )


        return fused.astype(
            "float32"
        )
