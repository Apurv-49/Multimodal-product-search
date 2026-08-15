from __future__ import annotations

import numpy as np
import open_clip
import torch
from PIL import Image


class CLIPEmbedder:
    """OpenCLIP encoder used for image/text retrieval and lightweight hints."""

    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "laion2b_s34b_b79k"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()
        self.dimension = int(self.model.visual.output_dim)

    @staticmethod
    def _normalize(features: torch.Tensor) -> torch.Tensor:
        return features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    def get_image_embedding(self, image_path: str) -> np.ndarray:
        with Image.open(image_path) as image:
            return self.get_image_embedding_from_pil(image)

    def get_image_embedding_from_pil(self, image: Image.Image) -> np.ndarray:
        return self.get_image_embeddings_from_pil([image])[0]

    def get_image_embeddings_from_pil(
        self, images: list[Image.Image], batch_size: int = 32
    ) -> np.ndarray:
        if not images:
            return np.empty((0, self.dimension), dtype="float32")

        outputs: list[np.ndarray] = []
        for start in range(0, len(images), batch_size):
            batch_images = images[start : start + batch_size]
            batch = torch.stack(
                [self.preprocess(image.convert("RGB")) for image in batch_images]
            ).to(self.device)
            with torch.inference_mode():
                features = self._normalize(self.model.encode_image(batch))
            outputs.append(features.cpu().numpy().astype("float32"))
        return np.concatenate(outputs, axis=0)

    def get_text_embedding(self, text: str) -> np.ndarray:
        return self.get_text_embeddings([text])[0]

    def get_text_embeddings(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype="float32")

        outputs: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            text_input = self.tokenizer(batch_texts).to(self.device)
            with torch.inference_mode():
                features = self._normalize(self.model.encode_text(text_input))
            outputs.append(features.cpu().numpy().astype("float32"))
        return np.concatenate(outputs, axis=0)

    def classify_image(
        self,
        image: Image.Image,
        labels: list[str],
        template: str,
    ) -> tuple[str, float, dict[str, float]]:
        """Zero-shot classification used as a candidate-generation hint, not ground truth."""
        image_input = self.preprocess(image.convert("RGB")).unsqueeze(0).to(self.device)
        texts = [template.format(label=label) for label in labels]
        text_input = self.tokenizer(texts).to(self.device)
        with torch.inference_mode():
            image_features = self._normalize(self.model.encode_image(image_input))
            text_features = self._normalize(self.model.encode_text(text_input))
            logits = (100.0 * image_features @ text_features.T).softmax(dim=-1)[0]

        probabilities = logits.cpu().numpy()
        scores = {label: float(prob) for label, prob in zip(labels, probabilities)}
        best_index = int(np.argmax(probabilities))
        return labels[best_index], float(probabilities[best_index]), scores

    def top_image_labels(
        self,
        image: Image.Image,
        labels: list[str],
        template: str,
        top_k: int = 3,
    ) -> list[tuple[str, float]]:
        """Return top label candidates for query expansion."""
        _, _, scores = self.classify_image(image, labels, template)
        return sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]

    def get_fused_embedding(
        self, image_path: str, metadata_text: str, alpha: float = 0.5
    ) -> np.ndarray:
        image_embedding = self.get_image_embedding(image_path)
        text_embedding = self.get_text_embedding(metadata_text)
        fused = alpha * image_embedding + (1.0 - alpha) * text_embedding
        fused /= np.linalg.norm(fused).clip(min=1e-12)
        return fused.astype("float32")
