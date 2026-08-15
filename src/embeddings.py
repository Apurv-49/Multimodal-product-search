import torch
import open_clip
from PIL import Image
import numpy as np


class CLIPEmbedder:
    def __init__(self, model_name="ViT-B-32", pretrained="laion2b_s34b_b79k"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()

    @staticmethod
    def _normalize(features: torch.Tensor) -> torch.Tensor:
        return features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    def get_image_embedding(self, image_path: str) -> np.ndarray:
        image = Image.open(image_path).convert("RGB")
        return self.get_image_embedding_from_pil(image)

    def get_image_embedding_from_pil(self, image: Image.Image) -> np.ndarray:
        image_input = self.preprocess(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            image_features = self._normalize(self.model.encode_image(image_input))
        return image_features.cpu().numpy().flatten().astype("float32")

    def get_image_embeddings_from_pil(self, images: list[Image.Image]) -> np.ndarray:
        if not images:
            return np.empty((0, 512), dtype="float32")
        batch = torch.stack([self.preprocess(image.convert("RGB")) for image in images]).to(self.device)
        with torch.inference_mode():
            features = self._normalize(self.model.encode_image(batch))
        return features.cpu().numpy().astype("float32")

    def get_text_embedding(self, text: str) -> np.ndarray:
        return self.get_text_embeddings([text])[0]

    def get_text_embeddings(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 512), dtype="float32")
        text_input = self.tokenizer(texts).to(self.device)
        with torch.inference_mode():
            text_features = self._normalize(self.model.encode_text(text_input))
        return text_features.cpu().numpy().astype("float32")

    def classify_image(self, image: Image.Image, labels: list[str], template: str) -> tuple[str, float, dict[str, float]]:
        """Zero-shot CLIP classification used only as a retrieval hint/reranker."""
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

    def get_fused_embedding(self, image_path: str, metadata_text: str, alpha=0.5) -> np.ndarray:
        img_emb = self.get_image_embedding(image_path)
        txt_emb = self.get_text_embedding(metadata_text)
        fused = alpha * img_emb + (1 - alpha) * txt_emb
        fused /= np.linalg.norm(fused).clip(min=1e-12)
        return fused.astype("float32")
