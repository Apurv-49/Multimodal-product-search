import torch
import open_clip
from PIL import Image
import numpy as np

class CLIPEmbedder:
    def __init__(self, model_name='ViT-B-32', pretrained='laion2b_s34b_b79k'):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()

    def get_image_embedding(self, image_path: str) -> np.ndarray:
        image = Image.open(image_path).convert("RGB")
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu().numpy().flatten()

    def get_text_embedding(self, text: str) -> np.ndarray:
        text_input = self.tokenizer([text]).to(self.device)
        
        with torch.no_grad():
            text_features = self.model.encode_text(text_input)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().numpy().flatten()
        
    def get_fused_embedding(self, image_path: str, metadata_text: str, alpha=0.5) -> np.ndarray:
        """Early Fusion Strategy: Merging Visual and Text Metadata Context"""
        img_emb = self.get_image_embedding(image_path)
        txt_emb = self.get_text_embedding(metadata_text)
        
        fused = alpha * img_emb + (1 - alpha) * txt_emb
        fused /= np.linalg.norm(fused)
        return fused
