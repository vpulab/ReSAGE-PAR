"""
Text-Image Representational Similarity (RSI)

This module provides a utility class that computes embeddings for texts and images
and computes similarities between text prompts and images using different scoring methods.

Supported score types:
- 'clip': CLIP-based embeddings with cosine similarity (default)
- 'blip': BLIP Image-Text Matching score

API:
- TextImageRepresentationalSimilarity(score_name: str = 'clip', model_name: Optional[str] = None)
    - embed_text(prompts: List[str]) -> np.ndarray (N x D) [CLIP only]
    - embed_images(images: List[PIL.Image]) -> np.ndarray (N x D) [CLIP only]
    - cosine_similarity_matrix(text_emb, image_emb) -> np.ndarray (N_text x N_image) [CLIP only]
    - score_prompts_images(prompts, images) -> dict with per-pair similarities and averages

Note: this code expects `transformers`, `torch`, and `PIL` to be installed in the environment.

"""

from typing import List, Optional, Tuple, Dict
import numpy as np
import torch

from transformers import CLIPProcessor, CLIPModel
from .scores import blipscore


class TextImageRepresentationalSimilarity:
    def __init__(self, score_name: str = "clip", model_name: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize the scoring model.
        
        Args:
            score_name: Type of score to use ('clip' or 'blip')
            model_name: Optional model name (uses default for each score type if None)
            device: Device to use ('cuda' or 'cpu', auto-detected if None)
        """
        self.score_name = score_name.lower()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None
        
        # Set default model names for each score type
        if model_name is None:
            if self.score_name == "clip":
                self.model_name = "openai/clip-vit-base-patch32"
            elif self.score_name == "blip":
                self.model_name = "Salesforce/blip-itm-base-coco"
            else:
                raise ValueError(f"Unknown score name: {score_name}. Supported: 'clip', 'blip'")
        else:
            self.model_name = model_name
        
        self._load_model()

    def _load_model(self):
        """Load the model based on score_name."""
        if self.score_name == "clip":
            self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(self.model_name)
            self.model.eval()
        elif self.score_name == "blip":
            # Initialize BLIP model using the blipscore module
            blipscore.setModelBlipScore()
        else:
            raise ValueError(f"Unknown score name: {self.score_name}")

    def _to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        return tensor.detach().cpu().numpy()

    def embed_text(self, prompts: List[str], batch_size: int = 32) -> np.ndarray:
        """Return L2-normalized text embeddings (N x D). Only for CLIP."""
        if self.score_name != "clip":
            raise NotImplementedError(f"embed_text is only available for CLIP, not {self.score_name}")
        
        all_embs = []
        with torch.no_grad():
            for i in range(0, len(prompts), batch_size):
                batch = prompts[i : i + batch_size]
                inputs = self.processor(text=batch, return_tensors="pt", padding=True).to(self.device)
                out = self.model.get_text_features(**inputs)
                # Normalize
                out = out / out.norm(p=2, dim=-1, keepdim=True)
                all_embs.append(self._to_numpy(out))
        if len(all_embs) == 0:
            return np.zeros((0, self.model.config.projection_dim))
        return np.vstack(all_embs)

    def embed_images(self, images: List, batch_size: int = 16) -> np.ndarray:
        """Return L2-normalized image embeddings (N x D). Accepts PIL images or numpy arrays. Only for CLIP."""
        if self.score_name != "clip":
            raise NotImplementedError(f"embed_images is only available for CLIP, not {self.score_name}")
        
        all_embs = []
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch = images[i : i + batch_size]
                inputs = self.processor(images=batch, return_tensors="pt", padding=True).to(self.device)
                out = self.model.get_image_features(**inputs)
                out = out / out.norm(p=2, dim=-1, keepdim=True)
                all_embs.append(self._to_numpy(out))
        if len(all_embs) == 0:
            return np.zeros((0, self.model.config.projection_dim))
        return np.vstack(all_embs)

    @staticmethod
    def cosine_similarity_matrix(text_emb: np.ndarray, image_emb: np.ndarray) -> np.ndarray:
        """Compute cosine similarity matrix between text and image embeddings.

        Returns matrix of shape (N_text, N_image) with values in [-1, 1].
        """
        if text_emb.size == 0 or image_emb.size == 0:
            return np.zeros((text_emb.shape[0], image_emb.shape[0]))
        # Both inputs are expected to be L2-normalized; then cosine similarity == dot product
        return np.dot(text_emb, image_emb.T)

    def score_prompts_images(self, prompts: List[str], image) -> Dict[str, object]:
        """Compute similarity scores for each prompt with a single image.

        Args:
            prompts: List of text prompts
            image: Single image (PIL Image or path string)

        Returns:
          For CLIP:
          {
            'text_emb': np.ndarray (N_text x D),
            'image_emb': np.ndarray (1 x D),
            'scores': np.ndarray (N_text,) - similarity score for each prompt
          }
          
          For BLIP:
          {
            'scores': np.ndarray (N_text,) - similarity score for each prompt
          }
        """
        if self.score_name == "clip":
            text_emb = self.embed_text(prompts)
            image_emb = self.embed_images([image])  # Wrap single image in list
            sim = self.cosine_similarity_matrix(text_emb, image_emb)
            scores = sim[:, 0]  # Extract scores for the single image
            return {
                "text_emb": text_emb,
                "image_emb": image_emb,
                "scores": scores,
            }
        
        elif self.score_name == "blip":
            # Compute BLIP scores for each prompt with the single image
            n_prompts = len(prompts)
            scores = np.zeros(n_prompts)
            
            # Handle image path or PIL Image
            if isinstance(image, str):
                # Image is already a path
                for i, prompt in enumerate(prompts):
                    scores[i] = blipscore.getBlipScore(prompt, image)
            else:
                # Image is PIL Image - save to temp file once
                import tempfile
                import os
                from PIL import Image as PILImage
                
                if isinstance(image, PILImage.Image):
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
                        image.save(tmp.name)
                        tmp_path = tmp.name
                    
                    try:
                        for i, prompt in enumerate(prompts):
                            scores[i] = blipscore.getBlipScore(prompt, tmp_path)
                    finally:
                        os.unlink(tmp_path)
                else:
                    raise ValueError(f"Image must be either a path string or PIL Image, got {type(image)}")
            
            return {
                "scores": scores,
            }
        
        else:
            raise ValueError(f"Unknown score name: {self.score_name}")


if __name__ == "__main__":
    # Quick smoke demo
    from PIL import Image

    print("Testing CLIP score...")
    model_clip = TextImageRepresentationalSimilarity(score_name="clip")
    prompts = ["a person wearing a red jacket", "a person wearing a blue shirt"]
    # Create a single white image for testing
    img = Image.new("RGB", (224, 224), color=(255, 255, 255))
    r = model_clip.score_prompts_images(prompts, img)
    print("CLIP Scores:", r["scores"])
    
    print("\nTesting BLIP score...")
    model_blip = TextImageRepresentationalSimilarity(score_name="blip")
    r_blip = model_blip.score_prompts_images(prompts, img)
    print("BLIP Scores:", r_blip["scores"])
