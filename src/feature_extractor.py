import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import TigerEmbeddingModel

class SnowLeopardFeatureExtractor:
    def __init__(self, device='cpu'):
        self.device = device

        WEIGHTS_PATH = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'models', 'tiger_finetuned_vit.pth'
        )

        print(f"Loading fine-tuned tiger model from: {WEIGHTS_PATH}")

        self.model = TigerEmbeddingModel()
        self.model.load_state_dict(
            torch.load(WEIGHTS_PATH, map_location=device)
        )
        self.model = self.model.to(device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        print(f"[OK] Fine-tuned tiger model loaded successfully on {device}")

    def extract_features(self, image_path):
        try:
            image = Image.open(image_path).convert('RGB')
            return self.extract_features_from_pil(image)
        except Exception as e:
            print(f"Error extracting features from {image_path}: {e}")
            return None

    def extract_features_from_pil(self, pil_image):
        """Extract features directly from a PIL Image (no file I/O)."""
        try:
            image = pil_image.convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                features = self.model(image_tensor)
            features = features.cpu().numpy().flatten()
            return features
        except Exception as e:
            print(f"Error extracting features from PIL image: {e}")
            return None

    def extract_features_multi_crop(self, pil_image, num_crops=5):
        """Extract features with test-time augmentation for robustness.

        Generates multiple crops/views of the image, extracts embeddings
        for each, and returns the L2-normalised average. This significantly
        improves matching when the query comes from a phone screen shown
        to a webcam (glare, moiré, angle distortion, etc.).
        """
        try:
            image = pil_image.convert('RGB')
            w, h = image.size

            crops = [image]  # original full image

            # Centre crop (90%)
            margin_w, margin_h = int(w * 0.05), int(h * 0.05)
            crops.append(image.crop((margin_w, margin_h, w - margin_w, h - margin_h)))

            # Slightly tighter centre crop (80%)
            margin_w2, margin_h2 = int(w * 0.10), int(h * 0.10)
            crops.append(image.crop((margin_w2, margin_h2, w - margin_w2, h - margin_h2)))

            # Left-biased crop
            crops.append(image.crop((0, margin_h, w - margin_w * 2, h - margin_h)))

            # Right-biased crop
            crops.append(image.crop((margin_w * 2, margin_h, w, h - margin_h)))

            all_embeddings = []
            for crop in crops[:num_crops]:
                emb = self.extract_features_from_pil(crop)
                if emb is not None:
                    all_embeddings.append(emb)

            if not all_embeddings:
                return None

            avg_embedding = np.mean(all_embeddings, axis=0)
            # L2 normalise
            norm = np.linalg.norm(avg_embedding)
            if norm > 0:
                avg_embedding = avg_embedding / norm
            return avg_embedding

        except Exception as e:
            print(f"Error in multi-crop extraction: {e}")
            return None