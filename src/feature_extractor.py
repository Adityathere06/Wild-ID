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

        print(f"✓ Fine-tuned tiger model loaded successfully on {device}")

    def extract_features(self, image_path):
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                features = self.model(image_tensor)
            features = features.cpu().numpy().flatten()
            return features
        except Exception as e:
            print(f"Error extracting features from {image_path}: {e}")
            return None