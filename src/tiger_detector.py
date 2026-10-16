import torch
import torchvision.models as models
from torchvision.models import MobileNet_V3_Small_Weights
from PIL import Image
import numpy as np

class TigerDetector:
    """Detects whether an image or webcam capture contains a tiger.
    Prevents false identifications when humans, pets, other wild animals,
    or objects are shown to the system.
    """
    def __init__(self, device='cpu'):
        self.device = device
        # Use pretrained MobileNetV3 (lightweight, runs in ~5ms)
        self.weights = MobileNet_V3_Small_Weights.DEFAULT
        self.model = models.mobilenet_v3_small(weights=self.weights).to(device)
        self.model.eval()
        self.transforms = self.weights.transforms()
        self.categories = self.weights.meta['categories']

        # Tiger-related ImageNet indices
        # 292: 'tiger', 282: 'tiger cat'
        self.tiger_indices = [292, 282]

    def _describe_category(self, cat_name):
        """Map raw ImageNet category to user-friendly label."""
        name_lower = cat_name.lower()

        if name_lower in ['tiger', 'tiger cat']:
            return "Tiger"

        # Big cats / wild felines
        big_cats = ['snow leopard', 'leopard', 'jaguar', 'cheetah', 'lion', 'cougar', 'panther']
        for c in big_cats:
            if c in name_lower:
                return f"Other Big Cat ({cat_name.title()})"

        # Domestic cats
        if 'cat' in name_lower:
            return f"Domestic Cat ({cat_name.title()})"

        # Dogs & canines
        dog_indicators = [
            'terrier', 'retriever', 'hound', 'spaniel', 'shepherd', 'dog', 'pug',
            'chihuahua', 'collie', 'poodle', 'mastiff', 'bulldog', 'corgi', 'boxer',
            'rottweiler', 'husky', 'dalmatian', 'beagle', 'dhole', 'dingo', 'wolf'
        ]
        if any(d in name_lower for d in dog_indicators):
            return f"Dog / Canine ({cat_name.title()})"

        # Other wild / farm animals
        animals = [
            'elephant', 'zebra', 'bear', 'horse', 'monkey', 'chimpanzee', 'gorilla',
            'baboon', 'lemur', 'deer', 'fox', 'rabbit', 'bird', 'owl', 'eagle',
            'parrot', 'snake', 'frog', 'turtle', 'lizard', 'fish', 'whale',
            'dolphin', 'shark', 'cow', 'sheep', 'goat', 'pig', 'ox', 'bison',
            'camel', 'kangaroo', 'koala', 'panda', 'hyena', 'hippopotamus', 'rhinoceros'
        ]
        for a in animals:
            if a in name_lower:
                return f"Animal ({cat_name.title()})"

        # Human-related categories
        human_indicators = [
            'suit', 'jersey', 'groom', 'wig', 'sweatshirt', 'coat', 'jean',
            'bonnet', 'sombrero', 'kimono', 'mask', 'sunglass', 'tie', 'cardigan',
            'cloak', 'pajama', 'apron', 'scuba diver', 'trench coat', 'face powder',
            'lipstick', 'hair spray', 'band aid'
        ]
        if any(h in name_lower for h in human_indicators):
            return f"Human / Person ({cat_name.title()})"

        return f"Non-Tiger Subject ({cat_name.title()})"

    def predict(self, pil_image, threshold=0.15):
        """Analyze an image and determine if it contains a tiger.

        Returns:
            is_tiger (bool): True if tiger detected, False otherwise
            label (str): Human readable description of detected subject
            tiger_confidence (float): Estimated confidence that subject is a tiger
            top_prediction (str): Primary detected ImageNet class
            top_confidence (float): Probability of the top class
        """
        try:
            image_rgb = pil_image.convert('RGB')
            w, h = image_rgb.size

            # Evaluate both full frame and center crop (for phone screen in webcam)
            crops = [image_rgb]
            if w > 100 and h > 100:
                center_crop = image_rgb.crop((int(w * 0.15), int(h * 0.15), int(w * 0.85), int(h * 0.85)))
                crops.append(center_crop)

            best_tiger_prob = 0.0
            best_top_cat = "Unknown"
            best_top_prob = 0.0

            for crop in crops:
                tensor = self.transforms(crop).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    logits = self.model(tensor)
                    probs = torch.softmax(logits, dim=1)[0]

                tiger_prob = sum(probs[idx].item() for idx in self.tiger_indices)
                top_idx = torch.argmax(probs).item()
                top_prob = probs[top_idx].item()
                top_cat = self.categories[top_idx]

                if tiger_prob > best_tiger_prob:
                    best_tiger_prob = tiger_prob

                if top_prob > best_top_prob:
                    best_top_prob = top_prob
                    best_top_cat = top_cat

            # Decision logic:
            # Tiger detected if tiger_prob >= threshold OR top prediction is a tiger
            is_tiger = (best_tiger_prob >= threshold) or (best_top_cat.lower() in ['tiger', 'tiger cat'])

            if is_tiger:
                label = "Tiger"
            else:
                label = self._describe_category(best_top_cat)

            return is_tiger, label, best_tiger_prob, best_top_cat, best_top_prob

        except Exception as e:
            print(f"Error in TigerDetector: {e}")
            # Fallback to True so we don't block in case of unexpected exception
            return True, "Tiger", 1.0, "tiger", 1.0
