import torch
import torch.nn as nn
import timm

class TigerEmbeddingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            'vit_base_patch16_224', pretrained=False, num_classes=0)
        self.embedding_head = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

    def forward(self, x):
        features = self.backbone(x)
        embedding = self.embedding_head(features)
        embedding = nn.functional.normalize(embedding, p=2, dim=1)
        return embedding