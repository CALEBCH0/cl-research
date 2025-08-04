import torch
import torch.nn as nn
from typing import Dict, Any, Optional


class FaceRecognitionBackbone(nn.Module):
    """Base class for face recognition backbones."""
    
    def __init__(self, backbone: nn.Module, feature_dim: int, embedding_dim: int = 512):
        super().__init__()
        self.backbone = backbone
        self.feature_dim = feature_dim
        self.embedding_dim = embedding_dim
        
        # Add embedding layer for face recognition
        self.embedding = nn.Sequential(
            nn.Linear(feature_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Extract features from backbone
        features = self.backbone(x)
        
        # Get embeddings for face recognition
        embeddings = self.embedding(features)
        
        # L2 normalize embeddings (common in face recognition)
        embeddings_norm = nn.functional.normalize(embeddings, p=2, dim=1)
        
        return {
            'features': features,
            'embeddings': embeddings,
            'embeddings_norm': embeddings_norm
        }
    
    def freeze_backbone(self):
        """Freeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True
    
    def freeze_until_layer(self, layer_name: str):
        """Freeze backbone until specified layer."""
        freeze = True
        for name, param in self.backbone.named_parameters():
            if layer_name in name:
                freeze = False
            param.requires_grad = not freeze