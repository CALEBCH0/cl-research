import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ClassificationHead(nn.Module):
    """Standard classification head for face recognition."""
    
    def __init__(self, embedding_dim: int, num_classes: int, dropout: float = 0.0):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
            
        self.fc = nn.Linear(embedding_dim, num_classes)
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc.weight)
        if self.fc.bias is not None:
            nn.init.constant_(self.fc.bias, 0)
    
    def forward(self, embeddings: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.dropout is not None:
            embeddings = self.dropout(embeddings)
        
        logits = self.fc(embeddings)
        return logits