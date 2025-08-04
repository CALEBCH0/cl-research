import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class CosFaceHead(nn.Module):
    """CosFace head for face recognition.
    
    Reference:
        CosFace: Large Margin Cosine Loss for Deep Face Recognition
        https://arxiv.org/abs/1801.09414
    """
    
    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        s: float = 30.0,
        m: float = 0.35
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.s = s
        self.m = m
        
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
    
    def forward(self, embeddings: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Normalize embeddings and weights
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weight_norm = F.normalize(self.weight, p=2, dim=1)
        
        # Compute cosine similarity
        cosine = F.linear(embeddings, weight_norm)
        
        if labels is None:
            # Inference mode
            return self.s * cosine
        
        # Training mode with cosine margin
        # One-hot encode labels
        one_hot = torch.zeros(cosine.size(), device=embeddings.device)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        
        # Apply margin to correct class
        output = cosine - one_hot * self.m
        output *= self.s
        
        return output