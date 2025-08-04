import torch
import numpy as np
from sklearn.metrics import roc_curve, auc
from typing import Dict, List, Tuple, Optional
import torch.nn.functional as F


class FaceRecognitionMetrics:
    """Metrics specific to face recognition tasks."""
    
    def __init__(self):
        self.embeddings = []
        self.labels = []
        self.predictions = []
    
    def update(self, embeddings: torch.Tensor, labels: torch.Tensor, predictions: Optional[torch.Tensor] = None):
        """Update metrics with new batch."""
        self.embeddings.append(embeddings.detach().cpu())
        self.labels.append(labels.detach().cpu())
        if predictions is not None:
            self.predictions.append(predictions.detach().cpu())
    
    def reset(self):
        """Reset accumulated metrics."""
        self.embeddings = []
        self.labels = []
        self.predictions = []
    
    def compute_accuracy(self) -> float:
        """Compute classification accuracy."""
        if not self.predictions:
            return 0.0
        
        all_predictions = torch.cat(self.predictions)
        all_labels = torch.cat(self.labels)
        
        _, predicted = torch.max(all_predictions, 1)
        correct = (predicted == all_labels).sum().item()
        total = all_labels.size(0)
        
        return correct / total if total > 0 else 0.0
    
    def compute_verification_metrics(self, distance_metric: str = 'cosine') -> Dict[str, float]:
        """Compute face verification metrics (TAR@FAR)."""
        if not self.embeddings:
            return {}
        
        all_embeddings = torch.cat(self.embeddings)
        all_labels = torch.cat(self.labels)
        
        # Compute pairwise distances
        distances, labels_pair = self._compute_pairwise_distances(
            all_embeddings, all_labels, distance_metric
        )
        
        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(labels_pair, -distances)
        roc_auc = auc(fpr, tpr)
        
        # Compute TAR @ FAR
        metrics = {
            'auc': roc_auc,
            'tar_at_far_0.001': self._tar_at_far(fpr, tpr, 0.001),
            'tar_at_far_0.01': self._tar_at_far(fpr, tpr, 0.01),
            'tar_at_far_0.1': self._tar_at_far(fpr, tpr, 0.1),
        }
        
        return metrics
    
    def _compute_pairwise_distances(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        metric: str = 'cosine'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute pairwise distances between embeddings."""
        n_samples = embeddings.shape[0]
        
        # Sample pairs to avoid memory issues
        max_pairs = 10000
        if n_samples * (n_samples - 1) // 2 > max_pairs:
            # Random sampling of pairs
            indices = np.random.choice(n_samples, size=(max_pairs, 2), replace=True)
            indices = indices[indices[:, 0] != indices[:, 1]]  # Remove same indices
        else:
            # All pairs
            indices = [(i, j) for i in range(n_samples) for j in range(i + 1, n_samples)]
            indices = np.array(indices)
        
        distances = []
        labels_pair = []
        
        for i, j in indices:
            if metric == 'cosine':
                # Cosine similarity (converted to distance)
                sim = F.cosine_similarity(
                    embeddings[i].unsqueeze(0),
                    embeddings[j].unsqueeze(0)
                )
                dist = 1 - sim.item()
            elif metric == 'euclidean':
                dist = torch.norm(embeddings[i] - embeddings[j]).item()
            else:
                raise ValueError(f"Unknown distance metric: {metric}")
            
            distances.append(dist)
            labels_pair.append(int(labels[i] == labels[j]))
        
        return np.array(distances), np.array(labels_pair)
    
    def _tar_at_far(self, fpr: np.ndarray, tpr: np.ndarray, far_target: float) -> float:
        """Compute True Accept Rate at given False Accept Rate."""
        idx = np.argmin(np.abs(fpr - far_target))
        return float(tpr[idx])
    
    def compute_all_metrics(self) -> Dict[str, float]:
        """Compute all face recognition metrics."""
        metrics = {}
        
        # Classification metrics
        if self.predictions:
            metrics['accuracy'] = self.compute_accuracy()
        
        # Verification metrics
        verification_metrics = self.compute_verification_metrics()
        metrics.update(verification_metrics)
        
        return metrics