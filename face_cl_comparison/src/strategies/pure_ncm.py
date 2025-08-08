"""Pure NCM (Nearest Class Mean) strategy for Avalanche."""
import torch
import torch.nn as nn
from avalanche.training.templates import SupervisedTemplate
from avalanche.training.plugins import SupervisedPlugin, EvaluationPlugin
from typing import Optional, List, Union
from torch.optim import Optimizer
from avalanche.models import FeatureExtractorBackbone


class NCMClassifier(nn.Module):
    """Nearest Class Mean classifier."""
    
    def __init__(self, feature_size: int, num_classes: int):
        super().__init__()
        self.feature_size = feature_size
        self.num_classes = num_classes
        # Class means: [num_classes, feature_size]
        self.register_buffer('class_means', torch.zeros(num_classes, feature_size))
        self.register_buffer('class_counts', torch.zeros(num_classes))
        
    def update_mean(self, features: torch.Tensor, labels: torch.Tensor):
        """Update class means with new features."""
        with torch.no_grad():
            unique_labels = labels.unique()
            for c in unique_labels:
                mask = labels == c
                if mask.any():
                    class_features = features[mask]
                    # Incremental mean update
                    old_count = self.class_counts[c]
                    new_count = old_count + len(class_features)
                    old_mean = self.class_means[c]
                    new_mean = (old_mean * old_count + class_features.sum(0)) / new_count
                    self.class_means[c] = new_mean
                    self.class_counts[c] = new_count
            
            # Debug: print update info
            if len(unique_labels) > 0:
                print(f"Updated means for classes: {unique_labels.tolist()}, "
                      f"counts: {[int(self.class_counts[c]) for c in unique_labels]}")
    
    def forward(self, features: torch.Tensor):
        """Classify based on nearest class mean."""
        # Check if we have any class means computed
        if self.class_counts.sum() == 0:
            # No means computed yet, return zeros
            return torch.zeros(features.size(0), self.num_classes, device=features.device)
        
        # Only compute distances for classes we've seen
        seen_classes = (self.class_counts > 0).nonzero(as_tuple=True)[0]
        
        if len(seen_classes) == 0:
            return torch.zeros(features.size(0), self.num_classes, device=features.device)
        
        # Normalize features and means for seen classes
        features_norm = nn.functional.normalize(features, dim=1)
        seen_means = self.class_means[seen_classes]
        seen_means_norm = nn.functional.normalize(seen_means, dim=1)
        
        # Compute distances: [batch_size, num_seen_classes]
        distances = torch.cdist(features_norm, seen_means_norm)
        
        # Create output tensor with large distances for unseen classes
        output = torch.full((features.size(0), self.num_classes), float('inf'), device=features.device)
        output[:, seen_classes] = distances
        
        # Return negative distances (so argmax gives nearest)
        return -output


class PureNCM(SupervisedTemplate):
    """Pure Nearest Class Mean strategy.
    
    This strategy:
    1. Uses a frozen feature extractor (no backprop)
    2. Only updates class means during training
    3. Classifies based on nearest mean in feature space
    """
    
    def __init__(
        self,
        *,  # Force keyword-only arguments as per Avalanche convention
        feature_extractor: nn.Module,
        feature_size: int,
        num_classes: int,
        train_mb_size: int = 1,
        train_epochs: int = 1,
        eval_mb_size: Optional[int] = None,
        device: Union[str, torch.device] = "cpu",
        plugins: Optional[List[SupervisedPlugin]] = None,
        evaluator: Optional[EvaluationPlugin] = None,
        eval_every: int = -1,
    ):
        # Create NCM classifier
        self.ncm_classifier = NCMClassifier(feature_size, num_classes).to(device)
        
        # Freeze feature extractor
        self.feature_extractor = feature_extractor.to(device)
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.feature_extractor.eval()
        
        # Dummy optimizer (required by template but not used)
        dummy_optimizer = torch.optim.SGD([torch.nn.Parameter(torch.tensor(0.0))], lr=0.0)
        
        # Dummy criterion (required by template but not used)
        dummy_criterion = nn.CrossEntropyLoss()
        
        # Combined model for evaluation
        class CombinedModel(nn.Module):
            def __init__(self, feat_ext, classifier):
                super().__init__()
                self.feature_extractor = feat_ext
                self.classifier = classifier
                
            def forward(self, x):
                with torch.no_grad():
                    features = self.feature_extractor(x)
                return self.classifier(features)
        
        self.model = CombinedModel(feature_extractor, self.ncm_classifier)
        
        super().__init__(
            model=self.model,
            optimizer=dummy_optimizer,
            criterion=dummy_criterion,
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            eval_mb_size=eval_mb_size,
            device=device,
            plugins=plugins,
            evaluator=evaluator,
            eval_every=eval_every,
        )
    
    def training_epoch(self, **kwargs):
        """Custom training epoch that only updates class means."""
        for i, self.mbatch in enumerate(self.dataloader):
            self._unpack_minibatch()
            
            # Extract features (no gradients needed)
            with torch.no_grad():
                features = self.feature_extractor(self.mb_x)
            
            # Debug info (only first batch)
            if i == 0:
                print(f"  Feature shape: {features.shape}, Labels: {self.mb_y.unique().tolist()}")
            
            # Update class means
            self.ncm_classifier.update_mean(features, self.mb_y)
            
            # Dummy forward pass for metrics
            self.mb_output = self.ncm_classifier(features)
            self.loss = torch.tensor(0.0).to(self.device)  # No real loss
            
            # Trigger plugins
            self._after_training_iteration(**kwargs)
    
    def backward(self):
        """No backward pass needed for NCM."""
        pass
    
    def optimizer_step(self):
        """No optimizer step needed for NCM."""
        pass
    