import numpy as np
from typing import Dict, List, Optional
from collections import defaultdict


class ContinualLearningMetrics:
    """Metrics for continual learning evaluation."""
    
    def __init__(self):
        self.accuracy_matrix = defaultdict(dict)  # accuracy_matrix[test_exp][train_exp]
        self.current_experience = 0
    
    def update_accuracy_matrix(self, train_exp: int, test_exp: int, accuracy: float):
        """Update accuracy matrix."""
        self.accuracy_matrix[test_exp][train_exp] = accuracy
        self.current_experience = max(self.current_experience, train_exp)
    
    def compute_average_accuracy(self) -> float:
        """Compute average accuracy across all experiences."""
        if not self.accuracy_matrix:
            return 0.0
        
        all_accuracies = []
        for test_exp in range(self.current_experience + 1):
            if test_exp in self.accuracy_matrix and self.current_experience in self.accuracy_matrix[test_exp]:
                all_accuracies.append(self.accuracy_matrix[test_exp][self.current_experience])
        
        return np.mean(all_accuracies) if all_accuracies else 0.0
    
    def compute_forgetting(self) -> float:
        """Compute average forgetting across experiences."""
        if self.current_experience == 0:
            return 0.0
        
        forgetting_scores = []
        
        for test_exp in range(self.current_experience):
            # Get accuracy right after training on this experience
            if test_exp in self.accuracy_matrix and test_exp in self.accuracy_matrix[test_exp]:
                initial_acc = self.accuracy_matrix[test_exp][test_exp]
                
                # Get current accuracy on this experience
                if test_exp in self.accuracy_matrix and self.current_experience in self.accuracy_matrix[test_exp]:
                    current_acc = self.accuracy_matrix[test_exp][self.current_experience]
                    forgetting = max(0, initial_acc - current_acc)
                    forgetting_scores.append(forgetting)
        
        return np.mean(forgetting_scores) if forgetting_scores else 0.0
    
    def compute_forward_transfer(self) -> float:
        """Compute forward transfer."""
        if self.current_experience == 0:
            return 0.0
        
        forward_transfer_scores = []
        
        for test_exp in range(1, self.current_experience + 1):
            # Random baseline (assuming uniform distribution)
            random_acc = 1.0 / (test_exp + 1) * 100  # Assuming percentage
            
            # Initial accuracy before training on this experience
            if test_exp in self.accuracy_matrix and (test_exp - 1) in self.accuracy_matrix[test_exp]:
                initial_acc = self.accuracy_matrix[test_exp][test_exp - 1]
                forward_transfer = initial_acc - random_acc
                forward_transfer_scores.append(forward_transfer)
        
        return np.mean(forward_transfer_scores) if forward_transfer_scores else 0.0
    
    def compute_backward_transfer(self) -> float:
        """Compute backward transfer."""
        if self.current_experience == 0:
            return 0.0
        
        backward_transfer_scores = []
        
        for test_exp in range(self.current_experience):
            for train_exp in range(test_exp + 1, self.current_experience + 1):
                # Accuracy before training on new experience
                if test_exp in self.accuracy_matrix and (train_exp - 1) in self.accuracy_matrix[test_exp]:
                    acc_before = self.accuracy_matrix[test_exp][train_exp - 1]
                    
                    # Accuracy after training on new experience
                    if test_exp in self.accuracy_matrix and train_exp in self.accuracy_matrix[test_exp]:
                        acc_after = self.accuracy_matrix[test_exp][train_exp]
                        backward_transfer = acc_after - acc_before
                        backward_transfer_scores.append(backward_transfer)
        
        return np.mean(backward_transfer_scores) if backward_transfer_scores else 0.0
    
    def get_accuracy_matrix(self) -> Dict[int, Dict[int, float]]:
        """Get the full accuracy matrix."""
        return dict(self.accuracy_matrix)
    
    def compute_all_metrics(self) -> Dict[str, float]:
        """Compute all continual learning metrics."""
        return {
            'average_accuracy': self.compute_average_accuracy(),
            'forgetting': self.compute_forgetting(),
            'forward_transfer': self.compute_forward_transfer(),
            'backward_transfer': self.compute_backward_transfer(),
        }