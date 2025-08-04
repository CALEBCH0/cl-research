import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
from typing import List, Tuple, Dict, Optional, Callable
import numpy as np


class FaceDataset(Dataset):
    """Base face recognition dataset wrapper."""
    
    def __init__(
        self,
        root: str,
        samples: List[Tuple[str, int]],
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None
    ):
        self.root = root
        self.samples = samples
        self.transform = transform
        self.target_transform = target_transform
        
        # Build class to indices mapping
        self.class_to_idx = {}
        self.classes = []
        for _, label in samples:
            if label not in self.class_to_idx:
                self.class_to_idx[label] = len(self.classes)
                self.classes.append(label)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, target = self.samples[idx]
        
        # Load image
        img_path = os.path.join(self.root, path)
        img = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform is not None:
            img = self.transform(img)
        
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return img, target
    
    def get_num_classes(self) -> int:
        return len(self.classes)