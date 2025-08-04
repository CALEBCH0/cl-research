"""Synthetic dataset loader for testing without downloads."""
import torch
import numpy as np
from typing import List, Tuple


def create_synthetic_face_dataset(
    num_samples: int = 1000,
    num_classes: int = 50,
    img_size: Tuple[int, int] = (64, 64),
    img_channels: int = 3,
    train_split: float = 0.7,
    val_split: float = 0.15
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]], List[Tuple[str, int]]]:
    """Create synthetic face dataset for testing."""
    
    print(f"Creating synthetic dataset with {num_samples} samples and {num_classes} classes...")
    
    # Generate synthetic sample paths and labels
    samples = []
    samples_per_class = num_samples // num_classes
    
    for class_idx in range(num_classes):
        for sample_idx in range(samples_per_class):
            # Fake path (won't actually exist)
            fake_path = f"synthetic/person_{class_idx:03d}/img_{sample_idx:04d}.jpg"
            samples.append((fake_path, class_idx))
    
    # Add remaining samples to make exact num_samples
    remaining = num_samples - len(samples)
    for i in range(remaining):
        class_idx = i % num_classes
        fake_path = f"synthetic/person_{class_idx:03d}/img_extra_{i:04d}.jpg"
        samples.append((fake_path, class_idx))
    
    # Shuffle
    np.random.seed(42)
    np.random.shuffle(samples)
    
    # Split data
    n_train = int(len(samples) * train_split)
    n_val = int(len(samples) * val_split)
    
    train_samples = samples[:n_train]
    val_samples = samples[n_train:n_train + n_val]
    test_samples = samples[n_train + n_val:]
    
    print(f"Synthetic dataset created:")
    print(f"  Total classes: {num_classes}")
    print(f"  Train samples: {len(train_samples)}")
    print(f"  Val samples: {len(val_samples)}")
    print(f"  Test samples: {len(test_samples)}")
    
    return train_samples, val_samples, test_samples


class SyntheticFaceDataset(torch.utils.data.Dataset):
    """Synthetic face dataset that generates random images."""
    
    def __init__(
        self,
        samples: List[Tuple[str, int]],
        img_size: Tuple[int, int] = (64, 64),
        img_channels: int = 3,
        transform=None
    ):
        self.samples = samples
        self.img_size = img_size
        self.img_channels = img_channels
        self.transform = transform
        # Add targets attribute for Avalanche compatibility
        self.targets = [label for _, label in samples]
        
        # Generate consistent random images for each class
        self.class_prototypes = {}
        np.random.seed(42)
        unique_classes = set(label for _, label in samples)
        
        for class_idx in unique_classes:
            # Create a prototype face for each class
            prototype = np.random.randn(img_channels, *img_size).astype(np.float32)
            prototype = (prototype - prototype.min()) / (prototype.max() - prototype.min())
            self.class_prototypes[class_idx] = prototype
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        _, label = self.samples[idx]
        
        # Generate image based on class prototype with some variation
        prototype = self.class_prototypes[label]
        noise = np.random.randn(*prototype.shape) * 0.1
        img = prototype + noise
        img = np.clip(img, 0, 1)
        
        # Convert to PIL Image format
        img = (img * 255).astype(np.uint8)
        img = np.transpose(img, (1, 2, 0))  # CHW to HWC
        
        from PIL import Image
        img = Image.fromarray(img)
        
        if self.transform:
            img = self.transform(img)
        
        return img, label