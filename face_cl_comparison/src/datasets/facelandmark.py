"""
FaceLandmark IR face dataset loader for Avalanche continual learning.

Dataset structure:
- Root: /home/dylee/data/data_fid/FaceID/FaceLandmark/data/IR/cropdata/
- Folders: aihub_dms_<number> (each folder is one person/identity)
- Images: Q_001_30_M_01_M0_G0_C0_01.jpg format
- Format: 428x428 RGB JPEG images
"""

import os
from pathlib import Path
from typing import Tuple, Optional, Union, List
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from collections import defaultdict

from avalanche.benchmarks import nc_benchmark
from avalanche.benchmarks.utils import as_classification_dataset
from src.utils.benchmark_info import BenchmarkInfo


class FaceLandmarkDataset(Dataset):
    """FaceLandmark IR face dataset."""
    
    def __init__(self, root_dir: str, transform=None, target_transform=None,
                 subset_indices: Optional[List[int]] = None):
        """
        Initialize FaceLandmark dataset.
        
        Args:
            root_dir: Path to cropdata directory
            transform: Image transformations
            target_transform: Target transformations
            subset_indices: Optional indices for subset selection
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.target_transform = target_transform
        
        # Load all images and labels
        self.samples = []
        self.targets = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        
        # Scan directory structure
        person_dirs = sorted([d for d in self.root_dir.iterdir() 
                            if d.is_dir() and d.name.startswith('aihub_dms_')])
        
        print(f"Found {len(person_dirs)} person directories in {root_dir}")
        
        for class_idx, person_dir in enumerate(person_dirs):
            self.class_to_idx[person_dir.name] = class_idx
            self.idx_to_class[class_idx] = person_dir.name
            
            # Get all images for this person
            image_files = sorted(list(person_dir.glob('*.jpg')) + 
                               list(person_dir.glob('*.JPG')) +
                               list(person_dir.glob('*.jpeg')) +
                               list(person_dir.glob('*.JPEG')))
            
            for img_path in image_files:
                self.samples.append(str(img_path))
                self.targets.append(class_idx)
        
        # Apply subset if specified
        if subset_indices is not None:
            self.samples = [self.samples[i] for i in subset_indices]
            self.targets = [self.targets[i] for i in subset_indices]
        
        print(f"Loaded {len(self.samples)} images from {len(self.class_to_idx)} identities")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path = self.samples[idx]
        target = self.targets[idx]
        
        # Load image
        img = Image.open(img_path).convert('RGB')
        
        # Apply transformations
        if self.transform:
            img = self.transform(img)
        
        if self.target_transform:
            target = self.target_transform(target)
        
        return img, target


def create_facelandmark_benchmark(
    root_dir: str = '/home/dylee/data/data_fid/FaceID/FaceLandmark/data/IR/cropdata',
    n_experiences: Optional[int] = None,
    test_split: float = 0.2,
    seed: int = 42,
    image_size: Tuple[int, int] = (112, 112),
    use_cache: bool = True,
    cache_dir: str = '.cache/facelandmark',
    preload_to_memory: bool = False
) -> Tuple:
    """
    Create FaceLandmark benchmark for continual learning.
    
    Args:
        root_dir: Path to FaceLandmark cropdata directory
        n_experiences: Number of experiences (if None, use one per identity)
        test_split: Fraction of data to use for testing
        seed: Random seed for reproducibility
        image_size: Target image size (height, width)
        use_cache: Whether to cache dataset splits
        cache_dir: Directory for caching
        preload_to_memory: Whether to preload all images to memory
        
    Returns:
        Tuple of (benchmark, BenchmarkInfo)
    """
    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Define transforms (428x428 -> target_size)
    train_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Create cache directory if needed
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    # Cache file names
    cache_file = cache_path / f'facelandmark_split_n{n_experiences}_test{test_split}_seed{seed}.pt'
    
    if use_cache and cache_file.exists():
        print(f"Loading cached dataset splits from {cache_file}")
        cache_data = torch.load(cache_file)
        train_indices = cache_data['train_indices']
        test_indices = cache_data['test_indices']
        n_classes = cache_data['n_classes']
        actual_n_experiences = cache_data['n_experiences']
    else:
        # Create full dataset to get structure
        full_dataset = FaceLandmarkDataset(root_dir)
        n_classes = len(full_dataset.class_to_idx)
        
        # Determine number of experiences
        if n_experiences is None:
            actual_n_experiences = n_classes  # One experience per identity
        else:
            actual_n_experiences = min(n_experiences, n_classes)
        
        print(f"Creating {actual_n_experiences} experiences from {n_classes} identities")
        
        # Split data by class
        class_indices = defaultdict(list)
        for idx, target in enumerate(full_dataset.targets):
            class_indices[target].append(idx)
        
        # Split train/test for each class
        train_indices = []
        test_indices = []
        
        for class_idx in range(n_classes):
            indices = np.array(class_indices[class_idx])
            np.random.shuffle(indices)
            
            n_samples = len(indices)
            n_test = max(1, int(n_samples * test_split))
            
            test_indices.extend(indices[:n_test].tolist())
            train_indices.extend(indices[n_test:].tolist())
        
        # Save cache
        if use_cache:
            print(f"Saving dataset splits to {cache_file}")
            torch.save({
                'train_indices': train_indices,
                'test_indices': test_indices,
                'n_classes': n_classes,
                'n_experiences': actual_n_experiences
            }, cache_file)
    
    # Create train and test datasets
    train_dataset = FaceLandmarkDataset(
        root_dir, 
        transform=train_transform,
        subset_indices=train_indices
    )
    
    test_dataset = FaceLandmarkDataset(
        root_dir,
        transform=test_transform,
        subset_indices=test_indices
    )
    
    # Convert to Avalanche datasets
    train_dataset = as_classification_dataset(train_dataset)
    test_dataset = as_classification_dataset(test_dataset)
    
    # Create class order for experiences
    if actual_n_experiences == n_classes:
        # One experience per class
        classes_per_exp = [[i] for i in range(n_classes)]
    else:
        # Distribute classes across experiences
        classes = list(range(n_classes))
        np.random.shuffle(classes)
        classes_per_exp = np.array_split(classes, actual_n_experiences)
        classes_per_exp = [list(exp) for exp in classes_per_exp]
    
    # Create benchmark
    benchmark = nc_benchmark(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        n_experiences=actual_n_experiences,
        task_labels=False,
        seed=seed,
        class_ids_from_zero_in_each_exp=False,
        one_dataset_per_exp=False,
        class_ids_from_zero_from_first_exp=True,
        reproducibility_data=None
    )
    
    # Create BenchmarkInfo
    benchmark_info = BenchmarkInfo(
        num_classes=n_classes,
        image_size=image_size,
        channels=3,  # RGB images
        num_train=len(train_indices),
        num_test=len(test_indices),
        n_experiences=actual_n_experiences,
        class_names=full_dataset.idx_to_class if hasattr(full_dataset, 'idx_to_class') else None
    )
    
    print(f"Created FaceLandmark benchmark:")
    print(f"  - Classes: {n_classes}")
    print(f"  - Experiences: {actual_n_experiences}")
    print(f"  - Train samples: {len(train_indices)}")
    print(f"  - Test samples: {len(test_indices)}")
    print(f"  - Image size: {image_size}")
    
    return benchmark, benchmark_info


# For backward compatibility
def get_facelandmark_benchmark(*args, **kwargs):
    """Alias for create_facelandmark_benchmark."""
    return create_facelandmark_benchmark(*args, **kwargs)