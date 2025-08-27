"""Cached SmartEye dataset loader with efficient pre-splitting."""
import os
import pickle
import hashlib
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, TensorDataset
from avalanche.benchmarks import nc_benchmark
from avalanche.benchmarks.utils import as_classification_dataset
from ..utils.benchmark_info import BenchmarkInfo

class CachedSmartEyeDataset(Dataset):
    """SmartEye dataset with lazy loading (doesn't load all images into memory)."""
    
    def __init__(self, image_paths, labels, image_size=(112, 112), transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.image_size = image_size
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image on-demand
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load as grayscale IR image
        image = Image.open(img_path).convert('L')
        image = image.resize(self.image_size, Image.BILINEAR)
        
        # Convert to tensor and normalize
        image = np.array(image, dtype=np.float32) / 255.0
        image = torch.FloatTensor(image).unsqueeze(0)
        
        if self.transform:
            image = self.transform(image)
            
        return image, label


def get_cache_key(root_dir, use_cropdata, test_split, seed, min_samples_per_class):
    """Generate unique cache key based on parameters."""
    params = f"{root_dir}_{use_cropdata}_{test_split}_{seed}_{min_samples_per_class}"
    return hashlib.md5(params.encode()).hexdigest()


def create_smarteye_benchmark_cached(
    root_dir: str = "/home/dylee/data/data_fid/FaceID/ARM",
    use_cropdata: bool = True,
    n_experiences: int = 17,
    image_size: Tuple[int, int] = (112, 112),
    test_split: float = 0.2,
    seed: int = 42,
    min_samples_per_class: int = 10,
    cache_dir: str = ".smarteye_cache",
    use_cache: bool = True,
    preload_to_memory: bool = False  # New option for backward compatibility
):
    """
    Create SmartEye benchmark with caching support.
    
    This function implements a two-level caching system:
    1. Experiment-level cache: Reuses dataset objects across runs within same experiment
    2. Disk cache: Persists split indices across different experiments
    
    Args:
        root_dir: Path to face_dataset directory
        use_cropdata: If True, use cropdata; if False, use rawdata
        n_experiences: Number of experiences (fixed at 17 for SmartEye)
        image_size: Target image size
        test_split: Fraction of data for testing
        seed: Random seed for reproducibility
        min_samples_per_class: Minimum samples per identity
        cache_dir: Directory to store cached splits
        use_cache: Whether to use cached splits if available
        preload_to_memory: If True, loads all images to memory (original behavior)
                          If False, uses lazy loading (memory efficient)
    
    Returns:
        Avalanche NCScenario benchmark
    """
    
    # Simple disk-based caching only
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(exist_ok=True)
    
    # Generate cache key
    cache_key = get_cache_key(root_dir, use_cropdata, test_split, seed, min_samples_per_class)
    cache_file = cache_dir / f"smarteye_{cache_key}.pkl"
    
    # Check if cached version exists
    if use_cache and cache_file.exists():
        print(f"Loading cached dataset split from {cache_file}")
        with open(cache_file, 'rb') as f:
            cache_data = pickle.load(f)
            
        # Verify cache validity
        if cache_data['image_size'] != image_size:
            print(f"Cache has different image_size ({cache_data['image_size']} vs {image_size}), regenerating...")
        else:
            train_paths = cache_data['train_paths']
            train_labels = cache_data['train_labels']
            test_paths = cache_data['test_paths']
            test_labels = cache_data['test_labels']
            class_order = cache_data['class_order']
            
            print(f"Loaded cached split: {len(train_paths)} train, {len(test_paths)} test samples")
            
            # Create datasets based on memory preference
            if preload_to_memory:
                # Original behavior: load all images to memory
                print("Preloading all images to memory...")
                X_train, X_test = [], []
                
                for path in train_paths:
                    img = Image.open(path).convert('L')
                    img = img.resize(image_size, Image.BILINEAR)
                    img = np.array(img, dtype=np.float32) / 255.0
                    img = torch.FloatTensor(img).unsqueeze(0)
                    X_train.append(img)
                
                for path in test_paths:
                    img = Image.open(path).convert('L')
                    img = img.resize(image_size, Image.BILINEAR)
                    img = np.array(img, dtype=np.float32) / 255.0
                    img = torch.FloatTensor(img).unsqueeze(0)
                    X_test.append(img)
                
                X_train = torch.stack(X_train)
                X_test = torch.stack(X_test)
                
                train_dataset = TensorDataset(X_train, train_labels)
                test_dataset = TensorDataset(X_test, test_labels)
                
                # Add targets attribute
                train_dataset.targets = train_labels.tolist()
                test_dataset.targets = test_labels.tolist()
            else:
                # Memory efficient: lazy loading
                train_dataset = CachedSmartEyeDataset(train_paths, train_labels, image_size)
                test_dataset = CachedSmartEyeDataset(test_paths, test_labels, image_size)
                
                # Add targets attribute for Avalanche
                train_dataset.targets = train_labels.tolist()
                test_dataset.targets = test_labels.tolist()
            
            # Convert to classification datasets
            train_dataset = as_classification_dataset(train_dataset)
            test_dataset = as_classification_dataset(test_dataset)
            
            # Create benchmark with cached class order
            benchmark = nc_benchmark(
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                n_experiences=n_experiences,
                task_labels=False,
                shuffle=False,  # Use cached order
                fixed_class_order=class_order,
                class_ids_from_zero_in_each_exp=False
            )
            
            # Create BenchmarkInfo object
            info = BenchmarkInfo(
                num_classes=cache_data['n_classes'],
                num_samples=len(train_paths) + len(test_paths),
                num_train=len(train_paths),
                num_test=len(test_paths),
                image_size=image_size,
                channels=1,  # Grayscale IR images
                n_experiences=n_experiences,
                dataset_name='smarteye_crop' if use_cropdata else 'smarteye_raw',
                data_type='cropdata' if use_cropdata else 'rawdata'
            )
            
            return benchmark, info
    
    # If not cached or cache invalid, create new split
    print("Creating new dataset split...")
    
    # Build dataset structure
    root_path = Path(root_dir)
    data_dir = root_path / ("cropdata" if use_cropdata else "rawdata")
    
    # Collect all samples and labels
    samples = []
    labels = []
    class_to_idx = {}
    
    identity_folders = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    
    # Filter identities by minimum samples
    valid_identities = []
    for identity_folder in identity_folders:
        image_files = list(identity_folder.glob("*.png"))
        if len(image_files) >= min_samples_per_class:
            valid_identities.append(identity_folder)
    
    print(f"Found {len(valid_identities)} valid identities (>= {min_samples_per_class} samples)")
    
    for idx, identity_folder in enumerate(valid_identities):
        identity_name = identity_folder.name
        class_to_idx[identity_name] = idx
        
        image_files = sorted(list(identity_folder.glob("*.png")))
        for img_file in image_files:
            samples.append(str(img_file))  # Store as string for pickling
            labels.append(idx)
    
    labels = torch.LongTensor(labels)
    
    # Create train/test split
    train_indices = []
    test_indices = []
    
    num_classes = len(class_to_idx)
    for class_id in range(num_classes):
        class_indices = torch.where(labels == class_id)[0].numpy()
        
        # Shuffle indices for this class
        rng = np.random.RandomState(seed + class_id)
        rng.shuffle(class_indices)
        
        # Split
        n_train = int((1 - test_split) * len(class_indices))
        train_indices.extend(class_indices[:n_train])
        test_indices.extend(class_indices[n_train:])
    
    # Create train/test data
    train_paths = [samples[i] for i in train_indices]
    train_labels = labels[train_indices]
    test_paths = [samples[i] for i in test_indices]
    test_labels = labels[test_indices]
    
    # Generate random class order for experiences
    rng = np.random.RandomState(seed)
    class_order = list(range(num_classes))
    rng.shuffle(class_order)
    
    # Save cache
    if use_cache:
        cache_data = {
            'train_paths': train_paths,
            'train_labels': train_labels,
            'test_paths': test_paths,
            'test_labels': test_labels,
            'class_order': class_order,
            'image_size': image_size,
            'n_classes': num_classes
        }
        
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        print(f"Cached split saved to {cache_file}")
    
    # Create datasets based on memory preference
    if preload_to_memory:
        # Original behavior: load all images to memory
        print("Preloading all images to memory...")
        X_train, X_test = [], []
        
        for path in train_paths:
            img = Image.open(path).convert('L')
            img = img.resize(image_size, Image.BILINEAR)
            img = np.array(img, dtype=np.float32) / 255.0
            img = torch.FloatTensor(img).unsqueeze(0)
            X_train.append(img)
        
        for path in test_paths:
            img = Image.open(path).convert('L')
            img = img.resize(image_size, Image.BILINEAR)
            img = np.array(img, dtype=np.float32) / 255.0
            img = torch.FloatTensor(img).unsqueeze(0)
            X_test.append(img)
        
        X_train = torch.stack(X_train)
        X_test = torch.stack(X_test)
        
        train_dataset = TensorDataset(X_train, train_labels)
        test_dataset = TensorDataset(X_test, test_labels)
        
        # Add targets attribute
        train_dataset.targets = train_labels.tolist()
        test_dataset.targets = test_labels.tolist()
    else:
        # Memory efficient: lazy loading
        train_dataset = CachedSmartEyeDataset(train_paths, train_labels, image_size)
        test_dataset = CachedSmartEyeDataset(test_paths, test_labels, image_size)
        
        # Add targets attribute for Avalanche
        train_dataset.targets = train_labels.tolist()
        test_dataset.targets = test_labels.tolist()
    
    # Convert to classification datasets
    train_dataset = as_classification_dataset(train_dataset)
    test_dataset = as_classification_dataset(test_dataset)
    
    # Create benchmark
    benchmark = nc_benchmark(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        n_experiences=n_experiences,
        task_labels=False,
        shuffle=False,  # Use our generated order
        fixed_class_order=class_order,
        class_ids_from_zero_in_each_exp=False
    )
    
    print(f"Created benchmark: {benchmark.n_experiences} experiences")
    print(f"Train/test split: {len(train_paths)} train, {len(test_paths)} test")
    
    # Create BenchmarkInfo object
    info = BenchmarkInfo(
        num_classes=num_classes,
        num_samples=len(train_paths) + len(test_paths),
        num_train=len(train_paths),
        num_test=len(test_paths),
        image_size=image_size,
        channels=1,  # Grayscale IR images
        n_experiences=n_experiences,
        dataset_name='smarteye_crop' if use_cropdata else 'smarteye_raw',
        data_type='cropdata' if use_cropdata else 'rawdata'
    )
    
    return benchmark, info