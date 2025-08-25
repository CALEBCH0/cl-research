"""SmartEye IR Face Dataset loader for Avalanche benchmarks."""
import os
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from avalanche.benchmarks import nc_benchmark
from avalanche.benchmarks.utils import as_classification_dataset
from collections import defaultdict


class SmartEyeFaceDataset(Dataset):
    """Dataset for SmartEye IR face images."""
    
    def __init__(self, root_dir: str, use_cropdata: bool = True, 
                 image_size: Tuple[int, int] = (112, 112),
                 transform=None):
        """
        Args:
            root_dir: Path to face_dataset directory
            use_cropdata: If True, use cropdata; if False, use rawdata
            image_size: Target image size (height, width)
            transform: Optional transform to apply
        """
        self.root_dir = Path(root_dir)
        self.use_cropdata = use_cropdata
        self.image_size = image_size
        self.transform = transform
        
        # Choose data directory
        if use_cropdata:
            self.data_dir = self.root_dir / "cropdata"
        else:
            self.data_dir = self.root_dir / "rawdata"
            
        # Load all images and labels
        self.samples = []
        self.labels = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        
        # Get all identity folders
        identity_folders = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])
        
        for idx, identity_folder in enumerate(identity_folders):
            identity_name = identity_folder.name
            self.class_to_idx[identity_name] = idx
            self.idx_to_class[idx] = identity_name
            
            # Get all images for this identity
            image_files = sorted(list(identity_folder.glob("*.png")))
            
            for img_file in image_files:
                self.samples.append(img_file)
                self.labels.append(idx)
                
        self.labels = torch.LongTensor(self.labels)
        print(f"Loaded {len(self.samples)} images from {len(identity_folders)} identities")
    
    # Print samples per identity for debugging
    samples_per_identity = {}
    for idx, identity_folder in enumerate(identity_folders):
        identity_name = identity_folder.name
        count = sum(1 for label in self.labels if label == idx)
        samples_per_identity[identity_name] = count
        
    print("Samples per identity:")
    for identity, count in sorted(samples_per_identity.items()):
        print(f"  {identity}: {count} images")
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.labels[idx]
        
        # Load image as grayscale (IR images)
        image = Image.open(img_path).convert('L')
        
        # Resize to target size
        image = image.resize(self.image_size, Image.BILINEAR)
        
        # Convert to tensor and normalize to [0, 1]
        image = np.array(image, dtype=np.float32) / 255.0
        image = torch.FloatTensor(image).unsqueeze(0)  # Add channel dimension
        
        if self.transform:
            image = self.transform(image)
            
        return image, label


def create_smarteye_benchmark(
    root_dir: str = "/Users/calebcho/data/face_dataset",
    use_cropdata: bool = True,
    n_experiences: int = 5,
    image_size: Tuple[int, int] = (112, 112),
    test_split: float = 0.2,
    seed: int = 42,
    min_samples_per_class: int = 10
):
    """
    Create an Avalanche benchmark from SmartEye face dataset.
    
    Args:
        root_dir: Path to face_dataset directory
        use_cropdata: If True, use cropdata; if False, use rawdata
        n_experiences: Number of experiences to split classes into
        image_size: Target image size
        test_split: Fraction of data to use for testing
        seed: Random seed
        min_samples_per_class: Minimum samples required per class
        
    Returns:
        benchmark: Avalanche benchmark
        info: Dictionary with dataset information
    """
    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create dataset
    dataset = SmartEyeFaceDataset(
        root_dir=root_dir,
        use_cropdata=use_cropdata,
        image_size=image_size
    )
    
    # Filter classes with too few samples
    samples_per_class = defaultdict(int)
    for label in dataset.labels:
        samples_per_class[label.item()] += 1
    
    valid_classes = [cls for cls, count in samples_per_class.items() 
                     if count >= min_samples_per_class]
    
    if len(valid_classes) < len(samples_per_class):
        print(f"Filtering classes: keeping {len(valid_classes)}/{len(samples_per_class)} "
              f"classes with >= {min_samples_per_class} samples")
        
        # Filter samples
        valid_indices = [i for i, label in enumerate(dataset.labels) 
                        if label.item() in valid_classes]
        
        dataset.samples = [dataset.samples[i] for i in valid_indices]
        dataset.labels = dataset.labels[valid_indices]
        
        # Remap class indices
        old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted(valid_classes))}
        dataset.labels = torch.LongTensor([old_to_new[label.item()] for label in dataset.labels])
        
        # Update class mappings
        new_class_to_idx = {}
        new_idx_to_class = {}
        for old_idx in sorted(valid_classes):
            identity_name = dataset.idx_to_class[old_idx]
            new_idx = old_to_new[old_idx]
            new_class_to_idx[identity_name] = new_idx
            new_idx_to_class[new_idx] = identity_name
        
        dataset.class_to_idx = new_class_to_idx
        dataset.idx_to_class = new_idx_to_class
    
    num_classes = len(dataset.class_to_idx)
    
    # Convert dataset to tensors
    all_images = []
    for sample_path in dataset.samples:
        # Load image as grayscale (IR images)
        image = Image.open(sample_path).convert('L')
        # Resize to target size
        image = image.resize(dataset.image_size, Image.BILINEAR)
        # Convert to tensor and normalize to [0, 1]
        image = np.array(image, dtype=np.float32) / 255.0
        image = torch.FloatTensor(image).unsqueeze(0)  # Add channel dimension
        all_images.append(image)
    
    X = torch.stack(all_images)  # Shape: (n_samples, 1, height, width)
    y = dataset.labels  # Shape: (n_samples,)
    
    # Create train/test split maintaining class balance
    train_indices = []
    test_indices = []
    
    for class_id in range(num_classes):
        class_indices = torch.where(y == class_id)[0].numpy()
        n_class_samples = len(class_indices)
        
        # Shuffle indices for this class
        rng = np.random.RandomState(seed + class_id)
        rng.shuffle(class_indices)
        
        # Split
        n_train = int((1 - test_split) * n_class_samples)
        train_indices.extend(class_indices[:n_train])
        test_indices.extend(class_indices[n_train:])
    
    # Create train/test tensors
    train_indices = torch.tensor(train_indices)
    test_indices = torch.tensor(test_indices)
    
    X_train = X[train_indices]
    y_train = y[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]
    
    # Create TensorDatasets
    from torch.utils.data import TensorDataset
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    # Add targets attribute for Avalanche
    train_dataset.targets = y_train.tolist()
    test_dataset.targets = y_test.tolist()
    
    # Convert to classification datasets
    train_dataset = as_classification_dataset(train_dataset)
    test_dataset = as_classification_dataset(test_dataset)
    
    # Find valid n_experiences
    valid_n_experiences = []
    for n in range(1, num_classes + 1):
        if num_classes % n == 0:
            valid_n_experiences.append(n)
    
    if n_experiences not in valid_n_experiences:
        old_n_experiences = n_experiences
        # Find closest valid value
        n_experiences = min(valid_n_experiences, 
                           key=lambda x: abs(x - old_n_experiences))
        print(f"Warning: {num_classes} classes cannot be evenly split into "
              f"{old_n_experiences} experiences.")
        print(f"Valid options: {valid_n_experiences}")
        print(f"Using closest valid n_experiences instead: {n_experiences}")
    
    # Create benchmark using nc_benchmark (New Classes benchmark)
    benchmark = nc_benchmark(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        n_experiences=n_experiences,
        task_labels=False,
        shuffle=True,
        seed=seed,
        class_ids_from_zero_in_each_exp=False
    )
    
    # Print train/test split details for debugging
    print(f"\nTrain/Test Split Details:")
    print(f"  Total samples: {len(X)}")
    print(f"  Train samples: {len(train_indices)} ({len(train_indices)/len(X)*100:.1f}%)")
    print(f"  Test samples: {len(test_indices)} ({len(test_indices)/len(X)*100:.1f}%)")
    
    # Check train/test split per class
    print(f"\nTrain/Test split per identity:")
    for class_id in range(num_classes):
        class_name = dataset.idx_to_class[class_id]
        train_count = sum(1 for idx in train_indices if y[idx] == class_id)
        test_count = sum(1 for idx in test_indices if y[idx] == class_id)
        total_count = train_count + test_count
        print(f"  {class_name}: {train_count} train, {test_count} test (total: {total_count})")
    
    # Create info dictionary
    info = {
        'num_classes': num_classes,
        'num_samples': len(X),
        'num_train': len(train_indices),
        'num_test': len(test_indices),
        'image_size': image_size,
        'channels': 1,  # Grayscale IR images
        'input_size': image_size[0] * image_size[1],
        'n_experiences': n_experiences,
        'class_names': dataset.idx_to_class,
        'data_type': 'cropdata' if use_cropdata else 'rawdata'
    }
    
    return benchmark, info


def create_smarteye_controlled_benchmark(
    subset_config: 'DatasetSubsetConfig',
    root_dir: str = "/Users/calebcho/data/face_dataset",
    use_cropdata: bool = True,
    image_size: Tuple[int, int] = (112, 112),
    test_split: float = 0.2
):
    """Create SmartEye benchmark with specific subset configuration."""
    # Similar to create_lfw_controlled_benchmark but for SmartEye dataset
    # This allows using the subset configuration system
    
    # For now, just use the standard benchmark creation
    # You can extend this to support subset configurations if needed
    return create_smarteye_benchmark(
        root_dir=root_dir,
        use_cropdata=use_cropdata,
        n_experiences=subset_config.n_experiences or 5,
        image_size=image_size,
        test_split=test_split,
        seed=subset_config.seed
    )


# Register the dataset name
SMARTEYE_DATASETS = {
    'smarteye_crop': {
        'use_cropdata': True,
        'min_samples_per_class': 10
    },
    'smarteye_raw': {
        'use_cropdata': False,
        'min_samples_per_class': 10
    }
}


def get_smarteye_config(dataset_name: str) -> Dict[str, Any]:
    """Get configuration for a SmartEye dataset variant."""
    if dataset_name not in SMARTEYE_DATASETS:
        raise ValueError(f"Unknown SmartEye dataset: {dataset_name}")
    return SMARTEYE_DATASETS[dataset_name]