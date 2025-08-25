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
from avalanche.benchmarks.scenarios.dataset_scenario import benchmark_from_datasets
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
    
    # Allow uneven splits for continual learning
    if n_experiences > num_classes:
        print(f"Warning: n_experiences ({n_experiences}) > num_classes ({num_classes})")
        print(f"Setting n_experiences = num_classes = {num_classes}")
        n_experiences = num_classes
    elif n_experiences < 1:
        n_experiences = 1
        
    print(f"Creating {n_experiences} experiences from {num_classes} classes")
    
    # Create benchmark with manual class splitting to handle uneven divisions
    if num_classes % n_experiences == 0:
        # Even split - use nc_benchmark
        benchmark = nc_benchmark(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            n_experiences=n_experiences,
            task_labels=False,
            shuffle=True,
            seed=seed,
            class_ids_from_zero_in_each_exp=False
        )
    else:
        # Uneven split - use ni_benchmark with manual experience creation
        print(f"Note: Uneven split - some experiences will have more classes than others")
        
        # Create class-to-experience mapping
        classes_per_exp = num_classes // n_experiences
        extra_classes = num_classes % n_experiences
        
        # Distribute classes across experiences
        experience_class_lists = []
        current_class = 0
        
        for exp_id in range(n_experiences):
            # Some experiences get an extra class
            exp_size = classes_per_exp + (1 if exp_id < extra_classes else 0)
            exp_classes = list(range(current_class, current_class + exp_size))
            experience_class_lists.append(exp_classes)
            current_class += exp_size
            
        print(f"Class distribution: {[len(exp_classes) for exp_classes in experience_class_lists]}")
        
        # Create experiences manually
        train_experiences = []
        test_experiences = []
        
        for exp_classes in experience_class_lists:
            # Filter train data for this experience
            train_mask = torch.tensor([label.item() in exp_classes for label in y_train])
            exp_train_X = X_train[train_mask]
            exp_train_y = y_train[train_mask]
            
            # Filter test data for this experience  
            test_mask = torch.tensor([label.item() in exp_classes for label in y_test])
            exp_test_X = X_test[test_mask]
            exp_test_y = y_test[test_mask]
            
            # Create experience datasets
            from torch.utils.data import TensorDataset
            exp_train_dataset = TensorDataset(exp_train_X, exp_train_y)
            exp_test_dataset = TensorDataset(exp_test_X, exp_test_y)
            
            # Add targets attribute
            exp_train_dataset.targets = exp_train_y.tolist()
            exp_test_dataset.targets = exp_test_y.tolist()
            
            # Convert to classification datasets
            exp_train_dataset = as_classification_dataset(exp_train_dataset)
            exp_test_dataset = as_classification_dataset(exp_test_dataset)
            
            train_experiences.append(exp_train_dataset)
            test_experiences.append(exp_test_dataset)
        
        # Create benchmark from experience list
        benchmark = benchmark_from_datasets(
            train_datasets=train_experiences,
            test_datasets=test_experiences,
            task_labels=False
        )
    
    print(f"Train/test split: {len(train_indices)} train, {len(test_indices)} test")
    
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