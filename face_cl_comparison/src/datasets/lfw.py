"""LFW (Labeled Faces in the Wild) dataset for continual learning."""
import numpy as np
import torch
from torch.utils.data import TensorDataset
from sklearn.datasets import fetch_lfw_people
from avalanche.benchmarks import nc_benchmark
from avalanche.benchmarks.utils import as_classification_dataset
from pathlib import Path
from src.datasets.subset_utils import DatasetSubsetConfig, create_cl_benchmark_from_subset
from src.utils.benchmark_info import BenchmarkInfo


def create_lfw_benchmark(n_experiences=10, min_faces_per_person=20, 
                        image_size=(64, 64), seed=42, test_split=0.2):
    """Create LFW benchmark for continual learning.
    
    Args:
        n_experiences: Number of CL tasks/experiences
        min_faces_per_person: Minimum images per identity to include
        image_size: Target size for images (height, width)
        seed: Random seed for reproducibility
        test_split: Fraction of data for test set
    
    Returns:
        benchmark: Avalanche benchmark
        dataset_info: Dict with dataset information
    """
    print(f"\nLoading LFW dataset (min {min_faces_per_person} faces per person)...")
    
    # Get sklearn data home directory
    from sklearn.datasets import get_data_home
    data_home = get_data_home()
    lfw_home = Path(data_home) / 'lfw_home'
    
    if not lfw_home.exists():
        print(f"First time loading LFW - downloading dataset (~200MB)...")
        print(f"Dataset will be cached in: {lfw_home}")
        print("This is a one-time download. Future runs will use the cached version.")
    else:
        print(f"Using cached LFW dataset from: {lfw_home}")
    
    # Fetch LFW data
    # Note: sklearn's fetch_lfw_people expects resize as a float ratio, not tuple
    # The original LFW images are 250x250, but the default output is 125x94
    # We'll skip sklearn's resize and do it ourselves for better control
    
    print("Fetching LFW data...")
    lfw_people = fetch_lfw_people(
        min_faces_per_person=min_faces_per_person,
        resize=None,  # Don't resize with sklearn, we'll do it ourselves
        color=False,  # Grayscale for consistency with Olivetti
        funneled=True,  # Use aligned faces
        download_if_missing=True,
        return_X_y=False
    )
    
    # Get data
    X = lfw_people.images  # Shape: (n_samples, height, width)
    y = lfw_people.target  # Shape: (n_samples,)
    
    n_classes = len(lfw_people.target_names)
    n_samples = len(X)
    
    print(f"Loaded {n_samples} images of {n_classes} people")
    print(f"Image shape: {X[0].shape} (requested: {image_size})")
    print(f"People included: {lfw_people.target_names[:5]}... ({n_classes} total)")
    
    # Convert to torch tensors
    X = torch.FloatTensor(X).unsqueeze(1)  # Add channel dimension
    y = torch.LongTensor(y)
    
    # Note: sklearn's fetch_lfw_people already returns normalized data in [0, 1] range
    # Do NOT divide by 255 again!
    
    # Always resize to target size since we're not using sklearn's resize
    import torch.nn.functional as F
    X = F.interpolate(X, size=image_size, mode='bilinear', align_corners=False)
    print(f"Resized images from {lfw_people.images[0].shape} to {image_size}")
    
    # Split train/test per class to ensure all classes in both sets
    train_indices = []
    test_indices = []
    
    for class_id in range(n_classes):
        # Get indices for this class
        class_indices = torch.where(y == class_id)[0].numpy()
        n_class_samples = len(class_indices)
        
        # Shuffle indices for this class
        rng = np.random.RandomState(seed + class_id)
        rng.shuffle(class_indices)
        
        # Split
        n_train = int((1 - test_split) * n_class_samples)
        train_indices.extend(class_indices[:n_train])
        test_indices.extend(class_indices[n_train:])
    
    # Convert to tensors
    train_indices = torch.tensor(train_indices)
    test_indices = torch.tensor(test_indices)
    
    # Shuffle indices
    train_perm = torch.randperm(len(train_indices), generator=torch.Generator().manual_seed(seed))
    test_perm = torch.randperm(len(test_indices), generator=torch.Generator().manual_seed(seed))
    
    train_indices = train_indices[train_perm]
    test_indices = test_indices[test_perm]
    
    # Create datasets
    train_dataset = TensorDataset(X[train_indices], y[train_indices])
    test_dataset = TensorDataset(X[test_indices], y[test_indices])
    
    # Add targets attribute for Avalanche
    train_dataset.targets = y[train_indices].tolist()
    test_dataset.targets = y[test_indices].tolist()
    
    # Convert to classification datasets
    train_dataset = as_classification_dataset(train_dataset)
    test_dataset = as_classification_dataset(test_dataset)
    
    # Adjust n_experiences to be a valid divisor of n_classes
    valid_n_experiences = n_experiences
    if n_classes % n_experiences != 0:
        # Find the closest valid divisor
        divisors = [i for i in range(1, n_classes + 1) if n_classes % i == 0]
        # Find divisor closest to requested n_experiences
        valid_n_experiences = min(divisors, key=lambda x: abs(x - n_experiences))
        print(f"Warning: {n_classes} classes cannot be evenly split into {n_experiences} experiences.")
        print(f"Using closest valid n_experiences instead: {valid_n_experiences}")
    
    # Create CL benchmark
    benchmark = nc_benchmark(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        n_experiences=valid_n_experiences,
        task_labels=False,
        seed=seed,
        class_ids_from_zero_in_each_exp=False  # Use global class IDs
    )
    
    # Create BenchmarkInfo object
    benchmark_info = BenchmarkInfo(
        num_classes=n_classes,
        image_size=image_size,
        channels=1,  # LFW is grayscale
        num_train=len(train_indices),
        num_test=len(test_indices),
        n_experiences=valid_n_experiences,
        class_names={i: name for i, name in enumerate(lfw_people.target_names)}
    )
    
    return benchmark, benchmark_info


# Cache for loaded LFW data to avoid reloading
_lfw_cache = {}

def _get_cached_lfw_data(min_faces_per_person, image_size):
    """Get cached LFW data or load if not cached."""
    cache_key = (min_faces_per_person, image_size)
    if cache_key not in _lfw_cache:
        benchmark, info = create_lfw_benchmark(
            n_experiences=1,  # Load as single experience
            min_faces_per_person=min_faces_per_person,
            image_size=image_size,
            seed=0  # Fixed seed for caching
        )
        _lfw_cache[cache_key] = (benchmark, info)
    return _lfw_cache[cache_key]

def create_lfw_subset_benchmark(n_identities=100, n_experiences=10, 
                               min_faces_per_person=20, image_size=(64, 64), 
                               seed=42):
    """Create a subset of LFW with specified number of identities.
    
    Useful for controlled experiments with specific number of classes.
    """
    # Use cached data if available
    benchmark_full, info_full = _get_cached_lfw_data(min_faces_per_person, image_size)
    
    # Get the data
    train_data = benchmark_full.train_stream[0].dataset
    test_data = benchmark_full.test_stream[0].dataset
    
    # Check if we have enough classes
    available_classes = info_full['num_classes']
    if n_identities > available_classes:
        print(f"Warning: Requested {n_identities} identities but only {available_classes} available.")
        print(f"Using all {available_classes} identities instead.")
        n_identities = available_classes
    
    # Select subset of identities
    all_classes = list(range(available_classes))
    rng = np.random.RandomState(seed)
    selected_classes = rng.choice(all_classes, size=n_identities, replace=False)
    selected_classes = sorted(selected_classes)
    
    # Filter datasets
    train_indices = [i for i, target in enumerate(train_data.targets) 
                    if target in selected_classes]
    test_indices = [i for i, target in enumerate(test_data.targets) 
                   if target in selected_classes]
    
    # Remap class IDs to 0...n_identities-1
    class_mapping = {old_id: new_id for new_id, old_id in enumerate(selected_classes)}
    
    # Create subset datasets
    train_X = torch.stack([train_data[i][0] for i in train_indices])
    train_y = torch.tensor([class_mapping[train_data.targets[i]] for i in train_indices])
    
    test_X = torch.stack([test_data[i][0] for i in test_indices])
    test_y = torch.tensor([class_mapping[test_data.targets[i]] for i in test_indices])
    
    # Create new datasets
    train_subset = TensorDataset(train_X, train_y)
    test_subset = TensorDataset(test_X, test_y)
    
    train_subset.targets = train_y.tolist()
    test_subset.targets = test_y.tolist()
    
    train_subset = as_classification_dataset(train_subset)
    test_subset = as_classification_dataset(test_subset)
    
    # Adjust n_experiences to be a valid divisor of n_identities
    valid_n_experiences = n_experiences
    if n_identities % n_experiences != 0:
        # Find the closest valid divisor
        divisors = [i for i in range(1, n_identities + 1) if n_identities % i == 0]
        # Find divisor closest to requested n_experiences
        valid_n_experiences = min(divisors, key=lambda x: abs(x - n_experiences))
        print(f"Warning: {n_identities} classes cannot be evenly split into {n_experiences} experiences.")
        print(f"Using closest valid n_experiences instead: {valid_n_experiences}")
    
    # Create benchmark with subset
    benchmark = nc_benchmark(
        train_dataset=train_subset,
        test_dataset=test_subset,
        n_experiences=valid_n_experiences,
        task_labels=False,
        seed=seed,
        class_ids_from_zero_in_each_exp=False
    )
    
    # Create BenchmarkInfo object
    benchmark_info = BenchmarkInfo(
        num_classes=n_identities,
        image_size=image_size,
        channels=1,  # LFW is grayscale
        num_train=len(train_indices),
        num_test=len(test_indices),
        n_experiences=valid_n_experiences,
        class_names={i: info_full.class_names[orig_i] for i, orig_i in enumerate(selected_classes)}
    )
    
    return benchmark, benchmark_info


def create_lfw_controlled_benchmark(
    subset_config: DatasetSubsetConfig,
    image_size=(64, 64),
    test_split=0.2
):
    """
    Create LFW benchmark with controlled subset using DatasetSubsetConfig.
    
    This is the new preferred way to create LFW benchmarks with specific
    number of classes and quality guarantees.
    
    Args:
        subset_config: Configuration for subset creation
        image_size: Target size for images (height, width)
        test_split: Fraction of data for test set
        
    Returns:
        benchmark: Avalanche benchmark
        dataset_info: Dict with dataset information
    """
    print(f"\nCreating controlled LFW benchmark:")
    print(f"  Target classes: {subset_config.target_classes}")
    print(f"  Min samples per class: {subset_config.min_samples_per_class}")
    print(f"  Selection strategy: {subset_config.selection_strategy}")
    
    # Load LFW data with quality threshold
    from sklearn.datasets import get_data_home
    data_home = get_data_home()
    lfw_home = Path(data_home) / 'lfw_home'
    
    if not lfw_home.exists():
        print(f"First time loading LFW - downloading dataset (~200MB)...")
        print(f"Dataset will be cached in: {lfw_home}")
    else:
        print(f"Using cached LFW dataset from: {lfw_home}")
    
    print("Fetching LFW data...")
    lfw_people = fetch_lfw_people(
        min_faces_per_person=subset_config.min_samples_per_class,
        resize=None,  # We'll resize ourselves
        color=False,
        funneled=True,
        download_if_missing=True,
        return_X_y=False
    )
    
    # Get data
    X = lfw_people.images  # Shape: (n_samples, height, width)
    y = lfw_people.target  # Shape: (n_samples,)
    class_names = lfw_people.target_names
    
    print(f"Loaded {len(X)} images of {len(class_names)} people")
    
    # Add channel dimension (data is already normalized by sklearn)
    X = torch.FloatTensor(X).unsqueeze(1)  # Add channel dimension
    # Note: sklearn's fetch_lfw_people already returns normalized data in [0, 1] range
    
    # Resize if needed
    import torch.nn.functional as F
    X = F.interpolate(X, size=image_size, mode='bilinear', align_corners=False)
    
    # Convert labels to torch
    y = torch.LongTensor(y)
    
    # Split into train/test maintaining class balance
    train_indices = []
    test_indices = []
    
    for class_id in range(len(class_names)):
        class_indices = torch.where(y == class_id)[0].numpy()
        n_class_samples = len(class_indices)
        
        # Shuffle indices for this class
        rng = np.random.RandomState(subset_config.seed + class_id)
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
    
    # Create controlled benchmark using subset utils
    benchmark, dataset_info = create_cl_benchmark_from_subset(
        X_train, y_train,
        X_test, y_test,
        subset_config,
        class_names=class_names.tolist()
    )
    
    # Convert dict info to BenchmarkInfo
    benchmark_info = BenchmarkInfo(
        num_classes=dataset_info['num_classes'],
        image_size=image_size,
        channels=1,  # LFW is grayscale  
        num_train=dataset_info['num_train_samples'],
        num_test=dataset_info['num_test_samples'],
        n_experiences=dataset_info['n_experiences'],
        class_names=dataset_info.get('selected_class_names', {})
    )
    
    print(f"\nCreated benchmark with:")
    print(f"  Classes: {benchmark_info.num_classes} "
          f"(selected from {len(class_names)})")
    print(f"  Experiences: {benchmark_info.n_experiences} "
          f"({benchmark_info.num_classes // benchmark_info.n_experiences} classes per exp)")
    print(f"  Train samples: {benchmark_info.num_train}")
    print(f"  Test samples: {benchmark_info.num_test}")
    
    return benchmark, benchmark_info