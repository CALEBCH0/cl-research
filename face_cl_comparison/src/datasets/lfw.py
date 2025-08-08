"""LFW (Labeled Faces in the Wild) dataset for continual learning."""
import numpy as np
import torch
from torch.utils.data import TensorDataset
from sklearn.datasets import fetch_lfw_people
from avalanche.benchmarks import nc_benchmark
from avalanche.benchmarks.utils import as_classification_dataset
from pathlib import Path


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
    print(f"Loading LFW dataset (min {min_faces_per_person} faces per person)...")
    
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
    # Default LFW images are 250x250, so we calculate the ratio
    resize_ratio = image_size[0] / 250.0  # Assuming square images
    
    print("Fetching LFW data...")
    lfw_people = fetch_lfw_people(
        min_faces_per_person=min_faces_per_person,
        resize=resize_ratio,
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
    
    # Normalize to [0, 1]
    X = X / 255.0
    
    # Ensure exact size if needed (sklearn might give slightly different size)
    if X.shape[2:] != image_size:
        import torch.nn.functional as F
        X = F.interpolate(X, size=image_size, mode='bilinear', align_corners=False)
        print(f"Resized images to exact size: {image_size}")
    
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
    
    # Create CL benchmark
    benchmark = nc_benchmark(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        n_experiences=n_experiences,
        task_labels=False,
        seed=seed,
        class_ids_from_zero_in_each_exp=False  # Use global class IDs
    )
    
    # Dataset info
    dataset_info = {
        'num_classes': n_classes,
        'num_train_samples': len(train_indices),
        'num_test_samples': len(test_indices),
        'image_shape': (1, image_size[0], image_size[1]),  # (C, H, W)
        'people_names': lfw_people.target_names.tolist()
    }
    
    return benchmark, dataset_info


def create_lfw_subset_benchmark(n_identities=100, n_experiences=10, 
                               min_faces_per_person=20, image_size=(64, 64), 
                               seed=42):
    """Create a subset of LFW with specified number of identities.
    
    Useful for controlled experiments with specific number of classes.
    """
    # First load full LFW
    benchmark_full, info_full = create_lfw_benchmark(
        n_experiences=1,  # Load as single experience first
        min_faces_per_person=min_faces_per_person,
        image_size=image_size,
        seed=seed
    )
    
    # Get the data
    train_data = benchmark_full.train_stream[0].dataset
    test_data = benchmark_full.test_stream[0].dataset
    
    # Select subset of identities
    all_classes = list(range(info_full['num_classes']))
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
    
    # Create benchmark with subset
    benchmark = nc_benchmark(
        train_dataset=train_subset,
        test_dataset=test_subset,
        n_experiences=n_experiences,
        task_labels=False,
        seed=seed,
        class_ids_from_zero_in_each_exp=False
    )
    
    # Updated dataset info
    dataset_info = {
        'num_classes': n_identities,
        'num_train_samples': len(train_indices),
        'num_test_samples': len(test_indices),
        'image_shape': (1, image_size[0], image_size[1]),
        'selected_people': [info_full['people_names'][i] for i in selected_classes]
    }
    
    return benchmark, dataset_info