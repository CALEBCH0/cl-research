"""Dataset-agnostic utilities for creating controlled subsets for CL benchmarking."""
import numpy as np
from typing import Optional, List, Dict, Tuple, Any
import torch
from torch.utils.data import TensorDataset
from avalanche.benchmarks import nc_benchmark
from avalanche.benchmarks.utils import as_classification_dataset


class DatasetSubsetConfig:
    """Configuration for creating dataset subsets with controlled properties."""
    
    def __init__(
        self,
        target_classes: int,
        min_samples_per_class: int = 10,
        n_experiences: Optional[int] = None,
        selection_strategy: str = "most_samples",
        seed: int = 42
    ):
        """
        Args:
            target_classes: Desired number of classes in subset
            min_samples_per_class: Minimum samples required per class
            n_experiences: Number of CL experiences (must divide target_classes)
            selection_strategy: How to select classes ('most_samples', 'balanced', 'random')
            seed: Random seed for reproducibility
        """
        self.target_classes = target_classes
        self.min_samples_per_class = min_samples_per_class
        self.n_experiences = n_experiences
        self.selection_strategy = selection_strategy
        self.seed = seed
        

def analyze_dataset_classes(X, y) -> Dict[int, int]:
    """Analyze class distribution in dataset.
    
    Returns:
        Dict mapping class_id -> sample_count
    """
    if isinstance(y, torch.Tensor):
        y_np = y.numpy()
    else:
        y_np = np.array(y)
    
    unique_classes, counts = np.unique(y_np, return_counts=True)
    return dict(zip(unique_classes.tolist(), counts.tolist()))


def create_subset_with_target_classes(
    X, y, 
    subset_config: DatasetSubsetConfig,
    class_names: Optional[List[str]] = None
) -> Tuple[Any, Any, Dict[str, Any]]:
    """
    Create a subset with target number of classes.
    
    Args:
        X: Input data (numpy array or torch tensor)
        y: Labels (numpy array or torch tensor)
        subset_config: Configuration for subset creation
        class_names: Optional list of class names
        
    Returns:
        X_subset: Subset of input data
        y_subset: Subset of labels (remapped to 0...target_classes-1)
        info: Dictionary with subset information
    """
    # Analyze class distribution
    class_counts = analyze_dataset_classes(X, y)
    
    # Filter classes by minimum samples
    eligible_classes = [
        cls for cls, count in class_counts.items() 
        if count >= subset_config.min_samples_per_class
    ]
    
    print(f"Found {len(eligible_classes)} classes with {subset_config.min_samples_per_class}+ samples")
    
    # Check if we have enough classes
    if len(eligible_classes) < subset_config.target_classes:
        print(f"Warning: Only {len(eligible_classes)} classes available, "
              f"requested {subset_config.target_classes}")
        selected_classes = eligible_classes
    else:
        # Select subset of classes based on strategy
        rng = np.random.RandomState(subset_config.seed)
        
        if subset_config.selection_strategy == "most_samples":
            # Sort by sample count and take top classes
            sorted_classes = sorted(eligible_classes, 
                                  key=lambda c: class_counts[c], 
                                  reverse=True)
            selected_classes = sorted_classes[:subset_config.target_classes]
            
        elif subset_config.selection_strategy == "balanced":
            # Try to select classes with similar sample counts
            sorted_classes = sorted(eligible_classes, 
                                  key=lambda c: class_counts[c])
            # Take from middle outward for balance
            mid = len(sorted_classes) // 2
            selected_indices = []
            for i in range(subset_config.target_classes):
                if i % 2 == 0:
                    idx = mid + i // 2
                else:
                    idx = mid - (i + 1) // 2
                if 0 <= idx < len(sorted_classes):
                    selected_indices.append(idx)
            selected_classes = [sorted_classes[i] for i in selected_indices]
            
        else:  # random
            selected_classes = rng.choice(eligible_classes, 
                                        size=subset_config.target_classes,
                                        replace=False).tolist()
    
    selected_classes = sorted(selected_classes)
    
    # Create class mapping (old -> new)
    class_mapping = {old_id: new_id for new_id, old_id in enumerate(selected_classes)}
    
    # Select samples from chosen classes
    if isinstance(y, torch.Tensor):
        mask = torch.zeros(len(y), dtype=torch.bool)
        for cls in selected_classes:
            mask |= (y == cls)
        indices = torch.where(mask)[0]
    else:
        mask = np.zeros(len(y), dtype=bool)
        for cls in selected_classes:
            mask |= (y == cls)
        indices = np.where(mask)[0]
    
    # Create subset
    if isinstance(X, torch.Tensor):
        X_subset = X[indices]
        y_subset = y[indices]
        # Remap labels
        y_subset_remapped = torch.zeros_like(y_subset)
        for old_cls, new_cls in class_mapping.items():
            y_subset_remapped[y_subset == old_cls] = new_cls
        y_subset = y_subset_remapped
    else:
        X_subset = X[indices]
        y_subset = y[indices]
        # Remap labels
        y_subset_remapped = np.zeros_like(y_subset)
        for old_cls, new_cls in class_mapping.items():
            y_subset_remapped[y_subset == old_cls] = new_cls
        y_subset = y_subset_remapped
    
    # Prepare info
    info = {
        'num_classes': len(selected_classes),
        'num_samples': len(indices),
        'class_mapping': class_mapping,
        'selected_classes': selected_classes,
        'samples_per_class': {
            new_id: int(class_counts[old_id]) 
            for old_id, new_id in class_mapping.items()
        }
    }
    
    if class_names:
        info['selected_class_names'] = [class_names[i] for i in selected_classes]
    
    return X_subset, y_subset, info


def create_cl_benchmark_from_subset(
    X_train, y_train,
    X_test, y_test,
    subset_config: DatasetSubsetConfig,
    class_names: Optional[List[str]] = None
):
    """
    Create a complete CL benchmark with controlled subset.
    
    Returns:
        benchmark: Avalanche benchmark
        dataset_info: Information about the dataset
    """
    # Create train subset
    X_train_subset, y_train_subset, train_info = create_subset_with_target_classes(
        X_train, y_train, subset_config, class_names
    )
    
    # Create test subset (using same selected classes)
    selected_classes = train_info['selected_classes']
    class_mapping = train_info['class_mapping']
    
    # Filter test set to only include selected classes
    if isinstance(y_test, torch.Tensor):
        mask = torch.zeros(len(y_test), dtype=torch.bool)
        for cls in selected_classes:
            mask |= (y_test == cls)
        test_indices = torch.where(mask)[0]
        X_test_subset = X_test[test_indices]
        y_test_subset = y_test[test_indices]
        # Remap labels
        y_test_remapped = torch.zeros_like(y_test_subset)
        for old_cls, new_cls in class_mapping.items():
            y_test_remapped[y_test_subset == old_cls] = new_cls
        y_test_subset = y_test_remapped
    else:
        mask = np.zeros(len(y_test), dtype=bool)
        for cls in selected_classes:
            mask |= (y_test == cls)
        test_indices = np.where(mask)[0]
        X_test_subset = X_test[test_indices]
        y_test_subset = y_test[test_indices]
        # Remap labels
        y_test_remapped = np.zeros_like(y_test_subset)
        for old_cls, new_cls in class_mapping.items():
            y_test_remapped[y_test_subset == old_cls] = new_cls
        y_test_subset = y_test_remapped
    
    # Convert to torch if needed
    if not isinstance(X_train_subset, torch.Tensor):
        X_train_subset = torch.FloatTensor(X_train_subset)
        y_train_subset = torch.LongTensor(y_train_subset)
    if not isinstance(X_test_subset, torch.Tensor):
        X_test_subset = torch.FloatTensor(X_test_subset)
        y_test_subset = torch.LongTensor(y_test_subset)
    
    # Create datasets
    train_dataset = TensorDataset(X_train_subset, y_train_subset)
    test_dataset = TensorDataset(X_test_subset, y_test_subset)
    
    train_dataset.targets = y_train_subset.tolist()
    test_dataset.targets = y_test_subset.tolist()
    
    train_dataset = as_classification_dataset(train_dataset)
    test_dataset = as_classification_dataset(test_dataset)
    
    # Determine n_experiences
    n_classes = train_info['num_classes']
    n_experiences = subset_config.n_experiences
    
    if n_experiences is None:
        # Auto-select reasonable n_experiences
        divisors = [i for i in range(2, min(21, n_classes + 1)) if n_classes % i == 0]
        if divisors:
            # Prefer around 10 experiences
            n_experiences = min(divisors, key=lambda x: abs(x - 10))
        else:
            n_experiences = n_classes  # Fallback: 1 class per experience
        print(f"Auto-selected n_experiences: {n_experiences}")
    elif n_classes % n_experiences != 0:
        # Find closest valid divisor
        divisors = [i for i in range(1, n_classes + 1) if n_classes % i == 0]
        old_n_exp = n_experiences
        n_experiences = min(divisors, key=lambda x: abs(x - old_n_exp))
        print(f"Warning: {n_classes} classes cannot be evenly split into {old_n_exp} experiences.")
        print(f"Using closest valid n_experiences instead: {n_experiences}")
    
    # Create benchmark
    benchmark = nc_benchmark(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        n_experiences=n_experiences,
        task_labels=False,
        seed=subset_config.seed,
        class_ids_from_zero_in_each_exp=False
    )
    
    # Dataset info
    dataset_info = {
        'num_classes': n_classes,
        'num_train_samples': len(X_train_subset),
        'num_test_samples': len(X_test_subset),
        'n_experiences': n_experiences,
        'classes_per_exp': n_classes // n_experiences,
        'class_mapping': class_mapping,
        'subset_config': subset_config.__dict__,
        'train_samples_per_class': train_info['samples_per_class'],
        'test_samples_per_class': analyze_dataset_classes(X_test_subset, y_test_subset)
    }
    
    if 'selected_class_names' in train_info:
        dataset_info['selected_class_names'] = train_info['selected_class_names']
    
    return benchmark, dataset_info