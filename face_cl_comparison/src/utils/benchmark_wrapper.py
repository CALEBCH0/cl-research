"""Wrapper functions to ensure all benchmarks return (benchmark, BenchmarkInfo) format."""

from .benchmark_info import BenchmarkInfo
from avalanche.benchmarks.classic import SplitMNIST, SplitCIFAR10, SplitCIFAR100


def wrap_avalanche_benchmark(benchmark, dataset_name: str = "unknown", **kwargs):
    """
    Wrap any Avalanche benchmark to return (benchmark, BenchmarkInfo) format.
    
    Args:
        benchmark: Avalanche benchmark object
        dataset_name: Name of the dataset
        **kwargs: Additional metadata
        
    Returns:
        (benchmark, BenchmarkInfo) tuple
    """
    
    # Extract info from benchmark
    try:
        # Get training stream info
        train_stream = benchmark.train_stream
        test_stream = benchmark.test_stream
        
        # Calculate totals
        total_train = sum(len(exp.dataset) for exp in train_stream)
        total_test = sum(len(exp.dataset) for exp in test_stream)
        n_experiences = len(train_stream)
        
        # Get sample info from first experience
        first_exp = train_stream[0]
        first_sample = first_exp.dataset[0]
        if isinstance(first_sample, (tuple, list)) and len(first_sample) >= 2:
            sample, _ = first_sample
        else:
            sample = first_sample
        
        if hasattr(sample, 'shape'):
            if len(sample.shape) == 3:  # (C, H, W)
                channels, height, width = sample.shape
                image_size = (height, width)
            elif len(sample.shape) == 2:  # (H, W) - grayscale
                height, width = sample.shape
                channels = 1
                image_size = (height, width)
            else:
                # Fallback
                channels = 1
                image_size = (28, 28)
        else:
            # Fallback for other data types
            channels = 1
            image_size = (28, 28)
        
        # Try to get number of classes
        num_classes = 0
        if hasattr(benchmark, 'n_classes'):
            num_classes = benchmark.n_classes
        else:
            # Estimate from experiences
            all_classes = set()
            for exp in train_stream:
                if hasattr(exp, 'classes_in_this_experience'):
                    all_classes.update(exp.classes_in_this_experience)
            num_classes = len(all_classes) if all_classes else 10  # fallback
        
    except Exception as e:
        # Fallback values if extraction fails
        print(f"Warning: Could not extract benchmark info ({e}), using defaults")
        total_train = 50000
        total_test = 10000
        n_experiences = 5
        channels = 1
        if dataset_name == 'mnist':
            image_size = (28, 28)
            num_classes = 10
        elif dataset_name == 'cifar10':
            image_size = (32, 32)
            num_classes = 10
            channels = 3
        elif dataset_name == 'cifar100':
            image_size = (32, 32)
            num_classes = 100
            channels = 3
        else:
            image_size = (28, 28)
            num_classes = 10
    
    # Filter kwargs to avoid conflicts
    filtered_kwargs = {k: v for k, v in kwargs.items() 
                      if k not in ['num_classes', 'num_samples', 'num_train', 'num_test', 
                                   'image_size', 'channels', 'n_experiences', 'dataset_name']}
    
    # Create BenchmarkInfo object
    info = BenchmarkInfo(
        num_classes=num_classes,
        num_samples=total_train + total_test,
        num_train=total_train,
        num_test=total_test,
        image_size=image_size,
        channels=channels,
        n_experiences=n_experiences,
        dataset_name=dataset_name,
        **filtered_kwargs
    )
    
    return benchmark, info


def create_mnist_benchmark(**kwargs):
    """Create MNIST benchmark with unified return format."""
    try:
        benchmark = SplitMNIST(
            n_experiences=kwargs.get('n_experiences', 5),
            seed=kwargs.get('seed', 42)
        )
        return wrap_avalanche_benchmark(benchmark, "mnist", **kwargs)
    except Exception as e:
        print(f"Error creating MNIST benchmark: {e}")
        raise


def create_cifar10_benchmark(**kwargs):
    """Create CIFAR-10 benchmark with unified return format.""" 
    benchmark = SplitCIFAR10(
        n_experiences=kwargs.get('n_experiences', 5),
        seed=kwargs.get('seed', 42)
    )
    return wrap_avalanche_benchmark(benchmark, "cifar10", **kwargs)


def create_cifar100_benchmark(**kwargs):
    """Create CIFAR-100 benchmark with unified return format."""
    benchmark = SplitCIFAR100(
        n_experiences=kwargs.get('n_experiences', 10),
        seed=kwargs.get('seed', 42)
    )
    return wrap_avalanche_benchmark(benchmark, "cifar100", **kwargs)