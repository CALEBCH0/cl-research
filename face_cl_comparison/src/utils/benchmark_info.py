"""Unified benchmark information container for all datasets."""

from typing import Optional, Dict, Any, Tuple, List


class BenchmarkInfo:
    """
    Unified benchmark information container - pure configuration/metadata.
    All benchmark creation functions should return (benchmark, BenchmarkInfo).
    """
    
    def __init__(self, 
                 num_classes: int,
                 image_size: Tuple[int, int] = (224, 224),
                 channels: int = 3,
                 num_train: int = 0,
                 num_test: int = 0,
                 n_experiences: int = 1,
                 class_names: Optional[Dict[int, str]] = None):
        """
        Initialize BenchmarkInfo with dataset metadata.
        
        Args:
            num_classes: Number of classes in the dataset
            image_size: Tuple of (height, width) for images
            channels: Number of channels (1 for grayscale, 3 for RGB)
            num_train: Number of training samples
            num_test: Number of test samples
            n_experiences: Number of experiences/tasks
            class_names: Optional mapping of class indices to names
        """
        self.num_classes = num_classes
        self.image_size = image_size
        self.channels = channels
        self.num_train = num_train
        self.num_test = num_test
        self.num_samples = num_train + num_test
        self.n_experiences = n_experiences
        self.class_names = class_names or {}
        self.classes = list(range(num_classes))
    
    @classmethod
    def from_avalanche_benchmark(cls, benchmark):
        """
        Create BenchmarkInfo by extracting info from an Avalanche benchmark object.
        Used for built-in Avalanche benchmarks where we need to extract the metadata.
        """
        # Extract num_classes
        if hasattr(benchmark, 'n_classes'):
            num_classes = benchmark.n_classes
        elif hasattr(benchmark, 'classes_order'):
            num_classes = len(benchmark.classes_order)
        else:
            num_classes = 10  # Default fallback
        
        # Calculate train/test sizes from streams
        num_train = 0
        num_test = 0
        if hasattr(benchmark, 'train_stream'):
            for exp in benchmark.train_stream:
                num_train += len(exp.dataset)
            n_experiences = len(benchmark.train_stream)
        else:
            n_experiences = 1
            
        if hasattr(benchmark, 'test_stream'):
            for exp in benchmark.test_stream:
                num_test += len(exp.dataset)
        
        # Try to get image info from first sample
        image_size = (28, 28)  # Default
        channels = 1  # Default
        if hasattr(benchmark, 'train_stream') and len(benchmark.train_stream) > 0:
            first_exp = benchmark.train_stream[0]
            if hasattr(first_exp, 'dataset') and len(first_exp.dataset) > 0:
                try:
                    sample = first_exp.dataset[0][0]  # Get first sample (input, target)
                    if hasattr(sample, 'shape') and len(sample.shape) >= 3:
                        channels = sample.shape[0]
                        image_size = (sample.shape[1], sample.shape[2])
                except:
                    pass  # Keep defaults if extraction fails
        
        return cls(
            num_classes=num_classes,
            image_size=image_size,
            channels=channels,
            num_train=num_train,
            num_test=num_test,
            n_experiences=n_experiences
        )