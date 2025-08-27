"""Unified BenchmarkInfo class for consistent benchmark metadata."""

class BenchmarkInfo:
    """
    Unified information object for all benchmarks.
    Provides consistent interface regardless of dataset type.
    """
    
    def __init__(self, 
                 num_classes: int = 0,
                 num_samples: int = 0, 
                 num_train: int = 0,
                 num_test: int = 0,
                 image_size: tuple = (28, 28),
                 channels: int = 1,
                 n_experiences: int = 1,
                 dataset_name: str = "unknown",
                 **kwargs):
        """
        Initialize BenchmarkInfo.
        
        Args:
            num_classes: Number of classes in dataset
            num_samples: Total number of samples
            num_train: Number of training samples
            num_test: Number of test samples  
            image_size: Input image size (height, width)
            channels: Number of input channels
            n_experiences: Number of continual learning experiences
            dataset_name: Name of the dataset
            **kwargs: Additional metadata
        """
        self.num_classes = num_classes
        self.num_samples = num_samples
        self.num_train = num_train
        self.num_test = num_test
        self.image_size = image_size
        self.channels = channels
        self.input_size = image_size[0] * image_size[1] * channels
        self.n_experiences = n_experiences
        self.dataset_name = dataset_name
        
        # Store any additional metadata
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def __repr__(self):
        return (f"BenchmarkInfo(dataset={self.dataset_name}, "
                f"classes={self.num_classes}, "
                f"train={self.num_train}, test={self.num_test}, "
                f"experiences={self.n_experiences})")
    
    def to_dict(self):
        """Convert to dictionary for backward compatibility."""
        return {
            'num_classes': self.num_classes,
            'num_samples': self.num_samples,
            'num_train': self.num_train,
            'num_test': self.num_test,
            'image_size': self.image_size,
            'channels': self.channels,
            'input_size': self.input_size,
            'n_experiences': self.n_experiences,
            'dataset_name': self.dataset_name
        }
    
    @classmethod
    def from_dict(cls, data: dict):
        """Create BenchmarkInfo from dictionary."""
        return cls(**data)