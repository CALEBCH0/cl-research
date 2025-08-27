"""Training functionality for face CL experiments."""
import warnings
warnings.filterwarnings('ignore', message='.*longdouble.*')
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message='.*OpenSSL.*')
warnings.filterwarnings('ignore', message='No loggers specified.*')

from collections import namedtuple
import copy
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from sklearn.datasets import fetch_olivetti_faces
from avalanche.benchmarks.classic import SplitMNIST, SplitCIFAR10, SplitFMNIST
from avalanche.benchmarks import nc_benchmark
from avalanche.benchmarks.utils import AvalancheDataset, as_classification_dataset
from avalanche.benchmarks.scenarios.dataset_scenario import benchmark_from_datasets
from torchvision import datasets, transforms
from avalanche.models import SimpleMLP, SimpleCNN, FeatureExtractorBackbone, MobilenetV1
from avalanche.training.supervised import (
    Naive, EWC, Replay, GEM, AGEM, LwF, 
    SynapticIntelligence as SI, MAS, GDumb,
    Cumulative, JointTraining, ICaRL, StreamingLDA
)
from src.strategies.pure_ncm import PureNCM
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics, forgetting_metrics
from avalanche.logging import InteractiveLogger, TensorboardLogger, TextLogger, BaseLogger
from avalanche.training.plugins import EvaluationPlugin
from src.plugin_factory import create_plugins

# Optional imports with fallback
try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False

BenchmarkInfo = namedtuple("BenchmarkInfo", ["input_size", "num_classes", "channels"])


def normalize_benchmark_info(benchmark_info):
    """Convert benchmark_info dict to BenchmarkInfo namedtuple if needed."""
    if hasattr(benchmark_info, 'num_classes'):
        # Already a namedtuple or similar object
        return benchmark_info
    else:
        # It's a dict, convert to namedtuple
        # Handle different possible keys in the dict
        if 'image_shape' in benchmark_info:
            # LFW format: (channels, height, width)
            channels, height, width = benchmark_info['image_shape']
            input_size = height * width
        elif 'input_size' in benchmark_info:
            input_size = benchmark_info['input_size']
            channels = benchmark_info.get('channels', 1)
        else:
            # Default values
            input_size = 64 * 64
            channels = 1
            
        return BenchmarkInfo(
            input_size=input_size,
            num_classes=benchmark_info['num_classes'],
            channels=channels
        )


def create_benchmark(benchmark_name, experiences=5, seed=42, subset_config=None):
    """Create benchmark separately for caching purposes."""
    return set_benchmark(benchmark_name, experiences, seed, subset_config)


def set_benchmark(benchmark_name, experiences=5, seed=42, subset_config=None):
    """Set the appropriate benchmark."""
    
    if benchmark_name == 'mnist':
        benchmark = SplitMNIST(
            n_experiences=experiences,
            return_task_id=False,
            seed=seed
        )
        input_size = 28 * 28
        num_classes = 10
        channels = 1
    elif benchmark_name == 'fmnist':
        benchmark = SplitFMNIST(
            n_experiences=experiences,
            return_task_id=False,
            seed=seed
        )
        input_size = 28 * 28
        num_classes = 10
        channels = 1
    elif benchmark_name == 'cifar10':
        benchmark = SplitCIFAR10(
            n_experiences=experiences,
            return_task_id=False,
            seed=seed
        )
        input_size = 32 * 32 * 3
        num_classes = 10
        channels = 3
    elif benchmark_name == 'olivetti':
        # Olivetti Faces dataset
        
        # Load Olivetti faces
        olivetti = fetch_olivetti_faces(shuffle=False)  # Don't shuffle, we'll do it manually
        X = olivetti.images  # Shape: (400, 64, 64)
        y = olivetti.target  # Shape: (400,)
        
        # Convert to torch tensors
        X = torch.FloatTensor(X).unsqueeze(1)  # Add channel dimension
        y = torch.LongTensor(y)
        
        # Split train/test per class to ensure all classes are in both sets
        train_indices = []
        test_indices = []
        
        # Olivetti has 10 images per person, split 8 train / 2 test per person
        for person_id in range(40):
            # Get indices for this person
            person_indices = torch.where(y == person_id)[0]
            # Shuffle indices for this person
            perm = torch.randperm(len(person_indices), generator=torch.Generator().manual_seed(seed + person_id))
            shuffled_indices = person_indices[perm]
            
            # Split 80/20 for this person
            n_train_person = int(0.8 * len(shuffled_indices))
            train_indices.extend(shuffled_indices[:n_train_person].tolist())
            test_indices.extend(shuffled_indices[n_train_person:].tolist())
        
        # Convert to tensors
        train_indices = torch.tensor(train_indices)
        test_indices = torch.tensor(test_indices)
        
        # Create TensorDatasets
        train_tensor_dataset = TensorDataset(X[train_indices], y[train_indices])
        test_tensor_dataset = TensorDataset(X[test_indices], y[test_indices])
        
        # Add targets attribute to the dataset (required by Avalanche)
        train_tensor_dataset.targets = y[train_indices].tolist()
        test_tensor_dataset.targets = y[test_indices].tolist()
        
        
        # Convert to classification datasets
        train_dataset = as_classification_dataset(train_tensor_dataset)
        test_dataset = as_classification_dataset(test_tensor_dataset)
        
        # Create benchmark using nc_benchmark
        benchmark = nc_benchmark(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            n_experiences=experiences,
            task_labels=False,
            seed=seed,
            class_ids_from_zero_in_each_exp=False  # Use global class IDs
        )
        
        input_size = 64 * 64
        num_classes = 40  # 40 people in Olivetti
        channels = 1
    elif benchmark_name.startswith('lfw'):
        # LFW (Labeled Faces in the Wild) dataset
        from src.datasets.lfw import create_lfw_benchmark, create_lfw_subset_benchmark, create_lfw_controlled_benchmark
        from src.datasets.lfw_configs import get_lfw_config, LFW_CONFIGS
        from src.datasets.subset_utils import DatasetSubsetConfig
        
        # Check if subset_config is provided (new way)
        if subset_config and isinstance(subset_config, dict):
            # Create DatasetSubsetConfig from dict
            config_obj = DatasetSubsetConfig(
                target_classes=subset_config['target_classes'],
                min_samples_per_class=subset_config.get('min_samples_per_class', 10),
                n_experiences=experiences,
                selection_strategy=subset_config.get('selection_strategy', 'most_samples'),
                seed=seed
            )
            benchmark, dataset_info = create_lfw_controlled_benchmark(
                config_obj,
                image_size=(64, 64)
            )
        # Check if it's a predefined config (old way)
        elif benchmark_name in LFW_CONFIGS:
            config = get_lfw_config(benchmark_name)
            
            # Use the exact number of classes if specified
            if 'num_classes' in config:
                # Create subset with exact number of classes
                benchmark, dataset_info = create_lfw_subset_benchmark(
                    n_identities=config['num_classes'],
                    n_experiences=experiences if experiences in config['valid_n_experiences'] else config['default_n_experiences'],
                    min_faces_per_person=config['min_faces_per_person'],
                    image_size=(64, 64),
                    seed=seed
                )
                if experiences not in config['valid_n_experiences']:
                    print(f"\nNote: {experiences} experiences not valid for {benchmark_name} ({config['num_classes']} classes)")
                    print(f"Valid options: {config['valid_n_experiences']}")
                    print(f"Using default: {config['default_n_experiences']} experiences")
            else:
                # Fallback to old method
                benchmark, dataset_info = create_lfw_benchmark(
                    n_experiences=experiences,
                    min_faces_per_person=config['min_faces_per_person'],
                    image_size=(64, 64),
                    seed=seed
                )
            print(f"\nUsing {benchmark_name}: {config['description']}")
        
        # Legacy number-based names (deprecated but still supported)
        elif benchmark_name == 'lfw' or '_' in benchmark_name:
            # Try to parse as number
            try:
                n_identities = int(benchmark_name.split('_')[1])
                print(f"\nWarning: Using deprecated format 'lfw_{n_identities}'.")
                print("Please use predefined configs instead:")
                for name, cfg in LFW_CONFIGS.items():
                    print(f"  - {name}: {cfg['description']}")
                
                # Map to closest predefined config
                if n_identities <= 20:
                    actual_config = 'lfw_20'
                elif n_identities <= 50:
                    actual_config = 'lfw_50'
                elif n_identities <= 80:
                    actual_config = 'lfw_80'
                elif n_identities <= 100:
                    actual_config = 'lfw_100'
                elif n_identities <= 150:
                    actual_config = 'lfw_150'
                else:
                    actual_config = 'lfw_all'
                
                print(f"Mapping to {actual_config}")
                config = get_lfw_config(actual_config)
                benchmark, dataset_info = create_lfw_benchmark(
                    n_experiences=experiences,
                    min_faces_per_person=config['min_faces_per_person'],
                    image_size=(64, 64),
                    seed=seed
                )
            except:
                # Default 'lfw' means lfw_80
                config = get_lfw_config('lfw_80')
                benchmark, dataset_info = create_lfw_benchmark(
                    n_experiences=experiences,
                    min_faces_per_person=config['min_faces_per_person'],
                    image_size=(64, 64),
                    seed=seed
                )
        
        input_size = 64 * 64
        num_classes = dataset_info['num_classes']
        channels = 1
        
        print(f"\nLFW Dataset loaded:")
        print(f"  Classes: {num_classes}")
        print(f"  Train samples: {dataset_info['num_train_samples']}")
        print(f"  Test samples: {dataset_info['num_test_samples']}")
        if 'n_experiences' in dataset_info:
            print(f"  Experiences: {dataset_info['n_experiences']} (classes per exp: {dataset_info['classes_per_exp']})")
            
    elif benchmark_name.startswith('smarteye'):
        # SmartEye IR face dataset
        from src.datasets.smarteye import create_smarteye_benchmark, get_smarteye_config
        
        # Handle both predefined configs and subset configs
        if subset_config is not None:
            from src.datasets.smarteye import create_smarteye_controlled_benchmark
            benchmark, dataset_info = create_smarteye_controlled_benchmark(
                subset_config,
                image_size=(112, 112)  # Default size for face models
            )
        elif benchmark_name in ['smarteye_crop', 'smarteye_raw']:
            config = get_smarteye_config(benchmark_name)
            benchmark, dataset_info = create_smarteye_benchmark(
                root_dir='/Users/calebcho/data/face_dataset',  # Default path
                n_experiences=experiences,
                use_cropdata=config['use_cropdata'],
                image_size=(112, 112),
                seed=seed,
                min_samples_per_class=config['min_samples_per_class']
            )
        else:
            # Default to cropdata
            benchmark, dataset_info = create_smarteye_benchmark(
                root_dir='/Users/calebcho/data/face_dataset',  # Default path
                n_experiences=experiences,
                use_cropdata=True,
                image_size=(112, 112),
                seed=seed
            )
        
        input_size = dataset_info['input_size']
        num_classes = dataset_info['num_classes']
        channels = dataset_info['channels']
        
        print(f"\nSmartEye Dataset loaded:")
        print(f"  Data type: {dataset_info['data_type']}")
        print(f"  Classes: {num_classes}")
        print(f"  Train samples: {dataset_info['num_train']}")
        print(f"  Test samples: {dataset_info['num_test']}")
        print(f"  Image size: {dataset_info['image_size']}")
        print(f"  Experiences: {dataset_info['n_experiences']}")
        
    else:
        raise ValueError(f"Unknown benchmark: {benchmark_name}")
    
    return benchmark, BenchmarkInfo(input_size, num_classes, channels)


def create_model(model_type, benchmark_info, **kwargs):
    """Create model based on type and benchmark."""
    if model_type == 'mlp':
        model = SimpleMLP(
            num_classes=benchmark_info.num_classes,
            input_size=benchmark_info.input_size,
            hidden_size=400,
            hidden_layers=2
        )
    elif model_type == 'cnn':
        # SimpleCNN expects 3 channels, but MNIST has 1
        # For MNIST/Fashion-MNIST, we need a different approach
        if benchmark_info.channels == 1:
            # Use a simple CNN that works with grayscale
            class SimpleGrayCNN(nn.Module):
                def __init__(self, num_classes):
                    super().__init__()
                    self.features = nn.Sequential(
                        nn.Conv2d(1, 32, 3, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(2),
                        nn.Conv2d(32, 64, 3, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(2),
                        nn.Flatten()
                    )
                    # Adjust for different input sizes
                    if benchmark_info.input_size == 28 * 28:  # MNIST
                        self.classifier = nn.Linear(64 * 7 * 7, num_classes)
                    elif benchmark_info.input_size == 64 * 64:  # Olivetti
                        self.classifier = nn.Linear(64 * 16 * 16, num_classes)
                    else:
                        self.classifier = nn.Linear(64 * 7 * 7, num_classes)
                
                def forward(self, x):
                    x = self.features(x)
                    return self.classifier(x)
            
            model = SimpleGrayCNN(num_classes=benchmark_info.num_classes)
        else:
            model = SimpleCNN(
                num_classes=benchmark_info.num_classes
            )
    elif model_type == 'mobilenetv1':
        # Use Avalanche's MobileNetV1
        model = MobilenetV1(num_classes=benchmark_info.num_classes)
        # MobileNetV1 expects 3 channels, wrap for grayscale
        if benchmark_info.channels == 1:
            class GrayToRGBWrapper(nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model
                
                def forward(self, x):
                    x = x.repeat(1, 3, 1, 1)  # Convert 1 channel to 3
                    return self.model(x)
            model = GrayToRGBWrapper(model)
    elif model_type.startswith('efficientnet') or model_type.startswith('resnet') or \
         model_type.startswith('mobilenet') or model_type in ['resnet18', 'resnet50']:
        # timm model support (EfficientNet, ResNet, MobileNet, etc.)
        if not TIMM_AVAILABLE:
            print(f"timm not installed. Cannot create {model_type}.")
            return create_model('cnn', benchmark_info)
        
        # Fix MobileNetV3 names for timm
        timm_model_name = model_type
        if model_type == 'mobilenetv3_small':
            timm_model_name = 'mobilenetv3_small_100'
        elif model_type == 'mobilenetv3_large':
            timm_model_name = 'mobilenetv3_large_100'
        
        # Handle grayscale images (MNIST, Fashion-MNIST, Olivetti)
        if benchmark_info.channels == 1:
            # Convert grayscale to RGB by repeating channels
            class GrayToRGBWrapper(nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model
                
                def forward(self, x):
                    x = x.repeat(1, 3, 1, 1)  # Convert 1 channel to 3
                    return self.model(x)
            
            try:
                base_model = timm.create_model(
                    timm_model_name,
                    pretrained=True,
                    num_classes=benchmark_info.num_classes
                )
                model = GrayToRGBWrapper(base_model)
            except Exception as e:
                print(f"Error creating {model_type}: {e}")
                raise ValueError(f"Failed to create model {model_type}")
        else:
            # RGB images
            try:
                model = timm.create_model(
                    timm_model_name,
                    pretrained=True,
                    num_classes=benchmark_info.num_classes
                )
            except Exception as e:
                print(f"Error creating {model_type}: {e}")
                raise ValueError(f"Failed to create model {model_type}")
    elif model_type in ['dwseesawfacev2', 'ghostfacenetv2', 'modified_mobilefacenet']:
        # Import custom backbone models from backbones directory
        import sys
        import os
        # Add cl-research directory to path to access backbones and utils
        cl_research_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
        sys.path.insert(0, cl_research_path)
        
        # Also add the parent directory to access utils.common
        parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
        if parent_path not in sys.path:
            sys.path.insert(0, parent_path)
        
        try:
            if model_type == 'dwseesawfacev2':
                from backbones.DWseesawfaceV2 import dwseesawfacev2
                embedding_size = kwargs.get('embedding_size', 512)
                model = dwseesawfacev2(embedding_size=embedding_size)
                
            elif model_type == 'ghostfacenetv2':
                from backbones.ghostfacenetV2 import ghostfacenetv2
                image_size = kwargs.get('image_size', (112, 112))
                # GhostFaceNetV2 expects a single integer for image size, not a tuple
                if isinstance(image_size, (list, tuple)):
                    image_size = image_size[0]  # Assume square images
                num_features = kwargs.get('num_features', 512)
                width = kwargs.get('width', 1.0)
                model = ghostfacenetv2(image_size=image_size, num_features=num_features, width=width)
                
            elif model_type == 'modified_mobilefacenet':
                from backbones.modified_mobilefacenet import modified_mbilefacenet
                # Pass all kwargs to the model
                model = modified_mbilefacenet(**kwargs)
                
        except ImportError as e:
            print(f"Error importing {model_type}: {e}")
            raise ValueError(f"Failed to import backbone model {model_type}")
        except Exception as e:
            print(f"Error creating {model_type}: {e}")
            raise ValueError(f"Failed to create backbone model {model_type}")
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


def create_strategy(strategy_name, model, optimizer, criterion, device, 
                   eval_plugin, mem_size=200, model_type=None, benchmark_info=None, 
                   plugins_config=None, **kwargs):
    """Create strategy with given parameters."""
    
    # Generic feature extractor that works with different architectures
    class GenericFeatureExtractor(nn.Module):
        def __init__(self, base_model, model_type):
            super().__init__()
            self.model_type = model_type
            
            # Handle GrayToRGBWrapper
            if hasattr(base_model, 'model'):  # GrayToRGBWrapper
                self.wrapper = True
                actual_model = base_model.model
            else:
                self.wrapper = False
                actual_model = base_model
            
            # Extract features based on model type
            if 'efficientnet' in model_type:
                # For EfficientNet models
                self.features = nn.Sequential(
                    actual_model.conv_stem,
                    actual_model.bn1,
                    actual_model.blocks,
                    actual_model.conv_head,
                    actual_model.bn2,
                    actual_model.global_pool,
                    nn.Flatten()
                )
                self.num_features = actual_model.num_features
                
            elif 'resnet' in model_type:
                # For ResNet models
                # Remove the final fc layer
                modules = list(actual_model.children())[:-1]
                self.features = nn.Sequential(*modules, nn.Flatten())
                # Get feature dimension
                self.num_features = actual_model.fc.in_features
                
            elif 'mobilenet' in model_type:
                # For MobileNet models
                if model_type == 'mobilenetv1':
                    # Avalanche's MobileNetV1 has a specific structure
                    # Extract all layers except the final classifier
                    self.features = nn.Sequential(
                        actual_model.features,
                        nn.AdaptiveAvgPool2d(1),
                        nn.Flatten()
                    )
                    # MobileNetV1 has 1024 features before the classifier
                    self.num_features = 1024
                elif hasattr(actual_model, 'features'):
                    # Standard MobileNet structure (V2, V3, etc.)
                    self.features = nn.Sequential(
                        actual_model.features,
                        actual_model.avgpool if hasattr(actual_model, 'avgpool') else nn.AdaptiveAvgPool2d(1),
                        nn.Flatten()
                    )
                    # Get feature dimension from classifier
                    if hasattr(actual_model, 'classifier'):
                        if isinstance(actual_model.classifier, nn.Sequential):
                            # Find the first Linear layer in classifier
                            for module in actual_model.classifier:
                                if isinstance(module, nn.Linear):
                                    self.num_features = module.in_features
                                    break
                        else:
                            self.num_features = actual_model.classifier.in_features
                    else:
                        self.num_features = 1280  # Default for MobileNetV3
                else:
                    # Alternative MobileNet structure
                    modules = list(actual_model.children())[:-1]
                    self.features = nn.Sequential(*modules, nn.Flatten())
                    self.num_features = 1280  # Default
            else:
                raise ValueError(f"Unsupported model type for feature extraction: {model_type}")
        
        def forward(self, x):
            if self.wrapper:
                x = x.repeat(1, 3, 1, 1)  # Convert grayscale to RGB
            return self.features(x)
    
    base_kwargs = {
        'model': model,
        'optimizer': optimizer,
        'criterion': criterion,
        'train_mb_size': kwargs.get('batch_size', 32),
        'train_epochs': kwargs.get('epochs', 1),
        'eval_mb_size': kwargs.get('batch_size', 32) * 2,
        'device': device,
        'evaluator': eval_plugin
    }
    
    # Create plugins from configuration
    plugins = []
    if plugins_config:
        # Get feature extractor for plugins that need it
        feature_extractor = None
        if hasattr(model, 'features'):
            feature_extractor = model.features
        elif hasattr(model, 'feature_extractor'):
            feature_extractor = model.feature_extractor
            
        plugins = create_plugins(
            plugins_config, 
            mem_size=mem_size, 
            device=device,
            optimizer=optimizer,
            feature_extractor=feature_extractor
        )
    
    if plugins:
        base_kwargs['plugins'] = plugins
    
    if strategy_name == 'naive':
        return Naive(**base_kwargs)
    elif strategy_name == 'ewc':
        return EWC(**base_kwargs, 
                   ewc_lambda=kwargs.get('ewc_lambda', 0.4), 
                   mode=kwargs.get('mode', 'online'), 
                   decay_factor=kwargs.get('decay_factor', 0.1))
    elif strategy_name == 'replay':
        return Replay(**base_kwargs, mem_size=mem_size)
    elif strategy_name == 'gem':
        return GEM(**base_kwargs, 
                   patterns_per_exp=kwargs.get('patterns_per_exp', 256), 
                   memory_strength=kwargs.get('memory_strength', 0.5))
    elif strategy_name == 'agem':
        return AGEM(**base_kwargs, 
                    patterns_per_exp=kwargs.get('patterns_per_exp', 256), 
                    sample_size=kwargs.get('sample_size', 256))
    elif strategy_name == 'lwf':
        return LwF(**base_kwargs, 
                   alpha=kwargs.get('alpha', 0.5), 
                   temperature=kwargs.get('temperature', 2))
    elif strategy_name == 'si':
        return SI(**base_kwargs, si_lambda=kwargs.get('si_lambda', 0.0001))
    elif strategy_name == 'mas':
        return MAS(**base_kwargs, 
                    lambda_reg=kwargs.get('lambda_reg', 1), 
                    alpha=kwargs.get('alpha', 0.5))
    elif strategy_name == 'gdumb':
        return GDumb(**base_kwargs, mem_size=mem_size)
    elif strategy_name == 'cumulative':
        return Cumulative(**base_kwargs)
    elif strategy_name == 'joint':
        return JointTraining(**base_kwargs)
    elif strategy_name == 'icarl':
        # ICaRL requires separate feature_extractor and classifier
        if model_type and model_type.startswith('efficientnet'):
            # For timm EfficientNet models, we need to properly split them
            import copy
            
            # Get the base EfficientNet model
            if hasattr(model, 'model'):  # GrayToRGBWrapper
                base_model = model.model
                needs_rgb_conversion = True
            else:
                base_model = model
                needs_rgb_conversion = False
            
            # Create feature extractor that outputs features before the classifier
            class EfficientNetFeatureExtractor(nn.Module):
                def __init__(self, efficientnet_model, convert_rgb=False):
                    super().__init__()
                    self.convert_rgb = convert_rgb
                    
                    # Copy all layers except the classifier
                    # Handle different EfficientNet versions that might have different attribute names
                    if hasattr(efficientnet_model, 'conv_stem'):
                        self.conv_stem = efficientnet_model.conv_stem
                        self.bn1 = efficientnet_model.bn1
                        self.act1 = getattr(efficientnet_model, 'act1', nn.SiLU(inplace=True))
                        self.blocks = efficientnet_model.blocks
                        self.conv_head = efficientnet_model.conv_head
                        self.bn2 = efficientnet_model.bn2
                        self.act2 = getattr(efficientnet_model, 'act2', nn.SiLU(inplace=True))
                        self.global_pool = efficientnet_model.global_pool
                    else:
                        # Fallback for different model structures
                        raise ValueError(f"EfficientNet model structure not recognized")
                    
                    # Store feature dimension
                    self.num_features = efficientnet_model.num_features
                    
                def forward(self, x):
                    if self.convert_rgb:
                        x = x.repeat(1, 3, 1, 1)  # Convert grayscale to RGB
                    
                    # Standard EfficientNet forward (without classifier)
                    x = self.conv_stem(x)
                    x = self.bn1(x)
                    x = self.act1(x)
                    x = self.blocks(x)
                    x = self.conv_head(x)
                    x = self.bn2(x)
                    x = self.act2(x)
                    x = self.global_pool(x)
                    x = x.flatten(1)  # Flatten for classifier
                    return x
            
            # Create the feature extractor
            feature_extractor = EfficientNetFeatureExtractor(base_model, needs_rgb_conversion)
            
            # Get the classifier (just the final linear layer)
            classifier = copy.deepcopy(base_model.classifier)
            
            # Verify dimensions match
            feature_dim = feature_extractor.num_features
            classifier_in_features = classifier.in_features if hasattr(classifier, 'in_features') else None
            
            if classifier_in_features and feature_dim != classifier_in_features:
                print(f"Warning: Feature dim {feature_dim} != Classifier input {classifier_in_features}")
            
            print(f"ICaRL: Split EfficientNet - Feature dim: {feature_dim}, Num classes: {benchmark_info.num_classes}")
            
        else:
            # For other models (MLP, CNN)
            print(f"Note: ICaRL with {model_type} using fallback split.")
            
            # For SimpleCNN and custom CNN models
            if hasattr(model, 'features') and hasattr(model, 'classifier'):
                # Model already has features/classifier split
                feature_extractor = copy.deepcopy(model.features)
                classifier = copy.deepcopy(model.classifier)
            else:
                # Try to split Sequential models
                all_children = list(model.children())
                if len(all_children) > 1:
                    # Find the last linear layer
                    classifier_idx = None
                    for i in reversed(range(len(all_children))):
                        if isinstance(all_children[i], nn.Linear):
                            classifier_idx = i
                            break
                    
                    if classifier_idx is not None:
                        feature_extractor = nn.Sequential(*all_children[:classifier_idx])
                        classifier = all_children[classifier_idx]
                    else:
                        raise ValueError(f"Cannot find classifier layer in {model_type} for ICaRL.")
                else:
                    # Model doesn't have clear separation
                    raise ValueError(f"Cannot split {model_type} for ICaRL. Consider using Replay instead.")
        
        # Move to device
        feature_extractor = feature_extractor.to(device)
        classifier = classifier.to(device)
        
        # Create new optimizer for the split model
        # Note: ICaRL trains both feature extractor and classifier
        icarl_params = list(feature_extractor.parameters()) + list(classifier.parameters())
        icarl_optimizer = torch.optim.Adam(icarl_params, lr=kwargs.get('lr', 0.001))
        
        # ICaRL specific parameters
        return ICaRL(
            feature_extractor=feature_extractor,
            classifier=classifier,
            optimizer=icarl_optimizer,  # Use new optimizer for split model
            memory_size=mem_size,
            buffer_transform=None,
            fixed_memory=True,
            train_mb_size=kwargs.get('batch_size', 32),
            train_epochs=kwargs.get('epochs', 1),
            eval_mb_size=kwargs.get('batch_size', 32) * 2,
            device=device,
            evaluator=eval_plugin
        )
    elif strategy_name == 'slda':
        # SLDA requires a feature extractor model, not a classifier
        
        # For standard Avalanche models, use them directly with SLDA
        if model_type in ['mlp', 'cnn'] and hasattr(model, 'features'):
            # Standard Avalanche models like SimpleMLP already have proper structure
            feature_extractor = model
            # For SimpleMLP, the feature size is the hidden layer size
            if hasattr(model, 'classifier') and isinstance(model.classifier, nn.Linear):
                feature_size = model.classifier.in_features
            else:
                feature_size = benchmark_info.input_size  # fallback
        else:
            # Create feature extractor for custom models
            feature_extractor = GenericFeatureExtractor(model, model_type)
            feature_size = feature_extractor.num_features
            
        feature_extractor = feature_extractor.to(device)
        
        print(f"SLDA: Created feature extractor for {model_type} with feature size {feature_size}")
        
        # SLDA kwargs
        slda_kwargs = {
            'slda_model': feature_extractor,
            'criterion': criterion,
            'input_size': feature_size,
            'num_classes': benchmark_info.num_classes,  # Use actual number of classes from benchmark
            'shrinkage_param': 1e-4,
            'streaming_update_sigma': True,
            'train_mb_size': kwargs.get('batch_size', 32),
            'eval_mb_size': kwargs.get('batch_size', 32) * 2,
            'device': device,
            'evaluator': eval_plugin,
            'train_epochs': 1,  # SLDA typically uses 1 epoch
            'plugins': plugins if plugins else None  # Add any plugins
        }
        return StreamingLDA(**slda_kwargs)
    elif strategy_name == 'pure_ncm':
        # Pure NCM requires a feature extractor
        # Reuse the same GenericFeatureExtractor class defined above
        if model_type:
            feature_extractor = GenericFeatureExtractor(model, model_type)
            feature_size = feature_extractor.num_features
        else:
            # Fallback for unknown model types
            raise ValueError(f"Pure NCM requires a known model type, got: {model_type}")
        
        feature_extractor = feature_extractor.to(device)
        
        # Pure NCM doesn't need optimizer/criterion in base_kwargs
        return PureNCM(
            feature_extractor=feature_extractor,
            feature_size=feature_size,
            num_classes=benchmark_info.num_classes,
            train_mb_size=kwargs.get('batch_size', 32),
            train_epochs=kwargs.get('epochs', 1),
            eval_mb_size=kwargs.get('batch_size', 32) * 2,
            device=device,
            evaluator=eval_plugin
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")


def run_training(benchmark_name='fmnist', strategy_name='naive', model_type='mlp',
                device='cuda', experiences=5, epochs=1, batch_size=32, 
                mem_size=200, lr=0.001, seed=42, verbose=True, 
                plugins_config=None, benchmark=None, benchmark_info=None,
                subset_config=None, model_config=None, strategy_config=None, **kwargs):
    """
    Run a complete training experiment and return results.
    
    Returns:
        dict: Results containing final accuracies and average accuracy
    """
    # Set seed for all random number generators
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For deterministic behavior (reduces performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Create benchmark if not provided (for caching)
    if benchmark is None or benchmark_info is None:
        if verbose:
            print(f"\n=== Creating benchmark ===")
            print(f"Dataset: {benchmark_name}")
            print(f"Experiences: {experiences}")
        benchmark, benchmark_info = set_benchmark(benchmark_name, experiences, seed, subset_config)
    
    # Normalize benchmark_info to ensure it's a namedtuple
    benchmark_info = normalize_benchmark_info(benchmark_info)
    
    if verbose:
        print(f"\n=== Running training ===")
        print(f"Dataset: {benchmark_name}")
        print(f"Strategy: {strategy_name}")
        print(f"Model: {model_type}")
        print(f"Experiences: {experiences}")
        print(f"Seed: {seed}")
    
    # Create model
    if model_config:
        # Use modular config to create model
        from src.utils.modular_config import create_model_from_config
        model = create_model_from_config(model_config, benchmark_info)
    else:
        # Use original model creation
        model = create_model(model_type, benchmark_info)
    model = model.to(device)
    
    # Create optimizer and criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # Create evaluation plugin
    if verbose:
        loggers = [InteractiveLogger()]
    else:
        # Use a minimal logger to suppress warning
        class SilentLogger(BaseLogger):
            def log_metrics(self, metrics):
                pass
            
            def log_single_metric(self, name, value, x_plot):
                pass
        loggers = [SilentLogger()]
    
    eval_plugin = EvaluationPlugin(
        accuracy_metrics(minibatch=False, epoch=True, experience=True, stream=True),
        loss_metrics(minibatch=False, epoch=True, experience=True, stream=True),
        forgetting_metrics(experience=True, stream=True),
        loggers=loggers
    )
    
    # Create strategy
    if strategy_config:
        # Use modular config to create strategy
        from src.utils.modular_config import create_strategy_from_config
        strategy = create_strategy_from_config(
            strategy_config=strategy_config,
            model=model,
            benchmark_info=benchmark_info,
            optimizer=optimizer,
            criterion=criterion,
            eval_plugin=eval_plugin,
            device=device,
            model_type=model_type,
            epochs=epochs,
            batch_size=batch_size,
            mem_size=mem_size
        )
    else:
        # Use original strategy creation
        strategy = create_strategy(
            strategy_name, model, optimizer, criterion, device, eval_plugin,
            mem_size=mem_size, model_type=model_type, benchmark_info=benchmark_info,
            epochs=epochs, batch_size=batch_size, 
            num_classes=benchmark_info.num_classes if hasattr(benchmark_info, 'num_classes') else benchmark_info['num_classes'],
            plugins_config=plugins_config
        )
    
    # Training loop
    # Get number of experiences (handle both benchmark types)
    n_experiences = getattr(benchmark, 'n_experiences', len(benchmark.train_stream))
    
    for i, train_exp in enumerate(benchmark.train_stream):
        if verbose:
            print(f"\n>>> Training Experience {i+1}/{n_experiences}")
        strategy.train(train_exp)
    
    # Final evaluation
    if verbose:
        print("\nFinal evaluation...")
    eval_results = strategy.eval(benchmark.test_stream)
    
    # Debug: print available keys
    if verbose:
        print(f"\nEvaluation results keys for {strategy_name} on {benchmark_name}:")
        for k in sorted(eval_results.keys()):
            if 'Top1_Acc' in k:
                print(f"  {k}: {eval_results[k]:.4f}")
    
    # Extract accuracies (use n_experiences from above)
    accuracies = []
    for i in range(n_experiences):
        key = f'Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp{i:03d}'
        if key in eval_results:
            accuracies.append(eval_results[key])
    
    # If no accuracies found with Task000, try without task ID
    if not accuracies:
        for i in range(n_experiences):
            key = f'Top1_Acc_Exp/eval_phase/test_stream/Exp{i:03d}'
            if key in eval_results:
                accuracies.append(eval_results[key])
    
    avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0.0
    
    if verbose:
        print(f"\nExtracted accuracies: {accuracies}")
        print(f"Average accuracy: {avg_accuracy:.4f}")
    
    return {
        'accuracies': accuracies,
        'average_accuracy': avg_accuracy,
        'strategy': strategy_name,
        'model': model_type,
        'benchmark': benchmark_name
    }