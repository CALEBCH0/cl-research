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
from avalanche.models import SimpleMLP, SimpleCNN, FeatureExtractorBackbone
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


def set_benchmark(benchmark_name, experiences=5, seed=42):
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
    else:
        raise ValueError(f"Unknown benchmark: {benchmark_name}")
    
    return benchmark, BenchmarkInfo(input_size, num_classes, channels)


def create_model(model_type, benchmark_info):
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
    elif model_type.startswith('efficientnet'):
        # EfficientNet support
        if not TIMM_AVAILABLE:
            print("timm not installed. Using CNN instead.")
            return create_model('cnn', benchmark_info)
        
        # For SLDA, we need a feature extractor
        if benchmark_info.channels == 1:
                # Convert grayscale to RGB by repeating channels
                class GrayToRGBWrapper(nn.Module):
                    def __init__(self, model):
                        super().__init__()
                        self.model = model
                    
                    def forward(self, x):
                        x = x.repeat(1, 3, 1, 1)  # Convert 1 channel to 3
                        return self.model(x)
                
                # Fix model name for timm - convert efficientnet-b1 to efficientnet_b1
                timm_model_name = model_type.replace('-', '_')
                base_model = timm.create_model(
                    timm_model_name,
                    pretrained=True,
                    num_classes=benchmark_info.num_classes
                )
                model = GrayToRGBWrapper(base_model)
        else:
            # Fix model name for timm - convert efficientnet-b1 to efficientnet_b1
            timm_model_name = model_type.replace('-', '_')
            model = timm.create_model(
                timm_model_name,
                pretrained=True,
                num_classes=benchmark_info.num_classes
            )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


def create_strategy(strategy_name, model, optimizer, criterion, device, 
                   eval_plugin, mem_size=200, model_type=None, benchmark_info=None, 
                   plugins_config=None, **kwargs):
    """Create strategy with given parameters."""
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
        # For EfficientNet, we need to get the feature extractor part
        
        class EfficientNetFeatureExtractor(nn.Module):
            def __init__(self, base_model):
                super().__init__()
                if hasattr(base_model, 'model'):  # GrayToRGBWrapper
                    self.wrapper = base_model
                    efficientnet = base_model.model
                else:
                    self.wrapper = None
                    efficientnet = base_model
                
                # For timm EfficientNet models
                self.features = efficientnet.conv_stem
                self.bn1 = efficientnet.bn1
                self.blocks = efficientnet.blocks
                self.conv_head = efficientnet.conv_head
                self.bn2 = efficientnet.bn2
                self.global_pool = efficientnet.global_pool
                
                # Get the feature dimension
                self.num_features = efficientnet.num_features
            
            def forward(self, x):
                if self.wrapper is not None:
                    x = x.repeat(1, 3, 1, 1)  # Convert grayscale to RGB
                
                # EfficientNet forward without the classifier
                x = self.features(x)
                x = self.bn1(x)
                x = self.blocks(x)
                x = self.conv_head(x)
                x = self.bn2(x)
                x = self.global_pool(x)
                x = x.flatten(1)
                return x
        
        # Create feature extractor
        feature_extractor = EfficientNetFeatureExtractor(model)
        feature_extractor = feature_extractor.to(device)
        feature_size = feature_extractor.num_features
        
        # SLDA kwargs
        slda_kwargs = {
            'slda_model': feature_extractor,
            'criterion': criterion,
            'input_size': feature_size,
            'num_classes': kwargs.get('num_classes', 10),
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
        if model_type.startswith('efficientnet'):
            # Create feature extractor for EfficientNet
            class EfficientNetFeatureExtractor(nn.Module):
                def __init__(self, base_model):
                    super().__init__()
                    if hasattr(base_model, 'model'):  # GrayToRGBWrapper
                        self.wrapper = base_model
                        efficientnet = base_model.model
                    else:
                        self.wrapper = None
                        efficientnet = base_model
                    
                    # For timm EfficientNet models
                    self.features = efficientnet.conv_stem
                    self.bn1 = efficientnet.bn1
                    self.blocks = efficientnet.blocks
                    self.conv_head = efficientnet.conv_head
                    self.bn2 = efficientnet.bn2
                    self.global_pool = efficientnet.global_pool
                    
                    # Get the feature dimension
                    self.num_features = efficientnet.num_features
                
                def forward(self, x):
                    if self.wrapper is not None:
                        x = x.repeat(1, 3, 1, 1)  # Convert grayscale to RGB
                    
                    # EfficientNet forward without the classifier
                    x = self.features(x)
                    x = self.bn1(x)
                    x = self.blocks(x)
                    x = self.conv_head(x)
                    x = self.bn2(x)
                    x = self.global_pool(x)
                    x = x.flatten(1)
                    return x
            
            feature_extractor = EfficientNetFeatureExtractor(model)
            feature_size = feature_extractor.num_features
        else:
            # For other models, create simple feature extractor
            if hasattr(model, 'features'):
                feature_extractor = model.features
                # Estimate feature size
                dummy_input = torch.randn(1, benchmark_info.channels, 64, 64).to(device)
                with torch.no_grad():
                    dummy_features = feature_extractor(dummy_input)
                feature_size = dummy_features.shape[1]
            else:
                # Fallback: use the whole model except last layer
                print(f"Warning: Pure NCM with {model_type} may not work optimally.")
                feature_extractor = nn.Sequential(*list(model.children())[:-1])
                feature_size = 512  # Default guess
        
        feature_extractor = feature_extractor.to(device)
        
        # Pure NCM doesn't need optimizer/criterion in base_kwargs
        return PureNCM(
            feature_extractor=feature_extractor,
            feature_size=feature_size,
            num_classes=kwargs.get('num_classes', 10),
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
                plugins_config=None, **kwargs):
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
    
    # Create benchmark
    benchmark, benchmark_info = set_benchmark(benchmark_name, experiences, seed)
    
    # Create model
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
    strategy = create_strategy(
        strategy_name, model, optimizer, criterion, device, eval_plugin,
        mem_size=mem_size, model_type=model_type, benchmark_info=benchmark_info,
        epochs=epochs, batch_size=batch_size, num_classes=benchmark_info.num_classes,
        plugins_config=plugins_config
    )
    
    # Training loop
    for i, train_exp in enumerate(benchmark.train_stream):
        if verbose:
            print(f"\n>>> Training Experience {i+1}/{benchmark.n_experiences}")
        strategy.train(train_exp)
    
    # Final evaluation
    if verbose:
        print("\nFinal evaluation...")
    eval_results = strategy.eval(benchmark.test_stream)
    
    # Extract accuracies
    accuracies = []
    for i in range(benchmark.n_experiences):
        key = f'Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp{i:03d}'
        if key in eval_results:
            accuracies.append(eval_results[key])
    
    avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0.0
    
    return {
        'accuracies': accuracies,
        'average_accuracy': avg_accuracy,
        'strategy': strategy_name,
        'model': model_type,
        'benchmark': benchmark_name
    }