"""Training functionality for face CL experiments."""
import warnings
warnings.filterwarnings('ignore', message='.*longdouble.*')
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message='.*OpenSSL.*')
warnings.filterwarnings('ignore', message='No loggers specified.*')

from collections import namedtuple
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
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics, forgetting_metrics
from avalanche.logging import InteractiveLogger, TensorboardLogger, TextLogger, BaseLogger
from avalanche.training.plugins import EvaluationPlugin

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
            class_ids_from_zero_in_each_exp=True
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
                
                base_model = timm.create_model(
                    model_type.replace('_', '-'),  # efficientnet_b1 -> efficientnet-b1
                    pretrained=True,
                    num_classes=benchmark_info.num_classes
                )
                model = GrayToRGBWrapper(base_model)
        else:
            model = timm.create_model(
                model_type.replace('_', '-'),
                pretrained=True,
                num_classes=benchmark_info.num_classes
            )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


def create_strategy(strategy_name, model, optimizer, criterion, device, 
                   eval_plugin, mem_size=200, **kwargs):
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
    
    if strategy_name == 'naive':
        return Naive(**base_kwargs)
    elif strategy_name == 'ewc':
        return EWC(**base_kwargs, ewc_lambda=0.4, mode='online', decay_factor=0.1)
    elif strategy_name == 'replay':
        return Replay(**base_kwargs, mem_size=mem_size)
    elif strategy_name == 'gem':
        return GEM(**base_kwargs, patterns_per_exp=256, memory_strength=0.5)
    elif strategy_name == 'agem':
        return AGEM(**base_kwargs, patterns_per_exp=256, sample_size=256)
    elif strategy_name == 'lwf':
        return LwF(**base_kwargs, alpha=0.5, temperature=2)
    elif strategy_name == 'si':
        return SI(**base_kwargs, si_lambda=0.0001)
    elif strategy_name == 'mas':
        return MAS(**base_kwargs, lambda_reg=1, alpha=0.5)
    elif strategy_name == 'gdumb':
        return GDumb(**base_kwargs, mem_size=mem_size)
    elif strategy_name == 'cumulative':
        return Cumulative(**base_kwargs)
    elif strategy_name == 'joint':
        return JointTraining(**base_kwargs)
    elif strategy_name == 'icarl':
        return ICaRL(**base_kwargs, mem_size_per_class=20, buffer_transform=None, fixed_memory=True)
    elif strategy_name == 'slda':
        # SLDA requires special setup
        
        # Extract feature size based on model
        if hasattr(model, 'fc'):
            feature_size = model.fc.in_features
        elif hasattr(model, 'classifier'):
            if isinstance(model.classifier, nn.Linear):
                feature_size = model.classifier.in_features
            else:
                # Find last linear layer
                for module in reversed(list(model.classifier.modules())):
                    if isinstance(module, nn.Linear):
                        feature_size = module.in_features
                        break
        else:
            feature_size = 512  # default
        
        # Wrap model as feature extractor
        if not isinstance(model, FeatureExtractorBackbone):
            # Create a feature extractor from the model
            class FeatureExtractor(nn.Module):
                def __init__(self, base_model):
                    super().__init__()
                    self.base_model = base_model
                    # Remove the last layer
                    if hasattr(base_model, 'fc'):
                        self.base_model.fc = nn.Identity()
                    elif hasattr(base_model, 'classifier'):
                        if isinstance(base_model.classifier, nn.Sequential):
                            self.base_model.classifier[-1] = nn.Identity()
                        else:
                            self.base_model.classifier = nn.Identity()
                
                def forward(self, x):
                    return self.base_model(x)
            
            feature_extractor = FeatureExtractor(model)
            feature_extractor = feature_extractor.to(device)
        
        # SLDA doesn't use optimizer/criterion in the same way
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
            'train_epochs': 1  # SLDA typically uses 1 epoch
        }
        return StreamingLDA(**slda_kwargs)
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")


def run_training(benchmark_name='fmnist', strategy_name='naive', model_type='mlp',
                device='cuda', experiences=5, epochs=1, batch_size=32, 
                mem_size=200, lr=0.001, seed=42, verbose=True, **kwargs):
    """
    Run a complete training experiment and return results.
    
    Returns:
        dict: Results containing final accuracies and average accuracy
    """
    # Set seed
    torch.manual_seed(seed)
    
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
        mem_size=mem_size, epochs=epochs, batch_size=batch_size,
        num_classes=benchmark_info.num_classes
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