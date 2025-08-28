"""Unified component factory that automatically uses Avalanche built-ins or custom implementations."""

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional
from src.utils.benchmark_info import BenchmarkInfo


def auto_detect_feature_size(model: nn.Module, benchmark_info) -> int:
    """
    Auto-detect the feature size of a model by doing a forward pass.
    """
    model.eval()
    
    # Get the device the model is on
    device = next(model.parameters()).device if len(list(model.parameters())) > 0 else torch.device('cpu')
    
    # Create a dummy input on the same device
    # Use 3 channels for compatibility with RGB models
    channels = 3 if benchmark_info.channels == 1 else benchmark_info.channels
    dummy_input = torch.randn(1, channels, 
                             benchmark_info.image_size[0], 
                             benchmark_info.image_size[1], 
                             device=device)
    
    try:
        with torch.no_grad():
            features = model(dummy_input)
            if features.dim() > 2:
                features = features.view(features.size(0), -1)
            feature_size = features.shape[1]
            return feature_size
    except Exception as e:
        print(f"Warning: Could not auto-detect feature size: {e}")
        return 512  # Safe fallback


# Model requirements dictionaries
MODEL_INPUT_SIZES = {
    'ghostfacenetv2': (112, 112),
    'modified_mobilefacenet': (256, 256),
    'dwseesawfacev2': (256, 256),
    'efficientnet_b0': (224, 224),
    'efficientnet_b1': (240, 240),
    'efficientnet_b2': (260, 260),
    'efficientnet_b3': (300, 300),
    'efficientnet_b4': (380, 380),
    'resnet18': (224, 224),
    'resnet50': (224, 224),
    'mobilenet_v2': (224, 224),
}

MODEL_FEATURE_SIZES = {
    'ghostfacenetv2': 256,
    'modified_mobilefacenet': 512,
    'dwseesawfacev2': 512,
    'efficientnet_b0': 1280,
    'efficientnet_b1': 1280,
    'efficientnet_b2': 1408,
    'efficientnet_b3': 1536,
    'efficientnet_b4': 1792,
    'resnet18': 512,
    'resnet50': 2048,
    'mobilenet_v2': 1280,
}

# Strategy mappings
AVALANCHE_STRATEGIES = {
    'naive': 'Naive',
    'cumulative': 'Cumulative', 
    'joint': 'JointTraining',
    'replay': 'Replay',
    'icarl': 'ICaRL',
    'lwf': 'LwF',
    'ewc': 'EWC',
    'slda': 'StreamingLDA',
    'agem': 'AGEM',
    'gem': 'GEM',
    'mir': 'MIR',
    'gdumb': 'GDumb',
    'gss': 'GSS_greedy',
    'der': 'DER',
    'bic': 'BiC',
    'mas': 'MAS',
    'si': 'SynapticIntelligence',
}

CUSTOM_STRATEGIES = {
    # Add custom strategies here if needed
}

AVALANCHE_MODELS = {
    'mlp': 'SimpleMLP',
    'cnn': 'SimpleCNN', 
}

TORCHVISION_MODELS = [
    'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
    'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 
    'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7',
    'mobilenet_v2', 'mobilenet_v3_large', 'mobilenet_v3_small',
    'densenet121', 'densenet161', 'densenet169', 'densenet201',
    'vgg11', 'vgg13', 'vgg16', 'vgg19',
]

CUSTOM_MODELS = {
    'ghostfacenetv2': ('ghostfacenetV2', 'GhostFaceNetV2'),  # (module_file, class_name)
    'modified_mobilefacenet': ('modified_mobilefacenet', 'Modified_MobileFaceNet'),
    'dwseesawfacev2': ('DWseesawfaceV2', 'DWSeesawFaceV2'),
}



def create_benchmark_from_config(dataset_config: Dict[str, Any]):
    """Create a benchmark from dataset config - returns (benchmark, benchmark_info) consistently."""
    dataset_type = dataset_config.get('type', dataset_config.get('name', 'unknown'))
    
    # Check if it's an Avalanche built-in benchmark
    if dataset_type == 'splitmnist':
        from avalanche.benchmarks.classic import SplitMNIST
        n_experiences = dataset_config.get('n_experiences', 5)
        seed = dataset_config.get('seed', 42)
        benchmark = SplitMNIST(n_experiences=n_experiences, seed=seed)
        benchmark_info = BenchmarkInfo.from_avalanche_benchmark(benchmark)
        return benchmark, benchmark_info
        
    elif dataset_type == 'mnist':
        from avalanche.benchmarks.classic import SplitMNIST  
        seed = dataset_config.get('seed', 42)
        benchmark = SplitMNIST(n_experiences=1, seed=seed)
        benchmark_info = BenchmarkInfo.from_avalanche_benchmark(benchmark)
        return benchmark, benchmark_info
        
    elif dataset_type == 'splitcifar10':
        from avalanche.benchmarks.classic import SplitCIFAR10
        n_experiences = dataset_config.get('n_experiences', 5)
        seed = dataset_config.get('seed', 42)
        benchmark = SplitCIFAR10(n_experiences=n_experiences, seed=seed)
        benchmark_info = BenchmarkInfo.from_avalanche_benchmark(benchmark)
        return benchmark, benchmark_info
        
    # Custom benchmarks - these already return (benchmark, BenchmarkInfo)
    elif dataset_type == 'smarteye_crop':
        from src.datasets.smarteye import create_smarteye_benchmark
        params = dataset_config.get('params', {})
        
        # Extract path from dataset config if provided
        if 'path' in dataset_config:
            params['root_dir'] = dataset_config['path']
        
        # Extract other common dataset parameters
        if 'test_split' in dataset_config:
            params['test_split'] = dataset_config['test_split']
        if 'n_experiences' in dataset_config:
            params['n_experiences'] = dataset_config['n_experiences']
        if 'seed' in dataset_config:
            params['seed'] = dataset_config['seed']
        if 'image_size' in dataset_config:
            params['image_size'] = tuple(dataset_config['image_size'])
            
        # Custom datasets already return (benchmark, BenchmarkInfo)
        return create_smarteye_benchmark(**params)
        
    elif dataset_type == 'lfw':
        from src.datasets.lfw import create_lfw_benchmark
        params = dataset_config.get('params', {})
        
        # Extract common dataset parameters
        if 'n_experiences' in dataset_config:
            params['n_experiences'] = dataset_config['n_experiences']
        if 'test_split' in dataset_config:
            params['test_split'] = dataset_config['test_split']
        if 'seed' in dataset_config:
            params['seed'] = dataset_config['seed']
        if 'image_size' in dataset_config:
            params['image_size'] = tuple(dataset_config['image_size'])
        if 'min_faces_per_person' in dataset_config:
            params['min_faces_per_person'] = dataset_config['min_faces_per_person']
            
        # LFW also returns (benchmark, BenchmarkInfo)
        return create_lfw_benchmark(**params)
        
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


def create_model_from_config(model_config: Dict[str, Any], benchmark_info: BenchmarkInfo):
    """Create a model from config."""
    model_type = model_config.get('type', model_config.get('name', 'unknown'))
    
    # Check Avalanche built-in models
    if model_type in AVALANCHE_MODELS:
        from avalanche.models import SimpleMLP, SimpleCNN
        avalanche_class = AVALANCHE_MODELS[model_type]
        
        if avalanche_class == 'SimpleMLP':
            input_size = benchmark_info.channels * benchmark_info.image_size[0] * benchmark_info.image_size[1]
            model = SimpleMLP(input_size=input_size, num_classes=benchmark_info.num_classes)
        elif avalanche_class == 'SimpleCNN':
            model = SimpleCNN(num_classes=benchmark_info.num_classes)
        
        print(f"Created Avalanche {avalanche_class}: {model_type}")
        return model
        
    # Check torchvision models
    elif model_type in TORCHVISION_MODELS:
        import torchvision.models as tv_models
        model_fn = getattr(tv_models, model_type)
        
        # Use new weights parameter instead of deprecated pretrained
        try:
            # Try to get the default weights for this model
            weights_attr = f"{model_type.upper()}_Weights"
            if hasattr(tv_models, weights_attr):
                weights_enum = getattr(tv_models, weights_attr)
                model = model_fn(weights=weights_enum.DEFAULT)
            else:
                # Fallback for models without weights enum
                model = model_fn(weights='DEFAULT')
        except:
            # Final fallback to None (no pretrained weights)
            model = model_fn(weights=None)
        
        # Update final layer for correct number of classes
        if hasattr(model, 'classifier'):
            if isinstance(model.classifier, nn.Sequential):
                in_features = model.classifier[-1].in_features
                model.classifier[-1] = nn.Linear(in_features, benchmark_info.num_classes)
            else:
                in_features = model.classifier.in_features
                model.classifier = nn.Linear(in_features, benchmark_info.num_classes)
        elif hasattr(model, 'fc'):
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, benchmark_info.num_classes)
            
        print(f"Created torchvision {model_type}: {model_type}")
        return model
        
    # Check custom models
    elif model_type in CUSTOM_MODELS:
        import importlib
        
        module_file, class_name = CUSTOM_MODELS[model_type]
        
        try:
            # Import from the backbones package in the project
            module = importlib.import_module(f'backbones.{module_file}')
            model_class = getattr(module, class_name)
            model = model_class(num_classes=benchmark_info.num_classes)
            print(f"Created custom model: {model_type}")
            return model
        except (ImportError, AttributeError) as e:
            print(f"Failed to import {model_type} from backbones.{module_file}: {e}")
            raise ValueError(f"Could not create custom model {model_type}: {e}")
        
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def create_strategy_from_config(strategy_config: Dict[str, Any], model, benchmark_info, 
                               optimizer, criterion, eval_plugin, device: str = 'cuda', **kwargs):
    """
    Create a strategy based on config using unified factory.
    
    Args:
        strategy_config: Strategy configuration with name, type, params
        model: The model to use 
        benchmark_info: Benchmark information
        optimizer: Optimizer instance
        criterion: Loss criterion  
        eval_plugin: Evaluation plugin
        device: Device to use
        **kwargs: Additional training parameters
        
    Returns:
        Strategy instance
    """
    strategy_type = strategy_config.get('type', strategy_config.get('name', 'unknown'))
    
    # Merge strategy params with kwargs, strategy params take precedence
    all_params = {**kwargs}
    all_params.update(strategy_config.get('params', {}))
    
    print(f"Creating strategy: {strategy_type}")
    
    # Check if it's an Avalanche built-in strategy  
    if strategy_type in AVALANCHE_STRATEGIES:
        avalanche_class_name = AVALANCHE_STRATEGIES[strategy_type]
        
        # Import the strategy class dynamically
        try:
            from avalanche.training.supervised import (
                Naive, Cumulative, JointTraining, Replay, ICaRL, MIR, GDumb, 
                GSS_greedy, GenerativeReplay, LwF, EWC, SynapticIntelligence, 
                MAS, LFL, AGEM, GEM, BiC, CoPE, CWRStar, PackNet, PNNStrategy, 
                DER, ER_ACE, ER_AML, LearningToPrompt, SCR, MER, AR1
            )
            # IL2M might not exist in all Avalanche versions
            try:
                from avalanche.training.supervised import IL2M
            except ImportError:
                IL2M = None
            from avalanche.training.supervised.deep_slda import StreamingLDA
            
            # Get the actual class
            strategy_class = locals()[avalanche_class_name]
            
            # Special handling for SLDA - needs custom model wrapper
            if strategy_type == 'slda':
                # Get model type from kwargs or strategy config
                model_type_name = kwargs.get('model_type', strategy_config.get('model_type', 'unknown'))
                
                # Create SLDA-specific model wrapper
                if hasattr(model, 'features') and hasattr(model, 'classifier') and hasattr(model, '_input_size'):
                    # Standard Avalanche models like SimpleMLP - use features + proper input handling
                    class SLDAFeatureWrapper(nn.Module):
                        def __init__(self, base_model):
                            super().__init__()
                            self.base_model = base_model
                            
                        def forward(self, x):
                            # Handle input reshaping like SimpleMLP does
                            x = x.contiguous()
                            x = x.view(x.size(0), self.base_model._input_size)
                            return self.base_model.features(x)
                    
                    slda_model = SLDAFeatureWrapper(model)
                    # Get feature size from classifier
                    if isinstance(model.classifier, nn.Sequential):
                        # Find the last Linear layer in Sequential
                        for layer in reversed(model.classifier):
                            if isinstance(layer, nn.Linear):
                                feature_size = layer.in_features
                                break
                        else:
                            feature_size = 1000  # fallback
                    else:
                        feature_size = model.classifier.in_features
                    print(f"Using Avalanche model features for SLDA: feature_size={feature_size}")
                else:
                    # Other models (torchvision, custom)
                    class SLDAModelWrapper(nn.Module):
                        """
                        Wrapper for models to handle:
                        1. Channel conversion (grayscale to RGB)
                        2. Image resizing to model requirements
                        3. Feature extraction from penultimate layer
                        """
                        def __init__(self, model, model_type):
                            super().__init__()
                            self.model = model
                            self.model_type = model_type
                            
                            # Get requirements from dictionaries
                            self.target_size = MODEL_INPUT_SIZES.get(model_type, None)
                            
                            # Determine if RGB conversion needed
                            # Face models that can handle grayscale
                            grayscale_capable = ['ghostfacenetv2']
                            self.needs_rgb = model_type not in grayscale_capable
                            
                        def forward(self, x):
                            # Handle grayscale to RGB conversion if needed
                            if x.shape[1] == 1 and self.needs_rgb:
                                # SmartEye IR images are grayscale (1 channel)
                                # Convert to pseudo-RGB by duplicating the channel 3 times: (gray, gray, gray)
                                # This is equivalent to torch.cat([x, x, x], dim=1) but more efficient
                                x = x.repeat(1, 3, 1, 1)
                                # Now x has shape [batch, 3, H, W] for RGB models
                            
                            # Resize input if needed (after channel conversion)
                            if self.target_size:
                                x = nn.functional.interpolate(x, size=self.target_size, mode='bilinear', align_corners=False)
                            
                            # Get penultimate layer features (not final classification)
                            return self.get_features(x)
                            
                        def get_features(self, x):
                            # Extract features from penultimate layer
                            if hasattr(self.model, 'classifier'):
                                # Models with classifier (EfficientNet, etc)
                                modules = list(self.model.children())[:-1]
                                feature_extractor = nn.Sequential(*modules)
                                features = feature_extractor(x)
                            elif hasattr(self.model, 'fc'):
                                # ResNet-style models
                                modules = list(self.model.children())[:-1]
                                feature_extractor = nn.Sequential(*modules)
                                features = feature_extractor(x)
                            else:
                                # Custom models - assume they output features directly
                                features = self.model(x)
                            
                            # Flatten if needed
                            if features.dim() > 2:
                                features = features.view(features.size(0), -1)
                                
                            return features
                    
                    slda_model = SLDAModelWrapper(model, model_type_name)
                    
                    # Move wrapper to same device as model
                    model_device = next(model.parameters()).device if len(list(model.parameters())) > 0 else device
                    slda_model = slda_model.to(model_device)
                    
                    # Look up feature size from dictionary, only auto-detect if unknown
                    feature_size = MODEL_FEATURE_SIZES.get(model_type_name)
                    if feature_size:
                        print(f"Using known feature size for {model_type_name}: {feature_size}")
                    else:
                        # Only auto-detect for unknown models
                        feature_size = auto_detect_feature_size(slda_model, benchmark_info)
                        print(f"Auto-detected feature size for {model_type_name}: {feature_size}")
                        
                    print(f"Using wrapped model for SLDA: feature_size={feature_size}")
                
                # SLDA-specific parameters
                base_params = {
                    'slda_model': slda_model,
                    'criterion': criterion,
                    'input_size': feature_size,
                    'num_classes': benchmark_info.num_classes,
                    'shrinkage_param': all_params.get('shrinkage_param', 1e-4),
                    'streaming_update_sigma': all_params.get('streaming_update_sigma', True),
                    'train_mb_size': all_params.get('batch_size', 32),
                    'eval_mb_size': all_params.get('batch_size', 32) * 2,
                    'device': device,
                    'evaluator': eval_plugin
                }
                
            else:
                # Standard parameters all other strategies accept
                base_params = {
                    'model': model,
                    'optimizer': optimizer, 
                    'criterion': criterion,
                    'train_mb_size': all_params.get('batch_size', 32),
                    'eval_mb_size': all_params.get('batch_size', 32) * 2,
                    'device': device,
                    'evaluator': eval_plugin,
                    'train_epochs': all_params.get('epochs', 1)
                }
                
                # Add strategy-specific parameters
                if strategy_type == 'replay':
                    base_params['mem_size'] = all_params.get('mem_size', 500)
                elif strategy_type == 'ewc':
                    base_params['ewc_lambda'] = all_params.get('ewc_lambda', 0.4)
                elif strategy_type == 'icarl':
                    base_params['memory_size'] = all_params.get('memory_size', 2000)
                elif strategy_type == 'lwf':
                    base_params['alpha'] = all_params.get('alpha', 1.0)
                    base_params['temperature'] = all_params.get('temperature', 2.0)
                elif strategy_type == 'agem':
                    base_params['patterns_per_exp'] = all_params.get('patterns_per_exp', 256)
                    base_params['sample_size'] = all_params.get('sample_size', 256) 
                elif strategy_type == 'gem':
                    base_params['patterns_per_exp'] = all_params.get('patterns_per_exp', 256)
                    base_params['memory_strength'] = all_params.get('memory_strength', 0.5)
                
            # Create the strategy
            strategy = strategy_class(**base_params)
            print(f"Created Avalanche {avalanche_class_name}: {strategy_type}")
            return strategy
            
        except (ImportError, KeyError) as e:
            print(f"Could not create Avalanche strategy {strategy_type}: {e}")
            # Fall through to custom handling but also check if we should raise
            if strategy_type not in CUSTOM_STRATEGIES:
                raise ValueError(f"Failed to create strategy {strategy_type}: {e}")
    
    # Check if it's a custom strategy
    elif strategy_type in CUSTOM_STRATEGIES:
        from src.training import create_strategy_legacy
        return create_strategy_legacy(strategy_type, model, optimizer, criterion, 
                                    eval_plugin, device, **all_params)
    
    else:
        raise ValueError(f"Unknown strategy type: {strategy_type}")