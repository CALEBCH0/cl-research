"""Unified component factory that automatically uses Avalanche built-ins or custom implementations."""

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional
from src.utils.benchmark_info import BenchmarkInfo


def _adapt_model_input_channels(model: nn.Module, expected_channels: int, model_type: str) -> nn.Module:
    """
    Efficiently adapt a model's first convolutional layer to handle different input channels.
    
    Args:
        model: The model to adapt
        expected_channels: Number of channels the model will actually receive (e.g., 3 after grayscale->RGB conversion)
        model_type: Type of model for specific handling
        
    Returns:
        Adapted model
    """
    import torch.nn as nn
    
    # Find the first convolutional layer
    first_conv = None
    first_conv_name = None
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            first_conv = module
            first_conv_name = name
            break
    
    if first_conv is None:
        print(f"Warning: No Conv2d layer found in {model_type}")
        return model
    
    current_in_channels = first_conv.in_channels
    
    if current_in_channels == expected_channels:
        # Already matches, no adaptation needed
        return model
    
    print(f"Adapting {model_type}: {current_in_channels} -> {expected_channels} input channels")
    
    # Create new first conv layer with correct input channels
    new_first_conv = nn.Conv2d(
        in_channels=expected_channels,
        out_channels=first_conv.out_channels,
        kernel_size=first_conv.kernel_size,
        stride=first_conv.stride,
        padding=first_conv.padding,
        dilation=first_conv.dilation,
        groups=first_conv.groups,
        bias=first_conv.bias is not None,
        padding_mode=first_conv.padding_mode
    )
    
    # Smart weight initialization: handle channel dimension change
    with torch.no_grad():
        if expected_channels > current_in_channels:
            # Expanding channels: replicate weights across new channels
            old_weight = first_conv.weight.data
            new_weight = old_weight.repeat(1, expected_channels // current_in_channels, 1, 1)
            # Handle remainder channels
            remainder = expected_channels % current_in_channels
            if remainder > 0:
                new_weight = torch.cat([new_weight, old_weight[:, :remainder, :, :]], dim=1)
            new_first_conv.weight.data = new_weight / (expected_channels / current_in_channels)  # Normalize to maintain magnitude
        else:
            # Reducing channels: average across input channels
            old_weight = first_conv.weight.data
            new_first_conv.weight.data = old_weight[:, :expected_channels, :, :].clone()
        
        # Copy bias if it exists
        if first_conv.bias is not None:
            new_first_conv.bias.data = first_conv.bias.data.clone()
    
    # Replace the first conv layer in the model
    _replace_module_by_name(model, first_conv_name, new_first_conv)
    
    return model


def _add_model_preprocessing(model: nn.Module, model_type: str, benchmark_info) -> nn.Module:
    """
    Add preprocessing wrapper to models to handle dataset-to-model requirements.
    
    Args:
        model: The base model
        model_type: Type of model
        benchmark_info: Dataset information
        
    Returns:
        Model with preprocessing wrapper
    """
    class ModelWithPreprocessing(nn.Module):
        def __init__(self, base_model, model_type, benchmark_info):
            super().__init__()
            self.base_model = base_model
            self.model_type = model_type
            
            # Get model's required input format (height, width, channels)
            model_input_spec = MODEL_INPUT_SIZES.get(model_type, (*benchmark_info.image_size, benchmark_info.channels))
            self.target_size = (model_input_spec[0], model_input_spec[1])  # (height, width)
            self.target_channels = model_input_spec[2] if len(model_input_spec) > 2 else benchmark_info.channels
            
            # Determine if channel conversion is needed
            self.needs_channel_conversion = (
                benchmark_info.channels != self.target_channels
            )
            
        def forward(self, x):
            # 1. Channel conversion if needed
            if self.needs_channel_conversion:
                if x.shape[1] == 1 and self.target_channels == 3:
                    # Convert grayscale to RGB by duplicating channels
                    x = x.repeat(1, 3, 1, 1)
                elif x.shape[1] == 3 and self.target_channels == 1:
                    # Convert RGB to grayscale (unlikely but handled)
                    x = x.mean(dim=1, keepdim=True)
            
            # 2. Resize to model's required size if needed
            if len(x.shape) >= 4:  # Check if x has spatial dimensions
                current_size = (x.shape[2], x.shape[3])
                if current_size != self.target_size:
                    x = nn.functional.interpolate(
                        x, size=self.target_size, 
                        mode='bilinear', align_corners=False
                    )
            
            # 3. Forward through base model
            return self.base_model(x)
    
    return ModelWithPreprocessing(model, model_type, benchmark_info)


def _create_feature_extractor(model: nn.Module, model_type: str, benchmark_info) -> nn.Module:
    """
    Create a feature extractor from a model for SLDA.
    Models already have preprocessing built-in, so just extract features.
    
    Args:
        model: Model with preprocessing
        model_type: Type of model
        benchmark_info: Dataset information
        
    Returns:
        Feature extractor model
    """
    class FeatureExtractor(nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model
            
        def forward(self, x):
            # If the base model has preprocessing wrapper, it handles input conversion
            # If it's an Avalanche model, handle reshaping
            if hasattr(self.base_model, '_input_size'):
                # Avalanche models like SimpleMLP need input reshaping
                x = x.contiguous().view(x.size(0), self.base_model._input_size)
                return self.base_model.features(x)
            elif hasattr(self.base_model, 'base_model'):
                # Our preprocessing wrapper - extract features from the wrapped model
                wrapped_model = self.base_model.base_model
                
                # Apply preprocessing manually to input (don't run full model)
                if hasattr(self.base_model, 'needs_channel_conversion') and self.base_model.needs_channel_conversion:
                    if hasattr(self.base_model, 'target_channels'):
                        if x.shape[1] == 1 and self.base_model.target_channels == 3:
                            x = x.repeat(1, 3, 1, 1)
                        elif x.shape[1] == 3 and self.base_model.target_channels == 1:
                            x = x.mean(dim=1, keepdim=True)
                
                if hasattr(self.base_model, 'target_size') and len(x.shape) >= 4:
                    current_size = (x.shape[2], x.shape[3])
                    if current_size != self.base_model.target_size:
                        x = nn.functional.interpolate(x, size=self.base_model.target_size, mode='bilinear', align_corners=False)
                
                # Extract features from preprocessed input
                return self._extract_features(wrapped_model, x)
            else:
                # Standard model - extract features directly
                return self._extract_features(self.base_model, x)
        
        def _extract_features(self, model, x):
            """Extract features from penultimate layer."""
            if hasattr(model, 'classifier'):
                # Models with classifier (EfficientNet, etc)
                modules = list(model.children())[:-1]
                feature_extractor = nn.Sequential(*modules)
                features = feature_extractor(x)
            elif hasattr(model, 'fc'):
                # ResNet-style models
                modules = list(model.children())[:-1]
                feature_extractor = nn.Sequential(*modules)
                features = feature_extractor(x)
            else:
                # For face recognition models, use final embeddings (fast & effective)
                # These models are designed to output meaningful feature embeddings
                features = model(x)
            
            # Flatten if needed
            if features.dim() > 2:
                features = features.view(features.size(0), -1)
                
            return features
    
    return FeatureExtractor(model)


def _replace_module_by_name(model: nn.Module, target_name: str, new_module: nn.Module):
    """Replace a module in the model by its name."""
    name_parts = target_name.split('.')
    parent = model
    
    # Navigate to the parent of the target module
    for part in name_parts[:-1]:
        parent = getattr(parent, part)
    
    # Replace the target module
    setattr(parent, name_parts[-1], new_module)


def auto_detect_feature_size(model: nn.Module, benchmark_info, model_type: str = None) -> int:
    """
    Auto-detect the feature size of a model by doing a forward pass.
    """
    model.eval()
    
    # Get the device the model is on
    device = next(model.parameters()).device if len(list(model.parameters())) > 0 else torch.device('cpu')
    
    # Create a dummy input on the same device
    # Get model's required input format
    model_input_spec = MODEL_INPUT_SIZES.get(model_type, (*benchmark_info.image_size, benchmark_info.channels))
    target_height, target_width = model_input_spec[0], model_input_spec[1]
    target_channels = model_input_spec[2] if len(model_input_spec) > 2 else benchmark_info.channels
    
    dummy_input = torch.randn(1, target_channels, 
                             target_height, 
                             target_width, 
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
    # Format: (height, width, channels) - channels: 1=grayscale, 3=RGB
    'ghostfacenetv2': (112, 112, 1),
    'modified_mobilefacenet': (256, 256, 1),
    'dwseesawfacev2': (256, 256, 1),
    'efficientnet_b0': (224, 224, 3),
    'efficientnet_b1': (240, 240, 3),
    'efficientnet_b2': (260, 260, 3),
    'efficientnet_b3': (300, 300, 3),
    'efficientnet_b4': (380, 380, 3),
    'resnet18': (224, 224, 3),
    'resnet50': (224, 224, 3),
    'mobilenet_v2': (224, 224, 3),
}

MODEL_FEATURE_SIZES = {
    # Face recognition models - confirmed by auto-detection
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
        
        # Store model type for later use 
        model._model_type = model_type
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
            
        # Add preprocessing wrapper to handle channel/size conversions
        model = _add_model_preprocessing(model, model_type, benchmark_info)
        
        # Store model type for later use 
        model._model_type = model_type
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
            
            # Handle different constructor signatures for each model
            if model_type == 'ghostfacenetv2':
                # GhostFaceNetV2: num_features is embedding dim (e.g. 256), not num_classes
                # Use model's required input size for optimal performance
                required_size = MODEL_INPUT_SIZES.get(model_type, benchmark_info.image_size)
                model = model_class(
                    image_size=required_size[0],  # Use model's required size (assuming square)
                    num_features=256,  # Standard face embedding dimension
                    channels=benchmark_info.channels
                )
            elif model_type == 'modified_mobilefacenet':
                # Modified_MobileFaceNet: num_features is embedding dim (e.g. 512), not num_classes  
                # Use model's required input size for proper architecture compatibility
                required_size = MODEL_INPUT_SIZES.get(model_type, benchmark_info.image_size)
                model = model_class(
                    input_size=(benchmark_info.channels, *required_size),
                    num_features=512  # Standard face embedding dimension
                )
            elif model_type == 'dwseesawfacev2':
                # Need to check DWSeesawFaceV2 constructor
                model = model_class(num_classes=benchmark_info.num_classes)
            else:
                # Default case
                model = model_class(num_classes=benchmark_info.num_classes)
            
            # Wrap model with preprocessing to handle dataset requirements
            model = _add_model_preprocessing(model, model_type, benchmark_info)
            
            # Store model type for later use (after preprocessing wrapper)
            model._model_type = model_type
            print(f"Created custom model: {model_type}")
            return model
        except (ImportError, AttributeError) as e:
            print(f"Failed to import {model_type} from backbones.{module_file}: {e}")
            raise ValueError(f"Could not create custom model {model_type}: {e}")
        except Exception as e:
            print(f"Failed to create {model_type} model: {e}")
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
            
            # SLDA uses different parameters but no longer needs custom wrapper
            if strategy_type == 'slda':
                # Get model type - try multiple sources in order of preference
                model_type_name = (
                    kwargs.get('model_type') or                    # 1. Explicit parameter
                    strategy_config.get('model_type') or          # 2. Strategy config  
                    getattr(model, '_model_type', None) or        # 3. Stored in model during creation
                    'unknown'                                      # 4. Fallback
                )
                
                print(f"Debug: Using model_type_name: {model_type_name}")
                
                # Create feature extractor for SLDA (penultimate layer)
                feature_extractor = _create_feature_extractor(model, model_type_name, benchmark_info)
                
                # Get feature size from dictionary or auto-detect
                feature_size = MODEL_FEATURE_SIZES.get(model_type_name)
                if not feature_size:
                    feature_size = auto_detect_feature_size(feature_extractor, benchmark_info, model_type_name)
                    print(f"Auto-detected feature size for {model_type_name}: {feature_size}")
                else:
                    print(f"Using known feature size for {model_type_name}: {feature_size}")
                
                # SLDA-specific parameters
                base_params = {
                    'slda_model': feature_extractor,
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