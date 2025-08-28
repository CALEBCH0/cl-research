"""Unified component factory that automatically uses Avalanche built-ins or custom implementations."""

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional


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
                             benchmark_info.input_size, benchmark_info.input_size).to(device)
    
    with torch.no_grad():
        try:
            # For wrapped models that handle channel conversion, use original channels
            if hasattr(model, 'get_features') or isinstance(model, nn.Module) and 'SLDAModelWrapper' in str(type(model)):
                dummy_input = torch.randn(1, benchmark_info.channels, 
                                         benchmark_info.input_size, benchmark_info.input_size).to(device)
                features = model(dummy_input)
            # Try to get features from the model
            elif hasattr(model, 'features'):
                features = model.features(dummy_input)
            elif hasattr(model, 'feature_extractor'):
                features = model.feature_extractor(dummy_input)
            else:
                # For models without separate feature extractor, get penultimate layer output
                features = get_penultimate_features(model, dummy_input)
            
            # Flatten and get feature dimension
            if features.dim() > 2:
                features = features.view(features.size(0), -1)
            
            feature_size = features.shape[1]
            
            # Sanity check - feature size should be reasonable
            if feature_size > 100000:
                print(f"Warning: Detected unreasonably large feature size {feature_size}, using fallback")
                # Common feature sizes for known models
                return 512  # Conservative default
                
            return feature_size
        except Exception as e:
            print(f"Warning: Could not auto-detect feature size: {e}")
            # Better fallback based on known models
            return 512  # Most face models use 512 features


def get_penultimate_features(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    Extract features from the penultimate layer of a model.
    """
    # For models with classifier/fc layers, extract features before final layer
    if hasattr(model, 'classifier'):
        # Remove the final classifier and run forward
        if isinstance(model.classifier, nn.Sequential):
            # Create a new model without the last layer
            feature_extractor = nn.Sequential(*list(model.children())[:-1])
            return feature_extractor(x)
        else:
            # Single classifier layer - need to find the layer before it
            modules = list(model.children())
            feature_extractor = nn.Sequential(*modules[:-1])
            features = feature_extractor(x)
            return features.view(features.size(0), -1) if features.dim() > 2 else features
    elif hasattr(model, 'fc'):
        # ResNet-style models
        modules = list(model.children())
        feature_extractor = nn.Sequential(*modules[:-1])
        features = feature_extractor(x)
        return features.view(features.size(0), -1) if features.dim() > 2 else features
    else:
        # Fallback: run full model and hope it's already features
        return model(x)


# =====================================
# AVALANCHE COMPONENT MAPPINGS
# =====================================

AVALANCHE_MODELS = {
    # Basic models
    'mlp': 'SimpleMLP',
    'cnn': 'SimpleCNN',
    'simplecnn': 'SimpleCNN',
    'simplemlp': 'SimpleMLP',
    
    # MobileNet variants
    'mobilenetv1': 'MobilenetV1',
    
    # ResNet variants  
    'resnet18': 'pytorchcv_resnet18',
    'resnet50': 'pytorchcv_resnet50', 
    'resnet32': 'resnet32',
    'slimresnet18': 'SlimResNet18',
    
    # Specialized models
    'lenet5': 'LeNet5',
    'ncm_classifier': 'NCMClassifier',
    'fecam_classifier': 'FeCAMClassifier',
    'sldaresnet': 'SLDAResNetModel',
    'icarl_resnet': 'IcarlNet',
    'expert_gate': 'ExpertGate',
}

# Model feature dimensions (output of penultimate layer)
MODEL_FEATURE_SIZES = {
    # Face recognition models
    'ghostfacenetv2': 256,
    'modified_mobilefacenet': 512,
    'dwseesawfacev2': 512,
    
    # EfficientNet family
    'efficientnet_b0': 1280,
    'efficientnet_b1': 1280,
    'efficientnet_b2': 1408,
    'efficientnet_b3': 1536,
    'efficientnet_b4': 1792,
    'efficientnet_b5': 2048,
    'efficientnet_b6': 2304,
    'efficientnet_b7': 2560,
    
    # ResNet family
    'resnet18': 512,
    'resnet34': 512,
    'resnet50': 2048,
    'resnet101': 2048,
    'resnet152': 2048,
    
    # MobileNet family
    'mobilenet_v2': 1280,
    'mobilenet_v3_large': 1280,
    'mobilenet_v3_small': 1024,
    
    # VGG family (before classifier)
    'vgg11': 4096,
    'vgg13': 4096,
    'vgg16': 4096,
    'vgg19': 4096,
    
    # DenseNet family
    'densenet121': 1024,
    'densenet161': 2208,
    'densenet169': 1664,
    'densenet201': 1920,
}

# Model input size requirements
MODEL_INPUT_SIZES = {
    # Strict requirements (architecture constraints)
    'modified_mobilefacenet': (256, 256),  # 16x16 kernel requires this
    'dwseesawfacev2': (256, 256),
    
    # Optimal sizes (trained on these)
    'efficientnet_b0': (224, 224),
    'efficientnet_b1': (240, 240),
    'efficientnet_b2': (260, 260),
    'efficientnet_b3': (300, 300),
    'efficientnet_b4': (380, 380),
    
    # Face models
    'ghostfacenetv2': (112, 112),
    
    # Standard models (flexible but optimal at 224)
    'resnet18': (224, 224),
    'resnet50': (224, 224),
    'mobilenet_v2': (224, 224),
    'vgg16': (224, 224),
}

TORCHVISION_MODELS = {
    # ResNet family
    'resnet18_tv': 'resnet18',
    'resnet34': 'resnet34', 
    'resnet50_tv': 'resnet50',
    'resnet101': 'resnet101',
    'resnet152': 'resnet152',
    'resnext50_32x4d': 'resnext50_32x4d',
    'resnext101_32x8d': 'resnext101_32x8d',
    'wide_resnet50_2': 'wide_resnet50_2',
    'wide_resnet101_2': 'wide_resnet101_2',
    
    # EfficientNet family
    'efficientnet_b0': 'efficientnet_b0',
    'efficientnet_b1': 'efficientnet_b1',
    'efficientnet_b2': 'efficientnet_b2',
    'efficientnet_b3': 'efficientnet_b3',
    'efficientnet_b4': 'efficientnet_b4',
    'efficientnet_b5': 'efficientnet_b5',
    'efficientnet_b6': 'efficientnet_b6',
    'efficientnet_b7': 'efficientnet_b7',
    'efficientnet_v2_s': 'efficientnet_v2_s',
    'efficientnet_v2_m': 'efficientnet_v2_m',
    'efficientnet_v2_l': 'efficientnet_v2_l',
    
    # MobileNet family
    'mobilenet_v2': 'mobilenet_v2',
    'mobilenet_v3_large': 'mobilenet_v3_large',
    'mobilenet_v3_small': 'mobilenet_v3_small',
    
    # VGG family
    'vgg11': 'vgg11',
    'vgg11_bn': 'vgg11_bn',
    'vgg13': 'vgg13', 
    'vgg13_bn': 'vgg13_bn',
    'vgg16': 'vgg16',
    'vgg16_bn': 'vgg16_bn',
    'vgg19': 'vgg19',
    'vgg19_bn': 'vgg19_bn',
    
    # DenseNet family
    'densenet121': 'densenet121',
    'densenet161': 'densenet161',
    'densenet169': 'densenet169',
    'densenet201': 'densenet201',
    
    # Vision Transformer
    'vit_b_16': 'vit_b_16',
    'vit_b_32': 'vit_b_32',
    'vit_l_16': 'vit_l_16',
    'vit_l_32': 'vit_l_32',
    'vit_h_14': 'vit_h_14',
    
    # ConvNeXt
    'convnext_tiny': 'convnext_tiny',
    'convnext_small': 'convnext_small',
    'convnext_base': 'convnext_base',
    'convnext_large': 'convnext_large',
    
    # RegNet
    'regnet_y_400mf': 'regnet_y_400mf',
    'regnet_y_800mf': 'regnet_y_800mf',
    'regnet_y_1_6gf': 'regnet_y_1_6gf',
    'regnet_y_3_2gf': 'regnet_y_3_2gf',
    'regnet_y_8gf': 'regnet_y_8gf',
    'regnet_y_16gf': 'regnet_y_16gf',
    'regnet_y_32gf': 'regnet_y_32gf',
    
    # MaxViT
    'maxvit_t': 'maxvit_t',
    
    # Swin Transformer
    'swin_t': 'swin_t',
    'swin_s': 'swin_s',
    'swin_b': 'swin_b',
    'swin_v2_t': 'swin_v2_t',
    'swin_v2_s': 'swin_v2_s',
    'swin_v2_b': 'swin_v2_b',
}

AVALANCHE_STRATEGIES = {
    # Basic strategies
    'naive': 'Naive',
    'cumulative': 'Cumulative',
    'joint_training': 'JointTraining',
    'fromscratch': 'FromScratchTraining',
    
    # Memory-based strategies
    'replay': 'Replay', 
    'icarl': 'ICaRL',
    'mir': 'MIR',
    'gdumb': 'GDumb',
    'gss_greedy': 'GSS_greedy',
    'generative_replay': 'GenerativeReplay',
    'il2m': 'IL2M',
    
    # Regularization strategies
    'lwf': 'LwF',
    'ewc': 'EWC', 
    'synaptic_intelligence': 'SynapticIntelligence',
    'mas': 'MAS',
    'lfl': 'LFL',
    
    # Gradient-based strategies
    'agem': 'AGEM',
    'gem': 'GEM',
    
    # Other strategies
    'slda': 'StreamingLDA',
    'bic': 'BiC',
    'cope': 'CoPE',
    'cwr_star': 'CWRStar',
    'packnet': 'PackNet',
    'pnn': 'PNNStrategy',
    'der': 'DER',
    'er_ace': 'ER_ACE',
    'er_aml': 'ER_AML',
    'l2p': 'LearningToPrompt',
    'scr': 'SCR',
    'mer': 'MER',
    'ar1': 'AR1',
}

AVALANCHE_BENCHMARKS = {
    'mnist': 'SplitMNIST',
    'cifar10': 'SplitCIFAR10', 
    'cifar100': 'SplitCIFAR100',
    'fashion_mnist': 'SplitFashionMNIST',
    'cub200': 'SplitCUB200',
    'tiny_imagenet': 'SplitTinyImageNet',
    'core50': 'CORe50',
    'openloris': 'OpenLORIS',
    'stream51': 'Stream51',
    'omniglot': 'SplitOmniglot',
    'clear': 'CLEARBenchmark',
    'endless_cl_sim': 'EndlessCLSimBenchmark',
}

CUSTOM_MODELS = {
    'dwseesawfacev2': 'src.training.create_model',
    'ghostfacenetv2': 'src.training.create_model',
    'modified_mobilefacenet': 'src.training.create_model'
}

CUSTOM_STRATEGIES = {
    'icarl': 'src.training.create_strategy',
    'ewc': 'src.training.create_strategy'
}

CUSTOM_BENCHMARKS = {
    'smarteye_crop': 'src.datasets.smarteye_cached.create_smarteye_benchmark_cached',
    'smarteye_raw': 'src.datasets.smarteye_cached.create_smarteye_benchmark_cached',
    'lfw': 'src.datasets.lfw.create_lfw_benchmark'
}


# =====================================
# COMPONENT FACTORY FUNCTIONS
# =====================================

def create_model_from_config(model_config: Dict[str, Any], benchmark_info) -> nn.Module:
    """
    Create model using Avalanche built-ins when possible, custom otherwise.
    
    Args:
        model_config: Config with 'type', 'params', etc.
        benchmark_info: BenchmarkInfo object with dataset info
        
    Returns:
        PyTorch model
    """
    model_type = model_config.get('type', model_config.get('name', 'mlp'))
    params = model_config.get('params', {})
    
    # Check if it's an Avalanche built-in model
    if model_type in AVALANCHE_MODELS:
        avalanche_class = AVALANCHE_MODELS[model_type]
        
        if model_type == 'mlp':
            from avalanche.models import SimpleMLP
            model = SimpleMLP(
                num_classes=benchmark_info.num_classes,
                input_size=benchmark_info.input_size,
                hidden_size=params.get('hidden_size', 400),
                hidden_layers=params.get('hidden_layers', 2)
            )
            print(f"Created Avalanche {avalanche_class}: {model_type}")
            return model
            
        elif model_type == 'cnn':
            from avalanche.models import SimpleCNN
            model = SimpleCNN(
                num_classes=benchmark_info.num_classes,
                input_channels=benchmark_info.channels
            )
            print(f"Created Avalanche {avalanche_class}: {model_type}")
            return model
            
        elif model_type == 'mobilenetv1':
            from avalanche.models import MobilenetV1
            model = MobilenetV1(
                num_classes=benchmark_info.num_classes
            )
            print(f"Created Avalanche {avalanche_class}: {model_type}")
            return model
            
        elif model_type == 'lenet5':
            from avalanche.models import LeNet5
            model = LeNet5(
                num_classes=benchmark_info.num_classes,
                input_channels=benchmark_info.channels
            )
            print(f"Created Avalanche {avalanche_class}: {model_type}")
            return model
            
        elif model_type == 'resnet32':
            from avalanche.models import resnet32
            model = resnet32(
                num_classes=benchmark_info.num_classes,
                pretrained=params.get('pretrained', False)
            )
            print(f"Created Avalanche {avalanche_class}: {model_type}")
            return model
            
        elif model_type == 'slimresnet18':
            from avalanche.models import SlimResNet18
            model = SlimResNet18(
                num_classes=benchmark_info.num_classes
            )
            print(f"Created Avalanche {avalanche_class}: {model_type}")
            return model
            
        elif model_type in ['resnet18', 'resnet50']:
            # Use pytorchcv resnet models
            from avalanche.models.pytorchcv_wrapper import resnet
            depth = int(model_type.replace('resnet', ''))
            model = resnet('imagenet', depth, pretrained=params.get('pretrained', True))
            
            # Adapt for the number of classes if needed
            if hasattr(model, 'output') and model.output.out_features != benchmark_info.num_classes:
                model.output = nn.Linear(model.output.in_features, benchmark_info.num_classes)
            elif hasattr(model, 'fc') and model.fc.out_features != benchmark_info.num_classes:
                model.fc = nn.Linear(model.fc.in_features, benchmark_info.num_classes)
                
            print(f"Created Avalanche {avalanche_class}: {model_type}")
            return model
            
    # Check if it's a torchvision model
    elif model_type in TORCHVISION_MODELS:
        import torchvision.models as models
        import torch.nn as nn
        
        torchvision_name = TORCHVISION_MODELS[model_type]
        model_fn = getattr(models, torchvision_name)
        model = model_fn(pretrained=params.get('pretrained', True))
        
        # Adapt the final layer for the number of classes
        if hasattr(model, 'classifier'):
            if isinstance(model.classifier, nn.Sequential):
                # Models like EfficientNet with Sequential classifier
                last_layer = model.classifier[-1]
                if isinstance(last_layer, nn.Linear):
                    model.classifier[-1] = nn.Linear(last_layer.in_features, benchmark_info.num_classes)
            elif isinstance(model.classifier, nn.Linear):
                # Models like VGG with Linear classifier
                model.classifier = nn.Linear(model.classifier.in_features, benchmark_info.num_classes)
        elif hasattr(model, 'fc'):
            # ResNet-style models with fc layer
            model.fc = nn.Linear(model.fc.in_features, benchmark_info.num_classes)
        elif hasattr(model, 'head'):
            # Vision Transformer models with head
            if hasattr(model.head, 'in_features'):
                model.head = nn.Linear(model.head.in_features, benchmark_info.num_classes)
        elif hasattr(model, 'heads'):
            # Some models have heads instead of head
            if hasattr(model.heads, 'head'):
                model.heads.head = nn.Linear(model.heads.head.in_features, benchmark_info.num_classes)
                
        print(f"Created torchvision {torchvision_name}: {model_type}")
        return model
            
    # Check if it's a custom model
    elif model_type in CUSTOM_MODELS:
        from src.training import create_model
        model = create_model(model_type, benchmark_info, **params)
        print(f"Created custom model: {model_type}")
        return model
        
    else:
        # Fallback to custom creation
        print(f"Unknown model type '{model_type}', trying custom creation")
        from src.training import create_model
        return create_model(model_type, benchmark_info, **params)


def create_strategy_from_config(strategy_config: Dict[str, Any], model: nn.Module, 
                               benchmark_info, optimizer, criterion, eval_plugin, device, **kwargs) -> Any:
    """
    Create strategy using Avalanche built-ins when possible, custom otherwise.
    
    Args:
        strategy_config: Config with 'type', 'params', etc.
        model: PyTorch model
        benchmark_info: BenchmarkInfo object
        optimizer: PyTorch optimizer  
        criterion: Loss function
        eval_plugin: Avalanche evaluation plugin
        device: Device (cuda/cpu)
        
    Returns:
        Avalanche strategy
    """
    strategy_type = strategy_config.get('type', strategy_config.get('name', 'naive'))
    params = strategy_config.get('params', {})
    
    # Merge with additional kwargs
    all_params = {**kwargs, **params}
    
    # Special handling for SLDA (needs feature extraction setup)
    slda_model = None
    feature_size = None
    if strategy_type == 'slda':
        if hasattr(model, 'features') and hasattr(model, 'classifier') and hasattr(model, '_input_size'):
            # Standard Avalanche models like SimpleMLP
            class SLDAFeatureWrapper(nn.Module):
                def __init__(self, base_model):
                    super().__init__()
                    self.base_model = base_model
                    
                def forward(self, x):
                    x = x.contiguous()
                    x = x.view(x.size(0), self.base_model._input_size)
                    return self.base_model.features(x)
            
            slda_model = SLDAFeatureWrapper(model)
            if isinstance(model.classifier, nn.Sequential):
                for layer in reversed(model.classifier):
                    if isinstance(layer, nn.Linear):
                        feature_size = layer.in_features
                        break
                else:
                    feature_size = 1000
            else:
                feature_size = model.classifier.in_features
        else:
            # Other models (torchvision, custom)
            class SLDAModelWrapper(nn.Module):
                def __init__(self, model, model_type):
                    super().__init__()
                    self.model = model
                    self.model_type = model_type
                    self.target_size = MODEL_INPUT_SIZES.get(model_type, None)
                    grayscale_capable = ['ghostfacenetv2']
                    self.needs_rgb = model_type not in grayscale_capable
                    
                def forward(self, x):
                    if x.shape[1] == 1 and self.needs_rgb:
                        x = x.repeat(1, 3, 1, 1)
                    if self.target_size:
                        x = nn.functional.interpolate(x, size=self.target_size, mode='bilinear', align_corners=False)
                    return self.get_features(x)
                
                def get_features(self, x):
                    if hasattr(self.model, 'classifier'):
                        modules = list(self.model.children())[:-1]
                        feature_extractor = nn.Sequential(*modules)
                        features = feature_extractor(x)
                    elif hasattr(self.model, 'fc'):
                        modules = list(self.model.children())[:-1]
                        feature_extractor = nn.Sequential(*modules)
                        features = feature_extractor(x)
                    else:
                        features = self.model(x)
                    if features.dim() > 2:
                        features = features.view(features.size(0), -1)
                    return features
            
            model_type_name = kwargs.get('model_type', 'unknown')
            slda_model = SLDAModelWrapper(model, model_type_name)
            model_device = next(model.parameters()).device if len(list(model.parameters())) > 0 else device
            slda_model = slda_model.to(model_device)
            
            feature_size = MODEL_FEATURE_SIZES.get(model_type_name)
            if not feature_size:
                feature_size = auto_detect_feature_size(slda_model, benchmark_info)
    
    # Check if it's an Avalanche built-in strategy  
    if strategy_type in AVALANCHE_STRATEGIES:
        avalanche_class_name = AVALANCHE_STRATEGIES[strategy_type]
        
        # Import the strategy class dynamically
        try:
            from avalanche.training.supervised import (
                Naive, Cumulative, JointTraining, Replay, ICaRL, MIR, GDumb, 
                GSS_greedy, GenerativeReplay, IL2M, LwF, EWC, SynapticIntelligence, 
                MAS, LFL, AGEM, GEM, BiC, CoPE, CWRStar, PackNet, PNNStrategy, 
                DER, ER_ACE, ER_AML, LearningToPrompt, SCR, MER, AR1
            )
            from avalanche.training.supervised.deep_slda import StreamingLDA
            
            # Get the actual class
            strategy_class = locals()[avalanche_class_name]
            
            # Standard parameters all strategies accept
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
            
            # Strategy-specific parameters
            if strategy_type == 'replay':
                base_params['mem_size'] = all_params.get('mem_size', 500)
            elif strategy_type == 'ewc':
                base_params['ewc_lambda'] = all_params.get('ewc_lambda', 0.4)
            elif strategy_type == 'icarl':
                base_params['memory_size'] = all_params.get('memory_size', 2000)
            elif strategy_type == 'lwf':
                base_params['alpha'] = all_params.get('alpha', 1.0)
                base_params['temperature'] = all_params.get('temperature', 2.0)
            elif strategy_type == 'slda':
                # SLDA has different parameters
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
            # Fall through to custom handling
    
    # Check if it's a custom strategy
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
                # For other models (torchvision, custom), create a wrapper that handles resizing
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
                
                # Get model type from kwargs or strategy config
                model_type_name = kwargs.get('model_type', strategy_config.get('model_type', 'unknown'))
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
            
            strategy = StreamingLDA(
                slda_model=slda_model,
                criterion=criterion,
                input_size=feature_size,
                num_classes=benchmark_info.num_classes,
                shrinkage_param=all_params.get('shrinkage_param', 1e-4),
                streaming_update_sigma=all_params.get('streaming_update_sigma', True),
                train_mb_size=all_params.get('batch_size', 32),
                eval_mb_size=all_params.get('batch_size', 32) * 2,
                device=device,
                evaluator=eval_plugin
            )
            print(f"Created Avalanche {avalanche_class}: {strategy_type}")
            return strategy
            
    # Check if it's a custom strategy
    elif strategy_type in CUSTOM_STRATEGIES:
        from src.training import create_strategy_legacy
        strategy = create_strategy_legacy(strategy_type, model, benchmark_info, 
                                        optimizer, criterion, eval_plugin, device, **all_params)
        print(f"Created custom strategy: {strategy_type}")
        return strategy
        
    else:
        # Fallback to custom creation
        print(f"Unknown strategy type '{strategy_type}', trying custom creation")
        from src.training import create_strategy_legacy
        return create_strategy_legacy(strategy_type, model, benchmark_info,
                                    optimizer, criterion, eval_plugin, device, **all_params)


def create_benchmark_from_config(dataset_config: Dict[str, Any]) -> Tuple[Any, Any]:
    """
    Create benchmark using Avalanche built-ins when possible, custom otherwise.
    
    Args:
        dataset_config: Config with 'name', 'type', 'params', etc.
        
    Returns:
        (benchmark, benchmark_info) tuple
    """
    dataset_name = dataset_config.get('name', dataset_config.get('type', 'mnist'))
    params = dataset_config.get('params', {})
    
    # Check if it's an Avalanche built-in benchmark
    if dataset_name in AVALANCHE_BENCHMARKS:
        from src.utils.benchmark_wrapper import (create_mnist_benchmark, 
                                                create_cifar10_benchmark, create_cifar100_benchmark)
        
        kwargs = {
            'n_experiences': dataset_config.get('n_experiences', 5),
            'seed': dataset_config.get('seed', 42)
        }
        
        if dataset_name == 'mnist':
            benchmark, info = create_mnist_benchmark(**kwargs)
        elif dataset_name == 'cifar10':
            benchmark, info = create_cifar10_benchmark(**kwargs)  
        elif dataset_name == 'cifar100':
            benchmark, info = create_cifar100_benchmark(**kwargs)
            
        print(f"Created Avalanche benchmark: {dataset_name}")
        return benchmark, info
        
    # Check if it's a custom benchmark
    elif dataset_name in CUSTOM_BENCHMARKS:
        if 'smarteye' in dataset_name:
            from src.datasets.smarteye_cached import create_smarteye_benchmark_cached
            benchmark, info = create_smarteye_benchmark_cached(
                root_dir=dataset_config.get('path', '/home/dylee/data/data_fid/FaceID/ARM'),
                use_cropdata=dataset_name == 'smarteye_crop',
                n_experiences=dataset_config.get('n_experiences', 17),
                image_size=tuple(dataset_config.get('image_size', [112, 112])),
                test_split=dataset_config.get('test_split', 0.2),
                seed=dataset_config.get('seed', 42),
                use_cache=dataset_config.get('use_cache', True),
                preload_to_memory=dataset_config.get('preload_to_memory', False)
            )
            print(f"Created custom benchmark: {dataset_name}")
            return benchmark, info
            
        elif dataset_name == 'lfw':
            from src.datasets.lfw import create_lfw_benchmark
            # Handle LFW creation
            print(f"Created custom benchmark: {dataset_name}")
            # Implementation would go here
            
    else:
        # Fallback
        print(f"Unknown benchmark '{dataset_name}', trying fallback")
        from src.utils.benchmark_wrapper import wrap_avalanche_benchmark
        from src.training import create_benchmark
        
        benchmark = create_benchmark(
            dataset_name,
            experiences=dataset_config.get('n_experiences', 5),
            seed=dataset_config.get('seed', 42),
            subset_config=None
        )
        return wrap_avalanche_benchmark(benchmark, dataset_name)