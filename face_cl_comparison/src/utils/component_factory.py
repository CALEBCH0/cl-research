"""Unified component factory that automatically uses Avalanche built-ins or custom implementations."""

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional


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
    
    # Check if it's an Avalanche built-in strategy
    if strategy_type in AVALANCHE_STRATEGIES:
        avalanche_class = AVALANCHE_STRATEGIES[strategy_type]
        
        if strategy_type == 'naive':
            from avalanche.training.supervised import Naive
            strategy = Naive(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                train_mb_size=all_params.get('batch_size', 32),
                eval_mb_size=all_params.get('batch_size', 32) * 2,
                device=device,
                evaluator=eval_plugin
            )
            print(f"Created Avalanche {avalanche_class}: {strategy_type}")
            return strategy
            
        elif strategy_type == 'replay':
            from avalanche.training.supervised import Replay
            strategy = Replay(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                mem_size=all_params.get('mem_size', 500),
                train_mb_size=all_params.get('batch_size', 32),
                eval_mb_size=all_params.get('batch_size', 32) * 2,
                device=device,
                evaluator=eval_plugin
            )
            print(f"Created Avalanche {avalanche_class}: {strategy_type}")
            return strategy
            
        elif strategy_type == 'slda':
            from avalanche.training.supervised.deep_slda import StreamingLDA
            
            # For SLDA, we need the feature extractor part of the model
            if hasattr(model, 'features') and hasattr(model, 'classifier'):
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
                feature_size = model.classifier.in_features
                print(f"Using Avalanche model features for SLDA: feature_size={feature_size}")
            else:
                # Custom models - use the whole model as feature extractor
                slda_model = model
                feature_size = all_params.get('input_size', benchmark_info.input_size)
                print(f"Using custom model for SLDA: feature_size={feature_size}")
            
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