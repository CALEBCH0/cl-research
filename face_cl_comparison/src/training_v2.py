"""Improved training functionality with robust strategy creation."""
import warnings
warnings.filterwarnings('ignore', message='.*longdouble.*')
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message='.*OpenSSL.*')
warnings.filterwarnings('ignore', message='No loggers specified.*')

import copy
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union


def create_strategy_v2(
    strategy_name: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: Union[str, torch.device],
    eval_plugin: Any,
    benchmark_info: Any,
    model_type: Optional[str] = None,
    plugins_config: Optional[list] = None,
    **strategy_params: Any
) -> Any:
    """
    Create a continual learning strategy with robust parameter handling.
    
    This improved version:
    1. Handles all strategy-specific parameters through **strategy_params
    2. Avoids parameter conflicts
    3. Provides clear separation between required and optional parameters
    4. Supports both legacy and modular config formats
    
    Args:
        strategy_name: Name of the strategy (e.g., 'naive', 'replay', 'slda')
        model: The neural network model
        optimizer: The optimizer instance
        criterion: The loss function
        device: Device to run on ('cuda' or 'cpu')
        eval_plugin: Avalanche evaluation plugin
        benchmark_info: Information about the benchmark (num_classes, input_size, etc.)
        model_type: Type of model (for feature extraction in NCM methods)
        plugins_config: List of plugin configurations
        **strategy_params: All strategy-specific parameters (mem_size, epochs, batch_size, etc.)
    
    Returns:
        Configured Avalanche strategy instance
    """
    from avalanche.training.supervised import (
        Naive, EWC, Replay, GEM, AGEM, LwF, 
        SynapticIntelligence as SI, MAS, GDumb,
        Cumulative, JointTraining, ICaRL, StreamingLDA
    )
    from src.strategies.pure_ncm import PureNCM
    from src.plugin_factory import create_plugins
    
    # Extract common training parameters
    batch_size = strategy_params.pop('batch_size', 32)
    epochs = strategy_params.pop('epochs', 1)
    eval_batch_size = strategy_params.pop('eval_batch_size', batch_size * 2)
    
    # Base configuration for all strategies
    base_kwargs = {
        'model': model,
        'optimizer': optimizer,
        'criterion': criterion,
        'train_mb_size': batch_size,
        'train_epochs': epochs,
        'eval_mb_size': eval_batch_size,
        'device': device,
        'evaluator': eval_plugin
    }
    
    # Create plugins if configured
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
            device=device,
            optimizer=optimizer,
            feature_extractor=feature_extractor,
            **strategy_params  # Pass any plugin-specific params
        )
    
    if plugins:
        base_kwargs['plugins'] = plugins
    
    # Strategy-specific creation
    if strategy_name == 'naive':
        return Naive(**base_kwargs)
        
    elif strategy_name == 'ewc':
        ewc_lambda = strategy_params.pop('ewc_lambda', 0.4)
        mode = strategy_params.pop('mode', 'online')
        decay_factor = strategy_params.pop('decay_factor', 0.1)
        return EWC(**base_kwargs, ewc_lambda=ewc_lambda, mode=mode, decay_factor=decay_factor)
        
    elif strategy_name == 'replay':
        mem_size = strategy_params.pop('mem_size', 200)
        return Replay(**base_kwargs, mem_size=mem_size)
        
    elif strategy_name == 'gem':
        patterns_per_exp = strategy_params.pop('patterns_per_exp', 256)
        memory_strength = strategy_params.pop('memory_strength', 0.5)
        return GEM(**base_kwargs, patterns_per_exp=patterns_per_exp, memory_strength=memory_strength)
        
    elif strategy_name == 'agem':
        patterns_per_exp = strategy_params.pop('patterns_per_exp', 256)
        sample_size = strategy_params.pop('sample_size', 256)
        return AGEM(**base_kwargs, patterns_per_exp=patterns_per_exp, sample_size=sample_size)
        
    elif strategy_name == 'lwf':
        alpha = strategy_params.pop('alpha', 0.5)
        temperature = strategy_params.pop('temperature', 2)
        return LwF(**base_kwargs, alpha=alpha, temperature=temperature)
        
    elif strategy_name == 'si':
        si_lambda = strategy_params.pop('si_lambda', 0.0001)
        return SI(**base_kwargs, si_lambda=si_lambda)
        
    elif strategy_name == 'mas':
        lambda_reg = strategy_params.pop('lambda_reg', 1)
        alpha = strategy_params.pop('alpha', 0.5)
        return MAS(**base_kwargs, lambda_reg=lambda_reg, alpha=alpha)
        
    elif strategy_name == 'gdumb':
        mem_size = strategy_params.pop('mem_size', 200)
        return GDumb(**base_kwargs, mem_size=mem_size)
        
    elif strategy_name == 'cumulative':
        return Cumulative(**base_kwargs)
        
    elif strategy_name == 'joint':
        return JointTraining(**base_kwargs)
        
    elif strategy_name == 'slda':
        # SLDA needs a feature extractor
        feature_extractor = _create_feature_extractor(model, model_type, device)
        
        # SLDA-specific parameters
        shrinkage_param = strategy_params.pop('shrinkage_param', 1e-4)
        streaming_update_sigma = strategy_params.pop('streaming_update_sigma', True)
        
        slda_kwargs = {
            'slda_model': feature_extractor,
            'criterion': criterion,
            'input_size': feature_extractor.num_features,
            'num_classes': benchmark_info.num_classes if hasattr(benchmark_info, 'num_classes') else benchmark_info['num_classes'],
            'shrinkage_param': shrinkage_param,
            'streaming_update_sigma': streaming_update_sigma,
            'train_mb_size': batch_size,
            'eval_mb_size': eval_batch_size,
            'device': device,
            'evaluator': eval_plugin,
            'train_epochs': 1,  # SLDA typically uses 1 epoch
            'plugins': plugins if plugins else None
        }
        return StreamingLDA(**slda_kwargs)
        
    elif strategy_name == 'icarl':
        # ICaRL needs split architecture
        mem_size = strategy_params.pop('memory_size', strategy_params.pop('mem_size', 2000))
        fixed_memory = strategy_params.pop('fixed_memory', True)
        
        feature_extractor, classifier = _split_model_for_icarl(
            model, model_type, benchmark_info, device
        )
        
        # Create new optimizer for the split model
        icarl_params = list(feature_extractor.parameters()) + list(classifier.parameters())
        lr = strategy_params.pop('lr', 0.001)
        icarl_optimizer = torch.optim.Adam(icarl_params, lr=lr)
        
        return ICaRL(
            feature_extractor=feature_extractor,
            classifier=classifier,
            optimizer=icarl_optimizer,
            memory_size=mem_size,
            buffer_transform=None,
            fixed_memory=fixed_memory,
            train_mb_size=batch_size,
            train_epochs=epochs,
            eval_mb_size=eval_batch_size,
            device=device,
            evaluator=eval_plugin
        )
        
    elif strategy_name == 'pure_ncm':
        # Pure NCM needs a feature extractor
        feature_extractor = _create_feature_extractor(model, model_type, device)
        
        return PureNCM(
            feature_extractor=feature_extractor,
            feature_size=feature_extractor.num_features,
            num_classes=benchmark_info.num_classes if hasattr(benchmark_info, 'num_classes') else benchmark_info['num_classes'],
            train_mb_size=batch_size,
            train_epochs=epochs,
            eval_mb_size=eval_batch_size,
            device=device,
            evaluator=eval_plugin
        )
        
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")


def _create_feature_extractor(model: nn.Module, model_type: str, device: Union[str, torch.device]) -> nn.Module:
    """Create a feature extractor from a model."""
    
    class GenericFeatureExtractor(nn.Module):
        def __init__(self, base_model, model_type):
            super().__init__()
            self.model_type = model_type
            
            # Handle GrayToRGBWrapper
            if hasattr(base_model, 'model'):
                self.wrapper = True
                actual_model = base_model.model
            else:
                self.wrapper = False
                actual_model = base_model
            
            # Extract features based on model type
            if 'efficientnet' in model_type:
                self._setup_efficientnet(actual_model)
            elif 'resnet' in model_type:
                self._setup_resnet(actual_model)
            elif 'mobilenet' in model_type:
                self._setup_mobilenet(actual_model, model_type)
            elif model_type in ['dwseesawfacev2', 'ghostfacenetv2', 'modified_mobilefacenet']:
                # Custom face models already output embeddings
                self.features = actual_model
                # Different models use different attribute names and structures
                if model_type == 'ghostfacenetv2':
                    # GhostFaceNetV2 outputs embeddings of size 'emb' (default 512) or 'num_features' if linear layer is used
                    # The actual output size is determined by the output_layer's configuration
                    if hasattr(actual_model, 'output_layer') and hasattr(actual_model.output_layer, 'linear'):
                        if isinstance(actual_model.output_layer.linear, nn.Linear):
                            self.num_features = actual_model.output_layer.linear.out_features
                        else:
                            # nn.Identity case - uses emb size (default 512)
                            self.num_features = 512
                    else:
                        self.num_features = 512
                elif hasattr(actual_model, 'num_features'):
                    self.num_features = actual_model.num_features
                elif hasattr(actual_model, 'embedding_size'):
                    self.num_features = actual_model.embedding_size
                else:
                    self.num_features = 512  # Default
            elif 'dwseesaw' in model_type or 'ghostface' in model_type or 'mobilefacenet' in model_type or 'modified_mobile' in model_type:
                # Handle variations in model naming
                self.features = actual_model
                self.num_features = 512  # Default embedding size for face models
            else:
                raise ValueError(f"Unsupported model type for feature extraction: {model_type}")
        
        def _setup_efficientnet(self, model):
            self.features = nn.Sequential(
                model.conv_stem,
                model.bn1,
                model.blocks,
                model.conv_head,
                model.bn2,
                model.global_pool,
                nn.Flatten()
            )
            self.num_features = model.num_features
            
        def _setup_resnet(self, model):
            modules = list(model.children())[:-1]
            self.features = nn.Sequential(*modules, nn.Flatten())
            self.num_features = model.fc.in_features
            
        def _setup_mobilenet(self, model, model_type):
            if model_type == 'mobilenetv1':
                self.features = nn.Sequential(
                    model.features,
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten()
                )
                self.num_features = 1024
            elif hasattr(model, 'features'):
                self.features = nn.Sequential(
                    model.features,
                    model.avgpool if hasattr(model, 'avgpool') else nn.AdaptiveAvgPool2d(1),
                    nn.Flatten()
                )
                self.num_features = self._get_mobilenet_features(model)
            else:
                modules = list(model.children())[:-1]
                self.features = nn.Sequential(*modules, nn.Flatten())
                self.num_features = 1280
                
        def _get_mobilenet_features(self, model):
            if hasattr(model, 'classifier'):
                if isinstance(model.classifier, nn.Sequential):
                    for module in model.classifier:
                        if isinstance(module, nn.Linear):
                            return module.in_features
                else:
                    return model.classifier.in_features
            return 1280  # Default
        
        def forward(self, x):
            if self.wrapper:
                x = x.repeat(1, 3, 1, 1)
            
            # Get features
            features = self.features(x)
            
            # L2 normalize if this is a face recognition model
            if self.model_type in ['dwseesawfacev2', 'ghostfacenetv2', 'modified_mobilefacenet', 
                                   'dwseesaw', 'ghostface', 'mobilefacenet', 'modified_mobile']:
                # Face recognition models should output L2-normalized embeddings
                features = nn.functional.normalize(features, p=2, dim=1)
            
            return features
    
    feature_extractor = GenericFeatureExtractor(model, model_type)
    return feature_extractor.to(device)


def _split_model_for_icarl(model: nn.Module, model_type: str, 
                          benchmark_info: Any, device: Union[str, torch.device]) -> tuple:
    """Split a model into feature extractor and classifier for ICaRL."""
    
    if model_type and 'efficientnet' in model_type:
        # Handle EfficientNet models
        base_model = model.model if hasattr(model, 'model') else model
        needs_rgb_conversion = hasattr(model, 'model')
        
        class EfficientNetFeatureExtractor(nn.Module):
            def __init__(self, efficientnet_model, convert_rgb=False):
                super().__init__()
                self.convert_rgb = convert_rgb
                
                # Copy EfficientNet layers
                self.conv_stem = efficientnet_model.conv_stem
                self.bn1 = efficientnet_model.bn1
                self.act1 = getattr(efficientnet_model, 'act1', nn.SiLU(inplace=True))
                self.blocks = efficientnet_model.blocks
                self.conv_head = efficientnet_model.conv_head
                self.bn2 = efficientnet_model.bn2
                self.act2 = getattr(efficientnet_model, 'act2', nn.SiLU(inplace=True))
                self.global_pool = efficientnet_model.global_pool
                self.num_features = efficientnet_model.num_features
                
            def forward(self, x):
                if self.convert_rgb:
                    x = x.repeat(1, 3, 1, 1)
                
                x = self.conv_stem(x)
                x = self.bn1(x)
                x = self.act1(x)
                x = self.blocks(x)
                x = self.conv_head(x)
                x = self.bn2(x)
                x = self.act2(x)
                x = self.global_pool(x)
                x = x.flatten(1)
                return x
        
        feature_extractor = EfficientNetFeatureExtractor(base_model, needs_rgb_conversion)
        classifier = copy.deepcopy(base_model.classifier)
        
    else:
        # Generic model splitting
        if hasattr(model, 'features') and hasattr(model, 'classifier'):
            feature_extractor = copy.deepcopy(model.features)
            classifier = copy.deepcopy(model.classifier)
        else:
            # Try to split Sequential models
            all_children = list(model.children())
            classifier_idx = None
            for i in reversed(range(len(all_children))):
                if isinstance(all_children[i], nn.Linear):
                    classifier_idx = i
                    break
            
            if classifier_idx is not None:
                feature_extractor = nn.Sequential(*all_children[:classifier_idx])
                classifier = all_children[classifier_idx]
            else:
                raise ValueError(f"Cannot split {model_type} for ICaRL.")
    
    return feature_extractor.to(device), classifier.to(device)


def create_strategy_from_config_v2(
    strategy_config: Dict[str, Any],
    model: nn.Module,
    benchmark_info: Any,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    eval_plugin: Any,
    device: str = 'cuda',
    model_type: Optional[str] = None,
    **training_params: Any
) -> Any:
    """
    Create a strategy from modular config with improved parameter handling.
    
    This version:
    1. Cleanly separates config params from training params
    2. Avoids parameter conflicts
    3. Provides clear precedence rules
    
    Args:
        strategy_config: Strategy configuration from modular config
        model: The model to use
        benchmark_info: Benchmark information
        optimizer: Optimizer instance
        criterion: Loss criterion
        eval_plugin: Evaluation plugin
        device: Device to use
        model_type: Type of model
        **training_params: Additional training parameters
    
    Returns:
        Configured strategy instance
    """
    strategy_name = strategy_config['type']
    config_params = strategy_config.get('params', {})
    plugins = strategy_config.get('plugins', [])
    
    # Process plugins to standard format
    processed_plugins = []
    for plugin in plugins:
        if isinstance(plugin, str):
            processed_plugins.append({'name': plugin})
        else:
            processed_plugins.append(plugin)
    
    # Merge parameters with config taking precedence
    all_params = {**training_params}
    all_params.update(config_params)
    
    # Call the improved create_strategy function
    return create_strategy_v2(
        strategy_name=strategy_name,
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        eval_plugin=eval_plugin,
        benchmark_info=benchmark_info,
        model_type=model_type,
        plugins_config=processed_plugins,
        **all_params
    )