"""Plugin factory for creating and configuring Avalanche plugins."""
import torch.nn as nn
from typing import Dict, List, Any, Optional
from avalanche.training.plugins import (
    ReplayPlugin, 
    EWCPlugin, 
    LwFPlugin,
    FeatureDistillationPlugin,
    GEMPlugin,
    MASPlugin,
    SynapticIntelligencePlugin,
    RWalkPlugin,
    LRSchedulerPlugin,
)
from avalanche.training.storage_policy import (
    ExperienceBalancedBuffer,
    ClassBalancedBuffer,
    ReservoirSamplingBuffer,
)
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau


def create_plugin(plugin_name: str, plugin_config: Dict[str, Any], 
                  mem_size: int = 200, device: str = 'cpu', 
                  **kwargs) -> Optional[Any]:
    """Create a single plugin based on name and config.
    
    Args:
        plugin_name: Name of the plugin
        plugin_config: Configuration dict for the plugin
        mem_size: Memory size for replay-based plugins
        device: Device to use
        **kwargs: Additional arguments (model, etc.)
    
    Returns:
        Configured plugin instance or None if unknown
    """
    
    if plugin_name == 'replay':
        # Storage policy configuration
        storage_policy_name = plugin_config.get('storage_policy', 'experience_balanced')
        buffer_size = plugin_config.get('mem_size', mem_size)
        
        if storage_policy_name == 'experience_balanced':
            storage_policy = ExperienceBalancedBuffer(
                max_size=buffer_size,
                adaptive_size=plugin_config.get('adaptive_size', True)
            )
        elif storage_policy_name == 'class_balanced':
            storage_policy = ClassBalancedBuffer(
                max_size=buffer_size,
                adaptive_size=plugin_config.get('adaptive_size', True)
            )
        elif storage_policy_name == 'reservoir':
            storage_policy = ReservoirSamplingBuffer(
                max_size=buffer_size
            )
        else:
            storage_policy = ExperienceBalancedBuffer(max_size=buffer_size)
        
        return ReplayPlugin(
            mem_size=buffer_size,
            storage_policy=storage_policy,
            batch_size=plugin_config.get('batch_size', None),
            batch_size_mem=plugin_config.get('batch_size_mem', None),
            task_balanced_dataloader=plugin_config.get('task_balanced', True),
        )
    
    elif plugin_name == 'ewc':
        return EWCPlugin(
            ewc_lambda=plugin_config.get('ewc_lambda', 0.4),
            mode=plugin_config.get('mode', 'online'),
            decay_factor=plugin_config.get('decay_factor', 0.1),
            keep_importance_data=plugin_config.get('keep_importance_data', False)
        )
    
    elif plugin_name == 'lwf':
        return LwFPlugin(
            alpha=plugin_config.get('alpha', 0.5),
            temperature=plugin_config.get('temperature', 2.0)
        )
    
    elif plugin_name == 'feature_distillation':
        # Feature distillation requires a feature extractor
        feature_extractor = kwargs.get('feature_extractor')
        if feature_extractor is None:
            print("Warning: FeatureDistillationPlugin requires feature_extractor")
            return None
            
        return FeatureDistillationPlugin(
            feature_extractor=feature_extractor,
            alpha=plugin_config.get('alpha', 0.5),
            temperature=plugin_config.get('temperature', 2.0),
            distillation_loss=plugin_config.get('loss', 'l2')  # 'l2' or 'kl'
        )
    
    elif plugin_name == 'gem':
        return GEMPlugin(
            patterns_per_exp=plugin_config.get('patterns_per_exp', 256),
            memory_strength=plugin_config.get('memory_strength', 0.5),
            device=device
        )
    
    elif plugin_name == 'mas':
        return MASPlugin(
            lambda_reg=plugin_config.get('lambda_reg', 1.0),
            alpha=plugin_config.get('alpha', 0.5)
        )
    
    elif plugin_name == 'si':
        return SynapticIntelligencePlugin(
            si_lambda=plugin_config.get('si_lambda', 0.0001)
        )
    
    elif plugin_name == 'rwalk':
        return RWalkPlugin(
            ewc_lambda=plugin_config.get('ewc_lambda', 0.4),
            ewc_alpha=plugin_config.get('ewc_alpha', 0.9),
            delta_t=plugin_config.get('delta_t', 10)
        )
    
    elif plugin_name == 'lr_scheduler':
        # Create scheduler based on type
        scheduler_type = plugin_config.get('scheduler_type', 'step')
        optimizer = kwargs.get('optimizer')
        
        if optimizer is None:
            print("Warning: LRSchedulerPlugin requires optimizer")
            return None
            
        if scheduler_type == 'step':
            scheduler = StepLR(
                optimizer,
                step_size=plugin_config.get('step_size', 30),
                gamma=plugin_config.get('gamma', 0.1)
            )
        elif scheduler_type == 'cosine':
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=plugin_config.get('T_max', 100),
                eta_min=plugin_config.get('eta_min', 0)
            )
        elif scheduler_type == 'reduce_on_plateau':
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode=plugin_config.get('mode', 'min'),
                factor=plugin_config.get('factor', 0.1),
                patience=plugin_config.get('patience', 10)
            )
        else:
            return None
            
        return LRSchedulerPlugin(
            scheduler,
            step_granularity=plugin_config.get('step_granularity', 'epoch'),
            metric=plugin_config.get('metric', None)
        )
    
    else:
        print(f"Unknown plugin: {plugin_name}")
        return None


def create_plugins(plugins_config: List[Dict[str, Any]], 
                  mem_size: int = 200, 
                  device: str = 'cpu',
                  **kwargs) -> List[Any]:
    """Create multiple plugins from configuration.
    
    Args:
        plugins_config: List of plugin configurations, each with 'name' and 'params'
        mem_size: Default memory size for replay-based plugins
        device: Device to use
        **kwargs: Additional arguments passed to plugins
    
    Returns:
        List of configured plugin instances
    """
    plugins = []
    
    for plugin_spec in plugins_config:
        if isinstance(plugin_spec, str):
            # Simple plugin name without params
            plugin_name = plugin_spec
            plugin_config = {}
        else:
            # Plugin with configuration
            plugin_name = plugin_spec.get('name')
            plugin_config = plugin_spec.get('params', {})
        
        if plugin_name:
            plugin = create_plugin(
                plugin_name, plugin_config, 
                mem_size=mem_size, device=device, **kwargs
            )
            if plugin is not None:
                plugins.append(plugin)
    
    return plugins