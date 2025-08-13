"""
Modular configuration support for inline definitions of datasets, models, and strategies.
"""
from typing import Dict, Any, List, Union
import copy


def process_modular_item(item: Union[str, Dict[str, Any]], item_type: str) -> Dict[str, Any]:
    """
    Process a modular config item (dataset, model, or strategy).
    
    Args:
        item: Either a string (predefined) or dict (custom definition)
        item_type: Type of item ('dataset', 'model', 'strategy')
        
    Returns:
        Normalized dict with 'name', 'type', and 'params'
    """
    if isinstance(item, str):
        # Simple predefined item
        return {
            'name': item,
            'type': item,  # For predefined, type == name
            'params': {},
            'is_predefined': True
        }
    
    elif isinstance(item, dict):
        # Check if it's just a name reference (no type or params)
        if 'name' in item and 'type' not in item and 'params' not in item:
            # This is a predefined item in dict format
            return {
                'name': item['name'],
                'type': item['name'],
                'params': {},
                'is_predefined': True
            }
        
        # Custom definition
        result = {
            'name': item.get('name', f'custom_{item_type}'),
            'type': item.get('type', item.get('name', 'unknown')),
            'params': item.get('params', {}),
            'is_predefined': False
        }
        
        # Handle plugins for strategies
        if item_type == 'strategy' and 'plugins' in item:
            result['plugins'] = item['plugins']
            
        # Handle other special fields
        for key in ['plugins', 'transforms', 'augmentations']:
            if key in item and key not in result:
                result[key] = item[key]
                
        return result
    
    else:
        raise ValueError(f"Invalid {item_type} config: {item}")


def expand_modular_config(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Expand a modular config into individual run configurations.
    
    Args:
        config: Full experiment config with 'vary' and 'fixed' sections
        
    Returns:
        List of individual run configurations
    """
    runs = []
    
    # Get vary and fixed sections
    vary = config.get('vary', {})
    fixed = config.get('fixed', {})
    
    # Process each vary dimension
    vary_dimensions = {}
    for key, values in vary.items():
        if key in ['dataset', 'model', 'strategy']:
            # Process modular items
            vary_dimensions[key] = [process_modular_item(v, key) for v in values]
        else:
            # Keep as-is for other parameters
            vary_dimensions[key] = values
    
    # Generate all combinations
    import itertools
    
    # Get dimension names and values
    dim_names = list(vary_dimensions.keys())
    dim_values = list(vary_dimensions.values())
    
    # Create cartesian product
    for combo in itertools.product(*dim_values):
        # Start with fixed config
        run_config = copy.deepcopy(fixed)
        
        # Build run name parts
        name_parts = []
        
        # Add vary parameters
        for dim_name, value in zip(dim_names, combo):
            if dim_name in ['dataset', 'model', 'strategy']:
                # Modular item
                run_config[dim_name] = value
                name_parts.append(f"{dim_name}={value['name']}")
            else:
                # Regular parameter
                # Parse nested paths like 'training.lr'
                parts = dim_name.split('.')
                current = run_config
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                current[parts[-1]] = value
                name_parts.append(f"{parts[-1]}={value}")
        
        # Set run name (use 'name' for compatibility with runner)
        run_config['name'] = "_".join(name_parts)
        run_config['run_name'] = run_config['name']  # Keep both for compatibility
        
        # Add experiment metadata
        run_config['experiment_name'] = config.get('name', 'unnamed')
        run_config['description'] = config.get('description', '')
        
        runs.append(run_config)
    
    return runs


def create_dataset_from_config(dataset_config: Dict[str, Any]):
    """
    Create a dataset based on modular config.
    
    Args:
        dataset_config: Processed dataset config with name, type, params
        
    Returns:
        Dataset benchmark and info
    """
    from src.datasets import create_dataset
    from src.datasets.lfw import create_lfw_controlled_benchmark
    from src.datasets.subset_utils import DatasetSubsetConfig
    
    if dataset_config['is_predefined']:
        # Use existing create_dataset function
        return create_dataset(
            dataset_config['name'],
            n_experiences=dataset_config.get('n_experiences', 10),
            seed=dataset_config.get('seed', 42)
        )
    
    else:
        # Custom dataset
        dataset_type = dataset_config['type']
        params = dataset_config['params']
        
        if dataset_type == 'lfw':
            # Create LFW with custom parameters
            subset_config = DatasetSubsetConfig(
                target_classes=params['target_classes'],
                min_samples_per_class=params.get('min_samples_per_class', 20),
                selection_strategy=params.get('selection_strategy', 'most_samples'),
                n_experiences=dataset_config.get('n_experiences'),
                seed=dataset_config.get('seed', 42)
            )
            
            return create_lfw_controlled_benchmark(
                subset_config,
                image_size=tuple(dataset_config.get('image_size', [64, 64])),
                test_split=dataset_config.get('test_split', 0.2)
            )
        
        else:
            raise ValueError(f"Unknown custom dataset type: {dataset_type}")


def create_model_from_config(model_config: Dict[str, Any], num_classes: int):
    """
    Create a model based on modular config.
    
    Args:
        model_config: Processed model config with name, type, params
        num_classes: Number of output classes
        
    Returns:
        Model instance
    """
    from src.models.backbones import create_backbone
    
    backbone_name = model_config['type']
    params = model_config['params']
    
    # Create backbone with parameters
    backbone = create_backbone(
        backbone_name,
        pretrained=params.get('pretrained', True),
        num_classes=num_classes
    )
    
    # Apply additional settings
    if params.get('freeze_backbone', False):
        # Freeze all parameters except classifier
        for name, param in backbone.named_parameters():
            if 'classifier' not in name and 'fc' not in name:
                param.requires_grad = False
    
    # Add dropout if specified
    if 'dropout_rate' in params:
        import torch.nn as nn
        # This would need model-specific implementation
        # For now, just return the backbone
        pass
    
    return backbone


def create_strategy_from_config(strategy_config: Dict[str, Any], model, 
                               num_classes: int, device: str = 'cuda'):
    """
    Create a strategy based on modular config.
    
    Args:
        strategy_config: Processed strategy config with name, type, params, plugins
        model: The model to use
        num_classes: Number of classes
        device: Device to use
        
    Returns:
        Strategy instance
    """
    from src.strategies import create_strategy
    
    strategy_name = strategy_config['type']
    params = strategy_config.get('params', {})
    plugins = strategy_config.get('plugins', [])
    
    # Process plugins if they're in modular format
    processed_plugins = []
    for plugin in plugins:
        if isinstance(plugin, str):
            processed_plugins.append({'name': plugin})
        else:
            processed_plugins.append(plugin)
    
    return create_strategy(
        strategy_name,
        model,
        num_classes,
        device=device,
        params=params,
        plugins=processed_plugins
    )