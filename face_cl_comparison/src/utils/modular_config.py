"""
Modular configuration support for inline definitions of datasets, models, and strategies.
"""
from typing import Dict, Any, List, Union
import copy


def process_modular_item(item: Union[str, Dict[str, Any]], item_type: str) -> Dict[str, Any]:
    """
    Process a modular config item (dataset, model, or strategy).
    
    Unified format:
    - Dict with only 'name' -> predefined
    - Dict with 'type' and/or 'params' -> custom
    - String (for backward compatibility) -> predefined
    
    Args:
        item: Either a string (backward compat) or dict
        item_type: Type of item ('dataset', 'model', 'strategy')
        
    Returns:
        Normalized dict with 'name', 'type', and 'params'
    """
    if isinstance(item, str):
        # Backward compatibility: string means predefined
        return {
            'name': item,
            'type': item,  # For predefined, type == name
            'params': {},
            'is_predefined': True
        }
    
    elif isinstance(item, dict):
        # Must have 'name' field
        if 'name' not in item:
            raise ValueError(f"{item_type} config must have 'name' field: {item}")
        
        # Check if it's predefined (only 'name' field or no 'type'/'params')
        # Allow additional metadata fields like comments
        core_fields = {'name', 'type', 'params', 'plugins'}
        item_core_fields = set(item.keys()) & core_fields
        
        if item_core_fields == {'name'} or (
            'type' not in item and 
            'params' not in item and 
            item_type + '_params' not in item
        ):
            # Predefined item - minimal dict with just name, but preserve special fields
            result = {
                'name': item['name'],
                'type': item['name'],  # For predefined, type == name
                'params': {},
                'is_predefined': True
            }
            # Preserve special fields like path, test_split, n_experiences, etc.
            for key in ['path', 'test_split', 'n_experiences', 'image_size', 'seed']:
                if key in item:
                    result[key] = item[key]
            return result
        
        # Custom definition
        result = {
            'name': item['name'],
            'type': item.get('type', item['name']),  # Default type to name
            'params': item.get('params', {}),
            'is_predefined': False
        }
        
        # Handle plugins for strategies
        if item_type == 'strategy' and 'plugins' in item:
            result['plugins'] = item['plugins']
            
        # Handle other special fields
        for key in ['plugins', 'transforms', 'augmentations', 'path', 'test_split', 'n_experiences', 'image_size', 'seed']:
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
    
    # Process fixed items that might be strings or dicts
    for key in ['model', 'strategy', 'dataset']:
        if key in fixed:
            value = fixed[key]
            # Process if it's a string or a dict with 'name'
            if isinstance(value, (str, dict)):
                if isinstance(value, dict) and 'name' in value:
                    # It's already in dict format, process it
                    fixed[key] = process_modular_item(value, key)
                elif isinstance(value, str):
                    # Convert string to dict format
                    fixed[key] = process_modular_item(value, key)
    
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
                # Modular item - merge with fixed config
                # Start with fixed config for this component
                if dim_name in fixed:
                    merged_config = copy.deepcopy(fixed[dim_name])
                    # Update with vary config (vary takes precedence)
                    merged_config.update(value)
                    # Preserve any fixed settings not in vary
                    for key in fixed[dim_name]:
                        if key not in value and key not in ['name', 'type', 'params']:
                            merged_config[key] = fixed[dim_name][key]
                    run_config[dim_name] = merged_config
                else:
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
    Create a dataset based on modular config using unified factory.
    
    Args:
        dataset_config: Processed dataset config with name, type, params
        
    Returns:
        Dataset benchmark and info
    """
    from .component_factory import create_benchmark_from_config
    return create_benchmark_from_config(dataset_config)


def create_model_from_config(model_config: Dict[str, Any], benchmark_info):
    """
    Create a model based on modular config using unified factory.
    
    Args:
        model_config: Processed model config with name, type, params
        benchmark_info: BenchmarkInfo object with dataset information
        
    Returns:
        Model instance
    """
    from .component_factory import create_model_from_config as factory_create_model
    return factory_create_model(model_config, benchmark_info)


def create_strategy_from_config(strategy_config: Dict[str, Any], model, 
                               benchmark_info, optimizer, criterion, 
                               eval_plugin, device: str = 'cuda', 
                               model_type: str = None, **kwargs):
    """
    Create a strategy based on modular config.
    
    Args:
        strategy_config: Processed strategy config with name, type, params, plugins
        model: The model to use
        benchmark_info: Benchmark information
        optimizer: Optimizer instance
        criterion: Loss criterion
        eval_plugin: Evaluation plugin
        device: Device to use
        model_type: Type of model (for feature extraction)
        **kwargs: Additional training parameters
        
    Returns:
        Strategy instance
    """
    from .component_factory import create_strategy_from_config as factory_create_strategy
    
    # Merge strategy params with kwargs, strategy params take precedence
    all_params = {**kwargs}
    all_params.update(strategy_config.get('params', {}))
    
    return factory_create_strategy(
        strategy_config=strategy_config,
        model=model,
        benchmark_info=benchmark_info,
        optimizer=optimizer,
        criterion=criterion,
        eval_plugin=eval_plugin,
        device=device,
        **all_params
    )