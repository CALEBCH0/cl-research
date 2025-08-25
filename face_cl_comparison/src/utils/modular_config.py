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
            # Predefined item - minimal dict with just name
            return {
                'name': item['name'],
                'type': item['name'],  # For predefined, type == name
                'params': {},
                'is_predefined': True
            }
        
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
    Create a dataset based on modular config.
    
    Args:
        dataset_config: Processed dataset config with name, type, params
        
    Returns:
        Dataset benchmark and info
    """
    from src.training import create_benchmark
    from src.datasets.lfw import create_lfw_controlled_benchmark
    from src.datasets.subset_utils import DatasetSubsetConfig
    
    if dataset_config['is_predefined']:
        # Check if it's an LFW variant that needs image_size
        if dataset_config['name'].startswith('lfw'):
            # Use LFW-specific creation with image_size support
            from src.datasets.lfw import create_lfw_benchmark
            from src.datasets.lfw_configs import get_lfw_config
            
            config = get_lfw_config(dataset_config['name'])
            return create_lfw_benchmark(
                n_experiences=dataset_config.get('n_experiences', 10),
                min_faces_per_person=config['min_faces_per_person'],
                image_size=tuple(dataset_config.get('image_size', [64, 64])),
                seed=dataset_config.get('seed', 42)
            )
        elif dataset_config['name'].startswith('smarteye'):
            # Use SmartEye-specific creation with image_size support
            from src.datasets.smarteye import create_smarteye_benchmark, get_smarteye_config
            
            if dataset_config['name'] in ['smarteye_crop', 'smarteye_raw']:
                config = get_smarteye_config(dataset_config['name'])
                return create_smarteye_benchmark(
                    n_experiences=dataset_config.get('n_experiences', 5),
                    use_cropdata=config['use_cropdata'],
                    image_size=tuple(dataset_config.get('image_size', [112, 112])),
                    seed=dataset_config.get('seed', 42),
                    min_samples_per_class=config['min_samples_per_class']
                )
            else:
                # Default SmartEye config
                return create_smarteye_benchmark(
                    n_experiences=dataset_config.get('n_experiences', 5),
                    use_cropdata=True,
                    image_size=tuple(dataset_config.get('image_size', [112, 112])),
                    seed=dataset_config.get('seed', 42)
                )
        else:
            # Use existing create_benchmark function for other datasets
            return create_benchmark(
                dataset_config['name'],
                experiences=dataset_config.get('n_experiences', 10),
                seed=dataset_config.get('seed', 42),
                subset_config=None
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


def create_model_from_config(model_config: Dict[str, Any], benchmark_info):
    """
    Create a model based on modular config.
    
    Args:
        model_config: Processed model config with name, type, params
        benchmark_info: Either a BenchmarkInfo namedtuple or dict with benchmark information
        
    Returns:
        Model instance
    """
    # Extract num_classes from benchmark_info (handle both dict and namedtuple)
    if hasattr(benchmark_info, 'num_classes'):
        num_classes = benchmark_info.num_classes
    else:
        num_classes = benchmark_info['num_classes']
    model_type = model_config['type']
    params = model_config.get('params', {})
    
    # Check if it's a custom backbone model
    if model_type in ['dwseesawfacev2', 'ghostfacenetv2', 'modified_mobilefacenet']:
        # Custom backbone models - pass all params as kwargs
        from src.training import create_model
        
        # Create a temporary benchmark info for compatibility
        from collections import namedtuple
        BenchmarkInfo = namedtuple("BenchmarkInfo", ["input_size", "num_classes", "channels"])
        
        # Determine input size based on model or params
        if model_type == 'ghostfacenetv2' and 'image_size' in params:
            # GhostFaceNetV2 expects integer image_size, not tuple
            img_size = params['image_size']
            if isinstance(img_size, int):
                input_size = img_size * img_size
            else:
                input_size = img_size[0] * img_size[1]
        else:
            input_size = 64 * 64  # Default
            
        benchmark_info = BenchmarkInfo(
            input_size=input_size,
            num_classes=num_classes,
            channels=1  # Most face models expect grayscale
        )
        
        # Create model with all params
        model = create_model(model_type, benchmark_info, **params)
        
        # For face models, we don't need to add a classifier here
        # The strategy (especially SLDA) will handle that
        # Just return the backbone model that outputs embeddings
        
    else:
        # Try original backbone creation
        try:
            from src.models.backbones import create_backbone
            
            backbone = create_backbone(
                model_type,
                pretrained=params.get('pretrained', True),
                num_classes=num_classes
            )
            model = backbone
        except:
            # Fallback to training.py create_model
            from src.training import create_model
            from collections import namedtuple
            
            BenchmarkInfo = namedtuple("BenchmarkInfo", ["input_size", "num_classes", "channels"])
            benchmark_info = BenchmarkInfo(
                input_size=64 * 64,
                num_classes=num_classes,
                channels=1
            )
            
            model = create_model(model_type, benchmark_info, **params)
    
    # Apply additional settings
    if params.get('freeze_backbone', False):
        # Freeze all parameters except classifier
        for name, param in model.named_parameters():
            if 'classifier' not in name and 'fc' not in name:
                param.requires_grad = False
    
    return model


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
    # Try to use the improved version first
    try:
        from src.training_v2 import create_strategy_from_config_v2
        return create_strategy_from_config_v2(
            strategy_config=strategy_config,
            model=model,
            benchmark_info=benchmark_info,
            optimizer=optimizer,
            criterion=criterion,
            eval_plugin=eval_plugin,
            device=device,
            model_type=model_type,
            **kwargs
        )
    except ImportError:
        # Fallback to original implementation
        from src.training import create_strategy
        
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
        
        # Merge all parameters, with strategy params taking precedence
        all_params = {**kwargs}
        all_params.update(params)
        
        # Extract specific parameters that create_strategy expects as positional/keyword args
        mem_size = all_params.pop('mem_size', 200)
        epochs = all_params.pop('epochs', kwargs.get('epochs', 1))
        batch_size = all_params.pop('batch_size', kwargs.get('batch_size', 32))
        
        return create_strategy(
            strategy_name=strategy_name,
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            eval_plugin=eval_plugin,
            mem_size=mem_size,
            model_type=model_type,
            benchmark_info=benchmark_info,
            plugins_config=processed_plugins,
            epochs=epochs,
            batch_size=batch_size,
            **all_params  # Pass any remaining params
        )