"""
Adaptive configuration utilities for optimizing training parameters based on 
model, dataset, and hardware constraints.
"""

import torch
import psutil
from typing import Dict, Any, Tuple, List


def get_memory_info() -> Dict[str, float]:
    """Get system and GPU memory information in GB."""
    memory_info = {}
    
    # System RAM
    system_memory = psutil.virtual_memory()
    memory_info['system_total'] = system_memory.total / (1024**3)
    memory_info['system_available'] = system_memory.available / (1024**3)
    
    # GPU memory
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        memory_info['gpu_total'] = gpu_memory / (1024**3)
        memory_info['gpu_free'] = (gpu_memory - torch.cuda.memory_allocated()) / (1024**3)
    else:
        memory_info['gpu_total'] = 0
        memory_info['gpu_free'] = 0
    
    return memory_info


def estimate_model_memory(model_type: str, image_size: Tuple[int, int], batch_size: int = 1) -> float:
    """
    Estimate memory usage for a model in MB.
    
    Args:
        model_type: Type of model (e.g., 'efficientnet_b0', 'resnet18')
        image_size: Input image size (H, W)
        batch_size: Batch size
        
    Returns:
        Estimated memory usage in MB
    """
    h, w = image_size
    input_size = batch_size * 3 * h * w * 4  # 4 bytes per float32
    
    # Model-specific memory estimates (approximate)
    model_memory = {
        'mlp': 10,
        'cnn': 20,
        'resnet18': 45,
        'resnet50': 98,
        'efficientnet_b0': 20,
        'efficientnet_b1': 30,
        'efficientnet_b2': 35,
        'efficientnet_b3': 50,
        'efficientnet_b4': 75,
        'mobilenetv1': 15,
        'mobilenet_v2': 15,
        'ghostfacenetv2': 25,
        'modified_mobilefacenet': 30,
        'dwseesawfacev2': 60,
    }
    
    base_model_mb = model_memory.get(model_type, 50)  # Default 50MB
    
    # Scale with batch size and image size
    input_mb = input_size / (1024**2)
    activation_mb = input_mb * 4  # Rough estimate for activations
    
    total_mb = base_model_mb + input_mb + activation_mb
    return total_mb


def get_adaptive_batch_size(model_type: str, image_size: Tuple[int, int], 
                           base_batch_size: int = 32, strategy_type: str = 'naive') -> int:
    """
    Calculate adaptive batch size based on model and hardware constraints.
    
    Args:
        model_type: Type of model
        image_size: Input image size (H, W)
        base_batch_size: Desired base batch size
        strategy_type: Training strategy (some need more memory)
        
    Returns:
        Optimized batch size
    """
    memory_info = get_memory_info()
    
    # Start with base batch size and scale down if needed
    current_batch = base_batch_size
    
    while current_batch >= 1:
        estimated_memory = estimate_model_memory(model_type, image_size, current_batch)
        
        # Convert to GB
        estimated_gb = estimated_memory / 1024
        
        # Strategy-specific memory multipliers
        strategy_multiplier = {
            'naive': 1.0,
            'slda': 1.2,  # SLDA needs extra memory for statistics
            'replay': 1.5,  # Replay stores extra samples
            'icarl': 1.3,
            'ewc': 1.1,
        }.get(strategy_type, 1.0)
        
        estimated_gb *= strategy_multiplier
        
        # Check if it fits in GPU memory (leave 20% buffer)
        gpu_limit = memory_info['gpu_free'] * 0.8
        
        if estimated_gb <= gpu_limit or current_batch == 1:
            break
            
        # Reduce batch size
        current_batch = max(1, current_batch // 2)
    
    print(f"Adaptive batch sizing: {base_batch_size} -> {current_batch} "
          f"(model: {model_type}, size: {image_size}, GPU free: {memory_info['gpu_free']:.1f}GB)")
    
    return current_batch


def get_optimal_image_size(model_types: List[str], dataset_default: Tuple[int, int]) -> Tuple[int, int]:
    """
    Find optimal image size that minimizes total resize overhead across models.
    
    Args:
        model_types: List of model types to optimize for
        dataset_default: Default dataset image size
        
    Returns:
        Optimal image size (H, W)
    """
    # Model preferred sizes
    model_sizes = {
        'efficientnet_b0': (224, 224),
        'efficientnet_b1': (240, 240),
        'efficientnet_b2': (260, 260),
        'efficientnet_b3': (300, 300),
        'resnet18': (224, 224),
        'resnet50': (224, 224),
        'ghostfacenetv2': (112, 112),
        'modified_mobilefacenet': (256, 256),
        'dwseesawfacev2': (256, 256),
        'mobilenet_v2': (224, 224),
    }
    
    # Get sizes for all models
    preferred_sizes = []
    for model_type in model_types:
        size = model_sizes.get(model_type, dataset_default)
        preferred_sizes.append(size)
    
    if not preferred_sizes:
        return dataset_default
    
    # Find the size that minimizes total upscaling cost
    candidate_sizes = list(set(preferred_sizes + [dataset_default]))
    
    best_size = dataset_default
    min_cost = float('inf')
    
    for candidate in candidate_sizes:
        total_cost = 0
        for preferred in preferred_sizes:
            # Cost is roughly proportional to the scaling factor squared
            scale_factor = (preferred[0] * preferred[1]) / (candidate[0] * candidate[1])
            total_cost += abs(scale_factor - 1.0) ** 2
        
        if total_cost < min_cost:
            min_cost = total_cost
            best_size = candidate
    
    print(f"Optimal image size: {dataset_default} -> {best_size} "
          f"(models: {model_types}, cost reduction: {min_cost:.2f})")
    
    return best_size


def adjust_config_for_hardware(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Automatically adjust configuration based on hardware constraints.
    
    Args:
        config: Experiment configuration
        
    Returns:
        Adjusted configuration
    """
    adjusted_config = config.copy()
    
    # Extract model info
    if 'vary' in config and 'model' in config['vary']:
        model_types = [m.get('type', m.get('name', 'unknown')) for m in config['vary']['model']]
    elif 'fixed' in config and 'model' in config['fixed']:
        model_types = [config['fixed']['model'].get('type', config['fixed']['model'].get('name', 'unknown'))]
    else:
        model_types = ['unknown']
    
    # Get dataset info
    dataset_config = config.get('fixed', {}).get('dataset', {})
    dataset_default = tuple(dataset_config.get('image_size', [112, 112]))
    
    # Optimize image size
    optimal_size = get_optimal_image_size(model_types, dataset_default)
    if optimal_size != dataset_default:
        if 'fixed' not in adjusted_config:
            adjusted_config['fixed'] = {}
        if 'dataset' not in adjusted_config['fixed']:
            adjusted_config['fixed']['dataset'] = {}
        adjusted_config['fixed']['dataset']['image_size'] = list(optimal_size)
    
    # Adjust batch size
    training_config = config.get('fixed', {}).get('training', {})
    base_batch_size = training_config.get('batch_size', 32)
    strategy_type = config.get('fixed', {}).get('strategy', {}).get('type', 'naive')
    
    # Use the most memory-intensive model for batch size calculation
    memory_intensive_models = ['dwseesawfacev2', 'efficientnet_b4', 'resnet50']
    worst_case_model = next((m for m in model_types if m in memory_intensive_models), model_types[0])
    
    adaptive_batch_size = get_adaptive_batch_size(
        worst_case_model, optimal_size, base_batch_size, strategy_type
    )
    
    if adaptive_batch_size != base_batch_size:
        if 'fixed' not in adjusted_config:
            adjusted_config['fixed'] = {}
        if 'training' not in adjusted_config['fixed']:
            adjusted_config['fixed']['training'] = {}
        adjusted_config['fixed']['training']['batch_size'] = adaptive_batch_size
    
    return adjusted_config