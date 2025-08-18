"""Model utilities for getting model configurations and requirements."""
from typing import Dict, Any, Tuple, Optional


def get_model_requirements(model_type: str, model_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Get the requirements/configuration for a specific model type.
    
    Args:
        model_type: Type of model (e.g., 'dwseesawfacev2', 'efficientnet_b0')
        model_params: Optional parameters passed to the model
        
    Returns:
        Dictionary with model requirements like input_size, channels, etc.
    """
    model_params = model_params or {}
    
    # Default requirements
    default_requirements = {
        'input_size': (64, 64),
        'channels': 1,  # Grayscale for face recognition
        'normalize': True,
        'embedding_size': 512
    }
    
    # Model-specific requirements
    model_requirements = {
        # Custom face recognition models
        'dwseesawfacev2': {
            'input_size': (256, 256),  # DWSeesawFaceV2 needs large images due to 16x16 kernel
            'channels': 1,
            'embedding_size': model_params.get('embedding_size', 512)
        },
        'ghostfacenetv2': {
            'input_size': (112, 112),  # Default size, override if provided
            'channels': 1,
            'num_features': model_params.get('num_features', 512)
        },
        'modified_mobilefacenet': {
            'input_size': (256, 256),  # Modified MobileFaceNet needs 256x256 images
            'channels': 1,
            'embedding_size': 512
        },
        
        # Standard models
        'efficientnet_b0': {
            'input_size': (64, 64),  # Can work with smaller images
            'channels': 3,  # RGB expected, but we handle grayscale conversion
            'normalize': True
        },
        'efficientnet_b1': {
            'input_size': (64, 64),
            'channels': 3,
            'normalize': True
        },
        'resnet18': {
            'input_size': (64, 64),
            'channels': 3,
            'normalize': True
        },
        'resnet50': {
            'input_size': (64, 64),
            'channels': 3,
            'normalize': True
        },
        'mobilenetv1': {
            'input_size': (64, 64),
            'channels': 3,
            'normalize': True
        },
        'mobilenetv3_small': {
            'input_size': (64, 64),
            'channels': 3,
            'normalize': True
        },
        'mobilenetv3_large': {
            'input_size': (64, 64),
            'channels': 3,
            'normalize': True
        },
        
        # Simple models
        'mlp': {
            'input_size': (64, 64),  # Will be flattened
            'channels': 1,
            'normalize': True
        },
        'cnn': {
            'input_size': (64, 64),
            'channels': 1,
            'normalize': True
        }
    }
    
    # Handle model name variations (e.g., 'dwseesaw_512' -> 'dwseesawfacev2')
    normalized_model_type = model_type.lower()
    if 'dwseesaw' in normalized_model_type:
        normalized_model_type = 'dwseesawfacev2'
    elif 'ghostface' in normalized_model_type:
        normalized_model_type = 'ghostfacenetv2'
    elif 'mobilefacenet' in normalized_model_type and 'modified' in normalized_model_type:
        normalized_model_type = 'modified_mobilefacenet'
    
    # Get requirements for the model, fall back to defaults
    requirements = model_requirements.get(normalized_model_type, default_requirements.copy())
    
    # Update with any model-specific params
    if model_params:
        for key in ['embedding_size', 'num_features', 'image_size']:
            if key in model_params:
                if key == 'image_size':
                    # Handle both int and tuple/list
                    img_size = model_params[key]
                    if isinstance(img_size, int):
                        requirements['input_size'] = (img_size, img_size)
                    else:
                        requirements['input_size'] = tuple(img_size)
                else:
                    requirements[key] = model_params[key]
    
    return requirements


def get_dataset_requirements_for_model(model_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get dataset requirements based on model configuration.
    
    Args:
        model_config: Model configuration dict with 'type' and optionally 'params'
        
    Returns:
        Dictionary with dataset requirements like image_size
    """
    model_type = model_config.get('type', model_config.get('name', 'mlp'))
    model_params = model_config.get('params', {})
    
    model_reqs = get_model_requirements(model_type, model_params)
    
    # Map model requirements to dataset parameters
    dataset_reqs = {
        'image_size': model_reqs['input_size'],
        'channels': model_reqs['channels']
    }
    
    return dataset_reqs


def merge_dataset_config_with_model_requirements(
    dataset_config: Dict[str, Any], 
    model_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Merge dataset configuration with model requirements.
    Model requirements take precedence for compatibility.
    
    Args:
        dataset_config: Original dataset configuration
        model_config: Model configuration
        
    Returns:
        Updated dataset configuration
    """
    dataset_reqs = get_dataset_requirements_for_model(model_config)
    
    # Create a copy of dataset config
    updated_config = dataset_config.copy()
    
    # Update with model requirements
    # Only override if not explicitly set in dataset config
    if 'image_size' not in updated_config or updated_config.get('auto_adjust_size', True):
        updated_config['image_size'] = dataset_reqs['image_size']
        print(f"Auto-adjusting dataset image size to {dataset_reqs['image_size']} for model {model_config.get('name')}")
    
    return updated_config