"""Predefined LFW dataset configurations with quality guarantees."""

# LFW dataset configurations
# Each config ensures minimum images per person for reliable CL experiments
LFW_CONFIGS = {
    'lfw_small': {
        'min_faces_per_person': 70,
        'description': 'High-quality small dataset (~7 people, 70+ images each)',
        'use_case': 'Quick testing with very reliable data'
    },
    'lfw_20': {
        'min_faces_per_person': 50,
        'description': '~20 people with 50+ images each',
        'use_case': 'Small-scale experiments with high quality'
    },
    'lfw_50': {
        'min_faces_per_person': 30,
        'description': '~50 people with 30+ images each',
        'use_case': 'Medium-scale experiments'
    },
    'lfw_80': {
        'min_faces_per_person': 20,
        'description': '~80 people with 20+ images each',
        'use_case': 'Larger experiments with good quality'
    },
    'lfw_100': {
        'min_faces_per_person': 15,
        'description': '~100 people with 15+ images each',
        'use_case': 'Large-scale experiments'
    },
    'lfw_150': {
        'min_faces_per_person': 10,
        'description': '~150 people with 10+ images each',
        'use_case': 'Very large experiments (minimum recommended)'
    },
    # Not recommended for CL but available:
    'lfw_all': {
        'min_faces_per_person': 5,
        'description': 'All people with 5+ images (not recommended for CL)',
        'use_case': 'Maximum scale but poor quality for some classes'
    }
}

def get_lfw_config(config_name):
    """Get LFW configuration by name."""
    if config_name not in LFW_CONFIGS:
        available = ', '.join(LFW_CONFIGS.keys())
        raise ValueError(
            f"Unknown LFW config: {config_name}. "
            f"Available configs: {available}"
        )
    return LFW_CONFIGS[config_name]

def estimate_dataset_size(min_faces_per_person):
    """Estimate number of identities and total images for given threshold."""
    # Based on LFW statistics
    estimates = {
        70: (7, 490),      # ~7 people with 70+ images
        50: (20, 1000),    # ~20 people with 50+ images
        30: (50, 1500),    # ~50 people with 30+ images
        20: (80, 1600),    # ~80 people with 20+ images
        15: (100, 1500),   # ~100 people with 15+ images
        10: (150, 1500),   # ~150 people with 10+ images
        5: (400, 2000),    # ~400 people with 5+ images
    }
    
    # Find closest estimate
    thresholds = sorted(estimates.keys(), reverse=True)
    for threshold in thresholds:
        if min_faces_per_person >= threshold:
            return estimates[threshold]
    
    # Default to lowest threshold
    return estimates[5]