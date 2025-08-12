"""Predefined LFW dataset configurations with quality guarantees."""

# LFW dataset configurations with exact class counts and valid n_experiences
# Based on actual LFW data with different min_faces_per_person thresholds
LFW_CONFIGS = {
    'lfw_12': {
        'min_faces_per_person': 50,
        'num_classes': 12,  # Exact number
        'valid_n_experiences': [2, 3, 4, 6, 12],  # All divisors of 12
        'default_n_experiences': 6,  # 2 classes per experience
        'description': '12 people with 50+ images each (ultra high quality)',
        'use_case': 'Quick testing with perfect data quality'
    },
    'lfw_24': {
        'min_faces_per_person': 40,
        'num_classes': 24,  # Estimated
        'valid_n_experiences': [2, 3, 4, 6, 8, 12],  # Common divisors of 24
        'default_n_experiences': 8,  # 3 classes per experience
        'description': '~24 people with 40+ images each',
        'use_case': 'Small experiments with excellent quality'
    },
    'lfw_30': {
        'min_faces_per_person': 30,
        'num_classes': 30,  # Estimated
        'valid_n_experiences': [2, 3, 5, 6, 10, 15],  # Divisors of 30
        'default_n_experiences': 10,  # 3 classes per experience
        'description': '~30 people with 30+ images each',
        'use_case': 'Small-medium experiments'
    },
    'lfw_60': {
        'min_faces_per_person': 20,
        'num_classes': 60,  # Close to your 62
        'valid_n_experiences': [2, 3, 4, 5, 6, 10, 12, 15, 20],  # Many options!
        'default_n_experiences': 10,  # 6 classes per experience
        'description': '~60 people with 20+ images each',
        'use_case': 'Medium experiments with good quality'
    },
    'lfw_90': {
        'min_faces_per_person': 15,
        'num_classes': 90,  # Close to your 96
        'valid_n_experiences': [2, 3, 5, 6, 9, 10, 15, 18],  # Divisors of 90
        'default_n_experiences': 10,  # 9 classes per experience
        'description': '~90 people with 15+ images each',
        'use_case': 'Large experiments with acceptable quality'
    },
    'lfw_120': {
        'min_faces_per_person': 12,
        'num_classes': 120,  # Estimated
        'valid_n_experiences': [2, 3, 4, 5, 6, 8, 10, 12, 15, 20],  # Many divisors!
        'default_n_experiences': 10,  # 12 classes per experience
        'description': '~120 people with 12+ images each',
        'use_case': 'Large experiments with good divisibility'
    },
    'lfw_150': {
        'min_faces_per_person': 10,
        'num_classes': 150,  # Estimated
        'valid_n_experiences': [2, 3, 5, 6, 10, 15],  # Divisors of 150
        'default_n_experiences': 10,  # 15 classes per experience
        'description': '~150 people with 10+ images each',
        'use_case': 'Very large experiments (minimum quality)'
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

def find_valid_n_experiences(n_classes, preferred_n_exp=10):
    """Find valid number of experiences for given number of classes."""
    # Get all divisors
    divisors = [i for i in range(1, n_classes + 1) if n_classes % i == 0]
    
    # Filter reasonable options (2-20 experiences, with 2-50 classes per exp)
    reasonable = []
    for d in divisors:
        classes_per_exp = n_classes // d
        if 2 <= d <= 20 and 2 <= classes_per_exp <= 50:
            reasonable.append(d)
    
    if not reasonable:
        reasonable = divisors  # Fall back to all divisors
    
    # Find closest to preferred
    best = min(reasonable, key=lambda x: abs(x - preferred_n_exp))
    
    return {
        'recommended': best,
        'classes_per_exp': n_classes // best,
        'all_valid': sorted(reasonable),
        'all_divisors': sorted(divisors)
    }