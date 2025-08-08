#!/usr/bin/env python3
"""Test script to find correct EfficientNet model names in timm."""

try:
    import timm
    
    # List all available models
    all_models = timm.list_models()
    
    # Find EfficientNet models
    efficientnet_models = [m for m in all_models if 'efficientnet' in m.lower()]
    
    print("Available EfficientNet models:")
    for model in sorted(efficientnet_models)[:20]:
        print(f"  - {model}")
    
    # Check specific variations
    print("\nChecking specific names:")
    test_names = [
        'efficientnet-b1',
        'efficientnet_b1', 
        'efficientnetb1',
        'efficientnet_b1_pruned',
        'tf_efficientnet_b1',
        'tf_efficientnet_b1_ns',
    ]
    
    for name in test_names:
        if name in all_models:
            print(f"  ✓ {name} - Found!")
        else:
            print(f"  ✗ {name} - Not found")
            
except ImportError:
    print("timm is not installed. Install with: pip install timm")