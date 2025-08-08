#!/usr/bin/env python3
"""Check available MobileNet models in timm."""
import timm

# List all available models
all_models = timm.list_models()

# Find MobileNet models
mobilenet_models = [m for m in all_models if 'mobile' in m.lower()]

print("Available MobileNet models in timm:")
for model in sorted(mobilenet_models):
print(f"  {model}")

# Check specific models
test_models = [
'mobilenetv3_small',
'mobilenetv3_large',
'mobilenetv3_small_100',
'mobilenetv3_large_100',
'mobilenetv3_small_075',
'mobilenetv3_large_075',
'mobilenetv2_100',
'mobilenetv2_050',
]

print("\nChecking specific model names:")
for model_name in test_models:
try:
    model = timm.create_model(model_name, pretrained=False)
    print(f"✓ {model_name} - Success")
except Exception as e:
    print(f"✗ {model_name} - Error: {e}")