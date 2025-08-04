"""Diagnose what's available in your Avalanche installation."""
import sys
import importlib
import avalanche

print(f"Python version: {sys.version}")
print(f"Avalanche version: {avalanche.__version__}")
print("\n" + "="*60 + "\n")

# Test different import paths
imports_to_test = [
    # Classic benchmarks
    ("avalanche.benchmarks.classic", "SplitMNIST"),
    ("avalanche.benchmarks.classic", "SplitCIFAR10"),
    ("avalanche.benchmarks.classic", "SplitCIFAR100"),
    ("avalanche.benchmarks.classic", "SplitFashionMNIST"),
    ("avalanche.benchmarks.classic", "SplitCelebA"),
    
    # Alternative paths
    ("avalanche.benchmarks", "SplitMNIST"),
    ("avalanche.benchmarks.scenarios", "SplitMNIST"),
    
    # Generators
    ("avalanche.benchmarks.generators", "nc_benchmark"),
    ("avalanche.benchmarks.generators", "ni_benchmark"),
    
    # Utils
    ("avalanche.benchmarks.utils", "AvalancheDataset"),
    ("avalanche.benchmarks.utils", "as_classification_dataset"),
    
    # Training strategies
    ("avalanche.training.supervised", "Naive"),
    ("avalanche.training.supervised", "EWC"),
    ("avalanche.training.supervised", "Replay"),
    ("avalanche.training", "Naive"),
    ("avalanche.training", "EWC"),
    
    # Models
    ("avalanche.models", "SimpleMLP"),
    ("avalanche.models", "SimpleCNN"),
]

print("Testing imports:")
print("-"*60)

available_imports = {}
for module_path, class_name in imports_to_test:
    try:
        module = importlib.import_module(module_path)
        if hasattr(module, class_name):
            available_imports[f"{module_path}.{class_name}"] = "✓ Available"
            print(f"✓ from {module_path} import {class_name}")
        else:
            print(f"✗ {class_name} not found in {module_path}")
    except ImportError as e:
        print(f"✗ Cannot import {module_path}: {e}")

print("\n" + "="*60 + "\n")

# Check what's actually in avalanche.benchmarks.classic
try:
    import avalanche.benchmarks.classic as classic
    print("Contents of avalanche.benchmarks.classic:")
    print("-"*60)
    for attr in sorted(dir(classic)):
        if not attr.startswith('_'):
            print(f"  - {attr}")
except ImportError:
    print("Cannot import avalanche.benchmarks.classic")

print("\n" + "="*60 + "\n")

# Check avalanche.benchmarks structure
print("Avalanche benchmarks module structure:")
print("-"*60)
import avalanche.benchmarks as benchmarks
for attr in sorted(dir(benchmarks)):
    if not attr.startswith('_'):
        obj = getattr(benchmarks, attr)
        print(f"  - {attr}: {type(obj).__name__}")