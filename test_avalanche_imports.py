"""Test what's actually available in Avalanche."""
import avalanche
print(f"Avalanche version: {avalanche.__version__}")

# Check what's in avalanche.benchmarks.classic
from avalanche.benchmarks import classic
print("\nAvailable benchmarks in avalanche.benchmarks.classic:")
print([attr for attr in dir(classic) if not attr.startswith('_')])

# Check what's actually there
try:
    from avalanche.benchmarks.classic import SplitMNIST
    print("\n✓ SplitMNIST available")
except ImportError:
    print("\n✗ SplitMNIST not available")

try:
    from avalanche.benchmarks.classic import SplitFashionMNIST
    print("✓ SplitFashionMNIST available")
except ImportError:
    print("✗ SplitFashionMNIST not available")

try:
    from avalanche.benchmarks.classic import SplitCIFAR10
    print("✓ SplitCIFAR10 available")
except ImportError:
    print("✗ SplitCIFAR10 not available")

# Check generators
from avalanche.benchmarks import generators
print("\nAvailable generators:")
print([attr for attr in dir(generators) if not attr.startswith('_')])