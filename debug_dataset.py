"""Debug dataset format."""
from avalanche.benchmarks.classic import SplitFMNIST
from torch.utils.data import DataLoader

# Create benchmark
benchmark = SplitFMNIST(n_experiences=5, return_task_id=False, seed=42)

print("Benchmark info:")
print(f"Type: {type(benchmark)}")
print(f"Experiences: {benchmark.n_experiences}")

# Check first experience
first_exp = benchmark.train_stream[0]
print(f"\nFirst experience:")
print(f"Type: {type(first_exp)}")
print(f"Dataset type: {type(first_exp.dataset)}")
print(f"Dataset length: {len(first_exp.dataset)}")

# Try different ways to access data
print("\nTrying to access data...")

# Method 1: Direct access
try:
    sample = first_exp.dataset[0]
    print(f"Direct access: type={type(sample)}, len={len(sample) if hasattr(sample, '__len__') else 'N/A'}")
    if isinstance(sample, (list, tuple)):
        for i, item in enumerate(sample):
            print(f"  Item {i}: type={type(item)}, shape={getattr(item, 'shape', 'No shape')}")
except Exception as e:
    print(f"Direct access failed: {e}")

# Method 2: DataLoader
try:
    loader = DataLoader(first_exp.dataset, batch_size=1)
    batch = next(iter(loader))
    print(f"\nDataLoader access: type={type(batch)}, len={len(batch) if hasattr(batch, '__len__') else 'N/A'}")
    if isinstance(batch, (list, tuple)):
        for i, item in enumerate(batch):
            print(f"  Item {i}: type={type(item)}, shape={item.shape if hasattr(item, 'shape') else 'No shape'}")
except Exception as e:
    print(f"DataLoader access failed: {e}")

# Method 3: Use the experience's dataloader
try:
    exp_loader = first_exp.dataset
    print(f"\nExperience dataset attributes:")
    for attr in dir(exp_loader):
        if not attr.startswith('_'):
            print(f"  {attr}")