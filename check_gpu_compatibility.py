"""Check GPU compatibility with PyTorch."""
import torch
import subprocess
import platform

print("System Information:")
print("-" * 60)
print(f"Platform: {platform.system()}")
print(f"Python: {platform.python_version()}")
print(f"PyTorch: {torch.__version__}")

print("\nGPU Information:")
print("-" * 60)

# Check CUDA availability
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version (PyTorch): {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Compute Capability: {props.major}.{props.minor}")
        print(f"  Memory: {props.total_memory / 1024**3:.2f} GB")
        print(f"  Multi-processors: {props.multi_processor_count}")
        
        # Check if GPU is supported
        if props.major < 3 or (props.major == 3 and props.minor < 5):
            print(f"  ⚠️  WARNING: This GPU may not be fully supported by PyTorch")
            print(f"  ⚠️  PyTorch typically requires Compute Capability 3.5 or higher")
else:
    print("CUDA is not available")
    
    # Try to get nvidia-smi output
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("\nnvidia-smi output:")
            print(result.stdout)
        else:
            print("\nnvidia-smi not found or failed")
    except:
        print("\nCould not run nvidia-smi")

print("\nGTX 730 Compatibility Notes:")
print("-" * 60)
print("GTX 730 comes in different versions:")
print("- Kepler-based (GK208): Compute Capability 3.5 ✓ (Should work)")
print("- Fermi-based (GF108): Compute Capability 2.1 ✗ (Too old for modern PyTorch)")
print("\nIf your GTX 730 has Compute Capability < 3.5, you'll need to:")
print("1. Use CPU instead: --device cpu")
print("2. Use an older PyTorch version (not recommended)")
print("3. Upgrade to a newer GPU")

# Test small tensor operation
if torch.cuda.is_available():
    try:
        print("\nTesting GPU...")
        x = torch.randn(10, 10).cuda()
        y = x @ x.T
        print("✓ Basic GPU operations work")
    except Exception as e:
        print(f"✗ GPU test failed: {e}")