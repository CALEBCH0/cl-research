"""Check CUDA availability and configuration."""
import torch
import sys

print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print("-" * 60)

# Check CUDA availability
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"cuDNN version: {torch.backends.cudnn.version()}")
print(f"Number of GPUs: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    print(f"Current GPU: {torch.cuda.current_device()}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Test tensor on GPU
    try:
        x = torch.tensor([1.0, 2.0, 3.0]).cuda()
        print(f"\nTest tensor on GPU: {x}")
        print(f"Tensor device: {x.device}")
        print("✓ GPU tensor creation successful")
    except Exception as e:
        print(f"✗ GPU tensor creation failed: {e}")
else:
    print("\nPossible reasons CUDA is not available:")
    print("1. PyTorch installed without CUDA support (CPU-only version)")
    print("2. CUDA drivers not installed or incompatible")
    print("3. No NVIDIA GPU available")
    
    print("\nTo install PyTorch with CUDA support:")
    print("1. Uninstall current PyTorch: pip uninstall torch torchvision")
    print("2. Visit https://pytorch.org/get-started/locally/")
    print("3. Select your CUDA version and get the install command")
    print("   For CUDA 11.8: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
    print("   For CUDA 12.1: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")