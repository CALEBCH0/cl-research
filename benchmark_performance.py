"""Benchmark to diagnose performance issues."""
import torch
import torch.nn as nn
import time
import numpy as np
from avalanche.benchmarks.classic import SplitMNIST
from avalanche.models import SimpleCNN
from avalanche.training.supervised import Naive
import platform


def benchmark_tensor_ops(device, size=1000):
    """Benchmark basic tensor operations."""
    print(f"\n{'='*60}")
    print(f"Benchmarking tensor operations on {device}")
    print(f"{'='*60}")
    
    # Matrix multiplication
    x = torch.randn(size, size, device=device)
    y = torch.randn(size, size, device=device)
    
    # Warmup
    for _ in range(3):
        _ = torch.matmul(x, y)
    
    if device != 'cpu':
        torch.cuda.synchronize()
    
    # Time matrix multiplication
    times = []
    for _ in range(10):
        start = time.time()
        result = torch.matmul(x, y)
        if device != 'cpu':
            torch.cuda.synchronize()
        times.append(time.time() - start)
    
    avg_time = np.mean(times) * 1000  # Convert to ms
    print(f"Matrix multiply ({size}x{size}): {avg_time:.2f} ms")
    
    # Measure FLOPS
    flops = 2 * size ** 3  # For matrix multiplication
    gflops = (flops / avg_time) / 1e6  # GFLOPS
    print(f"Performance: {gflops:.2f} GFLOPS")
    
    return gflops


def benchmark_training(device, batch_size=128, num_batches=50):
    """Benchmark actual training performance."""
    print(f"\n{'='*60}")
    print(f"Benchmarking training on {device}")
    print(f"Batch size: {batch_size}")
    print(f"{'='*60}")
    
    # Create model and data
    model = SimpleCNN(num_classes=10, input_channels=1).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    # Create fake data
    data = torch.randn(batch_size, 1, 28, 28, device=device)
    targets = torch.randint(0, 10, (batch_size,), device=device)
    
    # Warmup
    for _ in range(5):
        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    if device != 'cpu':
        torch.cuda.synchronize()
    
    # Time training iterations
    start_time = time.time()
    for _ in range(num_batches):
        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    if device != 'cpu':
        torch.cuda.synchronize()
    
    total_time = time.time() - start_time
    samples_per_sec = (num_batches * batch_size) / total_time
    time_per_batch = (total_time / num_batches) * 1000  # ms
    
    print(f"Time per batch: {time_per_batch:.2f} ms")
    print(f"Samples/second: {samples_per_sec:.0f}")
    
    return samples_per_sec


def check_gpu_state():
    """Check GPU state and potential issues."""
    if torch.cuda.is_available():
        print(f"\n{'='*60}")
        print("GPU State Check")
        print(f"{'='*60}")
        
        # Check GPU properties
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {props.total_memory / 1024**3:.1f} GB")
        print(f"CUDA Cores: {props.multi_processor_count * 128}")  # Approximate
        
        # Check current memory usage
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"Memory allocated: {allocated:.2f} GB")
        print(f"Memory reserved: {reserved:.2f} GB")
        
        # Check if GPU is in P2 state (power saving)
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi', '--query-gpu=pstate,clocks.gr,clocks.mem', 
                                   '--format=csv,noheader'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print(f"GPU State: {result.stdout.strip()}")
                if 'P2' in result.stdout or 'P8' in result.stdout:
                    print("⚠️  WARNING: GPU may be in power saving mode!")
                    print("   This can severely impact performance.")
                    print("   Try: sudo nvidia-smi -pm 1")
        except:
            pass
        
        # Check PyTorch settings
        print(f"\nPyTorch Configuration:")
        print(f"cudNN enabled: {torch.backends.cudnn.enabled}")
        print(f"cudNN benchmark: {torch.backends.cudnn.benchmark}")
        print(f"TF32 enabled: {torch.backends.cuda.matmul.allow_tf32}")
        
        # Common issues
        print(f"\nPotential Issues to Check:")
        print("1. GPU in power saving mode (P2/P8 state)")
        print("2. CPU bottleneck (data loading)")
        print("3. PCIe bandwidth limitations")
        print("4. Thermal throttling")
        print("5. Wrong PyTorch build (CPU-only accidentally installed)")


def main():
    print(f"System: {platform.system()} {platform.machine()}")
    print(f"Python: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__}")
    
    # Detect available devices
    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')
        # Enable optimizations for 3090
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
    
    results = {}
    
    # Benchmark each device
    for device in devices:
        print(f"\n{'#'*60}")
        print(f"TESTING ON: {device.upper()}")
        print(f"{'#'*60}")
        
        # Tensor operations benchmark
        gflops = benchmark_tensor_ops(device, size=2000)
        
        # Training benchmark
        samples_sec = benchmark_training(device, batch_size=128 if device == 'cuda' else 32)
        
        results[device] = {
            'gflops': gflops,
            'samples_per_sec': samples_sec
        }
    
    # Check GPU state if available
    if torch.cuda.is_available():
        check_gpu_state()
    
    # Summary
    print(f"\n{'='*60}")
    print("PERFORMANCE SUMMARY")
    print(f"{'='*60}")
    for device, perf in results.items():
        print(f"\n{device.upper()}:")
        print(f"  Compute: {perf['gflops']:.2f} GFLOPS")
        print(f"  Training: {perf['samples_per_sec']:.0f} samples/sec")
    
    if 'cuda' in results and 'cpu' in results:
        speedup_compute = results['cuda']['gflops'] / results['cpu']['gflops']
        speedup_training = results['cuda']['samples_per_sec'] / results['cpu']['samples_per_sec']
        print(f"\nGPU Speedup vs CPU:")
        print(f"  Compute: {speedup_compute:.1f}x")
        print(f"  Training: {speedup_training:.1f}x")
        
        if speedup_training < 5:
            print("\n⚠️  WARNING: GPU speedup is lower than expected!")
            print("   A 3090 should be 10-50x faster than CPU for deep learning.")


if __name__ == "__main__":
    main()