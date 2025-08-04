# Fixing RTX 3090 Performance Issues

## Common Causes & Solutions

### 1. **GPU Power State (Most Common)**
The GPU might be in power-saving mode (P2/P8 state).

**Check:**
```bash
nvidia-smi -q -d PERFORMANCE
```

**Fix:**
```bash
# Set to maximum performance mode
sudo nvidia-smi -pm 1  # Enable persistence mode
sudo nvidia-smi -pl 350  # Set power limit to 350W (3090 default)

# Windows:
nvidia-smi -pm 1
```

### 2. **CPU Bottleneck**
Data loading might be bottlenecking the GPU.

**Fix in code:**
```python
# Increase number of workers
DataLoader(..., num_workers=8, pin_memory=True)

# Enable CPU affinity
torch.set_num_threads(8)
```

### 3. **Wrong PyTorch Installation**
You might have CPU-only PyTorch.

**Check:**
```python
import torch
print(torch.cuda.is_available())  # Should be True
print(torch.version.cuda)  # Should show CUDA version
```

**Reinstall:**
```bash
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 4. **Small Batch Size**
Small batches don't fully utilize the GPU.

**Fix:**
```bash
# For 3090 (24GB), use larger batches
python train_working.py --batch_size 256 --benchmark cifar10
```

### 5. **PCIe Bandwidth**
Check if GPU is in correct PCIe slot.

**Check:**
```bash
nvidia-smi -q | grep -i pcie
# Should show PCIe Gen3 x16 or Gen4 x16
```

### 6. **Thermal Throttling**
Check GPU temperature.

**Monitor:**
```bash
watch -n 1 nvidia-smi
# Temperature should stay below 83°C
```

### 7. **CUDNN Not Optimized**
Enable cuDNN autotuner.

**Add to code:**
```python
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True  # For Ampere GPUs
```

## Quick Performance Test

Run this to verify GPU is working correctly:
```python
import torch
import time

# Should be ~10-50x faster on 3090 vs CPU
x = torch.randn(5000, 5000).cuda()
start = time.time()
for _ in range(100):
    y = x @ x
torch.cuda.synchronize()
print(f"Time: {time.time() - start:.2f}s")
```

## Expected Performance

RTX 3090 should achieve:
- Matrix ops: 200-400 GFLOPS (vs ~10-20 on CPU)
- Training: 10,000+ samples/sec on MNIST
- 10-50x speedup over CPU

If you're not seeing this, run the benchmark and check the warnings!