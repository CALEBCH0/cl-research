# Face Recognition Continual Learning Testing Framework

This framework allows you to test different backbone models and continual learning strategies for face recognition using the Avalanche library.

## Directory Structure

```
cl-research/
├── models/
│   ├── backbones/       # Backbone models (ResNet, MobileNet, ViT, etc.)
│   └── heads/           # Classification heads (Softmax, ArcFace, CosFace)
├── datasets/
│   └── face_datasets/   # Face recognition datasets (LFW, etc.)
├── strategies/          # Continual learning strategies
├── experiments/
│   ├── configs/         # Hydra configuration files
│   ├── results/         # Experiment results
│   └── logs/            # Training logs
├── utils/
│   ├── metrics/         # Evaluation metrics
│   └── visualization/   # Plotting utilities
└── scripts/             # Utility scripts
```

## Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# If you encounter issues with avalanche-lib, try:
pip install -r requirements_fixed.txt
```

## Quick Start

### Option 1: Simple Continual Learning (Recommended)

#### Available Benchmarks:
- `mnist` - Handwritten digits (0-9)
- `fmnist` - Fashion items (clothing, shoes)
- `cifar10` - Natural images (10 classes)
- `lfw` - Labeled Faces in the Wild (face recognition)
- `celeba` - Celebrity faces (1.4GB download)

#### Available Strategies:
- **Regularization-based**: `naive`, `ewc`, `si`, `mas`, `lwf`
- **Replay-based**: `replay`, `gem`, `agem`, `gdumb`, `icarl`
- **Other**: `cumulative`, `joint` (upper bound)

```bash
# Test different strategies on Fashion-MNIST
python train_working.py --benchmark fmnist --strategy naive --epochs 2
python train_working.py --benchmark fmnist --strategy replay --epochs 2
python train_working.py --benchmark fmnist --strategy lwf --epochs 2

# Test with face recognition
python train_working.py --benchmark lfw --strategy replay --mem_size 500

# Test different buffer sizes
python train_working.py --benchmark fmnist --strategy replay --mem_size 100
python train_working.py --benchmark fmnist --strategy replay --mem_size 1000
```

### Option 2: Advanced Face Recognition (Custom Framework)

```bash
# Run with synthetic face data (no download required)
python train_face_cl.py dataset=synthetic training.epochs_per_experience=2

# Run with real LFW dataset
python scripts/download_lfw.py  # Download first
python train_face_cl.py

# Compare different backbones and strategies
python train_face_cl.py backbone=resnet50 strategy=ewc
python train_face_cl.py backbone=mobilenet strategy=replay
```

### Option 3: Automated Comparisons

```bash
# Compare different replay buffer sizes
python compare_buffer_sizes.py

# Run all backbone/strategy combinations
bash scripts/run_experiments.sh
```

## Example Usage

### Testing Different Models and Strategies

```bash
# Test ResNet50 with EWC strategy
python train_face_cl.py backbone=resnet50 strategy=ewc

# Test MobileNet with Replay strategy (memory efficient)
python train_face_cl.py backbone=mobilenet strategy=replay strategy.mem_size=1000

# Test Vision Transformer with larger batch size
python train_face_cl.py backbone=vit training.batch_size=64 training.epochs_per_experience=5

# Quick test with reduced settings
python train_face_cl.py dataset=synthetic \
    dataset.num_samples=500 \
    dataset.n_experiences=3 \
    training.epochs_per_experience=2 \
    training.batch_size=16
```

### Running on Different Hardware

```bash
# For GTX 730 (2GB VRAM) - use smaller batch sizes
python train_face_cl.py training.batch_size=16 training.eval_batch_size=32 backbone=mobilenet

# For CPU only
python train_face_cl.py experiment.device=cpu training.batch_size=8

# For M1 Mac (if MPS is working)
python train_face_cl.py experiment.device=mps training.batch_size=32
```

### Batch Experiments

```bash
# Run all combinations of backbones and strategies
bash scripts/run_experiments.sh

# Run specific combination
python train_face_cl.py backbone=resnet50 strategy=ewc experiment.name="resnet50_ewc_test"
```

### Analyzing Results

```bash
# Analyze single experiment
python scripts/analyze_results.py experiments/results/resnet50_ewc/results.yaml

# Compare multiple strategies
python scripts/analyze_results.py experiments/results --compare resnet50_naive resnet50_ewc resnet50_replay

# Generate plots for a specific run
python scripts/analyze_results.py outputs/2024-01-01/12-00-00/results.yaml --output-dir plots/
```

## Configuration

### Backbone Models
- ResNet (18, 50, 101)
- MobileNetV2, MobileNetV3
- Vision Transformer (ViT)
- EfficientNet

### Continual Learning Strategies
- Naive (Fine-tuning)
- EWC (Elastic Weight Consolidation)
- Replay (Experience Replay)
- LwF (Learning without Forgetting)
- GEM (Gradient Episodic Memory)
- AGEM (Averaged GEM)
- SI (Synaptic Intelligence)

### Datasets
- LFW (Labeled Faces in the Wild)
- Additional datasets can be added by implementing loaders in `datasets/`

### Metrics
- Classification accuracy
- Face verification metrics (TAR@FAR)
- Continual learning metrics:
  - Average accuracy
  - Forgetting
  - Forward transfer
  - Backward transfer

## Adding New Components

### Add a New Backbone
1. Add configuration in `experiments/configs/backbone/`
2. Update `models/backbones/backbone_factory.py`

### Add a New Strategy
1. Add configuration in `experiments/configs/strategy/`
2. Update `strategies/strategy_factory.py`

### Add a New Dataset
1. Create loader in `datasets/`
2. Update `datasets/scenario_builder.py`
3. Add configuration in `experiments/configs/dataset/`

## Results

Results are saved in Hydra output directories with:
- Model checkpoints
- Training logs
- Metrics (results.yaml)
- Visualizations (if generated)

## Available Datasets in Avalanche

### Built-in Classic Benchmarks
**Class-Incremental (Split by classes):**
- `SplitMNIST` - Handwritten digits (10 classes)
- `SplitFMNIST` - Fashion items (10 classes)
- `SplitCIFAR10` - Natural images (10 classes)
- `SplitCIFAR100` - Natural images (100 classes)
- `SplitTinyImageNet` - Tiny ImageNet (200 classes)
- `SplitImageNet` - Full ImageNet (1000 classes)
- `SplitCUB200` - Birds dataset (200 species)
- `SplitOmniglot` - Handwritten characters (1623 classes)
- `SplitInaturalist` - Nature species dataset

**Domain-Incremental (Same classes, different domains):**
- `PermutedMNIST` - MNIST with pixel permutations
- `RotatedMNIST` - MNIST with rotations

**Advanced/Specialized:**
- `CORe50` - 50 objects in 11 contexts
- `OpenLORIS` - Objects with natural variations
- `CLStream51` - Stream-51 for online learning
- `CLEAR` - Real-world imagery with temporal shifts
- `EndlessCLSim` - Synthetic endless learning
- `MiniImageNet` - Subset of ImageNet (100 classes)

### Datasets for Face Recognition

**Available through Avalanche/PyTorch:**
- `CelebA` - 200K celebrity images (better for attribute classification)
- `LFWPeople` - Limited for recognition (many people have <5 images)

**Recommended External Datasets for Face Recognition:**
1. **For Quick Testing:**
   - **Olivetti Faces** - 40 people, 10 images each (via scikit-learn)
   - **Yale Face Database B** - 38 people, varied lighting

2. **For Real Experiments:**
   - **VGGFace2** - 3.31M images, 9K identities (need custom loader)
   - **MS-Celeb-1M** - Large scale (use cleaned versions)
   - **CASIA-WebFace** - 500K images, 10K identities

### Quick Start with Face Recognition

```python
# Using Olivetti Faces (recommended for testing)
from sklearn.datasets import fetch_olivetti_faces
from avalanche.benchmarks import nc_benchmark
from avalanche.benchmarks.utils import AvalancheDataset
from torch.utils.data import TensorDataset
import torch

# Load and prepare data
data = fetch_olivetti_faces()
X = torch.FloatTensor(data.images).unsqueeze(1)
y = torch.LongTensor(data.target)

# Create Avalanche benchmark
train_data = AvalancheDataset(TensorDataset(X[:320], y[:320]))
test_data = AvalancheDataset(TensorDataset(X[320:], y[320:]))

benchmark = nc_benchmark(
    train_dataset=train_data,
    test_dataset=test_data,
    n_experiences=5,
    task_labels=False
)
```

## Tips

1. **Start Simple**: Use `train_simple.py` or `train_faces_avalanche.py` for initial testing
2. **Memory Management**: 
   - GTX 730 (2GB): Use batch_size=8-16, mobilenet backbone
   - Jetson Nano: Not recommended for training
   - CPU: Use batch_size=4-8, expect slow training
3. **Quick Testing**: Use synthetic dataset with reduced samples and experiences
4. **Debugging**: Set `HYDRA_FULL_ERROR=1` for detailed error traces
5. **Network Issues**: Use synthetic data or built-in Avalanche datasets if downloads fail

## Troubleshooting

### Common Issues

1. **Download fails**: Use synthetic dataset or download manually
2. **Out of memory**: Reduce batch_size and use smaller models
3. **Import errors**: Try `pip install -r requirements_fixed.txt`
4. **WSL network issues**: Download on Windows and copy to WSL filesystem
5. **Numpy warnings on WSL**: The longdouble warning is harmless and has been suppressed