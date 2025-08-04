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

### Option 1: Using Avalanche Built-in Datasets (Recommended for Testing)

```bash
# Test with Fashion-MNIST (quick, no face data needed)
python train_faces_avalanche.py --dataset fmnist --epochs 2

# Test with CIFAR-10 (general benchmark)
python train_simple.py --benchmark cifar10 --epochs 2 --strategy ewc

# Test with different strategies
python train_simple.py --benchmark mnist --strategy replay --epochs 3
```

### Option 2: Using Synthetic Data (No Download Required)

```bash
# Run with synthetic face data
python train_face_cl.py dataset=synthetic training.epochs_per_experience=2

# Smaller synthetic dataset for quick testing
python train_face_cl.py dataset=synthetic dataset.num_samples=500 dataset.num_classes=20
```

### Option 3: Using Real Face Data (LFW Dataset)

```bash
# First, download LFW dataset (if you have internet)
python scripts/download_lfw.py

# Then run experiments
python train_face_cl.py

# Or manually download if behind firewall/proxy:
# 1. Download: http://vis-www.cs.umass.edu/lfw/lfw.tgz
# 2. Extract to: datasets/face_datasets/lfw/
# 3. Run: python train_face_cl.py
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