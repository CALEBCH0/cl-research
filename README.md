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
pip install -r requirements.txt
```

## Usage

### 1. Run a Single Experiment

```bash
python train_face_cl.py
```

### 2. Run with Different Configurations

```bash
# Use different backbone
python train_face_cl.py backbone=mobilenet

# Use different strategy
python train_face_cl.py strategy=ewc

# Combine multiple overrides
python train_face_cl.py backbone=vit strategy=replay training.batch_size=64
```

### 3. Run Multiple Experiments

```bash
bash scripts/run_experiments.sh
```

### 4. Analyze Results

```bash
# Analyze single experiment
python scripts/analyze_results.py experiments/results/resnet50_ewc/results.yaml

# Compare multiple strategies
python scripts/analyze_results.py experiments/results --compare resnet50_naive resnet50_ewc resnet50_replay
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

1. Start with a small number of experiences to test quickly
2. Use smaller batch sizes for memory-constrained environments
3. Monitor GPU memory usage when using replay strategies
4. Use TensorBoard or WandB for real-time monitoring