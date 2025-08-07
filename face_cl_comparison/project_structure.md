# Face Recognition CL Comparison Framework Structure

## Directory Layout
```
face_cl_comparison/
├── configs/
│   ├── base/
│   │   ├── base_config.yaml              # Master template
│   │   ├── face_datasets.yaml            # Dataset configs
│   │   └── training_defaults.yaml        # Training defaults
│   │
│   ├── experiments/
│   │   ├── backbone_comparison.yaml      # Compare backbones
│   │   ├── strategy_categories.yaml      # Compare strategy types
│   │   ├── replay_variants.yaml          # Within replay category
│   │   ├── regularization_variants.yaml  # Within regularization
│   │   └── optimization_variants.yaml    # Within optimization
│   │
│   └── templates/
│       ├── quick_test.yaml              # For quick testing
│       └── full_evaluation.yaml         # Comprehensive eval
│
├── src/
│   ├── models/
│   │   ├── backbones.py                # Face recognition backbones
│   │   ├── heads.py                    # ArcFace, CosFace, etc.
│   │   └── model_factory.py            # Model creation
│   │
│   ├── strategies/
│   │   ├── replay_variants.py          # All replay strategies
│   │   ├── regularization_variants.py  # All regularization strategies
│   │   ├── optimization_variants.py    # All optimization strategies
│   │   └── strategy_factory.py         # Strategy creation
│   │
│   ├── datasets/
│   │   ├── face_benchmarks.py          # Face dataset loaders
│   │   └── transforms.py               # Face-specific augmentations
│   │
│   └── evaluation/
│       ├── metrics.py                  # Face verification metrics
│       └── visualization.py            # Comparison plots
│
├── runner.py                           # Main runner script
├── analyze_results.py                  # Post-experiment analysis
└── dashboard.py                        # Interactive results viewer
```

## Workflow

### 1. Create Experiment Config
```yaml
# experiments/backbone_comparison.yaml
name: backbone_comparison_exp1
description: Compare backbones for face recognition CL

# What to compare
compare:
  models:
    - resnet50_arcface
    - mobilenet_v3_arcface
    - efficientnet_b0_arcface
  
  strategies:
    - replay_balanced
    - ewc_online
    - naive  # baseline

# Fixed settings
base_settings:
  dataset: olivetti_faces  # or lfw, vggface2_subset
  n_experiences: 10
  n_classes_per_exp: 4
  
  training:
    epochs: 20
    batch_size: 32
    optimizer: adamw
    lr: 0.001
    
  evaluation:
    metrics: 
      - accuracy
      - forgetting
      - face_verification_accuracy
      - inference_time
```

### 2. Run Experiment
```bash
python runner.py --config experiments/backbone_comparison.yaml --gpu 0
```

### 3. View Results
```bash
python analyze_results.py --exp backbone_comparison_exp1
# or
python dashboard.py  # Interactive dashboard
```

## Implementation Details

### Strategy Categories and Variants

**Replay-based:**
- `replay_random`: Random sampling
- `replay_balanced`: Class-balanced sampling  
- `replay_herding`: Herding selection
- `feature_replay`: Store embeddings only
- `generative_replay`: VAE/GAN-based

**Regularization-based:**
- `ewc_online`: Online EWC
- `ewc_separate`: Separate fisher per task
- `si_default`: Synaptic Intelligence
- `mas_default`: Memory Aware Synapses
- `lwf_default`: Learning without Forgetting

**Optimization-based:**
- `gem`: Gradient Episodic Memory
- `agem`: Averaged GEM
- `der`: Dark Experience Replay
- `der++`: DER with consistency

**Architecture-based:**
- `pnn`: Progressive Neural Networks
- `packnet`: PackNet pruning
- `hat`: Hard Attention to Task

### Model Configurations

**Backbones:**
- ResNet (18, 34, 50, 101)
- MobileNetV2, V3 (different width multipliers)
- EfficientNet (B0-B4)
- Vision Transformer (ViT-Tiny, Small)

**Heads:**
- Softmax
- ArcFace
- CosFace
- SphereFace

### Metrics for Face Recognition

1. **CL Metrics:**
   - Average accuracy
   - Forgetting
   - Forward/Backward transfer

2. **Face-specific:**
   - TAR@FAR (1%, 0.1%)
   - Face verification accuracy
   - Identity preservation score

3. **Efficiency:**
   - Inference time
   - Memory usage
   - Model size