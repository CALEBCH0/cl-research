# Face Recognition Continual Learning Framework

A flexible framework for comparing continual learning strategies on face recognition tasks using Avalanche.

## Features

- **Multiple CL Strategies**: Naive, EWC, Replay, GEM, AGEM, LwF, SI, MAS, GDumb, iCaRL, SLDA, Pure NCM
- **Flexible Plugin System**: Combine any strategy with multiple plugins
- **Face Recognition Datasets**: 
  - Olivetti Faces (40 identities) - Quick testing
  - LFW - Labeled Faces in the Wild (158+ identities) - Real-world conditions
  - Custom LFW subsets (e.g., lfw_50, lfw_100, lfw_200)
- **Efficient Experimentation**: YAML-based configuration, multi-seed evaluation, automatic result aggregation

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/face_cl_comparison.git
cd face_cl_comparison

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```bash
# Run a simple experiment with Olivetti (quick)
./run_exp.sh NCM_SLDA_iCaRL

# Run with larger LFW dataset
python runner.py --exp lfw_comparison

# Run large-scale face comparison across dataset sizes
python runner.py --exp large_scale_faces

# Or directly with Python
python runner.py --exp NCM_SLDA_iCaRL
```

**To stop experiments gracefully**: Type 'q' then press Enter. The current run will complete and results will be saved.

## Available Datasets

### NEW: Controlled Dataset Subsets (Phase 1)

Create controlled subsets with exact class counts:

```yaml
dataset:
  name: lfw
  subset:
    target_classes: 50          # Exactly 50 classes
    min_samples_per_class: 15   # Quality threshold
    selection_strategy: most_samples  # How to select classes
  n_experiences: 10
```

**Benefits**:
- **Exact control**: Get exactly the number of classes you request
- **Quality guaranteed**: Never go below minimum samples threshold
- **Fair comparison**: Compare methods on identical subsets
- **Dataset-agnostic**: Same config works for any dataset (coming soon)

**Selection strategies**:
- `most_samples`: Select classes with most samples (recommended)
- `balanced`: Select classes with similar sample counts
- `random`: Random selection

### Standard Datasets
| Dataset | Identities | Images/Person | Total Images | Use Case |
|---------|------------|---------------|--------------|----------|
| `olivetti` | 40 | 10 | 400 | Quick tests, prototyping |

### LFW Quality Subsets
Choose based on your quality vs. scale needs:

| Dataset | Classes | Min Imgs/Person | Valid n_experiences | Quality | Recommended For |
|---------|---------|-----------------|---------------------|---------|-----------------|
| `lfw_12` | 12 | 50+ | 2, 3, 4, 6, 12 | ⭐⭐⭐⭐⭐ | Algorithm debugging, perfect data |
| `lfw_24` | 24 | 40+ | 2, 3, 4, 6, 8, 12 | ⭐⭐⭐⭐⭐ | Small experiments with many exp options |
| `lfw_30` | 30 | 30+ | 2, 3, 5, 6, 10, 15 | ⭐⭐⭐⭐ | Small-medium experiments |
| `lfw_60` | 60 | 20+ | 2, 3, 4, 5, 6, 10, 12, 15, 20 | ⭐⭐⭐⭐ | **Default** - great flexibility |
| `lfw_90` | 90 | 15+ | 2, 3, 5, 6, 9, 10, 15, 18 | ⭐⭐⭐ | Large experiments |
| `lfw_120` | 120 | 12+ | 2, 3, 4, 5, 6, 8, 10, 12, 15, 20 | ⭐⭐⭐ | Very large with many divisors |
| `lfw_150` | 150 | 10+ | 2, 3, 5, 6, 10, 15 | ⭐⭐ | Maximum scale (minimum quality) |

**Rule of thumb**: For continual learning, use at least 10 images per person (15+ preferred).

**Choosing n_experiences**: Each dataset has specific valid n_experiences values listed above. If you request an invalid number, it will automatically use the default value for that dataset.

**Note**: LFW dataset will be automatically downloaded on first use (~200MB). It's cached locally after the first download, so subsequent runs will be much faster. The cache location is typically:
- macOS: `~/scikit_learn_data/lfw_home/`
- Linux: `~/.cache/scikit_learn_data/lfw_home/`
- Windows: `C:\Users\{username}\scikit_learn_data\lfw_home\`

The resize issue with sklearn's `fetch_lfw_people` has been fixed - it now properly accepts resize as a float ratio.

## Flexible Plugin System

The framework now supports flexible plugin combinations for any strategy:

### Configuration Format

```yaml
# configs/experiments/my_experiment.yaml
name: my_experiment
description: Test different plugin combinations

comparison:
  strategies:
    # Simple strategy without plugins
    - naive
    
    # Strategy with single plugin
    - name: ewc_replay
      base: ewc
      plugins:
        - name: replay
          params:
            mem_size: 200
            storage_policy: class_balanced
    
    # Strategy with multiple plugins
    - name: ewc_kitchen_sink
      base: ewc
      plugins:
        - name: replay
          params:
            mem_size: 200
        - name: lwf
          params:
            alpha: 0.5
            temperature: 2.0
        - name: mas
          params:
            lambda_reg: 1.0
```

### Available Plugins

| Plugin | Description | Key Parameters |
|--------|-------------|----------------|
| `replay` | Experience replay with different storage policies | `mem_size`, `storage_policy`, `batch_size_mem` |
| `ewc` | Elastic Weight Consolidation | `ewc_lambda`, `mode`, `decay_factor` |
| `lwf` | Learning without Forgetting | `alpha`, `temperature` |
| `feature_distillation` | Feature-level distillation | `alpha`, `temperature`, `loss` |
| `gem` | Gradient Episodic Memory | `patterns_per_exp`, `memory_strength` |
| `mas` | Memory Aware Synapses | `lambda_reg`, `alpha` |
| `si` | Synaptic Intelligence | `si_lambda` |
| `rwalk` | RWalk (EWC variant) | `ewc_lambda`, `ewc_alpha`, `delta_t` |
| `lr_scheduler` | Learning rate scheduling | `scheduler_type`, `step_size`, `gamma` |

### Storage Policies for Replay

- `experience_balanced`: Balance samples across experiences
- `class_balanced`: Balance samples across classes
- `reservoir`: Reservoir sampling

## Strategies

### NCM-Based Strategies
- **Pure NCM**: Nearest Class Mean without training
- **SLDA**: Streaming Linear Discriminant Analysis
- **iCaRL**: Incremental Classifier and Representation Learning

### Standard CL Strategies
- **Naive**: Fine-tuning without any CL mechanism
- **Replay**: Experience replay
- **EWC**: Elastic Weight Consolidation
- **LwF**: Learning without Forgetting
- **GEM**: Gradient Episodic Memory
- **AGEM**: Averaged GEM
- **SI**: Synaptic Intelligence
- **MAS**: Memory Aware Synapses
- **GDumb**: Greedy sampler and Dumb learner

## Configuration

### Basic Configuration Structure

```yaml
name: experiment_name
description: Experiment description

# Strategy comparison
comparison:
  strategies:
    - strategy_config_1
    - strategy_config_2

# Fixed settings
fixed:
  model:
    backbone:
      name: efficientnet_b1  # or mlp, cnn
      
  dataset:
    name: olivetti  # or mnist, cifar10, fmnist
    n_experiences: 8
    
  training:
    epochs_per_experience: 10
    batch_size: 16
    lr: 0.001
    
    # Debug mode: true = single seed, false = multi-seed
    debug: false
    seed: 42  # Used when debug=true
    seeds: [42, 123, 456, 789, 1011]  # Used when debug=false
```

## Results

The framework automatically:
- Runs experiments with multiple seeds (configurable)
- Computes mean ± std for each metric
- Saves results in CSV format
- Displays formatted results table

Example output:
```
==========================================================================================
RESULTS SUMMARY

run_name          strategy          model             dataset_name      average_accuracy 
==========================================================================================
naive             naive             efficientnet_b1   olivetti          0.125 ± 0.023
slda              slda              efficientnet_b1   olivetti          0.970 ± 0.017
slda_replay       slda              efficientnet_b1   olivetti          0.962 ± 0.019
icarl             icarl             efficientnet_b1   olivetti          0.960 ± 0.018
ewc_replay_lwf    ewc               efficientnet_b1   olivetti          0.650 ± 0.045
```

## Advanced Usage

### Custom Plugin Combinations

Create complex strategy combinations by mixing plugins:

```yaml
- name: advanced_ewc
  base: ewc
  plugins:
    - name: replay
      params:
        mem_size: 500
        storage_policy: class_balanced
    - name: lwf
      params:
        alpha: 0.3
    - name: lr_scheduler
      params:
        scheduler_type: cosine
        T_max: 100
```

### Adding New Plugins

To add a new Avalanche plugin:

1. Import it in `src/plugin_factory.py`
2. Add a case in `create_plugin()` function
3. Use it in your configuration

Example:
```python
# In plugin_factory.py
elif plugin_name == 'your_plugin':
    return YourPlugin(
        param1=plugin_config.get('param1', default_value),
        param2=plugin_config.get('param2', default_value)
    )
```

## Project Structure

```
face_cl_comparison/
├── configs/
│   └── experiments/        # Experiment configurations
├── src/
│   ├── training.py        # Core training logic
│   ├── plugin_factory.py  # Plugin creation
│   └── strategies/        # Custom strategies
│       └── pure_ncm.py    
├── results/               # Experiment results
├── runner.py              # Main experiment runner
├── run_exp.sh            # Convenience script
└── requirements.txt
```

## Tips

1. **Debug Mode**: Set `debug: true` for quick testing with single seed
2. **Plugin Warnings**: Some plugin combinations may show warnings (e.g., SLDA with Replay) but still work
3. **Memory Size**: For NCM methods, ensure sufficient memory (e.g., 10+ samples per class)
4. **Face Datasets**: 
   - Olivetti (40 classes) - Fast prototyping and algorithm testing
   - LFW predefined configs - Guaranteed quality with different scales
   - Use `lfw_20` to `lfw_100` for reliable CL experiments
   - Avoid `lfw_all` unless you need maximum scale
5. **Dataset Quality**: Higher min_faces_per_person = fewer identities but better quality

## Common Issues

1. **iCaRL Poor Performance**: Fixed by ensuring correct class ID handling (`class_ids_from_zero_in_each_exp=False`)
2. **Plugin Compatibility**: Some strategies (like SLDA) may warn about incompatible callbacks but still function
3. **Memory Requirements**: EfficientNet models require significant GPU memory
4. **LFW Resize Error**: Fixed - sklearn's `fetch_lfw_people` expects resize as a float ratio (e.g., 0.256 for 64x64 from 250x250)
5. **Dataset Download**: LFW will automatically download (~200MB) on first use

## Contributing

Feel free to:
- Add new strategies
- Implement additional plugins  
- Extend to new datasets
- Improve the framework

## License

MIT License