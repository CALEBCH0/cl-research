#!/bin/bash

# Script to run multiple experiments with different backbones and strategies

# Array of backbones to test
BACKBONES=("resnet50" "mobilenet" "vit")

# Array of strategies to test
STRATEGIES=("naive" "ewc" "replay")

# Base output directory
OUTPUT_DIR="experiments/results"

# Create output directory
mkdir -p $OUTPUT_DIR

echo "Starting experiments..."

# Loop through all combinations
for backbone in "${BACKBONES[@]}"; do
    for strategy in "${STRATEGIES[@]}"; do
        echo "======================================"
        echo "Running: Backbone=$backbone, Strategy=$strategy"
        echo "======================================"
        
        # Run experiment
        python train_face_cl.py \
            backbone=$backbone \
            strategy=$strategy \
            experiment.name="${backbone}_${strategy}" \
            hydra.run.dir="${OUTPUT_DIR}/${backbone}_${strategy}"
        
        echo "Completed: $backbone with $strategy"
        echo ""
    done
done

echo "All experiments completed!"
echo "Results saved in: $OUTPUT_DIR"

# Generate comparison plots
echo "Generating comparison plots..."
python -c "
from utils.visualization import compare_strategies
import os

output_dir = '$OUTPUT_DIR'
backbones = ['resnet50', 'mobilenet', 'vit']
strategies = ['naive', 'ewc', 'replay']

# Compare strategies for each backbone
for backbone in backbones:
    strategy_dirs = [f'{backbone}_{strategy}' for strategy in strategies]
    compare_strategies(output_dir, strategy_dirs, f'{output_dir}/plots_{backbone}')

print('Comparison plots generated!')
"