#!/bin/bash
# Run scale comparison experiments

echo "Running face recognition scale comparison experiments..."
echo "=================================================="

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Run experiments in order of scale
echo -e "\n1. Olivetti (40 classes)..."
python runner.py --exp scale_comparison_olivetti

echo -e "\n2. LFW subset (50 classes)..."
python runner.py --exp scale_comparison_lfw_50

echo -e "\n3. LFW subset (100 classes)..."
python runner.py --exp scale_comparison_lfw_100

echo -e "\n4. LFW full (~150 classes)..."
python runner.py --exp scale_comparison_lfw_full

echo -e "\nAll experiments completed!"
echo "Results saved in results/ directory"