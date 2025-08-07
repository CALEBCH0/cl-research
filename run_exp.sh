#!/bin/bash

# Face CL Experiment Runner Script
# Usage: ./run_exp.sh <experiment_name> [options]

# Check if experiment name is provided
if [ $# -eq 0 ]; then
    echo "Usage: ./run_exp.sh <experiment_name> [options]"
    echo ""
    echo "Available experiments:"
    for exp in face_cl_comparison/configs/experiments/*.yaml; do
        basename "$exp" .yaml | sed 's/^/  - /'
    done
    echo ""
    echo "Options:"
    echo "  --gpu <n>     GPU device to use (default: 0)"
    echo "  --dry-run     Show configs without running"
    echo ""
    echo "Example:"
    echo "  ./run_exp.sh NCM_SLDA_iCaRL"
    echo "  ./run_exp.sh backbone_comparison --gpu 1"
    echo "  ./run_exp.sh replay_variants --dry-run"
    exit 1
fi

# Store experiment name
EXPERIMENT=$1
shift  # Remove first argument, keeping the rest for options

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Face CL Experiment Runner ===${NC}"
echo -e "${BLUE}Experiment: ${GREEN}$EXPERIMENT${NC}"

# Check if venv exists
if [ ! -d ".venv" ]; then
    echo -e "${RED}Error: Virtual environment not found!${NC}"
    echo "Please create it with: python -m venv .venv"
    exit 1
fi

# Activate virtual environment
echo -e "${BLUE}Activating virtual environment...${NC}"
source .venv/bin/activate

# Check if activation was successful
if [ -z "$VIRTUAL_ENV" ]; then
    echo -e "${RED}Error: Failed to activate virtual environment!${NC}"
    exit 1
fi

# Change to face_cl_comparison directory
cd face_cl_comparison || exit 1

# Run the experiment
echo -e "${BLUE}Running experiment...${NC}"
echo ""

python runner.py --exp "$EXPERIMENT" "$@"

# Store exit code
EXIT_CODE=$?

# Deactivate virtual environment
deactivate

# Exit with the same code as the python script
exit $EXIT_CODE