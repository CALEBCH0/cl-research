"""Efficient training script for low-end GPUs or CPU."""
import torch
import torch.nn as nn
from avalanche.benchmarks.classic import SplitMNIST, SplitFMNIST
from avalanche.models import SimpleMLP
from avalanche.training.supervised import Naive
from avalanche.evaluation.metrics import accuracy_metrics
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin
import argparse
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')


def check_device():
    """Check and recommend best device."""
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = props.total_memory / 1024**3
        compute_cap = f"{props.major}.{props.minor}"
        
        print(f"GPU detected: {gpu_name}")
        print(f"Memory: {gpu_memory:.1f} GB")
        print(f"Compute Capability: {compute_cap}")
        
        if props.major < 3 or (props.major == 3 and props.minor < 5):
            print("⚠️  GPU may not be fully supported. Using CPU instead.")
            return 'cpu'
        elif gpu_memory < 3:  # Less than 3GB
            print("⚠️  Limited GPU memory. Using small batch sizes.")
            return 'cuda'
        else:
            return 'cuda'
    else:
        print("No GPU detected. Using CPU.")
        return 'cpu'


def get_efficient_model(num_classes=10, input_size=784):
    """Create a smaller model for limited resources."""
    return nn.Sequential(
        nn.Linear(input_size, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(64, num_classes)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', type=str, default='mnist',
                       choices=['mnist', 'fmnist'])
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--experiences', type=int, default=5)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--force_cpu', action='store_true',
                       help='Force CPU usage even if GPU is available')
    args = parser.parse_args()
    
    # Determine device
    if args.force_cpu:
        device = 'cpu'
        print("Forcing CPU usage")
    elif args.device == 'auto':
        device = check_device()
    else:
        device = args.device
    
    # Adjust batch size based on device
    if device == 'cpu':
        batch_size = min(args.batch_size, 32)
        print(f"Using batch size: {batch_size} (CPU)")
    elif 'cuda' in device and torch.cuda.get_device_properties(0).total_memory < 3e9:
        batch_size = min(args.batch_size, 16)
        print(f"Using batch size: {batch_size} (Limited GPU memory)")
    else:
        batch_size = args.batch_size
    
    print(f"\nFinal configuration:")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print("-" * 40)
    
    # Create benchmark
    if args.benchmark == 'mnist':
        benchmark = SplitMNIST(n_experiences=args.experiences, return_task_id=False, seed=42)
    else:
        benchmark = SplitFMNIST(n_experiences=args.experiences, return_task_id=False, seed=42)
    
    # Create efficient model
    model = get_efficient_model(num_classes=10, input_size=784)
    model = model.to(device)
    
    # Create optimizer with lower memory footprint
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    # Simple evaluation plugin
    eval_plugin = EvaluationPlugin(
        accuracy_metrics(minibatch=False, epoch=True, experience=True),
        loggers=[InteractiveLogger()]
    )
    
    # Create strategy
    strategy = Naive(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_mb_size=batch_size,
        train_epochs=args.epochs,
        eval_mb_size=batch_size * 2,
        device=device,
        evaluator=eval_plugin
    )
    
    # Training loop
    print(f"\nTraining on {benchmark.n_experiences} experiences...")
    for i, exp in enumerate(benchmark.train_stream):
        print(f"\nExperience {i+1}/{benchmark.n_experiences}")
        strategy.train(exp)
        
        # Quick evaluation
        print("Evaluating...")
        strategy.eval(benchmark.test_stream[i:i+1])  # Only eval current experience
    
    print("\nTraining completed!")
    
    # Final evaluation
    print("\nFinal evaluation on all experiences:")
    strategy.eval(benchmark.test_stream)


if __name__ == "__main__":
    main()