"""CPU-optimized training for systems without compatible GPUs."""
import torch
import torch.nn as nn
from avalanche.benchmarks.classic import SplitMNIST, SplitFMNIST
from avalanche.training.supervised import Naive
from avalanche.evaluation.metrics import accuracy_metrics
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin
import argparse
import os

# Optimize CPU performance
torch.set_num_threads(os.cpu_count())  # Use all CPU cores
torch.set_num_interop_threads(2)  # Inter-op parallelism


class LightweightCNN(nn.Module):
    """Lightweight CNN optimized for CPU."""
    def __init__(self, num_classes=10, input_channels=1):
        super().__init__()
        # Smaller kernel sizes and fewer channels for CPU
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Flatten()
        )
        
        # Calculate feature size
        if input_channels == 1:  # MNIST/FashionMNIST
            feature_size = 32 * 7 * 7
        else:  # CIFAR
            feature_size = 32 * 8 * 8
            
        self.classifier = nn.Sequential(
            nn.Linear(feature_size, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', type=str, default='fmnist',
                       choices=['mnist', 'fmnist'])
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.01)
    args = parser.parse_args()
    
    print(f"CPU-Optimized Training")
    print(f"Using {torch.get_num_threads()} CPU threads")
    print("-" * 50)
    
    # Create benchmark
    if args.benchmark == 'mnist':
        benchmark = SplitMNIST(n_experiences=5, return_task_id=False, seed=42)
    else:
        benchmark = SplitFMNIST(n_experiences=5, return_task_id=False, seed=42)
    
    print(f"Benchmark: {args.benchmark.upper()}")
    print(f"Experiences: {benchmark.n_experiences}")
    
    # Create lightweight model
    model = LightweightCNN(num_classes=10, input_channels=1)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Optimizer - SGD is faster on CPU than Adam
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    # Simple evaluation
    eval_plugin = EvaluationPlugin(
        accuracy_metrics(minibatch=False, epoch=True, experience=True),
        loggers=[InteractiveLogger()]
    )
    
    # Strategy
    strategy = Naive(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_mb_size=args.batch_size,
        train_epochs=args.epochs,
        eval_mb_size=args.batch_size * 2,
        device='cpu',
        evaluator=eval_plugin
    )
    
    # Fast training loop
    print(f"\nTraining with batch_size={args.batch_size}, epochs={args.epochs}")
    print("=" * 50)
    
    import time
    total_start = time.time()
    
    for i, exp in enumerate(benchmark.train_stream):
        print(f"\nExperience {i+1}/{benchmark.n_experiences}")
        exp_start = time.time()
        
        # Train
        strategy.train(exp)
        
        # Quick eval on current experience only
        strategy.eval(benchmark.test_stream[i:i+1])
        
        exp_time = time.time() - exp_start
        print(f"Experience time: {exp_time:.1f}s")
    
    total_time = time.time() - total_start
    print(f"\nTotal training time: {total_time:.1f}s")
    print(f"Average per experience: {total_time/benchmark.n_experiences:.1f}s")
    
    # Final evaluation
    print("\n" + "=" * 50)
    print("Final evaluation on all experiences:")
    strategy.eval(benchmark.test_stream)


if __name__ == "__main__":
    main()