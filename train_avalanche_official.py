"""Working Avalanche example based on official documentation."""
import torch
import torch.nn as nn
from avalanche.benchmarks.classic import SplitMNIST
from avalanche.models import SimpleMLP
from avalanche.training.supervised import Naive, EWC, Replay
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics, forgetting_metrics
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin
import argparse

# If SplitMNIST import fails, try alternative imports
try:
    from avalanche.benchmarks.classic import SplitMNIST
    SPLIT_MNIST_AVAILABLE = True
except ImportError:
    SPLIT_MNIST_AVAILABLE = False
    print("SplitMNIST not found in avalanche.benchmarks.classic")
    
    # Try alternative import paths
    try:
        from avalanche.benchmarks import SplitMNIST
    except ImportError:
        # Create it manually
        from avalanche.benchmarks.generators import nc_benchmark
        from avalanche.benchmarks.datasets import MNIST
        from avalanche.benchmarks.utils import as_classification_dataset
        
        def create_split_mnist(n_experiences=5, seed=42, return_task_id=False):
            """Create Split MNIST manually if not available."""
            train_MNIST = as_classification_dataset(MNIST('./data', train=True, download=True))
            test_MNIST = as_classification_dataset(MNIST('./data', train=False, download=True))
            
            return nc_benchmark(
                train_dataset=train_MNIST,
                test_dataset=test_MNIST,
                n_experiences=n_experiences,
                task_labels=return_task_id,
                seed=seed,
                class_ids_from_zero_in_each_exp=True
            )
        
        # Replace SplitMNIST with our function
        SplitMNIST = create_split_mnist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy', type=str, default='naive',
                       choices=['naive', 'ewc', 'replay'])
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--experiences', type=int, default=5)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    print(f"Running {args.strategy} strategy")
    print(f"Device: {args.device}")
    
    # Create benchmark
    print("\nCreating Split MNIST benchmark...")
    if SPLIT_MNIST_AVAILABLE:
        benchmark = SplitMNIST(
            n_experiences=args.experiences,
            return_task_id=False,
            seed=42
        )
    else:
        benchmark = SplitMNIST(
            n_experiences=args.experiences,
            seed=42,
            return_task_id=False
        )
    
    print(f"Created benchmark with {benchmark.n_experiences} experiences")
    
    # Create model
    model = SimpleMLP(
        num_classes=10,
        input_size=784,
        hidden_size=400,
        hidden_layers=2,
        drop_rate=0.0
    )
    model = model.to(args.device)
    
    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Evaluation plugin
    eval_plugin = EvaluationPlugin(
        accuracy_metrics(
            minibatch=False,
            epoch=True,
            experience=True,
            stream=True
        ),
        loss_metrics(
            minibatch=False,
            epoch=True,
            experience=True,
            stream=True
        ),
        forgetting_metrics(
            experience=True,
            stream=True
        ),
        loggers=[InteractiveLogger()]
    )
    
    # Create strategy
    base_kwargs = {
        'model': model,
        'optimizer': optimizer,
        'criterion': criterion,
        'train_mb_size': 128,
        'train_epochs': args.epochs,
        'eval_mb_size': 256,
        'device': args.device,
        'evaluator': eval_plugin
    }
    
    if args.strategy == 'naive':
        strategy = Naive(**base_kwargs)
    elif args.strategy == 'ewc':
        strategy = EWC(**base_kwargs, ewc_lambda=0.4)
    elif args.strategy == 'replay':
        strategy = Replay(**base_kwargs, mem_size=200)
    
    # Training loop
    print(f"\nStarting training on {benchmark.n_experiences} experiences...")
    print("="*60)
    
    for i, train_exp in enumerate(benchmark.train_stream):
        print(f"\n>>> Training on experience {i+1}/{benchmark.n_experiences}")
        print(f"Classes in this experience: {train_exp.classes_in_this_experience}")
        
        # Train on current experience
        strategy.train(train_exp)
        
        # Evaluate on all test experiences
        print(f"\n>>> Evaluating on test experiences")
        strategy.eval(benchmark.test_stream)
    
    print("\n" + "="*60)
    print("Training completed!")


if __name__ == "__main__":
    main()