"""Working Avalanche example with available components."""
import torch
import torch.nn as nn
from avalanche.benchmarks.classic import SplitMNIST, SplitCIFAR10, SplitFMNIST
from avalanche.models import SimpleMLP, SimpleCNN
from avalanche.training.supervised import Naive, EWC, Replay
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics, forgetting_metrics
from avalanche.logging import InteractiveLogger, TensorboardLogger
from avalanche.training.plugins import EvaluationPlugin
import argparse


def main():
    parser = argparse.ArgumentParser(description='Avalanche CL Training')
    parser.add_argument('--benchmark', type=str, default='mnist',
                       choices=['mnist', 'fmnist', 'cifar10'],
                       help='Benchmark to use')
    parser.add_argument('--strategy', type=str, default='naive',
                       choices=['naive', 'ewc', 'replay'],
                       help='CL strategy')
    parser.add_argument('--model', type=str, default='mlp',
                       choices=['mlp', 'cnn'],
                       help='Model architecture')
    parser.add_argument('--epochs', type=int, default=2,
                       help='Epochs per experience')
    parser.add_argument('--experiences', type=int, default=5,
                       help='Number of experiences')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size')
    parser.add_argument('--device', type=str, 
                       default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    args = parser.parse_args()
    
    print("="*60)
    print(f"Benchmark: {args.benchmark}")
    print(f"Strategy: {args.strategy}")
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print("="*60)
    
    # Create benchmark
    if args.benchmark == 'mnist':
        benchmark = SplitMNIST(
            n_experiences=args.experiences,
            return_task_id=False,
            seed=42
        )
        input_size = 28 * 28
        num_classes = 10
        channels = 1
    elif args.benchmark == 'fmnist':
        benchmark = SplitFMNIST(
            n_experiences=args.experiences,
            return_task_id=False,
            seed=42
        )
        input_size = 28 * 28
        num_classes = 10
        channels = 1
    elif args.benchmark == 'cifar10':
        benchmark = SplitCIFAR10(
            n_experiences=args.experiences,
            return_task_id=False,
            seed=42
        )
        input_size = 32 * 32 * 3
        num_classes = 10
        channels = 3
    
    print(f"\nBenchmark created with {benchmark.n_experiences} experiences")
    
    # Create model
    if args.model == 'mlp':
        model = SimpleMLP(
            num_classes=num_classes,
            input_size=input_size,
            hidden_size=400,
            hidden_layers=2
        )
    else:  # cnn
        model = SimpleCNN(
            num_classes=num_classes,
            input_channels=channels
        )
    
    model = model.to(args.device)
    print(f"Model: {model.__class__.__name__}")
    
    # Create optimizer and criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    # Create loggers
    loggers = [InteractiveLogger()]
    # Optionally add TensorboardLogger
    # loggers.append(TensorboardLogger())
    
    # Create evaluation plugin
    eval_plugin = EvaluationPlugin(
        accuracy_metrics(
            minibatch=True,
            epoch=True,
            experience=True,
            stream=True
        ),
        loss_metrics(
            minibatch=True,
            epoch=True,
            experience=True,
            stream=True
        ),
        forgetting_metrics(
            experience=True,
            stream=True
        ),
        loggers=loggers
    )
    
    # Create strategy
    base_kwargs = {
        'model': model,
        'optimizer': optimizer,
        'criterion': criterion,
        'train_mb_size': args.batch_size,
        'train_epochs': args.epochs,
        'eval_mb_size': args.batch_size * 2,
        'device': args.device,
        'evaluator': eval_plugin
    }
    
    if args.strategy == 'naive':
        strategy = Naive(**base_kwargs)
    elif args.strategy == 'ewc':
        strategy = EWC(
            **base_kwargs,
            ewc_lambda=0.4,
            mode='online',
            decay_factor=0.1
        )
    elif args.strategy == 'replay':
        strategy = Replay(
            **base_kwargs,
            mem_size=500
        )
    
    print(f"\nStrategy: {strategy.__class__.__name__}")
    
    # Training loop
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)
    
    results = {}
    
    for i, train_exp in enumerate(benchmark.train_stream):
        print(f"\n>>> Experience {i+1}/{benchmark.n_experiences}")
        print(f"Classes: {train_exp.classes_in_this_experience}")
        
        # Train
        strategy.train(train_exp)
        
        # Evaluate on test stream
        print("\nEvaluating on test set...")
        eval_results = strategy.eval(benchmark.test_stream)
        
        # Store results
        results[f'exp_{i}'] = eval_results
    
    print("\n" + "="*60)
    print("Training completed!")
    print("="*60)
    
    # Print final accuracies
    print("\nFinal test accuracies:")
    for i in range(benchmark.n_experiences):
        key = f'Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp{i:03d}'
        if key in eval_results:
            print(f"  Experience {i}: {eval_results[key]:.4f}")


if __name__ == "__main__":
    main()