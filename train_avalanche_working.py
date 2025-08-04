"""Working example with Avalanche - using what's actually available."""
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from avalanche.benchmarks import nc_benchmark
from avalanche.benchmarks.utils import AvalancheDataset
from avalanche.training.supervised import Naive, EWC, Replay
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics, forgetting_metrics
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.models import SimpleCNN
import argparse


def create_split_mnist(n_experiences=5):
    """Create Split MNIST benchmark manually."""
    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load MNIST
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    # Convert to AvalancheDataset
    train_dataset = AvalancheDataset(train_dataset)
    test_dataset = AvalancheDataset(test_dataset)
    
    # Create benchmark
    benchmark = nc_benchmark(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        n_experiences=n_experiences,
        task_labels=False,
        seed=42,
        class_ids_from_zero_in_each_exp=True
    )
    
    return benchmark


def create_split_fashionmnist(n_experiences=5):
    """Create Split Fashion-MNIST benchmark manually."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Load Fashion-MNIST
    train_dataset = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST('./data', train=False, download=True, transform=transform)
    
    # Convert to AvalancheDataset
    train_dataset = AvalancheDataset(train_dataset)
    test_dataset = AvalancheDataset(test_dataset)
    
    # Create benchmark
    benchmark = nc_benchmark(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        n_experiences=n_experiences,
        task_labels=False,
        seed=42,
        class_ids_from_zero_in_each_exp=True
    )
    
    return benchmark


def create_split_cifar10(n_experiences=5):
    """Create Split CIFAR-10 benchmark manually."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Load CIFAR-10
    train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10('./data', train=False, download=True, transform=transform)
    
    # Convert to AvalancheDataset
    train_dataset = AvalancheDataset(train_dataset)
    test_dataset = AvalancheDataset(test_dataset)
    
    # Create benchmark
    benchmark = nc_benchmark(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        n_experiences=n_experiences,
        task_labels=False,
        seed=42,
        class_ids_from_zero_in_each_exp=True
    )
    
    return benchmark


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mnist',
                       choices=['mnist', 'fashionmnist', 'cifar10'])
    parser.add_argument('--strategy', type=str, default='naive',
                       choices=['naive', 'ewc', 'replay'])
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--n_experiences', type=int, default=5)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    print(f"Running {args.strategy} on {args.dataset}")
    print(f"Device: {args.device}")
    
    # Create benchmark
    if args.dataset == 'mnist':
        benchmark = create_split_mnist(args.n_experiences)
        model = SimpleCNN(num_classes=10, input_channels=1)
    elif args.dataset == 'fashionmnist':
        benchmark = create_split_fashionmnist(args.n_experiences)
        model = SimpleCNN(num_classes=10, input_channels=1)
    elif args.dataset == 'cifar10':
        benchmark = create_split_cifar10(args.n_experiences)
        model = SimpleCNN(num_classes=10, input_channels=3)
    
    print(f"Created benchmark with {benchmark.n_experiences} experiences")
    
    # Move model to device
    model = model.to(args.device)
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Create evaluation plugin
    eval_plugin = EvaluationPlugin(
        accuracy_metrics(minibatch=False, epoch=True, experience=True, stream=True),
        loss_metrics(minibatch=False, epoch=True, experience=True, stream=True),
        forgetting_metrics(experience=True, stream=True),
        loggers=[InteractiveLogger()]
    )
    
    # Create strategy
    base_kwargs = {
        'model': model,
        'optimizer': optimizer,
        'criterion': criterion,
        'train_mb_size': 32,
        'train_epochs': args.epochs,
        'eval_mb_size': 128,
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
    print(f"\nTraining on {benchmark.n_experiences} experiences")
    print("="*50)
    
    results = []
    for i, experience in enumerate(benchmark.train_stream):
        print(f"\nExperience {i}/{benchmark.n_experiences-1}")
        print(f"Current classes: {experience.classes_in_this_experience}")
        
        # Train
        strategy.train(experience)
        
        # Evaluate on all seen experiences
        print(f"\nEvaluating on test set...")
        strategy.eval(benchmark.test_stream[:i+1])
    
    print("\n" + "="*50)
    print("Training completed!")


if __name__ == "__main__":
    main()