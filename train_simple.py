"""Simplified training script using Avalanche's built-in benchmarks."""
import torch
import torch.nn as nn
from avalanche.benchmarks.classic import SplitCIFAR10, SplitCIFAR100, SplitMNIST
from avalanche.training import Naive, EWC, Replay
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics, forgetting_metrics
from avalanche.logging import InteractiveLogger, TensorboardLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.models import SimpleMLP, SimpleCNN
import argparse
from models.backbones import get_backbone
from models.heads import ClassificationHead


def get_benchmark(name, n_experiences=5):
    """Get Avalanche benchmark."""
    if name == 'mnist':
        return SplitMNIST(n_experiences=n_experiences, seed=42)
    elif name == 'cifar10':
        return SplitCIFAR10(n_experiences=n_experiences, seed=42)
    elif name == 'cifar100':
        return SplitCIFAR100(n_experiences=n_experiences, seed=42)
    else:
        raise ValueError(f"Unknown benchmark: {name}")


def get_model(benchmark_name, model_type='simple'):
    """Get model based on benchmark."""
    if benchmark_name == 'mnist':
        if model_type == 'simple':
            return SimpleMLP(num_classes=10, input_size=28*28, hidden_size=512)
        else:
            return SimpleCNN(num_classes=10, input_channels=1)
    elif benchmark_name in ['cifar10', 'cifar100']:
        num_classes = 10 if benchmark_name == 'cifar10' else 100
        if model_type == 'simple':
            return SimpleCNN(num_classes=num_classes, input_channels=3)
        else:
            # Use your custom backbone
            backbone_config = {
                'name': 'resnet18',
                'type': 'torchvision',
                'pretrained': True,
                'embedding_dim': 512
            }
            backbone = get_backbone(backbone_config)
            
            class Model(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.backbone = backbone
                    self.classifier = ClassificationHead(512, num_classes)
                
                def forward(self, x):
                    features = self.backbone(x)
                    embeddings = features['embeddings']
                    return self.classifier(embeddings)
            
            return Model()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', type=str, default='cifar10',
                       choices=['mnist', 'cifar10', 'cifar100'])
    parser.add_argument('--strategy', type=str, default='naive',
                       choices=['naive', 'ewc', 'replay'])
    parser.add_argument('--model', type=str, default='simple',
                       choices=['simple', 'resnet'])
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--n_experiences', type=int, default=5)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    print(f"Running {args.strategy} on {args.benchmark} with {args.model} model")
    print(f"Device: {args.device}")
    
    # Create benchmark
    benchmark = get_benchmark(args.benchmark, args.n_experiences)
    
    # Create model
    model = get_model(args.benchmark, args.model)
    model = model.to(args.device)
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Create loggers and metrics
    loggers = [InteractiveLogger()]
    
    eval_plugin = EvaluationPlugin(
        accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        loss_metrics(minibatch=True, epoch=True, experience=True),
        forgetting_metrics(experience=True),
        loggers=loggers
    )
    
    # Create strategy
    if args.strategy == 'naive':
        strategy = Naive(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_mb_size=32,
            train_epochs=args.epochs,
            eval_mb_size=128,
            device=args.device,
            evaluator=eval_plugin
        )
    elif args.strategy == 'ewc':
        strategy = EWC(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            ewc_lambda=0.4,
            train_mb_size=32,
            train_epochs=args.epochs,
            eval_mb_size=128,
            device=args.device,
            evaluator=eval_plugin
        )
    elif args.strategy == 'replay':
        strategy = Replay(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            mem_size=2000,
            train_mb_size=32,
            train_epochs=args.epochs,
            eval_mb_size=128,
            device=args.device,
            evaluator=eval_plugin
        )
    
    # Training loop
    print(f"\nTraining on {benchmark.n_experiences} experiences")
    for i, experience in enumerate(benchmark.train_stream):
        print(f"\n--- Experience {i+1}/{benchmark.n_experiences} ---")
        strategy.train(experience)
        print("\nEvaluating on test set...")
        strategy.eval(benchmark.test_stream)
    
    print("\nTraining completed!")


if __name__ == "__main__":
    main()