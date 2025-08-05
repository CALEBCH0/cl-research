"""Refactored training script for easy importing and testing."""
import warnings
warnings.filterwarnings('ignore', message='.*longdouble.*')
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message='.*OpenSSL.*')

from collections import namedtuple
import torch
import torch.nn as nn
from avalanche.benchmarks.classic import SplitMNIST, SplitCIFAR10, SplitFMNIST
from avalanche.benchmarks import nc_benchmark
from avalanche.benchmarks.utils import AvalancheDataset
from torchvision import datasets, transforms
from avalanche.models import SimpleMLP, SimpleCNN
from avalanche.training.supervised import (
    Naive, EWC, Replay, GEM, AGEM, LwF, 
    SynapticIntelligence as SI, MAS, GDumb,
    Cumulative, JointTraining, ICaRL
)
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics, forgetting_metrics
from avalanche.logging import InteractiveLogger, TensorboardLogger, TextLogger
from avalanche.training.plugins import EvaluationPlugin

BenchmarkInfo = namedtuple("BenchmarkInfo", ["input_size", "num_classes", "channels"])


def set_benchmark(benchmark_name, experiences=5, seed=42):
    """Set the appropriate benchmark."""
    
    if benchmark_name == 'mnist':
        benchmark = SplitMNIST(
            n_experiences=experiences,
            return_task_id=False,
            seed=seed
        )
        input_size = 28 * 28
        num_classes = 10
        channels = 1
    elif benchmark_name == 'fmnist':
        benchmark = SplitFMNIST(
            n_experiences=experiences,
            return_task_id=False,
            seed=seed
        )
        input_size = 28 * 28
        num_classes = 10
        channels = 1
    elif benchmark_name == 'cifar10':
        benchmark = SplitCIFAR10(
            n_experiences=experiences,
            return_task_id=False,
            seed=seed
        )
        input_size = 32 * 32 * 3
        num_classes = 10
        channels = 3
    else:
        raise ValueError(f"Unknown benchmark: {benchmark_name}")
    
    return benchmark, BenchmarkInfo(input_size, num_classes, channels)


def create_model(model_type, benchmark_info):
    """Create model based on type and benchmark."""
    if model_type == 'mlp':
        model = SimpleMLP(
            num_classes=benchmark_info.num_classes,
            input_size=benchmark_info.input_size,
            hidden_size=400,
            hidden_layers=2
        )
    else:  # cnn
        model = SimpleCNN(
            num_classes=benchmark_info.num_classes,
            input_channels=benchmark_info.channels
        )
    return model


def create_strategy(strategy_name, model, optimizer, criterion, device, 
                   eval_plugin, mem_size=200, **kwargs):
    """Create strategy with given parameters."""
    base_kwargs = {
        'model': model,
        'optimizer': optimizer,
        'criterion': criterion,
        'train_mb_size': kwargs.get('batch_size', 32),
        'train_epochs': kwargs.get('epochs', 1),
        'eval_mb_size': kwargs.get('batch_size', 32) * 2,
        'device': device,
        'evaluator': eval_plugin
    }
    
    if strategy_name == 'naive':
        return Naive(**base_kwargs)
    elif strategy_name == 'ewc':
        return EWC(**base_kwargs, ewc_lambda=0.4, mode='online', decay_factor=0.1)
    elif strategy_name == 'replay':
        return Replay(**base_kwargs, mem_size=mem_size)
    elif strategy_name == 'gem':
        return GEM(**base_kwargs, patterns_per_exp=256, memory_strength=0.5)
    elif strategy_name == 'agem':
        return AGEM(**base_kwargs, patterns_per_exp=256, sample_size=256)
    elif strategy_name == 'lwf':
        return LwF(**base_kwargs, alpha=0.5, temperature=2)
    elif strategy_name == 'si':
        return SI(**base_kwargs, si_lambda=0.0001)
    elif strategy_name == 'mas':
        return MAS(**base_kwargs, lambda_reg=1, alpha=0.5)
    elif strategy_name == 'gdumb':
        return GDumb(**base_kwargs, mem_size=mem_size)
    elif strategy_name == 'cumulative':
        return Cumulative(**base_kwargs)
    elif strategy_name == 'joint':
        return JointTraining(**base_kwargs)
    elif strategy_name == 'icarl':
        return ICaRL(**base_kwargs, mem_size_per_class=20, buffer_transform=None, fixed_memory=True)
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")


def run_training(benchmark_name='fmnist', strategy_name='naive', model_type='mlp',
                device='cuda', experiences=5, epochs=1, batch_size=32, 
                mem_size=200, lr=0.001, seed=42, verbose=True):
    """
    Run a complete training experiment and return results.
    
    Returns:
        dict: Results containing final accuracies and average accuracy
    """
    # Set seed
    torch.manual_seed(seed)
    
    # Create benchmark
    benchmark, benchmark_info = set_benchmark(benchmark_name, experiences, seed)
    
    # Create model
    model = create_model(model_type, benchmark_info)
    model = model.to(device)
    
    # Create optimizer and criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # Create evaluation plugin (minimal logging if not verbose)
    if verbose:
        loggers = [InteractiveLogger()]
    else:
        # Use a minimal logger to suppress warning
        from avalanche.logging import BaseLogger
        class SilentLogger(BaseLogger):
            def log_single_metric(self, name, value, x_plot):
                pass
        loggers = [SilentLogger()]
    
    eval_plugin = EvaluationPlugin(
        accuracy_metrics(minibatch=False, epoch=True, experience=True, stream=True),
        loss_metrics(minibatch=False, epoch=True, experience=True, stream=True),
        forgetting_metrics(experience=True, stream=True),
        loggers=loggers
    )
    
    # Create strategy
    strategy = create_strategy(
        strategy_name, model, optimizer, criterion, device, eval_plugin,
        mem_size=mem_size, epochs=epochs, batch_size=batch_size
    )
    
    # Training loop
    for i, train_exp in enumerate(benchmark.train_stream):
        if verbose:
            print(f"\n>>> Training Experience {i+1}/{benchmark.n_experiences}")
        strategy.train(train_exp)
    
    # Final evaluation
    if verbose:
        print("\nFinal evaluation...")
    eval_results = strategy.eval(benchmark.test_stream)
    
    # Extract accuracies
    accuracies = []
    for i in range(benchmark.n_experiences):
        key = f'Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp{i:03d}'
        if key in eval_results:
            accuracies.append(eval_results[key])
    
    avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0.0
    
    return {
        'accuracies': accuracies,
        'average_accuracy': avg_accuracy,
        'strategy': strategy_name,
        'model': model_type,
        'benchmark': benchmark_name
    }


def main():
    """Main function for command line usage."""
    from utils.argparser import get_args
    args = get_args()
    
    results = run_training(
        benchmark_name=args.benchmark,
        strategy_name=args.strategy,
        model_type=args.model,
        device=args.device,
        experiences=args.experiences,
        epochs=args.epochs,
        batch_size=args.batch_size,
        mem_size=args.mem_size,
        lr=args.lr,
        verbose=True
    )
    
    print("\n" + "="*60)
    print("Final Results:")
    print("="*60)
    for i, acc in enumerate(results['accuracies']):
        print(f"Experience {i}: {acc:.4f}")
    print(f"\nAverage Accuracy: {results['average_accuracy']:.4f}")


if __name__ == "__main__":
    main()