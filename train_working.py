"""Working Avalanche example with available components."""
import warnings
# Suppress numpy longdouble warning (common on WSL/certain systems)
warnings.filterwarnings('ignore', message='.*longdouble.*')
# Suppress urllib3 OpenSSL warning (common on macOS)
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
from avalanche.logging import InteractiveLogger, TensorboardLogger
from avalanche.training.plugins import EvaluationPlugin

from utils.argparser import get_args

BenchmarkInfo = namedtuple("BenchmarkInfo", ["input_size", "num_classes", "channels"])

def set_benchmark(args):
    """Set the appropriate benchmark based on the command line argument."""
    
    
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
    elif args.benchmark == 'lfw':
        # LFW People dataset
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        try:
            train_dataset = datasets.LFWPeople(
                root='./data', split='train', download=True, transform=transform
            )
            test_dataset = datasets.LFWPeople(
                root='./data', split='test', download=True, transform=transform
            )
            
            # Wrap in Avalanche dataset
            train_dataset = AvalancheDataset(train_dataset)
            test_dataset = AvalancheDataset(test_dataset)
            
            # Create benchmark
            benchmark = nc_benchmark(
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                n_experiences=args.experiences,
                task_labels=False,
                seed=42,
                class_ids_from_zero_in_each_exp=True
            )
            
            input_size = 64 * 64 * 3
            num_classes = len(train_dataset.targets_task_labels.uniques)
            channels = 3
            
        except Exception as e:
            print(f"Error loading LFW: {e}")
            print("Falling back to Fashion-MNIST")
            benchmark = SplitFMNIST(n_experiences=args.experiences, return_task_id=False, seed=42)
            input_size = 28 * 28
            num_classes = 10
            channels = 1
            
    elif args.benchmark == 'celeba':
        # CelebA dataset
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        try:
            # CelebA for attribute prediction (40 binary attributes)
            train_dataset = datasets.CelebA(
                root='./data', split='train', download=True, 
                transform=transform, target_type='attr'
            )
            test_dataset = datasets.CelebA(
                root='./data', split='test', download=True,
                transform=transform, target_type='attr'
            )
            
            # For continual learning, we'll use the first 10 attributes as "classes"
            # This is a simplification - real face recognition would use identity
            
            # Wrap and create multi-class from attributes
            train_dataset = AvalancheDataset(train_dataset)
            test_dataset = AvalancheDataset(test_dataset)
            
            benchmark = nc_benchmark(
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                n_experiences=args.experiences,
                task_labels=False,
                seed=42
            )
            
            input_size = 64 * 64 * 3
            num_classes = 40  # 40 attributes
            channels = 3
            
        except Exception as e:
            print(f"Error loading CelebA: {e}")
            print("Note: CelebA is ~1.4GB download")
            print("Falling back to Fashion-MNIST")
            benchmark = SplitFMNIST(n_experiences=args.experiences, return_task_id=False, seed=42)
            input_size = 28 * 28
            num_classes = 10
            channels = 1
            
    else:
        raise ValueError(f"Unknown benchmark: {args.benchmark}")
    
    return benchmark, BenchmarkInfo(input_size, num_classes, channels)

def set_strategy(args, base_kwargs):
    """Set the appropriate strategy based on the command line argument."""
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
            mem_size=args.mem_size
        )
    elif args.strategy == 'gem':
        strategy = GEM(
            **base_kwargs,
            patterns_per_exp=256,
            memory_strength=0.5
        )
    elif args.strategy == 'agem':
        strategy = AGEM(
            **base_kwargs,
            patterns_per_exp=256,
            sample_size=256
        )
    elif args.strategy == 'lwf':
        strategy = LwF(
            **base_kwargs,
            alpha=0.5,
            temperature=2
        )
    elif args.strategy == 'si':
        strategy = SI(
            **base_kwargs,
            si_lambda=0.0001
        )
    elif args.strategy == 'mas':
        strategy = MAS(
            **base_kwargs,
            lambda_reg=1,
            alpha=0.5
        )
    elif args.strategy == 'gdumb':
        strategy = GDumb(
            **base_kwargs,
            mem_size=args.mem_size
        )
    elif args.strategy == 'cumulative':
        strategy = Cumulative(
            **base_kwargs
        )
    elif args.strategy == 'joint':
        strategy = JointTraining(
            **base_kwargs
        )
    elif args.strategy == 'icarl':
        strategy = ICaRL(
            **base_kwargs,
            mem_size_per_class=20,
            buffer_transform=None,
            fixed_memory=True
        )
    else:
        raise ValueError(f"Unknown strategy: {args.strategy}")
    
    return strategy

def main(args):
    print("="*60)
    print(f"Benchmark: {args.benchmark}")
    print(f"Strategy: {args.strategy}")
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    if args.strategy == 'replay':
        print(f"Replay buffer size: {args.mem_size}")
    print("="*60)
    
    # Create benchmark
    benchmark, benchmark_info = set_benchmark(args)
    
    print(f"\nBenchmark created with {benchmark.n_experiences} experiences")
    
    # Create model
    if args.model == 'mlp':
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

    strategy = set_strategy(args, base_kwargs)

    print(f"\nStrategy: {strategy.__class__.__name__}")
    
    # Training loop
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)
    
    results = {}
    all_metrics = []  # Store all metrics for analysis
    
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
    args = get_args()
    main(args)