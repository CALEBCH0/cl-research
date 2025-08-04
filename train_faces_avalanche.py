"""Face recognition using Avalanche's built-in datasets."""
import torch
import torch.nn as nn
from torchvision import transforms
from avalanche.benchmarks.classic import SplitCelebA, SplitFMNIST
from avalanche.benchmarks import nc_benchmark
from avalanche.training import Naive, EWC, Replay
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin
import argparse


def get_face_benchmark(name='fmnist', n_experiences=5):
    """Get face-related benchmark from Avalanche."""
    if name == 'fmnist':
        # Fashion-MNIST as a simple test (not faces but similar structure)
        return SplitFMNIST(n_experiences=n_experiences, seed=42)
    elif name == 'celeba':
        # CelebA - actual face dataset but large download
        # Note: This requires downloading ~1.4GB
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        return SplitCelebA(n_experiences=n_experiences, seed=42)
    else:
        # Use torchvision's LFWPeople dataset with Avalanche
        from torchvision.datasets import LFWPeople
        from avalanche.benchmarks.utils import AvalancheDataset
        
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # Download LFW dataset
        try:
            train_dataset = LFWPeople(
                root='./data',
                split='train',
                download=True,
                transform=transform
            )
            test_dataset = LFWPeople(
                root='./data',
                split='test',
                download=True,
                transform=transform
            )
            
            # Wrap in Avalanche dataset
            train_dataset = AvalancheDataset(train_dataset)
            test_dataset = AvalancheDataset(test_dataset)
            
            # Create class-incremental benchmark
            return nc_benchmark(
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                n_experiences=n_experiences,
                task_labels=False,
                seed=42
            )
        except Exception as e:
            print(f"Failed to load LFW: {e}")
            print("Falling back to Fashion-MNIST")
            return SplitFMNIST(n_experiences=n_experiences, seed=42)


class SimpleFaceModel(nn.Module):
    """Simple CNN for face recognition."""
    def __init__(self, num_classes=10, input_channels=1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='fmnist',
                       choices=['fmnist', 'celeba', 'lfw'])
    parser.add_argument('--strategy', type=str, default='naive',
                       choices=['naive', 'ewc', 'replay'])
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--n_experiences', type=int, default=5)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    print(f"Running {args.strategy} on {args.dataset}")
    print(f"Device: {args.device}")
    
    # Create benchmark
    benchmark = get_face_benchmark(args.dataset, args.n_experiences)
    
    # Determine input channels and classes
    sample_input, _ = benchmark.train_stream[0].dataset[0]
    input_channels = sample_input.shape[0]
    num_classes = benchmark.n_classes
    
    print(f"Input shape: {sample_input.shape}")
    print(f"Number of classes: {num_classes}")
    
    # Create model
    model = SimpleFaceModel(num_classes=num_classes, input_channels=input_channels)
    model = model.to(args.device)
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Create evaluation plugin
    eval_plugin = EvaluationPlugin(
        accuracy_metrics(minibatch=True, epoch=True, experience=True),
        loss_metrics(minibatch=True, epoch=True),
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
        strategy = Replay(**base_kwargs, mem_size=500)
    
    # Training loop
    print(f"\nTraining on {benchmark.n_experiences} experiences")
    for i, experience in enumerate(benchmark.train_stream):
        print(f"\n--- Experience {i+1}/{benchmark.n_experiences} ---")
        print(f"Classes: {experience.classes_in_this_experience}")
        strategy.train(experience)
        print("\nEvaluating on test set...")
        strategy.eval(benchmark.test_stream)
    
    print("\nTraining completed!")


if __name__ == "__main__":
    main()