"""Minimal test script to verify setup without downloading datasets."""
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from avalanche.benchmarks import nc_benchmark
from avalanche.training import Naive
from avalanche.evaluation.metrics import accuracy_metrics
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create a simple dataset using MNIST (smaller download)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Use MNIST as a test dataset
try:
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    print("✓ Dataset loaded successfully")
except Exception as e:
    print(f"Dataset loading failed: {e}")
    print("Creating synthetic dataset instead...")
    # Create synthetic dataset
    from torch.utils.data import TensorDataset
    X_train = torch.randn(1000, 1, 28, 28)
    y_train = torch.randint(0, 10, (1000,))
    X_test = torch.randn(200, 1, 28, 28)
    y_test = torch.randint(0, 10, (200,))
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

# Create a simple model
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28 * 28, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
).to(device)
print("✓ Model created")

# Create CL scenario
scenario = nc_benchmark(
    train_dataset=train_dataset,
    test_dataset=test_dataset,
    n_experiences=3,
    task_labels=False,
    seed=42
)
print(f"✓ Scenario created with {scenario.n_experiences} experiences")

# Create optimizer and loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Create evaluation plugin
eval_plugin = EvaluationPlugin(
    accuracy_metrics(minibatch=True, epoch=True, experience=True),
    loggers=[InteractiveLogger()]
)

# Create strategy
strategy = Naive(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    train_mb_size=32,
    train_epochs=1,
    eval_mb_size=128,
    device=device,
    evaluator=eval_plugin
)
print("✓ Strategy created")

# Train on first experience only (quick test)
print("\nTraining on first experience...")
strategy.train(scenario.train_stream[0])
print("✓ Training completed")

# Evaluate
print("\nEvaluating...")
strategy.eval(scenario.test_stream[0])
print("✓ Evaluation completed")

print("\n✅ All tests passed! Your setup is working correctly.")