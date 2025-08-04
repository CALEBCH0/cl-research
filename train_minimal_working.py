"""Minimal working example with Avalanche."""
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from avalanche.benchmarks import nc_benchmark
from avalanche.benchmarks.utils import AvalancheDataset
from avalanche.training.supervised import Naive
from avalanche.evaluation.metrics import accuracy_metrics
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin


# Create synthetic data
print("Creating synthetic data...")
n_samples = 1000
n_features = 784  # Like MNIST
n_classes = 10

X = torch.randn(n_samples, n_features)
y = torch.randint(0, n_classes, (n_samples,))

# Create train/test split
train_size = int(0.8 * n_samples)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Create datasets
train_dataset = AvalancheDataset(TensorDataset(X_train, y_train))
test_dataset = AvalancheDataset(TensorDataset(X_test, y_test))

# Create benchmark
print("Creating benchmark...")
benchmark = nc_benchmark(
    train_dataset=train_dataset,
    test_dataset=test_dataset,
    n_experiences=5,
    task_labels=False,
    seed=42
)

# Create simple model
model = nn.Sequential(
    nn.Linear(n_features, 128),
    nn.ReLU(),
    nn.Linear(128, n_classes)
)

# Setup training
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# Create evaluation plugin
eval_plugin = EvaluationPlugin(
    accuracy_metrics(epoch=True, experience=True),
    loggers=[InteractiveLogger()]
)

# Create strategy
strategy = Naive(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    train_mb_size=32,
    train_epochs=2,
    eval_mb_size=128,
    device=device,
    evaluator=eval_plugin
)

# Train
print("\nStarting training...")
for i, exp in enumerate(benchmark.train_stream):
    print(f"\nExperience {i}")
    strategy.train(exp)
    strategy.eval(benchmark.test_stream[:i+1])

print("\nDone!")