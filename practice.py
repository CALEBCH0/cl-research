from avalanche.models import SimpleCNN, SimpleMLP, SimpleMLP_TinyImageNet, MobilenetV1, IncrementalClassifier, MultiHeadClassifier, as_multitask
from avalanche.benchmarks.classic import PermutedMNIST
from avalanche.models.utils import avalanche_model_adaptation
from avalanche.benchmarks import SplitMNIST
from avalanche.training import Naive

# Example 1
print("Example 1: SimpleCNN\n=====================")
benchmark = SplitMNIST(5, shuffle=False, class_ids_from_zero_in_each_exp=False)
model = SimpleCNN()
print(model)

# Example 2
print("Example 2: Dynamic model expansion\n=====================")
model = IncrementalClassifier(in_features=784)
print(model)
for exp in benchmark.train_stream:
    model.adaptation(exp)
    print(model)
    
# Example 3
print("Example 3: Multitask model\n=====================")
benchmark = SplitMNIST(5, shuffle=False, class_ids_from_zero_in_each_exp=True, return_task_id=True)
model = MultiHeadClassifier(in_features=784)
print(model)
for exp in benchmark.train_stream:
    model.adaptation(exp)
    print(model)
    
    
import random
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
import numpy as np
seeds = [42, 123, 456, 789, 101112]
results = []

for seed in seeds:
    # set seed
    torch.manual_seed(seed)
    random.seed(seed)
    
    # config
    device = torch.device("cude:0" if torch.cuda.is_available() else "cpu")
    
    # create model
    model = SimpleMLP(num_classes=10)
    
    # create benchmark
    benchmark = PermutedMNIST(n_experiences=3, seed=seed)
    
    # create optimizer and criterion
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = CrossEntropyLoss()
    
    # create strategy
    strategy = Naive(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_mb_size=32,
        train_epochs=2,
        eval_mb_size=32,
        device=device
    )
    
    # train and get results
    for exp in benchmark.train_stream:
        strategy.train(exp)
        results.append(strategy.eval(benchmark.test_stream))

print("Results for each seed:")
for seed, result in zip(seeds, results):
    print(f"Seed {seed}: {result}")