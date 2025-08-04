"""Test EWC with better parameters."""
import torch
from avalanche.benchmarks.classic import SplitFMNIST
from avalanche.models import SimpleMLP
from avalanche.training.supervised import EWC
from avalanche.evaluation.metrics import accuracy_metrics
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin
import matplotlib.pyplot as plt

# Test different EWC configurations
configs = [
    {"name": "EWC (λ=0.4)", "ewc_lambda": 0.4, "mode": "online"},
    {"name": "EWC (λ=100)", "ewc_lambda": 100, "mode": "online"},
    {"name": "EWC (λ=5000)", "ewc_lambda": 5000, "mode": "online"},
    {"name": "EWC Offline", "ewc_lambda": 100, "mode": "offline"}
]

results = {}

for config in configs:
    print(f"\nTesting {config['name']}...")
    
    # Create benchmark
    benchmark = SplitFMNIST(n_experiences=5, return_task_id=False, seed=42)
    
    # Create model
    model = SimpleMLP(num_classes=10, input_size=784, hidden_size=256)
    
    # Create strategy
    strategy = EWC(
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
        criterion=torch.nn.CrossEntropyLoss(),
        ewc_lambda=config['ewc_lambda'],
        mode=config['mode'],
        decay_factor=0.1 if config['mode'] == 'online' else None,
        train_mb_size=128,
        train_epochs=2,
        eval_mb_size=256,
        device='cpu',
        evaluator=EvaluationPlugin(
            accuracy_metrics(experience=True),
            loggers=[InteractiveLogger()]
        )
    )
    
    # Train
    for train_exp in benchmark.train_stream:
        strategy.train(train_exp)
    
    # Evaluate
    accuracies = []
    for i, test_exp in enumerate(benchmark.test_stream):
        result = strategy.eval([test_exp])
        key = f'Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp{i:03d}'
        if key in result:
            accuracies.append(result[key])
    
    results[config['name']] = accuracies
    print(f"Final accuracies: {[f'{acc:.3f}' for acc in accuracies]}")

# Visualize
plt.figure(figsize=(10, 6))
x = list(range(5))

for name, accs in results.items():
    plt.plot(x, accs, marker='o', linewidth=2, markersize=8, label=name)

plt.xlabel('Experience')
plt.ylabel('Test Accuracy')
plt.title('EWC Parameter Comparison\n(Higher values = less forgetting)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(0, 1.05)
plt.savefig('ewc_comparison.png', dpi=150)

print("\n" + "="*60)
print("EXPLANATION:")
print("="*60)
print("EWC Lambda (λ) controls the strength of regularization:")
print("- λ = 0.4 (too small): Acts like naive, forgets everything")
print("- λ = 100-5000 (better): Balances learning and remembering")
print("- λ too large: Can't learn new tasks well")
print("\nYour EWC used λ=0.4, which is why it forgot everything!")
print("\nRecommendation: Use Replay for CPU, or increase EWC lambda to 1000+")