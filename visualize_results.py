"""Visualize your continual learning results."""
import matplotlib.pyplot as plt
import numpy as np

# Your results
naive_results = [0.0, 0.0, 0.0, 0.0, 0.9995]
ewc_results = [0.0, 0.0, 0.0, 0.0, 0.9995]
replay_results = [0.514, 0.9155, 0.7265, 0.7895, 0.9785]

experiences = [0, 1, 2, 3, 4]

# Create figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Strategy Comparison
ax1.plot(experiences, naive_results, 'r-o', linewidth=2, markersize=8, label='Naive')
ax1.plot(experiences, ewc_results, 'b--o', linewidth=2, markersize=8, label='EWC (λ=0.4)')
ax1.plot(experiences, replay_results, 'g-o', linewidth=2, markersize=8, label='Replay')

ax1.set_xlabel('Experience (Task)', fontsize=12)
ax1.set_ylabel('Test Accuracy', fontsize=12)
ax1.set_title('Continual Learning Strategy Comparison\nYour Results on Fashion-MNIST', fontsize=14)
ax1.legend(fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(-0.05, 1.05)
ax1.set_xticks(experiences)

# Add annotations
ax1.annotate('Complete\nForgetting', xy=(1, 0.0), xytext=(1, 0.2),
            arrowprops=dict(arrowstyle='->', color='red'),
            fontsize=10, ha='center', color='red')
ax1.annotate('Good\nRetention!', xy=(1, 0.9155), xytext=(1, 0.7),
            arrowprops=dict(arrowstyle='->', color='green'),
            fontsize=10, ha='center', color='green')

# Plot 2: Average Accuracy Comparison
strategies = ['Naive', 'EWC (λ=0.4)', 'Replay']
avg_accuracies = [
    np.mean(naive_results),
    np.mean(ewc_results),
    np.mean(replay_results)
]

bars = ax2.bar(strategies, avg_accuracies, color=['red', 'blue', 'green'], alpha=0.7)
ax2.set_ylabel('Average Accuracy Across All Tasks', fontsize=12)
ax2.set_title('Overall Performance Comparison', fontsize=14)
ax2.set_ylim(0, 1.0)
ax2.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, acc in zip(bars, avg_accuracies):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{acc:.3f}', ha='center', va='bottom', fontsize=12)

# Add explanation text
fig.text(0.5, 0.02, 
         'Replay maintains 74.4% average accuracy across all tasks, while Naive and EWC (λ=0.4) only remember the last task',
         ha='center', fontsize=11, style='italic')

plt.tight_layout()
plt.savefig('your_results_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# Print detailed analysis
print("="*70)
print("DETAILED ANALYSIS OF YOUR RESULTS")
print("="*70)

print("\n1. CATASTROPHIC FORGETTING (Naive & EWC)")
print("-"*40)
print("Both strategies show 0% on tasks 0-3:")
print("- They completely forgot T-shirts, Trousers, Pullovers, etc.")
print("- Only remember the final task (Bags & Ankle boots) at 99.9%")
print("- This is classic catastrophic forgetting")

print("\n2. SUCCESSFUL CONTINUAL LEARNING (Replay)")
print("-"*40)
print("Replay maintains accuracy on old tasks:")
print("- Task 0 (T-shirt/Trouser): 51.4% - Some forgetting but not catastrophic")
print("- Task 1 (Pullover/Dress): 91.6% - Excellent retention!")
print("- Task 2 (Coat/Sandal): 72.7% - Good retention")
print("- Task 3 (Shirt/Sneaker): 78.9% - Good retention")
print("- Task 4 (Bag/Ankle boot): 97.9% - Current task")

print("\n3. WHY EWC FAILED")
print("-"*40)
print("EWC with λ=0.4 is too weak:")
print("- The regularization penalty is insignificant")
print("- Network ignores the constraint and overwrites old knowledge")
print("- Typical good values: λ=100 to λ=10000")

print("\n4. RECOMMENDATIONS")
print("-"*40)
print("For your setup (CPU):")
print("1. Replay is the best choice - simple and effective")
print("2. Try larger replay buffer: --mem_size 2000")
print("3. If using EWC, increase lambda significantly")
print("4. Consider other strategies like LwF or SI")

print("\nAverage accuracy across all tasks:")
print(f"- Naive: {np.mean(naive_results):.3f} (20%)")
print(f"- EWC: {np.mean(ewc_results):.3f} (20%)")
print(f"- Replay: {np.mean(replay_results):.3f} (74%)")
print("\nReplay is 3.7x better than the baselines!")