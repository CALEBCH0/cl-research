"""Compare different replay buffer sizes."""
import subprocess
import matplotlib.pyplot as plt
import numpy as np
import time
import os

# Buffer sizes to test
buffer_sizes = [50, 100, 200, 500, 1000, 2000]

print("="*70)
print("COMPARING REPLAY BUFFER SIZES")
print("="*70)
print(f"Testing buffer sizes: {buffer_sizes}")
print("This will take a while... grab some coffee! ☕")

results = {}
times = {}

for size in buffer_sizes:
    print(f"\n{'='*50}")
    print(f"Testing buffer size: {size}")
    print(f"{'='*50}")
    
    start_time = time.time()
    
    # Run experiment
    cmd = [
        "python", "train_working.py",
        "--benchmark", "fmnist",
        "--strategy", "replay",
        "--mem_size", str(size),
        "--epochs", "2",
        "--device", "cpu"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        if result.returncode != 0:
            print(f"Error with buffer size {size}: {result.stderr}")
            continue
            
        # Parse results
        output_lines = result.stdout.split('\n')
        accuracies = []
        
        # Look for "Experience X: Y.ZZZZ" pattern
        for line in output_lines:
            if "Experience" in line and ":" in line and "Final test accuracies" not in line:
                try:
                    # Extract accuracy value after the colon
                    parts = line.strip().split(':')
                    if len(parts) >= 2:
                        acc_str = parts[-1].strip()
                        accuracy = float(acc_str)
                        accuracies.append(accuracy)
                except:
                    continue
        
        if len(accuracies) >= 5:  # Should have 5 experiences
            results[size] = accuracies[-5:]  # Take last 5 values
            elapsed = time.time() - start_time
            times[size] = elapsed
            
            avg_acc = np.mean(results[size])
            print(f"✓ Buffer {size}: Avg accuracy = {avg_acc:.3f}")
            print(f"  Individual: {[f'{acc:.3f}' for acc in results[size]]}")
            print(f"  Time: {elapsed:.1f}s")
        else:
            print(f"✗ Failed to parse results for buffer size {size}")
            
    except subprocess.TimeoutExpired:
        print(f"✗ Timeout for buffer size {size}")
    except Exception as e:
        print(f"✗ Error with buffer size {size}: {e}")

if not results:
    print("No successful results! Check if train_working.py is working.")
    exit(1)

print(f"\n{'='*70}")
print("RESULTS SUMMARY")
print(f"{'='*70}")

# Calculate metrics
avg_accuracies = []
forgetting_scores = []
final_accuracies = []

for size in buffer_sizes:
    if size in results:
        accs = results[size]
        avg_acc = np.mean(accs)
        # Forgetting = how much accuracy dropped on first task
        forgetting = max(0, 1.0 - accs[0])  # Assume perfect would be 1.0
        final_acc = accs[-1]  # Last task accuracy
        
        avg_accuracies.append(avg_acc)
        forgetting_scores.append(forgetting)
        final_accuracies.append(final_acc)
        
        print(f"Buffer {size:4d}: Avg={avg_acc:.3f}, Forgetting={forgetting:.3f}, Final={final_acc:.3f}, Time={times[size]:.1f}s")
    else:
        avg_accuracies.append(0)
        forgetting_scores.append(1)
        final_accuracies.append(0)

# Create visualizations
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Average accuracy vs buffer size
valid_sizes = [s for s in buffer_sizes if s in results]
valid_avgs = [np.mean(results[s]) for s in valid_sizes]

ax1.plot(valid_sizes, valid_avgs, 'bo-', linewidth=2, markersize=8)
ax1.set_xlabel('Buffer Size')
ax1.set_ylabel('Average Accuracy')
ax1.set_title('Average Accuracy vs Buffer Size')
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 1)

# Plot 2: Individual experience accuracies
colors = ['red', 'orange', 'green', 'blue', 'purple']
for exp_idx in range(5):
    exp_accs = [results[s][exp_idx] for s in valid_sizes]
    ax2.plot(valid_sizes, exp_accs, 'o-', color=colors[exp_idx], 
             label=f'Experience {exp_idx}', linewidth=2, markersize=6)

ax2.set_xlabel('Buffer Size')
ax2.set_ylabel('Test Accuracy')
ax2.set_title('Accuracy per Experience vs Buffer Size')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, 1)

# Plot 3: Training time vs buffer size
valid_times = [times[s] for s in valid_sizes]
ax3.bar(valid_sizes, valid_times, alpha=0.7, color='skyblue')
ax3.set_xlabel('Buffer Size')
ax3.set_ylabel('Training Time (seconds)')
ax3.set_title('Training Time vs Buffer Size')

# Plot 4: Memory efficiency (accuracy per unit buffer)
efficiency = [np.mean(results[s]) / s * 1000 for s in valid_sizes]  # Accuracy per 1000 samples
ax4.plot(valid_sizes, efficiency, 'go-', linewidth=2, markersize=8)
ax4.set_xlabel('Buffer Size')
ax4.set_ylabel('Accuracy per 1000 Buffer Samples')
ax4.set_title('Memory Efficiency')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('buffer_size_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# Find optimal buffer size
if valid_avgs:
    best_idx = np.argmax(valid_avgs)
    best_size = valid_sizes[best_idx]
    best_acc = valid_avgs[best_idx]
    
    # Find most efficient (good accuracy with small buffer)
    efficiency_scores = [valid_avgs[i] / (valid_sizes[i] / 100) for i in range(len(valid_sizes))]
    efficient_idx = np.argmax(efficiency_scores)
    efficient_size = valid_sizes[efficient_idx]
    efficient_acc = valid_avgs[efficient_idx]
    
    print(f"\n{'='*70}")
    print("RECOMMENDATIONS")
    print(f"{'='*70}")
    print(f"🏆 Best overall accuracy: Buffer size {best_size} (avg accuracy: {best_acc:.3f})")
    print(f"⚡ Most efficient: Buffer size {efficient_size} (accuracy: {efficient_acc:.3f})")
    
    if best_size <= 500:
        print(f"💡 Small buffers work well! No need for huge memory.")
    else:
        print(f"📈 Larger buffers help significantly.")
    
    print(f"\nTo use the best setting:")
    print(f"python train_working.py --benchmark fmnist --strategy replay --mem_size {best_size}")

print(f"\n✓ Results saved to 'buffer_size_comparison.png'")
print(f"✓ Experiment completed in {sum(times.values()):.1f} seconds total")