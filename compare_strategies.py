"""Compare multiple strategies and models using the refactored training script."""
import time
from train_working_refactored import run_training
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def compare_strategies(strategies, benchmark='fmnist', model='mlp', device='cuda', 
                      epochs=2, experiences=5, mem_size=500):
    """Compare multiple strategies on the same benchmark."""
    results = []
    
    for strategy in strategies:
        print(f"\nTesting {strategy}...")
        start_time = time.time()
        
        result = run_training(
            benchmark_name=benchmark,
            strategy_name=strategy,
            model_type=model,
            device=device,
            epochs=epochs,
            experiences=experiences,
            mem_size=mem_size,
            verbose=False  # Quiet mode for comparison
        )
        
        elapsed = time.time() - start_time
        result['time'] = elapsed
        results.append(result)
        
        print(f"  Average accuracy: {result['average_accuracy']:.4f} (time: {elapsed:.1f}s)")
    
    return results


def compare_models(models, strategy='replay', benchmark='fmnist', device='cuda'):
    """Compare different models with the same strategy."""
    results = []
    
    for model in models:
        print(f"\nTesting {model} with {strategy}...")
        
        result = run_training(
            benchmark_name=benchmark,
            strategy_name=strategy,
            model_type=model,
            device=device,
            verbose=False
        )
        
        results.append(result)
        print(f"  Average accuracy: {result['average_accuracy']:.4f}")
    
    return results


def plot_results(results, title="Strategy Comparison"):
    """Plot comparison results."""
    # Create DataFrame
    data = []
    for result in results:
        for i, acc in enumerate(result['accuracies']):
            data.append({
                'Strategy': result['strategy'],
                'Experience': i,
                'Accuracy': acc
            })
    
    df = pd.DataFrame(data)
    
    # Plot
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='Experience', y='Accuracy', hue='Strategy', marker='o')
    plt.title(title)
    plt.xlabel('Experience')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('strategy_comparison.png')
    plt.close()
    
    # Bar plot of average accuracies
    avg_data = []
    for result in results:
        avg_data.append({
            'Strategy': result['strategy'],
            'Average Accuracy': result['average_accuracy']
        })
    
    avg_df = pd.DataFrame(avg_data)
    
    plt.figure(figsize=(8, 6))
    sns.barplot(data=avg_df, x='Strategy', y='Average Accuracy')
    plt.title(f"{title} - Average Accuracy")
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('average_accuracy_comparison.png')
    plt.close()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda', help='Device to use')
    parser.add_argument('--benchmark', default='fmnist', help='Benchmark dataset')
    parser.add_argument('--epochs', type=int, default=2, help='Epochs per experience')
    parser.add_argument('--experiences', type=int, default=5, help='Number of experiences')
    args = parser.parse_args()
    
    print("="*60)
    print("Continual Learning Strategy Comparison")
    print("="*60)
    
    # Test different strategies
    strategies = ['naive', 'ewc', 'replay', 'lwf', 'si', 'cumulative']
    results = compare_strategies(
        strategies, 
        benchmark=args.benchmark,
        device=args.device,
        epochs=args.epochs,
        experiences=args.experiences
    )
    
    # Print summary table
    print("\n" + "="*60)
    print("Summary Results:")
    print("="*60)
    print(f"{'Strategy':<15} {'Avg Accuracy':<15} {'Time (s)':<10}")
    print("-"*40)
    for result in results:
        print(f"{result['strategy']:<15} {result['average_accuracy']:<15.4f} {result.get('time', 0):<10.1f}")
    
    # Save detailed results
    df_results = pd.DataFrame(results)
    df_results.to_csv('strategy_comparison_results.csv', index=False)
    
    # Plot results
    plot_results(results)
    print("\nPlots saved: strategy_comparison.png, average_accuracy_comparison.png")
    print("Detailed results saved: strategy_comparison_results.csv")


if __name__ == "__main__":
    main()