import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from omegaconf import OmegaConf
import os


def plot_cl_results(results_path: str, output_dir: str = "plots"):
    """Plot continual learning results from saved YAML file."""
    # Load results
    results = OmegaConf.load(results_path)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract metrics
    experiences = []
    avg_accuracies = []
    forgetting = []
    forward_transfer = []
    backward_transfer = []
    
    for result in results:
        experiences.append(result['experience'])
        metrics = result['metrics']
        avg_accuracies.append(metrics.get('average_accuracy', 0))
        forgetting.append(metrics.get('forgetting', 0))
        forward_transfer.append(metrics.get('forward_transfer', 0))
        backward_transfer.append(metrics.get('backward_transfer', 0))
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Continual Learning Results', fontsize=16)
    
    # Plot average accuracy
    axes[0, 0].plot(experiences, avg_accuracies, 'b-o', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('Experience')
    axes[0, 0].set_ylabel('Average Accuracy')
    axes[0, 0].set_title('Average Accuracy over Experiences')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim([0, 1])
    
    # Plot forgetting
    axes[0, 1].plot(experiences, forgetting, 'r-o', linewidth=2, markersize=8)
    axes[0, 1].set_xlabel('Experience')
    axes[0, 1].set_ylabel('Forgetting')
    axes[0, 1].set_title('Forgetting over Experiences')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 1])
    
    # Plot forward transfer
    axes[1, 0].plot(experiences, forward_transfer, 'g-o', linewidth=2, markersize=8)
    axes[1, 0].set_xlabel('Experience')
    axes[1, 0].set_ylabel('Forward Transfer')
    axes[1, 0].set_title('Forward Transfer over Experiences')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot backward transfer
    axes[1, 1].plot(experiences, backward_transfer, 'm-o', linewidth=2, markersize=8)
    axes[1, 1].set_xlabel('Experience')
    axes[1, 1].set_ylabel('Backward Transfer')
    axes[1, 1].set_title('Backward Transfer over Experiences')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cl_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plots saved to {output_dir}/cl_metrics.png")


def plot_accuracy_matrix(results_path: str, output_dir: str = "plots"):
    """Plot accuracy matrix heatmap."""
    # Load results
    results = OmegaConf.load(results_path)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract accuracy matrix
    n_experiences = len(results)
    accuracy_matrix = np.zeros((n_experiences, n_experiences))
    
    for result in results:
        train_exp = result['experience']
        metrics = result['metrics']
        
        for test_exp in range(n_experiences):
            key = f'test_exp_{test_exp}'
            if key in metrics:
                accuracy_matrix[test_exp, train_exp] = metrics[key]
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    
    # Mask upper triangle (future experiences)
    mask = np.zeros_like(accuracy_matrix)
    for i in range(n_experiences):
        for j in range(n_experiences):
            if j > i:
                mask[i, j] = True
    
    sns.heatmap(
        accuracy_matrix,
        mask=mask,
        annot=True,
        fmt='.3f',
        cmap='RdYlGn',
        vmin=0,
        vmax=1,
        cbar_kws={'label': 'Accuracy'},
        square=True
    )
    
    plt.xlabel('Train Experience', fontsize=12)
    plt.ylabel('Test Experience', fontsize=12)
    plt.title('Accuracy Matrix - Test vs Train Experience', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Accuracy matrix saved to {output_dir}/accuracy_matrix.png")


def compare_strategies(results_dir: str, strategy_names: List[str], output_dir: str = "plots"):
    """Compare results from multiple strategies."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Load results for each strategy
    all_results = {}
    for strategy in strategy_names:
        results_path = os.path.join(results_dir, strategy, 'results.yaml')
        if os.path.exists(results_path):
            all_results[strategy] = OmegaConf.load(results_path)
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Strategy Comparison', fontsize=16)
    
    metrics_to_plot = [
        ('average_accuracy', 'Average Accuracy', axes[0, 0]),
        ('forgetting', 'Forgetting', axes[0, 1]),
        ('forward_transfer', 'Forward Transfer', axes[1, 0]),
        ('backward_transfer', 'Backward Transfer', axes[1, 1])
    ]
    
    for metric_key, metric_name, ax in metrics_to_plot:
        for strategy, results in all_results.items():
            experiences = []
            values = []
            
            for result in results:
                experiences.append(result['experience'])
                values.append(result['metrics'].get(metric_key, 0))
            
            ax.plot(experiences, values, '-o', label=strategy, linewidth=2, markersize=6)
        
        ax.set_xlabel('Experience')
        ax.set_ylabel(metric_name)
        ax.set_title(f'{metric_name} Comparison')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        if metric_key in ['average_accuracy', 'forgetting']:
            ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'strategy_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Strategy comparison saved to {output_dir}/strategy_comparison.png")