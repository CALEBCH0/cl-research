#!/usr/bin/env python3
"""Script to analyze and visualize experiment results."""

import argparse
import os
from omegaconf import OmegaConf
from utils.visualization import plot_cl_results, plot_accuracy_matrix, compare_strategies


def main():
    parser = argparse.ArgumentParser(description='Analyze continual learning experiment results')
    parser.add_argument('results_path', type=str, help='Path to results.yaml file or directory')
    parser.add_argument('--output-dir', type=str, default='plots', help='Output directory for plots')
    parser.add_argument('--compare', nargs='+', help='List of strategies to compare')
    
    args = parser.parse_args()
    
    if os.path.isfile(args.results_path):
        # Single experiment results
        print(f"Analyzing results from: {args.results_path}")
        
        # Load and print summary
        results = OmegaConf.load(args.results_path)
        
        print("\nSummary:")
        print("-" * 50)
        
        for result in results:
            exp = result['experience']
            metrics = result['metrics']
            
            print(f"\nExperience {exp}:")
            print(f"  Average Accuracy: {metrics.get('average_accuracy', 0):.4f}")
            print(f"  Forgetting: {metrics.get('forgetting', 0):.4f}")
            print(f"  Forward Transfer: {metrics.get('forward_transfer', 0):.4f}")
            print(f"  Backward Transfer: {metrics.get('backward_transfer', 0):.4f}")
        
        # Generate plots
        print(f"\nGenerating plots in: {args.output_dir}")
        plot_cl_results(args.results_path, args.output_dir)
        plot_accuracy_matrix(args.results_path, args.output_dir)
        
    elif os.path.isdir(args.results_path) and args.compare:
        # Compare multiple strategies
        print(f"Comparing strategies: {args.compare}")
        compare_strategies(args.results_path, args.compare, args.output_dir)
    
    else:
        print("Please provide either a results.yaml file or a directory with --compare option")


if __name__ == '__main__':
    main()