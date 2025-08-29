#!/usr/bin/env python3
"""
CSV Results Viewer - Display experiment results in the same format as runner.py output.

Usage:
    python view_results.py                    # View latest results
    python view_results.py results.csv        # View specific CSV file
    python view_results.py output/exp_name/   # View results from experiment directory
    python view_results.py --sort accuracy    # Sort by accuracy (descending)
    python view_results.py --sort name        # Sort by run name
    python view_results.py --filter slda      # Filter for specific strategy
"""

import os
import sys
import csv
import json
import argparse
from pathlib import Path


def find_latest_results(base_dir='face_cl_comparison/results'):
    """Find the most recent results.csv file."""
    results_path = Path(base_dir)
    if not results_path.exists():
        return None
    
    # Find all results.csv files in experiment directories
    csv_files = list(results_path.glob('*/results.csv'))
    
    if not csv_files:
        return None
    
    # Get the most recent one
    latest = max(csv_files, key=lambda p: p.stat().st_mtime)
    return latest


def load_results(file_path):
    """Load results from CSV file."""
    if not Path(file_path).exists():
        # Try different possible paths
        possible_paths = [
            Path('face_cl_comparison/results') / file_path / 'results.csv',  # results/experiment_name
            Path('face_cl_comparison/results') / file_path,  # results/experiment_name/results.csv
            Path(file_path) / 'results.csv',  # Direct path to experiment dir
        ]
        
        found = False
        for possible_path in possible_paths:
            if possible_path.exists():
                file_path = possible_path
                found = True
                break
        
        if not found:
            print(f"Error: File not found: {file_path}")
            print("Tried the following paths:")
            for p in possible_paths:
                print(f"  - {p}")
            return None
    
    try:
        results = []
        with open(file_path, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                results.append(row)
        return results
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None


def parse_accuracies(acc_str):
    """Parse the accuracies list string."""
    try:
        # Remove quotes and parse as JSON array
        acc_str = acc_str.strip('"')
        accuracies = json.loads(acc_str)
        return accuracies
    except:
        return []


def format_accuracy(acc):
    """Format accuracy value for display."""
    if acc is None or acc == '':
        return "N/A"
    try:
        acc_float = float(acc)
        return f"{acc_float:.4f}"
    except:
        return str(acc)


def safe_float(value, default=0.0):
    """Safely convert to float."""
    try:
        return float(value)
    except:
        return default


def filter_results(results, filter_str):
    """Filter results by string match."""
    if not filter_str:
        return results
    
    filtered = []
    filter_lower = filter_str.lower()
    
    for result in results:
        # Check common fields for match
        fields_to_check = ['run_name', 'strategy', 'model', 'dataset_name']
        match_found = False
        
        for field in fields_to_check:
            if field in result and filter_lower in result[field].lower():
                match_found = True
                break
        
        if match_found:
            filtered.append(result)
    
    return filtered


def sort_results(results, sort_by):
    """Sort results by specified field."""
    if not sort_by:
        return results
    
    if sort_by == 'accuracy':
        return sorted(results, key=lambda x: safe_float(x.get('average_accuracy', 0)), reverse=True)
    elif sort_by == 'name':
        return sorted(results, key=lambda x: x.get('run_name', ''))
    elif sort_by == 'strategy':
        return sorted(results, key=lambda x: x.get('strategy', ''))
    elif sort_by == 'model':
        return sorted(results, key=lambda x: x.get('model', ''))
    
    return results


def display_results_table(results, sort_by=None, filter_str=None):
    """Display results in the same format as runner.py."""
    
    # Apply filter if specified
    if filter_str:
        results = filter_results(results, filter_str)
        if not results:
            print(f"No results matching filter: '{filter_str}'")
            return
    
    # Sort if specified
    results = sort_results(results, sort_by)
    
    # Print header
    print("\n" + "="*95)
    print("RESULTS SUMMARY")
    print()
    
    # Column widths
    col_widths = {
        'run_name': 28,
        'strategy': 10,
        'model': 22,
        'dataset_name': 15,
        'average_accuracy': 16
    }
    
    # Print column headers
    header_line = ""
    header_line += 'run_name'.ljust(col_widths['run_name'])
    header_line += 'strategy'.ljust(col_widths['strategy'])
    header_line += 'model'.ljust(col_widths['model'])
    header_line += 'dataset_name'.ljust(col_widths['dataset_name'])
    header_line += 'average_accuracy'
    print(header_line)
    
    print("="*95)
    
    # Print data rows
    for result in results:
        line = ""
        
        # Run name
        run_name = result.get('run_name', 'N/A')[:col_widths['run_name']-1]
        line += run_name.ljust(col_widths['run_name'])
        
        # Strategy
        strategy = result.get('strategy', 'N/A')[:col_widths['strategy']-1]
        line += strategy.ljust(col_widths['strategy'])
        
        # Model
        model = result.get('model', 'N/A')[:col_widths['model']-1]
        line += model.ljust(col_widths['model'])
        
        # Dataset name
        dataset_name = result.get('dataset_name', 'N/A')[:col_widths['dataset_name']-1]
        line += dataset_name.ljust(col_widths['dataset_name'])
        
        # Average accuracy
        avg_acc = result.get('average_accuracy', 'N/A')
        line += format_accuracy(avg_acc)
        
        print(line)
    
    print("="*95)
    
    # Print summary statistics
    total_runs = len(results)
    
    # Count successful runs (those with accuracy values)
    successful_results = []
    for result in results:
        acc = result.get('average_accuracy')
        if acc and acc != '' and acc != 'N/A':
            try:
                float(acc)
                successful_results.append(result)
            except:
                pass
    
    successful_runs = len(successful_results)
    failed_runs = total_runs - successful_runs
    
    if successful_runs > 0:
        # Calculate statistics
        accuracies = [safe_float(r['average_accuracy']) for r in successful_results]
        avg_acc = sum(accuracies) / len(accuracies)
        
        # Calculate standard deviation
        variance = sum((x - avg_acc) ** 2 for x in accuracies) / len(accuracies)
        std_acc = variance ** 0.5
        
        max_acc = max(accuracies)
        min_acc = min(accuracies)
        
        # Find best run
        best_result = max(successful_results, key=lambda x: safe_float(x['average_accuracy']))
        
        print(f"EXPERIMENT COMPLETE: {successful_runs}/{total_runs} runs successful")
        print()
        
        # Statistics
        print("STATISTICS:")
        print(f"  Average Accuracy: {avg_acc:.4f} ± {std_acc:.4f}")
        print(f"  Best Accuracy:    {max_acc:.4f} ({best_result['run_name']})")
        print(f"  Worst Accuracy:   {min_acc:.4f}")
        
        # Strategy breakdown if multiple strategies
        strategies = {}
        for result in successful_results:
            strategy = result.get('strategy', 'unknown')
            if strategy not in strategies:
                strategies[strategy] = []
            strategies[strategy].append(safe_float(result['average_accuracy']))
        
        if len(strategies) > 1:
            print("\nSTRATEGY BREAKDOWN:")
            for strategy, accs in strategies.items():
                mean_acc = sum(accs) / len(accs)
                if len(accs) > 1:
                    var = sum((x - mean_acc) ** 2 for x in accs) / len(accs)
                    std = var ** 0.5
                    print(f"  {strategy:15s}: {mean_acc:.4f} ± {std:.4f} (n={len(accs)})")
                else:
                    print(f"  {strategy:15s}: {mean_acc:.4f} (n={len(accs)})")
        
        # Model breakdown if multiple models
        models = {}
        for result in successful_results:
            model = result.get('model', 'unknown')
            if model not in models:
                models[model] = []
            models[model].append(safe_float(result['average_accuracy']))
        
        if len(models) > 1:
            print("\nMODEL BREAKDOWN:")
            for model, accs in models.items():
                mean_acc = sum(accs) / len(accs)
                if len(accs) > 1:
                    var = sum((x - mean_acc) ** 2 for x in accs) / len(accs)
                    std = var ** 0.5
                    print(f"  {model:20s}: {mean_acc:.4f} ± {std:.4f} (n={len(accs)})")
                else:
                    print(f"  {model:20s}: {mean_acc:.4f} (n={len(accs)})")
    else:
        print(f"NO SUCCESSFUL RUNS (0/{total_runs})")
    
    if failed_runs > 0:
        print(f"\n{failed_runs} runs failed")
    
    print("="*95)


def main():
    parser = argparse.ArgumentParser(description='View experiment results from CSV files')
    parser.add_argument('file', nargs='?', help='CSV file or experiment directory to view')
    parser.add_argument('--sort', choices=['accuracy', 'name', 'strategy', 'model'],
                       help='Sort results by specified column')
    parser.add_argument('--filter', help='Filter results by string match')
    parser.add_argument('--latest', action='store_true', 
                       help='Automatically load latest results')
    
    args = parser.parse_args()
    
    # Determine which file to load
    if args.file:
        csv_file = args.file
    elif args.latest or not sys.stdin.isatty():
        # Find latest results
        csv_file = find_latest_results()
        if not csv_file:
            print("No results.csv files found in face_cl_comparison/results/ directory")
            return
        print(f"Loading latest results: {csv_file}")
    else:
        # Default to looking for results.csv in current directory
        if Path('results.csv').exists():
            csv_file = 'results.csv'
        else:
            csv_file = find_latest_results()
            if not csv_file:
                print("No results.csv found. Specify a file or use --latest")
                print("\nUsage:")
                print("  python view_results.py results.csv")
                print("  python view_results.py face_cl_comparison/results/experiment_name_datetime/")
                print("  python view_results.py --latest")
                return
            print(f"Loading latest results: {csv_file}")
    
    # Load and display results
    results = load_results(csv_file)
    if results is not None:
        print(f"\nResults from: {csv_file}")
        display_results_table(results, sort_by=args.sort, filter_str=args.filter)


if __name__ == '__main__':
    main()