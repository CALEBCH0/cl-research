#!/usr/bin/env python3
"""
Simplified runner that works with existing code.
"""
import argparse
import itertools
import traceback
import yaml
import threading
import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import torch
from src.training import run_training

# Global flag for graceful shutdown
stop_requested = False

def check_for_quit():
    """Check for 'q' input in a separate thread."""
    global stop_requested
    while not stop_requested:
        try:
            user_input = input()
            if user_input.lower() == 'q':
                print("\nStop requested! Finishing current run and saving results...")
                stop_requested = True
                break
        except:
            pass


def parse_config(config_path):
    """Parse YAML config file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def generate_runs(config):
    """Generate run configurations from comparison config."""
    # Check for modular config format
    if 'vary' in config and 'fixed' in config:
        # New modular format
        from src.utils.modular_config import expand_modular_config
        return expand_modular_config(config)
    
    runs = []
    
    # Check for new strategies format
    if 'comparison' in config and 'strategies' in config['comparison']:
        # New format: each strategy has its own plugin config
        strategies = config['comparison']['strategies']
        strategy_runs = []
        
        for strategy_config in strategies:
            if isinstance(strategy_config, str):
                # Simple strategy name
                run_config = {
                    'name': strategy_config,
                    'strategy': strategy_config,
                    'plugins': []
                }
            else:
                # Strategy with configuration
                strategy_name = strategy_config.get('name')
                base_strategy = strategy_config.get('base', strategy_name)
                plugins = strategy_config.get('plugins', [])  # Default to empty list if not specified
                
                # Create readable name - always show base strategy and plugins
                if plugins:
                    plugin_names = "_".join([p['name'] if isinstance(p, dict) else p for p in plugins])
                    display_name = f"{base_strategy}_{plugin_names}"
                else:
                    display_name = base_strategy
                
                run_config = {
                    'name': display_name,
                    'strategy': base_strategy,
                    'plugins': plugins
                }
                
                # Add strategy-specific parameters
                if 'params' in strategy_config:
                    # Store strategy params separately to pass to create_strategy
                    run_config['strategy_params'] = strategy_config['params']
            
            strategy_runs.append(run_config)
        
        # Check if there's also a 'vary' section to combine with strategies
        if 'vary' in config['comparison']:
            vary_params = config['comparison']['vary']
            param_names = list(vary_params.keys())
            param_values = list(vary_params.values())
            
            # Create cartesian product of strategies and vary parameters
            for strategy_run in strategy_runs:
                for values in itertools.product(*param_values):
                    # Copy the strategy run config
                    run_config = strategy_run.copy()
                    
                    # Add the vary parameters
                    vary_suffix_parts = []
                    for param_path, value in zip(param_names, values):
                        parts = param_path.split('.')
                        if parts[0] == 'model' and parts[-1] == 'name':
                            run_config['model'] = value
                            vary_suffix_parts.append(f"model={value}")
                        elif parts[0] == 'dataset' and parts[-1] == 'name':
                            run_config['dataset'] = value
                            vary_suffix_parts.append(f"name={value}")
                        # Add other vary parameters as needed
                    
                    # Update the run name to include vary parameters
                    if vary_suffix_parts:
                        vary_suffix = "_".join(vary_suffix_parts)
                        run_config['name'] = f"{run_config['name']}_{vary_suffix}"
                    
                    runs.append(run_config)
        else:
            # No vary section, just use strategy runs
            runs = strategy_runs
    
    elif 'comparison' in config and 'vary' in config['comparison']:
        # Original format: grid search over parameters
        vary_params = config['comparison']['vary']
        param_names = list(vary_params.keys())
        param_values = list(vary_params.values())
        
        for values in itertools.product(*param_values):
            run_config = {
                'name': '_'.join([f"{p.split('.')[-1]}={v}" for p, v in zip(param_names, values)])
            }
            
            # Parse parameter paths and set values
            for param_path, value in zip(param_names, values):
                parts = param_path.split('.')
                if parts[0] == 'model' and parts[-1] == 'name':
                    run_config['model'] = value
                elif parts[0] == 'dataset' and parts[-1] == 'name':
                    run_config['dataset'] = value
                elif parts[0] == 'strategy' and parts[-1] == 'name':
                    run_config['strategy'] = value
                elif param_path == 'strategy.params.mem_size':
                    run_config['mem_size'] = value
                elif param_path == 'strategy.plugins':
                    run_config['plugins'] = value
                    # Create a readable name for plugin combinations
                    if value is None or value == []:
                        # No plugins, use strategy name only
                        pass
                    else:
                        plugin_names = "_".join([p['name'] if isinstance(p, dict) else p for p in value])
                        run_config['name'] = f"{run_config.get('strategy', 'unknown')}_{plugin_names}"
            
            runs.append(run_config)
    
    return runs


def main():
    parser = argparse.ArgumentParser(description="Face CL Experiment Runner")
    parser.add_argument('--exp', type=str, help='Experiment name')
    parser.add_argument('--config', type=str, help='Config file path')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device')
    parser.add_argument('--dry-run', action='store_true', help='Show configs without running')
    
    args = parser.parse_args()
    
    global stop_requested
    stop_requested = False
    
    # Determine config path
    if args.exp:
        if args.exp.endswith(('.yml', '.yaml')):
            config_path = Path(f'configs/experiments/{args.exp}')
        else:
            config_path = Path(f'configs/experiments/{args.exp}.yaml')
    else:
        config_path = Path(args.config) if args.config else None
    
    if not config_path or not config_path.exists():
        print(f"Error: Config file not found!")
        return
    
    # Load config
    print(f"\nLoading config: {config_path}")
    config = parse_config(config_path)
    
    # Generate runs
    runs = generate_runs(config)
    
    print(f"\nExperiment: {config.get('name', 'unnamed')}")
    print(f"Descrption: {config.get('description', 'No description')}")
    print(f"Number of runs: {len(runs)}")
    
    if args.dry_run:
        print("\nDRY RUN - Configurations:")
        for i, run in enumerate(runs):
            print(f"\nRun {i+1}: {run}")
        return
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path('results') / f"{config.get('name', 'exp')}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine if we're in debug mode
    training_config = config.get('fixed', {}).get('training', {})
    debug_mode = training_config.get('debug', True)
    
    if debug_mode:
        # Single seed for debugging
        seeds = [training_config.get('seed', 42)]
        print("\nDEBUG MODE: Running with single seed")
    else:
        # Multiple seeds for evaluation
        seeds = training_config.get('seeds', [42, 123, 456, 789, 1011])
        print(f"\nEVALUATION MODE: Running with {len(seeds)} seeds")
    
    # Start input monitoring thread
    if not args.dry_run:
        print("\nPress 'q' + Enter at any time to stop after current run completes.")
        input_thread = threading.Thread(target=check_for_quit, daemon=True)
        input_thread.start()
    
    # Check if dataset is fixed across all runs
    is_modular = 'vary' in config and 'fixed' in config
    if is_modular:
        # For modular configs, extract dataset names from dicts
        dataset_names = []
        for run in runs:
            dataset = run.get('dataset', {})
            if isinstance(dataset, dict):
                dataset_names.append(dataset.get('name', 'unknown'))
            else:
                dataset_names.append(dataset)
    else:
        # Original format
        dataset_names = [run.get('dataset', config.get('fixed', {}).get('dataset', {}).get('name', 'mnist')) 
                         for run in runs]
    fixed_dataset = len(set(dataset_names)) == 1
    
    # Cache for benchmarks to avoid reloading
    benchmark_cache = {}
    
    # Run experiments
    all_results = []
    
    for i, run_config in enumerate(runs):
        if stop_requested:
            print("\nStopping experiments as requested...")
            break
        print(f"\n[{i+1}/{len(runs)}] Running: {run_config['name']}")
        
        run_results_by_seed = []
        
        for seed_idx, seed in enumerate(seeds):
            if stop_requested:
                print("\nStopping seed iterations as requested...")
                break
                
            if not debug_mode:
                print(f"  Seed {seed_idx+1}/{len(seeds)}: {seed}")
            
            try:
                # Map to run_training parameters
                # Check CUDA availability
                if args.gpu >= 0 and torch.cuda.is_available():
                    device = f'cuda:{args.gpu}'
                else:
                    device = 'cpu'
                    if args.gpu >= 0 and seed_idx == 0:  # Only warn once
                        print(f"Warning: CUDA not available, using CPU instead")
                
                # Get fixed settings
                fixed = config.get('fixed', {})
                strategy_config = fixed.get('strategy', {})
                
                # Handle modular configs
                is_modular = 'vary' in config and 'fixed' in config
                
                if is_modular:
                    # Modular format - configs are dicts with name, type, params
                    dataset_config = run_config.get('dataset', {})
                    model_config = run_config.get('model', {})
                    strategy_config = run_config.get('strategy', {})
                    
                    dataset_name = dataset_config.get('name', 'mnist')
                    model_name = model_config.get('name', 'mlp')
                    strategy_name = strategy_config.get('name', 'naive')
                    
                    # Store full configs for later use
                    run_config['dataset_config'] = dataset_config
                    run_config['model_config'] = model_config
                    run_config['strategy_config'] = strategy_config
                else:
                    # Original format
                    # Get model name from fixed config if not in run_config
                    model_name = run_config.get('model')
                    if not model_name and 'model' in fixed:
                        model_name = fixed['model'].get('backbone', {}).get('name', 'mlp')
                    
                    # Get dataset name from run_config first, then fixed config
                    dataset_name = run_config.get('dataset')
                    if not dataset_name:
                        dataset_name = fixed.get('dataset', {}).get('name', 'mnist')
                    if dataset_name == 'splitmnist':
                        dataset_name = 'mnist'
                    
                    # Get strategy name from run_config or fixed config
                    strategy_name = run_config.get('strategy')
                    if not strategy_name and 'strategy' in fixed:
                        strategy_name = fixed['strategy'].get('name', 'naive')
                
                # Merge strategy-specific params with defaults
                strategy_specific_params = run_config.get('strategy_params', {})
                default_params = strategy_config.get('params', {})
                
                # Get subset config if present
                subset_config = fixed.get('dataset', {}).get('subset', None)
                
                # Get or create benchmark (with caching)
                n_experiences = fixed.get('dataset', {}).get('n_experiences', 5)
                
                if is_modular and 'dataset_config' in run_config:
                    # Handle modular dataset creation
                    dataset_config = run_config['dataset_config']
                    
                    # Override n_experiences if specified in params
                    if 'n_experiences' in dataset_config.get('params', {}):
                        n_experiences = dataset_config['params']['n_experiences']
                    # Also check if n_experiences is directly in dataset_config (from fixed merge)
                    elif 'n_experiences' in dataset_config:
                        n_experiences = dataset_config['n_experiences']
                    
                    # Create cache key for modular dataset
                    # Include image_size in cache key if it will be auto-adjusted
                    cache_params = dataset_config.get('params', {}).copy()
                    if 'model_config' in run_config:
                        from src.utils.model_utils import get_dataset_requirements_for_model
                        model_reqs = get_dataset_requirements_for_model(run_config['model_config'])
                        cache_params['image_size'] = model_reqs['image_size']
                    
                    cache_key = (
                        dataset_config['name'],
                        n_experiences,
                        frozenset(cache_params.items())
                    )
                    
                    if cache_key in benchmark_cache:
                        cached_benchmark, cached_info = benchmark_cache[cache_key]
                        if debug_mode and seed_idx == 0:
                            print(f"  Using cached benchmark for {dataset_config['name']}")
                    else:
                        # Create modular dataset
                        from src.utils.modular_config import create_dataset_from_config
                        from src.utils.model_utils import merge_dataset_config_with_model_requirements
                        
                        # Merge fixed dataset settings with modular config
                        full_dataset_config = dataset_config.copy()
                        for key, value in fixed.get('dataset', {}).items():
                            if key not in full_dataset_config:
                                full_dataset_config[key] = value
                        
                        # Auto-adjust dataset config based on model requirements
                        if 'model_config' in run_config:
                            full_dataset_config = merge_dataset_config_with_model_requirements(
                                full_dataset_config, run_config['model_config']
                            )
                        
                        cached_benchmark, cached_info = create_dataset_from_config(full_dataset_config)
                        benchmark_cache[cache_key] = (cached_benchmark, cached_info)
                        if debug_mode:
                            print(f"  Created and cached modular benchmark for {dataset_config['name']}")
                else:
                    # Original dataset creation
                    # Include subset config in cache key if present
                    if subset_config:
                        cache_key = (dataset_name, n_experiences, 
                                    frozenset(subset_config.items()) if subset_config else None)
                    else:
                        cache_key = (dataset_name, n_experiences)
                    
                    if cache_key in benchmark_cache:
                        # Reuse cached benchmark
                        cached_benchmark, cached_info = benchmark_cache[cache_key]
                        if debug_mode and seed_idx == 0:
                            print(f"  Using cached benchmark for {dataset_name}")
                    else:
                        # Create and cache benchmark
                        from src.training import create_benchmark
                        cached_benchmark, cached_info = create_benchmark(
                            dataset_name, n_experiences, seed, subset_config
                        )
                        benchmark_cache[cache_key] = (cached_benchmark, cached_info)
                        if debug_mode:
                            print(f"  Created and cached benchmark for {dataset_name}")
                
                # Build kwargs for run_training
                training_kwargs = {
                    'benchmark_name': dataset_name,
                    'strategy_name': strategy_name,
                    'model_type': model_name or 'mlp',
                    'device': device,
                    'experiences': n_experiences,
                    'epochs': fixed.get('training', {}).get('epochs_per_experience', 10),
                    'batch_size': fixed.get('training', {}).get('batch_size', 32),
                    'mem_size': strategy_specific_params.get('mem_size', default_params.get('mem_size', 500)),
                    'lr': fixed.get('training', {}).get('lr', 0.001),
                    'seed': seed,
                    'verbose': debug_mode,
                    'plugins_config': run_config.get('plugins', None),
                    'benchmark': cached_benchmark,
                    'benchmark_info': cached_info,
                    'subset_config': subset_config
                }
                
                # Add modular configs if present
                if is_modular:
                    if 'model_config' in run_config:
                        training_kwargs['model_config'] = run_config['model_config']
                    if 'strategy_config' in run_config:
                        training_kwargs['strategy_config'] = run_config['strategy_config']
                        # Extract strategy params for compatibility
                        if 'params' in run_config['strategy_config']:
                            for key, value in run_config['strategy_config']['params'].items():
                                if key not in training_kwargs:
                                    training_kwargs[key] = value
                
                # Add any additional strategy-specific parameters
                # These will be passed as **kwargs to create_strategy
                for key, value in strategy_specific_params.items():
                    if key not in training_kwargs:
                        training_kwargs[key] = value
                
                result = run_training(**training_kwargs)
                
                # Add run info to result
                result['run_name'] = run_config['name']
                result['model_name'] = run_config.get('model', 'mlp')
                result['dataset_name'] = dataset_name
                result['seed'] = seed
                run_results_by_seed.append(result)
                
            except Exception as e:
                print(f"Error in run {run_config['name']} with seed {seed}: {str(e)}")
                print(f"Error type: {type(e).__name__}")
                if debug_mode or 'pure_ncm' in run_config['name']:  # Always show traceback for pure_ncm
                    print("Traceback:")
                    traceback.print_exc()
                continue
        
        # Aggregate results for this run
        if run_results_by_seed:
            if debug_mode:
                # Single seed - just add the result
                all_results.extend(run_results_by_seed)
            else:
                # Multiple seeds - compute statistics
                accuracies = [r['average_accuracy'] for r in run_results_by_seed]
                mean_acc = np.mean(accuracies)
                std_acc = np.std(accuracies)
                
                # Create aggregated result
                agg_result = {
                    'run_name': run_config['name'],
                    'strategy': run_results_by_seed[0]['strategy'],
                    'model': run_results_by_seed[0]['model'],
                    'dataset_name': run_results_by_seed[0]['dataset_name'],
                    'average_accuracy': mean_acc,
                    'accuracy_std': std_acc,
                    'accuracy_mean': mean_acc,
                    'num_seeds': len(accuracies),
                    'individual_accuracies': accuracies
                }
                all_results.append(agg_result)
                
                print(f"  → {run_config['name']}: {mean_acc:.3f} ± {std_acc:.3f}")
    
    # Save results and show completion summary
    total_runs = len(runs)
    successful_runs = len(all_results)
    failed_runs = total_runs - successful_runs
    
    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(output_dir / 'results.csv', index=False)
        
        tab_size = 17
        print(f"\n{'='*90}")
        print("RESULTS SUMMARY\n")
        print(f"{'run_name':<{tab_size}} {'strategy':<{tab_size}} {'model':<{tab_size}} {'dataset_name':<{tab_size}} {'average_accuracy':<{tab_size}}")
        print('='*90)
        
        if debug_mode:
            # Single seed - show simple table
            for _, row in df.iterrows():
                print(f"{row['run_name']:<{tab_size}} {row['strategy']:<{tab_size}} {row['model']:<{tab_size}} {row['dataset_name']:<{tab_size}} "
                      f"{row['average_accuracy']:.4f}")
        else:
            # Multiple seeds - show mean ± std
            for _, row in df.iterrows():
                print(f"{row['run_name']:<{tab_size}} {row['strategy']:<{tab_size}} {row['model']:<{tab_size}} {row['dataset_name']:<{tab_size}} "
                      f"{row['accuracy_mean']:.3f} ± {row['accuracy_std']:.3f}")
        
        print(f"\n{'='*90}")
        print(f"EXPERIMENT COMPLETE: {successful_runs}/{total_runs} runs successful")
        if failed_runs > 0:
            print(f"⚠️  {failed_runs} runs failed - check output above for error details")
        print(f"Results saved to: {output_dir}")
        
        if stop_requested:
            print("\nExperiment stopped by user request.")
    else:
        print(f"\n{'='*90}")
        print(f"EXPERIMENT FAILED: 0/{total_runs} runs successful")
        print("❌ All runs failed - check error messages above")


if __name__ == '__main__':
    main()