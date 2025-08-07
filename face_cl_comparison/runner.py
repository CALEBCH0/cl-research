#!/usr/bin/env python3
"""
Simplified runner that works with existing code.
"""
import argparse
import yaml
from pathlib import Path
from datetime import datetime
import pandas as pd
from src.training import run_training


def parse_config(config_path):
    """Parse YAML config file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def generate_runs(config):
    """Generate run configurations from comparison config."""
    runs = []
    
    if 'comparison' in config and 'vary' in config['comparison']:
        # Grid search over parameters
        import itertools
        
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
                elif parts[0] == 'strategy' and parts[-1] == 'name':
                    run_config['strategy'] = value
                elif param_path == 'strategy.params.mem_size':
                    run_config['mem_size'] = value
            
            runs.append(run_config)
    
    return runs


def main():
    parser = argparse.ArgumentParser(description="Face CL Experiment Runner")
    parser.add_argument('--exp', type=str, help='Experiment name')
    parser.add_argument('--config', type=str, help='Config file path')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device')
    parser.add_argument('--dry-run', action='store_true', help='Show configs without running')
    
    args = parser.parse_args()
    
    # Determine config path
    if args.exp:
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
    
    # Run experiments
    results = []
    for i, run_config in enumerate(runs):
        print(f"\n[{i+1}/{len(runs)}] Running: {run_config['name']}")
        
        try:
            # Map to run_training parameters
            # Check CUDA availability
            import torch
            if args.gpu >= 0 and torch.cuda.is_available():
                device = f'cuda:{args.gpu}'
            else:
                device = 'cpu'
                if args.gpu >= 0:
                    print(f"Warning: CUDA not available, using CPU instead")
            
            # Get fixed settings
            fixed = config.get('fixed', {})
            strategy_config = fixed.get('strategy', {})
            
            # Get model name from fixed config if not in run_config
            model_name = run_config.get('model')
            if not model_name and 'model' in fixed:
                model_name = fixed['model'].get('backbone', {}).get('name', 'mlp')
            
            # Get dataset name
            dataset_name = fixed.get('dataset', {}).get('name', 'mnist')
            if dataset_name == 'splitmnist':
                dataset_name = 'mnist'
            
            result = run_training(
                benchmark_name=dataset_name,
                strategy_name=strategy_config.get('name', run_config.get('strategy', 'naive')),
                model_type=model_name or 'mlp',
                device=device,
                experiences=fixed.get('dataset', {}).get('n_experiences', 5),
                epochs=fixed.get('training', {}).get('epochs_per_experience', 10),
                batch_size=fixed.get('training', {}).get('batch_size', 32),
                mem_size=strategy_config.get('params', {}).get('mem_size', run_config.get('mem_size', 500)),
                lr=0.001,
                verbose=True
            )
            
            # Add run info to result
            result['run_name'] = run_config['name']
            result['model_name'] = run_config.get('model', 'mlp')
            results.append(result)
            
        except Exception as e:
            print(f"Error in run {run_config['name']}: {str(e)}")
            continue
    
    # Save results
    if results:
        df = pd.DataFrame(results)
        df.to_csv(output_dir / 'results.csv', index=False)
        
        print(f"\n{'='*60}")
        print("RESULTS SUMMARY")
        print('='*60)
        print(df[['run_name', 'strategy', 'model', 'average_accuracy']].to_string(index=False))
        
        print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()