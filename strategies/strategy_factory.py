import torch
import torch.nn as nn
from avalanche.training import Naive, EWC, Replay, LwF, GEM, AGEM, SynapticIntelligence
from avalanche.training.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics, forgetting_metrics
from avalanche.logging import InteractiveLogger, TensorboardLogger
from typing import Dict, Any, Optional


def get_strategy(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    strategy_config: Dict[str, Any],
    eval_plugin: Optional[EvaluationPlugin] = None,
    device: str = 'cuda'
):
    """Factory function to create continual learning strategies."""
    
    strategy_name = strategy_config['name']
    strategy_type = strategy_config['type']
    
    # Common parameters
    common_params = {
        'model': model,
        'optimizer': optimizer,
        'criterion': criterion,
        'train_mb_size': strategy_config['train_mb_size'],
        'train_epochs': strategy_config['train_epochs'],
        'eval_mb_size': strategy_config['eval_mb_size'],
        'device': device,
        'evaluator': eval_plugin
    }
    
    if strategy_type == 'Naive':
        strategy = Naive(**common_params)
    
    elif strategy_type == 'EWC':
        strategy = EWC(
            **common_params,
            ewc_lambda=strategy_config.get('ewc_lambda', 0.4),
            mode=strategy_config.get('mode', 'online'),
            decay_factor=strategy_config.get('decay_factor', 0.1) if strategy_config.get('mode') == 'online' else None
        )
    
    elif strategy_type == 'Replay':
        strategy = Replay(
            **common_params,
            mem_size=strategy_config.get('mem_size', 5000),
            batch_size_mem=strategy_config.get('batch_size_mem', 10),
        )
    
    elif strategy_type == 'LwF':
        strategy = LwF(
            **common_params,
            alpha=strategy_config.get('alpha', 0.5),
            temperature=strategy_config.get('temperature', 2.0)
        )
    
    elif strategy_type == 'GEM':
        strategy = GEM(
            **common_params,
            patterns_per_exp=strategy_config.get('patterns_per_exp', 256),
            memory_strength=strategy_config.get('memory_strength', 0.5)
        )
    
    elif strategy_type == 'AGEM':
        strategy = AGEM(
            **common_params,
            patterns_per_exp=strategy_config.get('patterns_per_exp', 256),
            sample_size=strategy_config.get('sample_size', 256)
        )
    
    elif strategy_type == 'SI':
        strategy = SynapticIntelligence(
            **common_params,
            si_lambda=strategy_config.get('si_lambda', 0.0001),
            eps=strategy_config.get('eps', 0.0000001)
        )
    
    else:
        raise ValueError(f"Unknown strategy type: {strategy_type}")
    
    return strategy