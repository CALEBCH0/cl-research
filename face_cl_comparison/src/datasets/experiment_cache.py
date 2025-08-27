"""Experiment-level dataset caching to reuse datasets across runs within an experiment."""
import hashlib
from typing import Dict, Any, Optional

# Experiment-level cache for datasets (reused across runs in same experiment)
_EXPERIMENT_CACHE: Dict[str, Any] = {}


def get_dataset_cache_key(dataset_config: Dict[str, Any]) -> str:
    """
    Generate a unique cache key for a dataset configuration.
    This key identifies when two runs can share the same dataset.
    """
    # Include all parameters that affect dataset creation
    # Exclude parameters that don't affect the actual data
    key_params = {
        'name': dataset_config.get('name'),
        'path': dataset_config.get('path'),
        'seed': dataset_config.get('seed', 42),
        'test_split': dataset_config.get('test_split', 0.2),
        'n_experiences': dataset_config.get('n_experiences'),
        'image_size': dataset_config.get('image_size'),
        'preload_to_memory': dataset_config.get('preload_to_memory', False),
        'use_cropdata': dataset_config.get('use_cropdata', True)
    }
    
    # Create stable string representation
    key_str = str(sorted(key_params.items()))
    return hashlib.md5(key_str.encode()).hexdigest()


def get_cached_dataset(dataset_config: Dict[str, Any]) -> Optional[Any]:
    """
    Get dataset from experiment cache if available.
    This avoids recreating the dataset for each run in the experiment.
    """
    cache_key = get_dataset_cache_key(dataset_config)
    
    if cache_key in _EXPERIMENT_CACHE:
        print(f"[Experiment Cache] Reusing dataset from memory for this run (key: {cache_key[:8]}...)")
        return _EXPERIMENT_CACHE[cache_key]
    
    return None


def cache_dataset(dataset_config: Dict[str, Any], dataset: Any) -> None:
    """
    Store dataset in experiment cache for reuse across runs.
    """
    cache_key = get_dataset_cache_key(dataset_config)
    _EXPERIMENT_CACHE[cache_key] = dataset
    print(f"[Experiment Cache] Stored dataset in memory for reuse across runs (key: {cache_key[:8]}...)")


def clear_experiment_cache() -> None:
    """
    Clear all cached datasets from memory.
    Typically called at the end of an experiment or when memory is needed.
    """
    global _EXPERIMENT_CACHE
    num_cached = len(_EXPERIMENT_CACHE)
    _EXPERIMENT_CACHE.clear()
    if num_cached > 0:
        print(f"[Experiment Cache] Cleared {num_cached} cached dataset(s) from memory")


def get_cache_info() -> Dict[str, Any]:
    """Get information about current cache state."""
    return {
        'num_cached_datasets': len(_EXPERIMENT_CACHE),
        'cache_keys': list(_EXPERIMENT_CACHE.keys())[:5],  # Show first 5 keys
        'cache_keys_total': len(_EXPERIMENT_CACHE)
    }


def cache_stats_summary() -> str:
    """Get a summary of cache statistics for logging."""
    info = get_cache_info()
    return (f"Experiment cache: {info['num_cached_datasets']} dataset(s) in memory. "
            f"This cache speeds up multiple runs with the same dataset configuration.")