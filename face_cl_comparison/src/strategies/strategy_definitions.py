"""
Comprehensive strategy definitions for face recognition CL.
All strategies are organized by category with variants.
"""
from typing import Dict, Any, List
from dataclasses import dataclass, field


@dataclass
class StrategyDefinition:
    """Define a CL strategy with its parameters."""
    name: str
    category: str
    base_strategy: str
    description: str
    params: Dict[str, Any] = field(default_factory=dict)
    requirements: List[str] = field(default_factory=list)


class StrategyRegistry:
    """Registry of all available CL strategies for face recognition."""
    
    # ============== REPLAY-BASED STRATEGIES ==============
    REPLAY_STRATEGIES = {
        # Standard Replay Variants
        "replay_random": StrategyDefinition(
            name="replay_random",
            category="replay",
            base_strategy="replay",
            description="Random sampling from memory buffer",
            params={"mem_size": 1000, "selection": "random"}
        ),
        
        "replay_balanced": StrategyDefinition(
            name="replay_balanced",
            category="replay",
            base_strategy="replay",
            description="Class-balanced sampling from memory",
            params={"mem_size": 1000, "selection": "class_balanced"}
        ),
        
        "replay_herding": StrategyDefinition(
            name="replay_herding",
            category="replay",
            base_strategy="replay",
            description="Herding-based sample selection",
            params={"mem_size": 1000, "selection": "herding"}
        ),
        
        "replay_uncertainty": StrategyDefinition(
            name="replay_uncertainty",
            category="replay",
            base_strategy="replay",
            description="Select most uncertain samples",
            params={"mem_size": 1000, "selection": "uncertainty"}
        ),
        
        # Feature Replay Variants
        "feature_replay": StrategyDefinition(
            name="feature_replay",
            category="replay",
            base_strategy="feature_replay",
            description="Store face embeddings instead of images",
            params={
                "mem_size": 2000,
                "feature_size": 512,
                "freeze_extractor": False
            }
        ),
        
        "feature_replay_frozen": StrategyDefinition(
            name="feature_replay_frozen",
            category="replay",
            base_strategy="feature_replay",
            description="Feature replay with frozen backbone",
            params={
                "mem_size": 2000,
                "feature_size": 512,
                "freeze_extractor": True
            }
        ),
        
        # Advanced Replay
        "gradient_replay": StrategyDefinition(
            name="gradient_replay",
            category="replay",
            base_strategy="gem",
            description="Gradient-based episodic memory",
            params={
                "patterns_per_exp": 256,
                "memory_strength": 0.5
            }
        ),
        
        "dark_experience_replay": StrategyDefinition(
            name="dark_experience_replay",
            category="replay",
            base_strategy="der",
            description="DER with logit matching",
            params={
                "mem_size": 500,
                "alpha": 0.5
            }
        ),
        
        "der_plus": StrategyDefinition(
            name="der_plus",
            category="replay",
            base_strategy="der",
            description="DER++ with consistency regularization",
            params={
                "mem_size": 500,
                "alpha": 0.5,
                "beta": 0.5
            }
        ),
    }
    
    # ============== REGULARIZATION STRATEGIES ==============
    REGULARIZATION_STRATEGIES = {
        "ewc_online": StrategyDefinition(
            name="ewc_online",
            category="regularization",
            base_strategy="ewc",
            description="Online Elastic Weight Consolidation",
            params={
                "ewc_lambda": 0.4,
                "mode": "online",
                "decay_factor": 0.1
            }
        ),
        
        "ewc_separate": StrategyDefinition(
            name="ewc_separate",
            category="regularization",
            base_strategy="ewc",
            description="EWC with separate Fisher per task",
            params={
                "ewc_lambda": 0.4,
                "mode": "separate"
            }
        ),
        
        "si": StrategyDefinition(
            name="si",
            category="regularization",
            base_strategy="si",
            description="Synaptic Intelligence",
            params={"si_lambda": 0.0001}
        ),
        
        "mas": StrategyDefinition(
            name="mas",
            category="regularization",
            base_strategy="mas",
            description="Memory Aware Synapses",
            params={
                "lambda_reg": 1,
                "alpha": 0.5
            }
        ),
        
        "lwf": StrategyDefinition(
            name="lwf",
            category="regularization",
            base_strategy="lwf",
            description="Learning without Forgetting",
            params={
                "alpha": 0.5,
                "temperature": 2
            }
        ),
        
        "lwf_mc": StrategyDefinition(
            name="lwf_mc",
            category="regularization",
            base_strategy="lwf_mc",
            description="LwF for multi-class",
            params={
                "alpha": 0.5,
                "temperature": 2
            }
        ),
    }
    
    # ============== OPTIMIZATION STRATEGIES ==============
    OPTIMIZATION_STRATEGIES = {
        "gem": StrategyDefinition(
            name="gem",
            category="optimization",
            base_strategy="gem",
            description="Gradient Episodic Memory",
            params={
                "patterns_per_exp": 256,
                "memory_strength": 0.5
            }
        ),
        
        "agem": StrategyDefinition(
            name="agem",
            category="optimization",
            base_strategy="agem",
            description="Averaged GEM (more efficient)",
            params={
                "patterns_per_exp": 256,
                "sample_size": 256
            }
        ),
        
        "er_ace": StrategyDefinition(
            name="er_ace",
            category="optimization",
            base_strategy="er_ace",
            description="Experience Replay with Asymmetric Cross-Entropy",
            params={"mem_size": 500}
        ),
    }
    
    # ============== ARCHITECTURE STRATEGIES ==============
    ARCHITECTURE_STRATEGIES = {
        "packnet": StrategyDefinition(
            name="packnet",
            category="architecture",
            base_strategy="packnet",
            description="Network pruning and packing",
            params={
                "prune_percentage": 0.5,
                "post_prune_epochs": 5
            },
            requirements=["prunable_model"]
        ),
        
        "pnn": StrategyDefinition(
            name="pnn",
            category="architecture",
            base_strategy="pnn",
            description="Progressive Neural Networks",
            params={
                "lateral_connections": True
            },
            requirements=["pnn_model"]
        ),
        
        "hat": StrategyDefinition(
            name="hat",
            category="architecture",
            base_strategy="hat",
            description="Hard Attention to Task",
            params={
                "lamb": 0.75,
                "smax": 400
            }
        ),
    }
    
    # ============== BASELINE STRATEGIES ==============
    BASELINE_STRATEGIES = {
        "naive": StrategyDefinition(
            name="naive",
            category="baseline",
            base_strategy="naive",
            description="Fine-tuning without any CL method",
            params={}
        ),
        
        "cumulative": StrategyDefinition(
            name="cumulative",
            category="baseline",
            base_strategy="cumulative",
            description="Train on all data seen so far",
            params={}
        ),
        
        "joint": StrategyDefinition(
            name="joint",
            category="baseline",
            base_strategy="joint",
            description="Upper bound - train on all data",
            params={}
        ),
    }
    
    # ============== FACE-SPECIFIC STRATEGIES ==============
    FACE_SPECIFIC_STRATEGIES = {
        "icarl": StrategyDefinition(
            name="icarl",
            category="face_specific",
            base_strategy="icarl",
            description="Class-incremental learning with nearest-mean classifier",
            params={
                "mem_size_per_class": 20,
                "fixed_memory": True,
                "herding": True
            }
        ),
        
        "bic": StrategyDefinition(
            name="bic",
            category="face_specific",
            base_strategy="bic",
            description="Bias Correction for class imbalance",
            params={
                "mem_size": 1000,
                "T": 2
            }
        ),
        
        "podnet": StrategyDefinition(
            name="podnet",
            category="face_specific",
            base_strategy="podnet",
            description="Pooled Outputs Distillation",
            params={
                "lambda_c": 1.0,
                "lambda_f": 1.0,
                "nb_proxy": 10
            },
            requirements=["podnet_model"]
        ),
    }
    
    @classmethod
    def get_all_strategies(cls) -> Dict[str, StrategyDefinition]:
        """Get all registered strategies."""
        all_strategies = {}
        for strategy_dict in [
            cls.REPLAY_STRATEGIES,
            cls.REGULARIZATION_STRATEGIES,
            cls.OPTIMIZATION_STRATEGIES,
            cls.ARCHITECTURE_STRATEGIES,
            cls.BASELINE_STRATEGIES,
            cls.FACE_SPECIFIC_STRATEGIES
        ]:
            all_strategies.update(strategy_dict)
        return all_strategies
    
    @classmethod
    def get_by_category(cls, category: str) -> Dict[str, StrategyDefinition]:
        """Get all strategies in a category."""
        all_strategies = cls.get_all_strategies()
        return {k: v for k, v in all_strategies.items() if v.category == category}
    
    @classmethod
    def get_categories(cls) -> List[str]:
        """Get all available categories."""
        return ["replay", "regularization", "optimization", "architecture", "baseline", "face_specific"]


# Convenience functions
def get_strategy_config(name: str) -> Dict[str, Any]:
    """Get strategy configuration by name."""
    strategy = StrategyRegistry.get_all_strategies().get(name)
    if not strategy:
        raise ValueError(f"Unknown strategy: {name}")
    
    return {
        "name": strategy.base_strategy,
        "params": strategy.params
    }


def list_strategies_by_category() -> Dict[str, List[str]]:
    """List all strategies organized by category."""
    result = {}
    for category in StrategyRegistry.get_categories():
        strategies = StrategyRegistry.get_by_category(category)
        result[category] = list(strategies.keys())
    return result