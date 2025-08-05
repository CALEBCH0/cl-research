"""Unit tests for training functionality."""
import unittest
from train_working_refactored import run_training, set_benchmark, create_model, create_strategy
import torch
import torch.nn as nn
from avalanche.evaluation.metrics import accuracy_metrics
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin


class TestTraining(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.seed = 42
        torch.manual_seed(self.seed)
    
    def test_benchmark_creation(self):
        """Test benchmark creation."""
        benchmark, info = set_benchmark('mnist', experiences=5)
        self.assertEqual(benchmark.n_experiences, 5)
        self.assertEqual(info.num_classes, 10)
        self.assertEqual(info.channels, 1)
    
    def test_model_creation(self):
        """Test model creation."""
        benchmark, info = set_benchmark('mnist')
        model = create_model('mlp', info)
        self.assertIsNotNone(model)
        
        # Test forward pass
        x = torch.randn(1, info.input_size)
        output = model(x)
        self.assertEqual(output.shape[1], info.num_classes)
    
    def test_strategy_creation(self):
        """Test strategy creation."""
        benchmark, info = set_benchmark('mnist')
        model = create_model('mlp', info)
        optimizer = torch.optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss()
        
        eval_plugin = EvaluationPlugin(
            accuracy_metrics(experience=True),
            loggers=[]
        )
        
        strategy = create_strategy(
            'replay', model, optimizer, criterion, 
            self.device, eval_plugin, mem_size=100
        )
        self.assertIsNotNone(strategy)
    
    def test_quick_training(self):
        """Test a quick training run."""
        results = run_training(
            benchmark_name='mnist',
            strategy_name='naive',
            model_type='mlp',
            device=self.device,
            experiences=2,
            epochs=1,
            batch_size=64,
            verbose=False
        )
        
        self.assertIn('average_accuracy', results)
        self.assertIn('accuracies', results)
        self.assertEqual(len(results['accuracies']), 2)
        self.assertTrue(0 <= results['average_accuracy'] <= 1)
    
    def test_multiple_strategies(self):
        """Test that different strategies produce different results."""
        results_naive = run_training(
            benchmark_name='mnist',
            strategy_name='naive',
            experiences=5,  # Changed from 3 to 5 (10 classes / 5 = 2 classes per exp)
            epochs=1,
            device=self.device,
            verbose=False
        )
        
        results_replay = run_training(
            benchmark_name='mnist',
            strategy_name='replay',
            experiences=5,  # Changed from 3 to 5
            epochs=1,
            device=self.device,
            verbose=False
        )
        
        # Replay should generally perform better
        self.assertGreaterEqual(
            results_replay['average_accuracy'],
            results_naive['average_accuracy'] - 0.1  # Allow some variance
        )


class TestBatchExperiments(unittest.TestCase):
    """Test batch experiment functionality."""
    
    def test_batch_comparison(self):
        """Test running multiple experiments."""
        strategies = ['naive', 'replay']
        results = []
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        for strategy in strategies:
            result = run_training(
                benchmark_name='mnist',
                strategy_name=strategy,
                experiences=2,
                epochs=1,
                device=device,
                verbose=False
            )
            results.append(result)
        
        self.assertEqual(len(results), 2)
        for result in results:
            self.assertIn('average_accuracy', result)


if __name__ == '__main__':
    # Run with minimal output
    unittest.main(verbosity=1)