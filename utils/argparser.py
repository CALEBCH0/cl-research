import argparse
import torch

def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Avalanche CL Training')
    parser.add_argument('--benchmark', type=str, default='mnist',
                       choices=['mnist', 'fmnist', 'cifar10', 'lfw', 'celeba'],
                       help='Benchmark to use')
    parser.add_argument('--strategy', type=str, default='naive',
                       choices=['naive', 'ewc', 'replay', 'gem', 'agem', 'lwf', 
                               'si', 'mas', 'gdumb', 'cumulative', 'joint', 'icarl'],
                       help='CL strategy')
    parser.add_argument('--model', type=str, default='mlp',
                       choices=['mlp', 'cnn'],
                       help='Model architecture')
    parser.add_argument('--epochs', type=int, default=2,
                       help='Epochs per experience')
    parser.add_argument('--experiences', type=int, default=5,
                       help='Number of experiences')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size')
    parser.add_argument('--device', type=str, 
                       default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    parser.add_argument('--mem_size', type=int, default=500,
                       help='Replay buffer size (only for replay strategy)')
    args = parser.parse_args()
    return args