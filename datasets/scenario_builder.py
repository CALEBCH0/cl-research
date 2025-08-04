import torch
from torch.utils.data import random_split
from torchvision import transforms
from avalanche.benchmarks import nc_benchmark
from avalanche.benchmarks.utils import AvalancheDataset
from typing import Dict, Any, List, Tuple
import numpy as np
from .face_dataset import FaceDataset
from .lfw_loader import load_lfw_dataset


def get_transforms(config: Dict[str, Any]) -> Tuple[transforms.Compose, transforms.Compose]:
    """Get train and test transforms based on config."""
    img_size = config['img_size']
    normalize_mean = config['normalize_mean']
    normalize_std = config['normalize_std']
    
    # Base transforms
    base_transforms = [
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=normalize_mean, std=normalize_std)
    ]
    
    # Train transforms with augmentation
    train_transforms_list = [transforms.Resize((int(img_size[0] * 1.1), int(img_size[1] * 1.1)))]
    
    if config.get('train_augmentation', {}).get('random_crop', False):
        train_transforms_list.append(transforms.RandomCrop(img_size))
    else:
        train_transforms_list.append(transforms.CenterCrop(img_size))
    
    if config.get('train_augmentation', {}).get('random_horizontal_flip', 0) > 0:
        train_transforms_list.append(
            transforms.RandomHorizontalFlip(p=config['train_augmentation']['random_horizontal_flip'])
        )
    
    if config.get('train_augmentation', {}).get('color_jitter', 0) > 0:
        jitter = config['train_augmentation']['color_jitter']
        train_transforms_list.append(
            transforms.ColorJitter(brightness=jitter, contrast=jitter, saturation=jitter, hue=jitter/2)
        )
    
    train_transforms_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=normalize_mean, std=normalize_std)
    ])
    
    if config.get('train_augmentation', {}).get('random_erasing', 0) > 0:
        train_transforms_list.append(
            transforms.RandomErasing(p=config['train_augmentation']['random_erasing'])
        )
    
    train_transform = transforms.Compose(train_transforms_list)
    test_transform = transforms.Compose(base_transforms)
    
    return train_transform, test_transform


def build_face_cl_scenario(config: Dict[str, Any]):
    """Build continual learning scenario for face recognition."""
    dataset_name = config.get('name', 'lfw')
    root = config.get('root', 'datasets/face_datasets/lfw')
    n_experiences = config.get('n_experiences', 5)
    scenario_type = config.get('scenario_type', 'class_incremental')
    
    # Get transforms
    train_transform, test_transform = get_transforms(config)
    
    # Load dataset
    if dataset_name == 'lfw':
        train_samples, val_samples, test_samples = load_lfw_dataset(
            root=root,
            download=config.get('download', True),
            train_split=config['train_split'],
            val_split=config['val_split']
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Create dataset objects
    train_dataset = FaceDataset(root, train_samples, transform=train_transform)
    val_dataset = FaceDataset(root, val_samples, transform=test_transform)
    test_dataset = FaceDataset(root, test_samples, transform=test_transform)
    
    # Convert to Avalanche datasets
    train_dataset = AvalancheDataset(train_dataset)
    val_dataset = AvalancheDataset(val_dataset)
    test_dataset = AvalancheDataset(test_dataset)
    
    # Create continual learning scenario
    if scenario_type == 'class_incremental':
        # Class-incremental scenario
        scenario = nc_benchmark(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            n_experiences=n_experiences,
            task_labels=False,
            shuffle=config.get('shuffle_classes', True),
            seed=42
        )
    elif scenario_type == 'domain_incremental':
        # Domain-incremental scenario (would need different implementation)
        raise NotImplementedError("Domain-incremental scenario not yet implemented")
    else:
        raise ValueError(f"Unknown scenario type: {scenario_type}")
    
    return scenario, val_dataset