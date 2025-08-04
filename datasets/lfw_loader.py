import os
import numpy as np
from typing import List, Tuple
from sklearn.model_selection import train_test_split
import urllib.request
import tarfile


def download_lfw(root: str):
    """Download LFW dataset if not exists."""
    if not os.path.exists(root):
        os.makedirs(root)
    
    # Check if already downloaded
    if os.path.exists(os.path.join(root, 'lfw')):
        print("LFW dataset already exists")
        return
    
    # Download URL
    url = "http://vis-www.cs.umass.edu/lfw/lfw.tgz"
    filename = os.path.join(root, "lfw.tgz")
    
    try:
        print("Downloading LFW dataset...")
        # Add timeout and better error handling
        import socket
        socket.setdefaulttimeout(30)
        urllib.request.urlretrieve(url, filename)
        
        print("Extracting...")
        with tarfile.open(filename, 'r:gz') as tar:
            tar.extractall(root)
        
        os.remove(filename)
        print("LFW dataset downloaded successfully")
    except Exception as e:
        print(f"Failed to download LFW dataset: {e}")
        print("\nTo manually download:")
        print(f"1. Download from: {url}")
        print(f"2. Extract to: {root}")
        print("3. Ensure the structure is: {root}/lfw/<person_name>/<images>")
        raise RuntimeError(
            "Could not download LFW dataset. Please download manually or check your internet connection."
        )


def load_lfw_dataset(
    root: str,
    download: bool = True,
    train_split: float = 0.7,
    val_split: float = 0.15,
    min_samples_per_class: int = 5
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]], List[Tuple[str, int]]]:
    """Load LFW dataset and create train/val/test splits."""
    
    if download:
        download_lfw(root)
    
    lfw_dir = os.path.join(root, 'lfw')
    
    # Collect all samples
    samples = []
    class_names = []
    
    for person_name in sorted(os.listdir(lfw_dir)):
        person_dir = os.path.join(lfw_dir, person_name)
        if not os.path.isdir(person_dir):
            continue
        
        images = [f for f in os.listdir(person_dir) if f.endswith('.jpg')]
        
        # Filter classes with too few samples
        if len(images) < min_samples_per_class:
            continue
        
        class_idx = len(class_names)
        class_names.append(person_name)
        
        for img_name in images:
            img_path = os.path.join('lfw', person_name, img_name)
            samples.append((img_path, class_idx))
    
    # Shuffle samples
    np.random.seed(42)
    np.random.shuffle(samples)
    
    # Split by class to ensure each split has all classes
    class_samples = {}
    for path, label in samples:
        if label not in class_samples:
            class_samples[label] = []
        class_samples[label].append((path, label))
    
    train_samples = []
    val_samples = []
    test_samples = []
    
    for label, class_sample_list in class_samples.items():
        n_samples = len(class_sample_list)
        n_train = int(n_samples * train_split)
        n_val = int(n_samples * val_split)
        
        train_samples.extend(class_sample_list[:n_train])
        val_samples.extend(class_sample_list[n_train:n_train + n_val])
        test_samples.extend(class_sample_list[n_train + n_val:])
    
    print(f"Loaded LFW dataset:")
    print(f"  Total classes: {len(class_names)}")
    print(f"  Train samples: {len(train_samples)}")
    print(f"  Val samples: {len(val_samples)}")
    print(f"  Test samples: {len(test_samples)}")
    
    return train_samples, val_samples, test_samples