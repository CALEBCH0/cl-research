# Face Recognition Datasets for Continual Learning

## Currently Implemented
- **Olivetti Faces**: 40 people × 10 images = 400 images (good for quick tests)

## Popular Face Datasets for CL

### 1. **LFW (Labeled Faces in the Wild)**
- **Size**: 13,233 images of 5,749 people
- **Features**: Real-world conditions, various poses/lighting
- **CL Setup**: Can use people with 10+ images (~158 identities)
- **Download**: Available via scikit-learn or direct download

### 2. **CelebA**
- **Size**: 202,599 images of 10,177 celebrities
- **Features**: 40 binary attributes, landmark annotations
- **CL Setup**: Great for large-scale experiments
- **Download**: Requires agreement to terms

### 3. **VGGFace2**
- **Size**: 3.31M images of 9,131 subjects
- **Features**: Large variations in pose, age, illumination
- **CL Setup**: Can create subsets (e.g., 1000 identities)
- **Download**: Academic use only

### 4. **MS-Celeb-1M** (MS1M)
- **Size**: 10M images of 100K celebrities
- **Features**: Largest public face dataset
- **CL Setup**: Usually use cleaned subset
- **Note**: Original withdrawn, cleaned versions available

### 5. **CASIA-WebFace**
- **Size**: 494,414 images of 10,575 subjects
- **Features**: Good quality, widely used
- **CL Setup**: Excellent for medium-scale experiments
- **Download**: Registration required

### 6. **AgeDB**
- **Size**: 16,488 images of 568 subjects
- **Features**: Age variations (1-101 years)
- **CL Setup**: Good for age-invariant learning
- **Download**: Publicly available

### 7. **RAF-DB (Real-world Affective Faces)**
- **Size**: 29,672 images
- **Features**: Facial expressions in the wild
- **CL Setup**: Can combine identity + expression
- **Download**: Academic use

## Implementation Example

Here's how to add LFW to the framework:

```python
# src/datasets/lfw.py
import numpy as np
from sklearn.datasets import fetch_lfw_people
from torch.utils.data import TensorDataset
import torch
from avalanche.benchmarks import nc_benchmark

def create_lfw_benchmark(n_experiences=10, min_faces_per_person=20, 
                        resize=(64, 64), seed=42):
    """Create LFW benchmark for continual learning.
    
    Args:
        n_experiences: Number of CL tasks
        min_faces_per_person: Minimum images per identity
        resize: Target size for images
        seed: Random seed
    
    Returns:
        CL benchmark
    """
    # Fetch LFW data
    lfw_people = fetch_lfw_people(
        min_faces_per_person=min_faces_per_person,
        resize=resize,
        color=False,  # Grayscale
        funneled=True,  # Aligned faces
        download_if_missing=True
    )
    
    # Convert to tensors
    X = torch.FloatTensor(lfw_people.images).unsqueeze(1)  # Add channel
    y = torch.LongTensor(lfw_people.target)
    
    # Split train/test (80/20)
    n_samples = len(X)
    n_train = int(0.8 * n_samples)
    
    # Shuffle
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_samples)
    
    train_X, train_y = X[perm[:n_train]], y[perm[:n_train]]
    test_X, test_y = X[perm[n_train:]], y[perm[n_train:]]
    
    # Create datasets
    train_dataset = TensorDataset(train_X, train_y)
    test_dataset = TensorDataset(test_X, test_y)
    
    # Create CL benchmark
    benchmark = nc_benchmark(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        n_experiences=n_experiences,
        task_labels=False,
        seed=seed,
        class_ids_from_zero_in_each_exp=False
    )
    
    return benchmark, len(lfw_people.target_names)
```

## Adding Custom Face Datasets

### Step 1: Create Dataset Loader
```python
# src/datasets/custom_faces.py
class CustomFaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.targets = []
        
        # Load your dataset structure
        # Expected: root_dir/person_name/image.jpg
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path = self.samples[idx]
        target = self.targets[idx]
        
        # Load and transform image
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
            
        return image, target
```

### Step 2: Update training.py
```python
elif benchmark_name == 'lfw':
    from src.datasets.lfw import create_lfw_benchmark
    benchmark, num_classes = create_lfw_benchmark(
        n_experiences=experiences,
        min_faces_per_person=20,
        seed=seed
    )
    input_size = 64 * 64
    channels = 1
    return benchmark, BenchmarkInfo(input_size, num_classes, channels)
```

## Recommended Datasets by Use Case

### Quick Prototyping (< 1 min per run)
- Olivetti Faces (400 images)
- LFW subset (1,000-5,000 images)

### Medium-Scale Experiments (5-30 min per run)
- Full LFW (13K images)
- AgeDB (16K images)
- RAF-DB (30K images)

### Large-Scale Experiments (1+ hours per run)
- CelebA subset (50K-200K images)
- CASIA-WebFace subset (100K images)
- VGGFace2 subset (100K-500K images)

### Production/Research (GPU cluster recommended)
- Full VGGFace2 (3.3M images)
- MS-Celeb-1M cleaned (1-5M images)

## Data Preprocessing Tips

1. **Face Alignment**: Use MTCNN or dlib for detection/alignment
2. **Normalization**: Consistent lighting/contrast
3. **Augmentation**: Careful with face-specific augmentations
4. **Privacy**: Some datasets have restrictions - check licenses

## Example Config for Large Dataset

```yaml
# configs/experiments/large_face_cl.yaml
name: large_face_cl
description: Large-scale face CL with VGGFace2 subset

comparison:
  strategies:
    - slda
    - icarl
    - replay_large

fixed:
  dataset:
    name: vggface2_subset
    n_identities: 1000  # Subset of identities
    n_experiences: 20   # 50 identities per experience
    samples_per_identity: 100  # Max samples
    
  model:
    backbone:
      name: efficientnet_b2  # Larger model for more classes
      
  training:
    epochs_per_experience: 5
    batch_size: 64  # Larger batch
    
  strategy:
    params:
      mem_size: 5000  # Larger memory for more classes
```

## Integration Checklist

- [ ] Choose dataset based on scale requirements
- [ ] Check license/usage restrictions
- [ ] Implement dataset loader
- [ ] Add to benchmark creation in training.py
- [ ] Test with small subset first
- [ ] Optimize batch size and memory for larger datasets
- [ ] Consider multi-GPU for very large datasets