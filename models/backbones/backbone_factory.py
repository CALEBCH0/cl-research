import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, Any
from .face_backbone import FaceRecognitionBackbone

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False


def get_torchvision_backbone(name: str, pretrained: bool = True, **kwargs) -> nn.Module:
    """Get backbone from torchvision."""
    if name == "resnet18":
        model = models.resnet18(pretrained=pretrained)
        feature_dim = model.fc.in_features
        model.fc = nn.Identity()
    elif name == "resnet50":
        model = models.resnet50(pretrained=pretrained)
        feature_dim = model.fc.in_features
        model.fc = nn.Identity()
    elif name == "resnet101":
        model = models.resnet101(pretrained=pretrained)
        feature_dim = model.fc.in_features
        model.fc = nn.Identity()
    elif name == "mobilenet_v2":
        model = models.mobilenet_v2(pretrained=pretrained)
        feature_dim = model.classifier[1].in_features
        model.classifier = nn.Identity()
    elif name == "mobilenetv3_large":
        model = models.mobilenet_v3_large(pretrained=pretrained)
        feature_dim = model.classifier[0].in_features
        model.classifier = nn.Identity()
    elif name == "efficientnet_b0":
        model = models.efficientnet_b0(pretrained=pretrained)
        feature_dim = model.classifier[1].in_features
        model.classifier = nn.Identity()
    else:
        raise ValueError(f"Unknown torchvision backbone: {name}")
    
    return model, feature_dim


def get_timm_backbone(name: str, pretrained: bool = True, **kwargs) -> nn.Module:
    """Get backbone from timm library."""
    if not TIMM_AVAILABLE:
        raise ImportError("timm is not installed. Please install it to use timm backbones.")
    
    model = timm.create_model(name, pretrained=pretrained, num_classes=0)
    feature_dim = model.num_features
    
    return model, feature_dim


def get_backbone(config: Dict[str, Any]) -> FaceRecognitionBackbone:
    """Factory function to create face recognition backbone."""
    backbone_type = config.get('type', 'torchvision')
    backbone_name = config['name']
    pretrained = config.get('pretrained', True)
    embedding_dim = config.get('embedding_dim', 512)
    
    # Get base backbone
    if backbone_type == 'torchvision':
        backbone, feature_dim = get_torchvision_backbone(
            backbone_name, pretrained=pretrained
        )
    elif backbone_type == 'timm':
        backbone, feature_dim = get_timm_backbone(
            backbone_name, pretrained=pretrained
        )
    else:
        raise ValueError(f"Unknown backbone type: {backbone_type}")
    
    # Override feature_dim if specified in config
    feature_dim = config.get('feature_dim', feature_dim)
    
    # Create face recognition backbone wrapper
    face_backbone = FaceRecognitionBackbone(
        backbone=backbone,
        feature_dim=feature_dim,
        embedding_dim=embedding_dim
    )
    
    # Apply modifications
    if config.get('freeze_backbone', False):
        face_backbone.freeze_backbone()
    elif config.get('freeze_until_layer'):
        face_backbone.freeze_until_layer(config['freeze_until_layer'])
    
    return face_backbone