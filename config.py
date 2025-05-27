#!/usr/bin/env python3
"""
Configuration settings for the AI Computer Vision App.
"""

class Config:
    """Configuration settings class"""
    
    # Available models
    AVAILABLE_MODELS = {
        "resnet18": "ResNet18",
        "resnet34": "ResNet34",
        "resnet50": "ResNet50",
        "mobilenet_v2": "MobileNetV2",
        "efficientnet_b0": "EfficientNet-B0",
        "vgg16": "VGG16",
        "densenet121": "DenseNet121",
        "inception_v3": "Inception V3",
    }
    
    # URL for ImageNet labels
    IMAGENET_LABELS_URLS = {
        "imagenet_class_index": "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json",
        "imagenet1000_clsidx_to_labels": "https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/raw/238f720ff059c1f82f368259d1ca4ffa5dd8f9f5/imagenet1000_clsidx_to_labels.txt"
    }
    
    # Default directories
    DEFAULT_MODELS_DIR = "./models"
    DEFAULT_LABELS_DIR = "./labels"
    
    # Processing modes
    VALID_MODES = [
        "classification",
        "detection",
        "pose",
        "action",
        "segmentation"
    ]
    
    # Default mode
    DEFAULT_MODE = "classification"
    
    # Server settings
    DEFAULT_HOST = "0.0.0.0"
    DEFAULT_PORT = 5000
    DEBUG_MODE = True