"""
Model management package for AI Computer Vision App.
"""

from models.model_handler import ModelHandler, MODEL_TYPES
from models.download_models import download_model, AVAILABLE_MODELS
from models.download_imagenet_labels import download_imagenet_labels, IMAGENET_LABELS_URLS

__all__ = [
    'ModelHandler',
    'MODEL_TYPES',
    'download_model',
    'AVAILABLE_MODELS',
    'download_imagenet_labels',
    'IMAGENET_LABELS_URLS'
]