#!/usr/bin/env python3
"""
Script to download pre-trained models for image classification.
Supports downloading ResNet18, ResNet34, ResNet50, MobileNetV2, etc.

Usage:
    python download_models.py --model=resnet18 --model-type=classification --output=./models
"""

import os
import sys
import argparse
import torch
from torchvision import models
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Available models mapping
AVAILABLE_MODELS = {
    # Classification models
    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    "resnet50": models.resnet50, 
    "mobilenet_v2": models.mobilenet_v2,
    "efficientnet_b0": models.efficientnet_b0,
    "vgg16": models.vgg16,
    "densenet121": models.densenet121,
    "inception_v3": models.inception_v3,
    
    # These would be imported from other modules in a real implementation
    # Placeholder functions for other model types
    # Detection models
    "ssd300": lambda pretrained: None,
    "fasterrcnn": lambda pretrained: None,
    
    # Segmentation models
    "deeplabv3": lambda pretrained: None,
    "fcn": lambda pretrained: None,
    
    # Pose estimation models
    "openpose": lambda pretrained: None,
    "hrnet": lambda pretrained: None,
    
    # Action recognition models
    "i3d": lambda pretrained: None,
    "slowfast": lambda pretrained: None,
}

def download_model(model_name, output_path, model_type=None, use_pretrained=True):
    """
    Download the specified model and save it to the output directory.
    
    Args:
        model_name (str): Name of the model to download (must be in AVAILABLE_MODELS).
        output_path (str): Path to save the downloaded model.
        model_type (str): Type of model (e.g., 'classification', 'detection').
        use_pretrained (bool): Whether to use pre-trained weights.
    
    Returns:
        str: Path where the model was saved.
    """
    if model_name not in AVAILABLE_MODELS:
        available_models = ", ".join(AVAILABLE_MODELS.keys())
        raise ValueError(f"Model '{model_name}' not supported. Available models: {available_models}")
    
    # If model_type is provided, create a type-specific directory
    if model_type:
        model_dir = os.path.join(output_path, model_type)
    else:
        model_dir = output_path
    
    # Create output directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    logger.info(f"Downloading {model_name} model" + 
                (f" for {model_type}" if model_type else "") + 
                f" (pretrained={use_pretrained})...")
    
    # Download the model with or without pre-trained weights
    try:
        model_fn = AVAILABLE_MODELS[model_name]
        
        # For classification models, we can actually download and save them
        if model_name in ["resnet18", "resnet34", "resnet50", "mobilenet_v2", 
                          "efficientnet_b0", "vgg16", "densenet121", "inception_v3"]:
            model = model_fn(pretrained=use_pretrained)
            
            # Save the model
            model_path = os.path.join(model_dir, f"{model_name}.pth")
            torch.save(model.state_dict(), model_path)
            
            logger.info(f"Model saved to {model_path}")
            return model_path
        else:
            # For other model types, we would implement their specific download logic
            # This is a placeholder - in a real implementation, each model type would have its own download function
            logger.warning(f"Note: '{model_name}' is a placeholder. No actual model will be downloaded.")
            
            # Create a dummy model file to simulate the download
            model_path = os.path.join(model_dir, f"{model_name}.pth")
            with open(model_path, 'w') as f:
                f.write(f"Placeholder for {model_name} model")
            
            logger.info(f"Placeholder model file created at {model_path}")
            return model_path
    
    except Exception as e:
        logger.error(f"Error downloading model: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Download pre-trained image classification models")
    
    parser.add_argument("--model", type=str, required=True, 
                        help=f"Model to download. Available models: {', '.join(AVAILABLE_MODELS.keys())}")
    parser.add_argument("--model-type", type=str, default=None,
                        help="Type of model (e.g., 'classification', 'detection')")
    parser.add_argument("--output", type=str, default="./models", 
                        help="Directory to save the downloaded model (default: ./models)")
    parser.add_argument("--no-pretrained", action="store_true",
                        help="Download the model without pre-trained weights")
    
    args = parser.parse_args()
    
    try:
        download_model(
            model_name=args.model, 
            output_path=args.output,
            model_type=args.model_type,
            use_pretrained=not args.no_pretrained
        )
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()