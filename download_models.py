"""
Script to download pretrained ONNX models for testing
"""

import os
import sys
import argparse
import requests
import zipfile
import io
import shutil
from tqdm import tqdm

# Dictionary of available models
# Format: 'model_key': ('url', 'model_filename', 'class_names_url')
AVAILABLE_MODELS = {
    'resnet18': (
        'https://github.com/onnx/models/raw/main/vision/classification/resnet/model/resnet18-v1-7.onnx',
        'resnet18.onnx',
        'https://raw.githubusercontent.com/onnx/models/main/vision/classification/synset.txt'
    ),
    'mobilenet': (
        'https://github.com/onnx/models/raw/main/vision/classification/mobilenet/model/mobilenetv2-7.onnx',
        'mobilenet.onnx',
        'https://raw.githubusercontent.com/onnx/models/main/vision/classification/synset.txt'
    ),
    'ssd-mobilenet': (
        'https://github.com/onnx/models/raw/main/vision/object_detection_segmentation/ssd-mobilenetv1/model/ssd_mobilenet_v1_10.onnx',
        'ssd-mobilenet.onnx',
        'https://raw.githubusercontent.com/onnx/models/main/vision/object_detection_segmentation/ssd-mobilenetv1/dependencies/coco_classes.txt'
    ),
    'yolov4': (
        'https://github.com/onnx/models/raw/main/vision/object_detection_segmentation/yolov4/model/yolov4.onnx',
        'yolov4.onnx',
        'https://raw.githubusercontent.com/onnx/models/main/vision/object_detection_segmentation/yolov4/dependencies/coco.names'
    )
}

def download_file(url, filename, desc=None):
    """
    Download a file from a URL with progress bar
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise exception for HTTP errors
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192
        
        with open(filename, 'wb') as f, tqdm(
            desc=desc,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=block_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

def download_model(model_key, output_dir):
    """
    Download a model and its class names
    """
    if model_key not in AVAILABLE_MODELS:
        print(f"Model '{model_key}' not found. Available models: {', '.join(AVAILABLE_MODELS.keys())}")
        return False
    
    model_url, model_filename, class_names_url = AVAILABLE_MODELS[model_key]
    
    # Create output directory structure
    os.makedirs(output_dir, exist_ok=True)
    
    # Full paths for model and class names
    model_path = os.path.join(output_dir, model_filename)
    class_names_path = os.path.splitext(model_path)[0] + '.txt'
    
    # Download model
    print(f"Downloading model: {model_key}")
    if not download_file(model_url, model_path, desc=f"Downloading {model_filename}"):
        return False
    
    # Download class names
    print(f"Downloading class names for {model_key}")
    if not download_file(class_names_url, class_names_path, desc="Downloading class names"):
        return False
    
    print(f"Successfully downloaded {model_key} to {model_path}")
    return True

def main():
    parser = argparse.ArgumentParser(description='Download pretrained ONNX models')
    parser.add_argument('--model', choices=list(AVAILABLE_MODELS.keys()) + ['all'], default='resnet18',
                        help='Model to download (default: resnet18)')
    parser.add_argument('--output', type=str, default='models',
                        help='Output directory (default: models)')
    
    args = parser.parse_args()
    
    # Create base output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Determine which models to download
    if args.model == 'all':
        models_to_download = AVAILABLE_MODELS.keys()
    else:
        models_to_download = [args.model]
    
    # Download each model
    for model_key in models_to_download:
        model_type = 'classification'
        if 'ssd' in model_key or 'yolo' in model_key:
            model_type = 'detection'
        
        # Create model-specific output directory
        model_output_dir = os.path.join(args.output, model_type)
        os.makedirs(model_output_dir, exist_ok=True)
        
        download_model(model_key, model_output_dir)
    
    print("\nDone!")

if __name__ == "__main__":
    main()