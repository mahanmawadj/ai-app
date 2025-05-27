#!/usr/bin/env python3
"""
Script to download ImageNet class labels and save them as a JSON file.

Usage:
    python download_imagenet_labels.py --output=./labels
"""

import os
import sys
import json
import argparse
import urllib.request
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# URLs for ImageNet labels
IMAGENET_LABELS_URLS = {
    "imagenet1000_clsidx_to_labels": "https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/raw/238f720ff059c1f82f368259d1ca4ffa5dd8f9f5/imagenet1000_clsidx_to_labels.txt",
    "imagenet_class_index": "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
}

def download_imagenet_labels(label_type, output_path):
    """
    Download ImageNet class labels and save them as a JSON file.
    
    Args:
        label_type (str): Type of labels to download.
        output_path (str): Path to save the downloaded labels.
    
    Returns:
        str: Path where the labels were saved.
    """
    if label_type not in IMAGENET_LABELS_URLS:
        available_types = ", ".join(IMAGENET_LABELS_URLS.keys())
        raise ValueError(f"Label type '{label_type}' not supported. Available types: {available_types}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    url = IMAGENET_LABELS_URLS[label_type]
    logger.info(f"Downloading ImageNet labels from {url}...")
    
    try:
        # Download the labels
        response = urllib.request.urlopen(url)
        content = response.read().decode('utf-8')
        
        # Parse and save the labels
        if label_type == "imagenet1000_clsidx_to_labels":
            # Parse text format
            labels_dict = {}
            for line in content.splitlines():
                if ':' in line:
                    idx, label = line.split(':', 1)
                    idx = idx.strip()
                    label = label.strip().strip("'")
                    labels_dict[idx] = label
        else:
            # Already in JSON format
            labels_dict = json.loads(content)
        
        # Save the labels
        output_file = os.path.join(output_path, f"{label_type}.json")
        with open(output_file, 'w') as f:
            json.dump(labels_dict, f, indent=2)
        
        logger.info(f"Labels saved to {output_file}")
        return output_file
    
    except Exception as e:
        logger.error(f"Error downloading labels: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Download ImageNet class labels")
    
    parser.add_argument("--type", type=str, default="imagenet_class_index", 
                        help=f"Type of labels to download. Available types: {', '.join(IMAGENET_LABELS_URLS.keys())}")
    parser.add_argument("--output", type=str, default="./labels", 
                        help="Directory to save the downloaded labels (default: ./labels)")
    
    args = parser.parse_args()
    
    try:
        download_imagenet_labels(
            label_type=args.type, 
            output_path=args.output
        )
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()