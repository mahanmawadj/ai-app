"""
Script to download and format ImageNet labels
"""

import os
import requests
import argparse
from tqdm import tqdm

# URLs for ImageNet class names
SYNSET_URL = "https://raw.githubusercontent.com/tensorflow/models/master/research/slim/datasets/imagenet_lsvrc_2015_synsets.txt"
METADATA_URL = "https://raw.githubusercontent.com/tensorflow/models/master/research/slim/datasets/imagenet_metadata.txt"

def download_file(url, output_path):
    """Download a file from a URL with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(output_path, 'wb') as f, tqdm(
        desc=os.path.basename(output_path),
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            bar.update(size)

def create_imagenet_labels_file(output_path):
    """Create ImageNet labels file by combining synsets and metadata"""
    # Create temp directory
    os.makedirs("temp", exist_ok=True)
    
    # Download synsets file
    synsets_path = os.path.join("temp", "synsets.txt")
    print(f"Downloading ImageNet synsets...")
    download_file(SYNSET_URL, synsets_path)
    
    # Download metadata file
    metadata_path = os.path.join("temp", "metadata.txt")
    print(f"Downloading ImageNet metadata...")
    download_file(METADATA_URL, metadata_path)
    
    # Read synsets (list of wordnet IDs in order)
    with open(synsets_path, 'r') as f:
        synsets = [line.strip() for line in f.readlines()]
    
    # Read metadata (wordnet ID to human-readable labels)
    metadata = {}
    with open(metadata_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                metadata[parts[0]] = parts[1]
    
    # Create labels file with format: "index label"
    print(f"Creating labels file: {output_path}")
    with open(output_path, 'w') as f:
        for i, synset in enumerate(synsets):
            label = metadata.get(synset, f"Unknown ({synset})")
            f.write(f"{label}\n")
    
    print(f"Created labels file with {len(synsets)} classes")
    
    # Clean up temp files
    os.remove(synsets_path)
    os.remove(metadata_path)
    os.rmdir("temp")

def main():
    parser = argparse.ArgumentParser(description='Download and format ImageNet labels')
    parser.add_argument('--output', type=str, default='models/classification/resnet18.txt',
                        help='Output path for labels file')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Create labels file
    create_imagenet_labels_file(args.output)
    
    print("Done!")

if __name__ == "__main__":
    main()