"""
Test script for the Classifier class
"""
import os
import sys
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path to import inference module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the Classifier class
from inference import Classifier

def main():
    parser = argparse.ArgumentParser(description='Test Classifier class')
    parser.add_argument('--model', required=True, help='Path to ONNX model file')
    parser.add_argument('--image', required=True, help='Path to input image file')
    parser.add_argument('--size', type=int, default=224, help='Input image size (default: 224)')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model file not found: {args.model}")
    
    # Check if image exists
    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image file not found: {args.image}")
    
    # Create classifier
    print(f"Creating classifier with model: {args.model}")
    classifier = Classifier(args.model, input_size=(args.size, args.size))
    
    # Load image
    print(f"Loading image: {args.image}")
    image = cv2.imread(args.image)
    if image is None:
        raise ValueError(f"Failed to read image: {args.image}")
    
    # Run inference
    print("Running inference...")
    results = classifier.infer(image)
    
    # Draw results
    output_image = classifier.draw_results(image, results)
    
    # Display results
    print("\nClassification results:")
    for i, result in enumerate(results):
        print(f"{i+1}. {result['class_name']}: {result['probability']*100:.2f}%")
    
    # Convert BGR to RGB for display
    output_image_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
    
    # Display the image
    plt.figure(figsize=(10, 8))
    plt.imshow(output_image_rgb)
    plt.axis('off')
    plt.title('Classification Results')
    plt.tight_layout()
    
    # Save the output image
    output_path = "classification_result.jpg"
    cv2.imwrite(output_path, output_image)
    print(f"\nSaved result image to: {output_path}")
    
    # Show the plot
    plt.show()
    
    print("\nDone!")

if __name__ == "__main__":
    main()