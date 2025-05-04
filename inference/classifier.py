"""
Image classification implementation using TensorRT.
Compatible with TensorRT 10.10.0.31, with improved label handling.
"""

import os
import cv2
import numpy as np
from .base import TRTBase

class Classifier(TRTBase):
    """Image classification model implementation using TensorRT"""
    
    def __init__(self, model_path, input_size=(224, 224)):
        """Initialize the classifier with model path and parameters"""
        self.input_size = input_size
        
        # Call parent constructor first
        super().__init__(model_path)
        
        # Load class names if available
        self.class_names = self._load_class_names()
    
    def _load_class_names(self):
        """Load class names from a file if available, with improved processing"""
        class_file = os.path.splitext(self.model_path)[0] + '.txt'
        if os.path.exists(class_file):
            with open(class_file, 'r') as f:
                raw_names = [line.strip() for line in f.readlines()]
            
            # Process ImageNet labels to make them more readable
            processed_names = []
            for name in raw_names:
                # For ImageNet labels, they are often in the format "n01440764 tench, Tinca tinca"
                if ' ' in name:
                    # Remove the synset ID and keep only the description
                    parts = name.split(' ', 1)
                    if len(parts) > 1:
                        name = parts[1]
                processed_names.append(name)
            
            print(f"Loaded {len(processed_names)} class names from {class_file}")
            return processed_names
        
        # Default ImageNet classes as fallback (shortened version)
        print(f"Class names file not found: {class_file}")
        print("Using generic class names as fallback")
        return [f"class_{i}" for i in range(1000)]
    
    def _preprocess_image(self, image):
        """Preprocess the input image for classification"""
        # Resize image
        input_image = cv2.resize(image, self.input_size)
        
        # Convert to RGB
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        
        # Normalize using ImageNet mean and std
        mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
        std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
        input_image = (input_image / 255.0 - mean) / std
        
        # HWC to CHW format
        input_image = input_image.transpose((2, 0, 1))
        
        # Add batch dimension
        input_image = np.expand_dims(input_image, axis=0)
        
        return input_image.astype(np.float32).flatten()
    
    def _postprocess_results(self, outputs):
        """Postprocess classification results"""
        # Get the raw scores
        scores = outputs[0]
        
        # Apply softmax to get probabilities
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / exp_scores.sum()
        
        # Get top-5 predictions
        top_indices = np.argsort(probs)[-5:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                'class_id': idx,
                'class_name': self.class_names[idx] if idx < len(self.class_names) else f"Class {idx}",
                'probability': float(probs[idx])
            })
        
        return results
    
    def draw_results(self, image, results):
        """Draw classification results on the image"""
        output_image = image.copy()
        
        # Display top predictions
        y_offset = 30
        for i, result in enumerate(results):
            class_name = result['class_name']
            probability = result['probability']
            label = f"{class_name}: {probability:.2f}"
            
            cv2.putText(output_image, label, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += 30
        
        return output_image