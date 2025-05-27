"""
Image classification implementation using TensorRT.
Compatible with TensorRT 10.10.0.31, with improved label handling.
"""

import os
import cv2
import numpy as np
import logging
from .base import TRTBase

# Configure logging
logger = logging.getLogger(__name__)

class Classifier(TRTBase):
    """Image classification model implementation using TensorRT"""
    
    def __init__(self, model_info=None, labels=None, input_size=(224, 224)):
        """
        Initialize the classifier with model info and parameters
        
        Args:
            model_info (dict): Dictionary containing model information including paths
            labels (dict): Dictionary containing class labels
            input_size (tuple): Image input size (width, height)
        """
        self.input_size = input_size
        self.class_names = None
        
        # Call parent constructor first
        super().__init__(model_info, labels)
        
        # Load class names if available
        self.class_names = self._load_class_names()
    
    def update_model(self, model_info, labels):
        """
        Update the model with new model info and labels
        
        Args:
            model_info (dict): Dictionary containing model information including paths
            labels (dict): Dictionary containing class labels
        
        Returns:
            bool: True if successful, False otherwise
        """
        result = super().update_model(model_info, labels)
        
        if result:
            # Reload class names
            self.class_names = self._load_class_names()
        
        return result
    
    def _load_class_names(self):
        """Load class names from labels or from a file if available"""
        # First, try to use provided labels
        if self.labels is not None:
            logger.info(f"Using provided labels with {len(self.labels)} classes")
            
            # Check the format and convert if needed
            if isinstance(self.labels, dict):
                # For ImageNet format: {"0": ["n01440764", "tench"], "1": [...]}
                class_names = []
                for i in range(1000):  # Assume max 1000 classes
                    idx = str(i)
                    if idx in self.labels:
                        # Get the readable name, which is typically the second element
                        if isinstance(self.labels[idx], list) and len(self.labels[idx]) > 1:
                            class_names.append(self.labels[idx][1])
                        else:
                            class_names.append(str(self.labels[idx]))
                    else:
                        # If we've run out of labels, stop
                        break
                
                if class_names:
                    return class_names
            
            # If the labels are not in the expected format, try to convert them
            try:
                return [str(label) for label in self.labels.values()]
            except (AttributeError, TypeError):
                logger.warning("Could not convert labels to class names")
        
        # If no labels provided or conversion failed, try to load from file
        if self.model_path:
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
                
                logger.info(f"Loaded {len(processed_names)} class names from {class_file}")
                return processed_names
        
        # Default ImageNet classes as fallback (shortened version)
        logger.warning("Using generic class names as fallback")
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
                'class_id': int(idx),
                'class_name': self.class_names[idx] if idx < len(self.class_names) else f"Class {idx}",
                'probability': float(probs[idx])
            })
        
        return results
    
    def draw_results(self, image, results):
        """Draw classification results on the image"""
        if results is None:
            logger.warning("No results to draw")
            return image
            
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