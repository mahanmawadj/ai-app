"""
Semantic segmentation implementation using TensorRT.
"""

import os
import cv2
import numpy as np
import logging
from .base import TRTBase

# Configure logging
logger = logging.getLogger(__name__)

class Segmenter(TRTBase):
    """Semantic segmentation model implementation using TensorRT"""
    
    def __init__(self, model_info=None, labels=None, input_size=(512, 512)):
        """
        Initialize the segmenter with model info and parameters
        
        Args:
            model_info (dict): Dictionary containing model information including paths
            labels (dict): Dictionary containing class labels
            input_size (tuple): Image input size (width, height)
        """
        self.input_size = input_size
        
        # Call parent constructor
        super().__init__(model_info, labels)
        
        # Load class names and colors
        self.class_names, self.colors = self._load_class_info()
    
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
            # Reload class names and colors
            self.class_names, self.colors = self._load_class_info()
        
        return result
    
    def _load_class_info(self):
        """Load class names and colors from labels or files if available"""
        class_names = []
        
        # First, try to use provided labels
        if self.labels is not None:
            logger.info("Using provided labels for segmentation")
            
            # Check the format and convert if needed
            if isinstance(self.labels, dict):
                try:
                    # Try to extract class names from labels
                    class_names = [str(label) for label in self.labels.values()]
                except (AttributeError, TypeError):
                    logger.warning("Could not convert labels to class names")
        
        # If no labels provided or conversion failed, try to load from file
        if not class_names and hasattr(self, 'model_path') and self.model_path:
            class_file = os.path.splitext(self.model_path)[0] + '.txt'
            if os.path.exists(class_file):
                with open(class_file, 'r') as f:
                    class_names = [line.strip() for line in f.readlines()]
                logger.info(f"Loaded {len(class_names)} class names from {class_file}")
        
        # Default classes as fallback
        if not class_names:
            logger.warning("Using default segmentation classes as fallback")
            class_names = [
                'background', 'person', 'bicycle', 'car', 'motorcycle', 
                'airplane', 'bus', 'train', 'truck', 'boat'
            ]
        
        # Generate random colors for each class
        colors = []
        np.random.seed(42)  # For reproducible colors
        for i in range(len(class_names)):
            colors.append((
                np.random.randint(0, 255),
                np.random.randint(0, 255),
                np.random.randint(0, 255)
            ))
        
        return class_names, colors
    
    def _preprocess_image(self, image):
        """Preprocess the input image for segmentation"""
        # Resize image
        input_image = cv2.resize(image, self.input_size)
        
        # Convert to RGB
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0,1]
        input_image = input_image.astype(np.float32) / 255.0
        
        # Normalize using mean and std
        mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
        std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
        input_image = (input_image - mean) / std
        
        # HWC to CHW format
        input_image = input_image.transpose((2, 0, 1))
        
        # Add batch dimension
        input_image = np.expand_dims(input_image, axis=0)
        
        return input_image.astype(np.float32).flatten()
    
    def _postprocess_results(self, outputs):
        """Postprocess segmentation results"""
        # Assuming output is a segmentation mask
        # Shape: [batch_size, num_classes, height, width]
        mask = outputs[0].reshape((len(self.class_names), self.input_size[1], self.input_size[0]))
        
        # Get the class with highest probability for each pixel
        class_mask = np.argmax(mask, axis=0)
        
        return class_mask
    
    def infer(self, image):
        """Run inference on an image"""
        if self.context is None or self.engine is None:
            logger.warning("Engine or context not initialized, cannot run inference")
            return None
            
        return super().infer(image)
    
    def draw_results(self, image, class_mask):
        """Draw segmentation results on the image"""
        if class_mask is None:
            logger.warning("No segmentation mask to draw")
            return image
            
        # Resize class mask to match input image
        height, width = image.shape[:2]
        resized_mask = cv2.resize(
            class_mask.astype(np.uint8), 
            (width, height), 
            interpolation=cv2.INTER_NEAREST
        )
        
        # Create a colored segmentation mask
        colored_mask = np.zeros((height, width, 3), dtype=np.uint8)
        for class_id in range(len(self.class_names)):
            colored_mask[resized_mask == class_id] = self.colors[class_id]
        
        # Blend with original image
        alpha = 0.5
        output_image = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)
        
        # Add legend
        y_offset = 30
        for i, class_name in enumerate(self.class_names):
            # Skip background class
            if i == 0 and class_name.lower() == 'background':
                continue
                
            # Draw color swatch and class name
            color = self.colors[i]
            cv2.rectangle(output_image, (width - 200, y_offset - 15), 
                         (width - 180, y_offset + 5), color, -1)
            cv2.putText(output_image, class_name, (width - 175, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            y_offset += 25
        
        return output_image