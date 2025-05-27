"""
Object detection implementation using TensorRT.
"""

import os
import cv2
import numpy as np
import logging
from .base import TRTBase

# Configure logging
logger = logging.getLogger(__name__)

class ObjectDetector(TRTBase):
    """Object detection model implementation using TensorRT"""
    
    def __init__(self, model_info=None, labels=None, conf_threshold=0.5, input_size=(300, 300)):
        """
        Initialize the object detector with model info and parameters
        
        Args:
            model_info (dict): Dictionary containing model information including paths
            labels (dict): Dictionary containing class labels
            conf_threshold (float): Confidence threshold for detections
            input_size (tuple): Image input size (width, height)
        """
        self.conf_threshold = conf_threshold
        self.input_size = input_size
        
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
            logger.info(f"Using provided labels for object detection")
            
            # Check the format and convert if needed
            if isinstance(self.labels, dict):
                try:
                    # Try to extract class names from labels
                    return [str(label) for label in self.labels.values()]
                except (AttributeError, TypeError):
                    logger.warning("Could not convert labels to class names")
        
        # If no labels provided or conversion failed, try to load from file
        if hasattr(self, 'model_path') and self.model_path:
            class_file = os.path.splitext(self.model_path)[0] + '.txt'
            if os.path.exists(class_file):
                with open(class_file, 'r') as f:
                    class_names = [line.strip() for line in f.readlines()]
                logger.info(f"Loaded {len(class_names)} class names from {class_file}")
                return class_names
        
        # Default COCO classes as fallback
        logger.warning("Using default COCO classes as fallback")
        return [
            'background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
            'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
            'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
            'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
            'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
            'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
    
    def _preprocess_image(self, image):
        """Preprocess the input image for object detection"""
        # Resize image
        input_image = cv2.resize(image, self.input_size)
        
        # Convert to RGB
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0,1]
        input_image = input_image.astype(np.float32) / 255.0
        
        # HWC to CHW format
        input_image = input_image.transpose((2, 0, 1))
        
        # Add batch dimension
        input_image = np.expand_dims(input_image, axis=0)
        
        return input_image.flatten()
    
    def _postprocess_results(self, outputs):
        """Postprocess detection results"""
        # Assuming outputs format is compatible with SSD/YOLO detections:
        # [batch_id, class_id, confidence, x_min, y_min, x_max, y_max]
        detections = outputs[0].reshape(-1, 7)
        
        results = []
        for detection in detections:
            # Skip low confidence detections
            confidence = detection[2]
            if confidence < self.conf_threshold:
                continue
            
            # Extract class ID and bounding box
            class_id = int(detection[1])
            x_min = detection[3]
            y_min = detection[4]
            x_max = detection[5]
            y_max = detection[6]
            
            # Add detection to results
            results.append({
                'class_id': class_id,
                'class_name': self.class_names[class_id] if class_id < len(self.class_names) else f"Class {class_id}",
                'confidence': float(confidence),
                'bbox': [x_min, y_min, x_max, y_max]
            })
        
        return results
    
    def infer(self, image):
        """Run inference on an image"""
        if self.context is None or self.engine is None:
            logger.warning("Engine or context not initialized, cannot run inference")
            return []
            
        return super().infer(image)
    
    def draw_results(self, image, results):
        """Draw detection results on the image"""
        if results is None:
            logger.warning("No results to draw")
            return image
            
        output_image = image.copy()
        
        for detection in results:
            # Extract data
            class_name = detection['class_name']
            confidence = detection['confidence']
            bbox = detection['bbox']
            
            # Convert normalized coordinates [0,1] to image coordinates
            height, width = output_image.shape[:2]
            x_min = int(bbox[0] * width)
            y_min = int(bbox[1] * height)
            x_max = int(bbox[2] * width)
            y_max = int(bbox[3] * height)
            
            # Draw bounding box
            cv2.rectangle(output_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(output_image, label, (x_min, y_min - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return output_image