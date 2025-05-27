"""
Pose estimation implementation using TensorRT.
"""

import cv2
import numpy as np
import logging
from .base import TRTBase

# Configure logging
logger = logging.getLogger(__name__)

class PoseEstimator(TRTBase):
    """Pose estimation model implementation using TensorRT"""
    
    def __init__(self, model_info=None, labels=None, input_size=(256, 256), conf_threshold=0.3):
        """
        Initialize the pose estimator with model info and parameters
        
        Args:
            model_info (dict): Dictionary containing model information including paths
            labels (dict): Dictionary containing keypoint labels
            input_size (tuple): Image input size (width, height)
            conf_threshold (float): Confidence threshold for keypoints
        """
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        
        # Standard COCO keypoint names
        self.keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        
        # Load custom keypoint names from labels if available
        if labels is not None and isinstance(labels, dict):
            try:
                keypoint_names = []
                for i in range(len(self.keypoint_names)):
                    key = str(i)
                    if key in labels:
                        keypoint_names.append(str(labels[key]))
                if keypoint_names:
                    logger.info(f"Using custom keypoint names from labels: {keypoint_names}")
                    self.keypoint_names = keypoint_names
            except Exception as e:
                logger.warning(f"Error loading keypoint names from labels: {e}")
        
        # Call parent constructor
        super().__init__(model_info, labels)
    
    def update_model(self, model_info, labels):
        """
        Update the model with new model info and labels
        
        Args:
            model_info (dict): Dictionary containing model information including paths
            labels (dict): Dictionary containing keypoint labels
        
        Returns:
            bool: True if successful, False otherwise
        """
        # Update keypoint names if provided
        if labels is not None and isinstance(labels, dict):
            try:
                keypoint_names = []
                for i in range(len(self.keypoint_names)):
                    key = str(i)
                    if key in labels:
                        keypoint_names.append(str(labels[key]))
                if keypoint_names:
                    logger.info(f"Updating keypoint names from labels")
                    self.keypoint_names = keypoint_names
            except Exception as e:
                logger.warning(f"Error updating keypoint names from labels: {e}")
        
        # Call parent update method
        return super().update_model(model_info, labels)
    
    def _preprocess_image(self, image):
        """Preprocess the input image for pose estimation"""
        # Resize image
        input_image = cv2.resize(image, self.input_size)
        
        # Convert to RGB
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0,1]
        input_image = input_image.astype(np.float32) / 255.0
        
        # Normalize using model mean and std
        mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
        std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
        input_image = (input_image - mean) / std
        
        # HWC to CHW format
        input_image = input_image.transpose((2, 0, 1))
        
        # Add batch dimension
        input_image = np.expand_dims(input_image, axis=0)
        
        return input_image.astype(np.float32).flatten()
    
    def _postprocess_results(self, outputs):
        """Postprocess pose estimation results"""
        # Assuming output format is heatmaps for keypoints and offsets
        # This implementation is simplified and may need adjustment based on the model
        
        # Example: output[0] contains keypoint heatmaps, output[1] contains offsets
        heatmaps = outputs[0].reshape((17, self.input_size[1] // 4, self.input_size[0] // 4))
        
        keypoints = []
        confidences = []
        
        # Process each keypoint heatmap
        for i in range(17):
            # Find the location of maximum confidence
            heatmap = heatmaps[i]
            max_val = np.max(heatmap)
            
            if max_val >= self.conf_threshold:
                idx = np.argmax(heatmap)
                y, x = np.unravel_index(idx, heatmap.shape)
                
                # Convert to original image coordinates (normalized [0,1])
                x_norm = float(x) / (self.input_size[0] // 4)
                y_norm = float(y) / (self.input_size[1] // 4)
                
                keypoints.append((x_norm, y_norm))
                confidences.append(float(max_val))
            else:
                keypoints.append((0, 0))
                confidences.append(0)
        
        # Combine results
        results = []
        for i in range(min(17, len(self.keypoint_names))):
            if confidences[i] >= self.conf_threshold:
                results.append({
                    'keypoint_id': i,
                    'keypoint_name': self.keypoint_names[i] if i < len(self.keypoint_names) else f"Keypoint {i}",
                    'position': keypoints[i],
                    'confidence': confidences[i]
                })
        
        return results
    
    def infer(self, image):
        """Run inference on an image"""
        if self.context is None or self.engine is None:
            logger.warning("Engine or context not initialized, cannot run inference")
            return []
            
        return super().infer(image)
    
    def draw_results(self, image, results):
        """Draw pose estimation results on the image"""
        if results is None:
            logger.warning("No results to draw")
            return image
            
        output_image = image.copy()
        height, width = output_image.shape[:2]
        
        # Define connections between keypoints for drawing limbs
        limbs = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Face
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
            (5, 11), (6, 12), (11, 12),  # Torso
            (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
        ]
        
        # Create keypoints dictionary for easier access
        keypoints_dict = {result['keypoint_id']: result for result in results}
        
        # Draw keypoints
        for result in results:
            keypoint_id = result['keypoint_id']
            x_norm, y_norm = result['position']
            x = int(x_norm * width)
            y = int(y_norm * height)
            
            # Draw circle at keypoint
            cv2.circle(output_image, (x, y), 5, (0, 255, 0), -1)
            
            # Draw keypoint name
            cv2.putText(output_image, result['keypoint_name'], (x + 5, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Draw limbs
        for limb in limbs:
            # Check if both keypoints of the limb are detected
            if limb[0] in keypoints_dict and limb[1] in keypoints_dict:
                # Get keypoint positions
                x1_norm, y1_norm = keypoints_dict[limb[0]]['position']
                x2_norm, y2_norm = keypoints_dict[limb[1]]['position']
                
                # Convert to image coordinates
                x1 = int(x1_norm * width)
                y1 = int(y1_norm * height)
                x2 = int(x2_norm * width)
                y2 = int(y2_norm * height)
                
                # Draw line between keypoints
                cv2.line(output_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        return output_image