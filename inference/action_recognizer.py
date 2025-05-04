"""
Action recognition implementation using TensorRT.
"""

import os
import cv2
import numpy as np
import pycuda.driver as cuda
from .base import TRTBase

class ActionRecognizer(TRTBase):
    """Action recognition model implementation using TensorRT"""
    
    def __init__(self, model_path, input_size=(224, 224), num_frames=16):
        """Initialize the action recognizer with model path and parameters"""
        self.input_size = input_size
        self.num_frames = num_frames
        self.frame_buffer = []
        
        # Call parent constructor
        super().__init__(model_path)
        
        # Load class names
        self.class_names = self._load_class_names()
    
    def _load_class_names(self):
        """Load action class names from a file if available"""
        class_file = os.path.splitext(self.model_path)[0] + '.txt'
        if os.path.exists(class_file):
            with open(class_file, 'r') as f:
                return [line.strip() for line in f.readlines()]
        
        # Default Kinetics actions as fallback (just a few examples)
        return [
            'walking', 'running', 'jumping', 'standing', 'sitting',
            'clapping', 'waving', 'dancing', 'typing', 'eating',
            'drinking', 'reading', 'writing', 'cooking', 'playing'
        ]
    
    def _preprocess_image(self, image):
        """Add frame to buffer and preprocess frames for action recognition"""
        # Add current frame to buffer
        processed_frame = cv2.resize(image, self.input_size)
        processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        
        self.frame_buffer.append(processed_frame)
        
        # Keep only the most recent frames
        if len(self.frame_buffer) > self.num_frames:
            self.frame_buffer.pop(0)
        
        # If we don't have enough frames yet, return None
        if len(self.frame_buffer) < self.num_frames:
            return None
        
        # Stack frames into a clip
        clip = np.stack(self.frame_buffer, axis=0)
        
        # Normalize
        clip = clip.astype(np.float32) / 255.0
        
        # Normalize using mean and std
        mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 1, 3)
        std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 1, 3)
        clip = (clip - mean) / std
        
        # Reshape to model input format (NCTHW or similar)
        # This may need adjustment based on the model's expected input format
        clip = clip.transpose((3, 0, 1, 2))  # THWC -> CTHW
        
        # Add batch dimension
        clip = np.expand_dims(clip, axis=0)
        
        return clip.astype(np.float32).flatten()
    
    def _postprocess_results(self, outputs):
        """Postprocess action recognition results"""
        # Get raw scores
        scores = outputs[0]
        
        # Apply softmax
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / exp_scores.sum()
        
        # Get top-5 predictions
        top_indices = np.argsort(probs)[-5:][::-1]
        
        results = []
        for idx in top_indices:
            if idx < len(self.class_names):
                results.append({
                    'action_id': idx,
                    'action_name': self.class_names[idx],
                    'probability': float(probs[idx])
                })
        
        return results
    
    def infer(self, image):
        """Run inference on an image frame"""
        # Preprocess image
        input_data = self._preprocess_image(image)
        
        # If we don't have enough frames yet, return an empty result
        if input_data is None:
            return []
        
        # Copy input data to device
        cuda.memcpy_htod_async(self.inputs[0]['mem'], input_data, self.stream)
        
        # Run inference
        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle
        )
        
        # Create output arrays
        output_data = []
        for output in self.outputs:
            output_array = np.empty(output['size'], dtype=np.float32)
            cuda.memcpy_dtoh_async(output_array, output['mem'], self.stream)
            output_data.append(output_array)
        
        # Synchronize the stream
        self.stream.synchronize()
        
        # Postprocess results
        return self._postprocess_results(output_data)
    
    def draw_results(self, image, results):
        """Draw action recognition results on the image"""
        output_image = image.copy()
        
        # Display top predictions
        y_offset = 30
        for i, result in enumerate(results):
            action_name = result['action_name']
            probability = result['probability']
            label = f"{action_name}: {probability:.2f}"
            
            cv2.putText(output_image, label, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            y_offset += 30
        
        return output_image