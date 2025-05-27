#!/usr/bin/env python3
"""
Model handler module for managing models with TensorRT compatibility.
Handles model loading, checking, conversion, and switching across various model types.
"""

import os
import threading
import logging
import json
import torch
import numpy as np
from pathlib import Path
import importlib.util

# Import model downloading utilities
from models.download_models import download_model, AVAILABLE_MODELS
from models.download_imagenet_labels import download_imagenet_labels, IMAGENET_LABELS_URLS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Define model types and their default models
MODEL_TYPES = {
    "classification": {
        "default_model": "resnet18",
        "available_models": ["resnet18", "resnet50", "vgg16"],
        "label_type": "imagenet_class_index"
    },
    "detection": {
        "default_model": "ssd300",
        "available_models": ["ssd300", "fasterrcnn"],
        "label_type": "coco_labels"
    },
    "segmentation": {
        "default_model": "deeplabv3",
        "available_models": ["deeplabv3", "fcn"],
        "label_type": "coco_labels"
    },
    "pose": {
        "default_model": "openpose",
        "available_models": ["openpose", "hrnet"],
        "label_type": "coco_keypoints"
    },
    "action": {
        "default_model": "i3d",
        "available_models": ["i3d", "slowfast"],
        "label_type": "kinetics_labels"
    },
    # Placeholders for future model types
    "llm": {
        "default_model": None,
        "available_models": [],
        "label_type": None
    },
    "voice": {
        "default_model": None,
        "available_models": [],
        "label_type": None
    }
}


class ModelHandler:
    """
    Handler class for managing models with TensorRT compatibility.
    Provides functionality for loading, checking, converting, and switching models.
    """
    
    def __init__(self, model_type=None, model_name=None, models_dir="./models", labels_dir="./labels"):
        """
        Initialize the model handler.
    
        Args:
            model_type (str): Type of model to load (classification, detection, etc.)
            model_name (str): Name of the specific model to load (if None, uses default for type)
            models_dir (str): Directory for storing models
            labels_dir (str): Directory for storing label files
        """
        self.models_dir = models_dir
        self.labels_dir = labels_dir
    
        # Allow initialization without a model type for lazy loading
        if model_type is None:
            logger.info("Initializing ModelHandler without a default model (lazy loading mode)")
            self.current_model_type = None
            self.current_model_name = None
            self.current_model_info = None
            self.labels = None
        else:
            # Validate model type
            if model_type not in MODEL_TYPES:
                logger.error(f"Invalid model type: {model_type}. Available types: {', '.join(MODEL_TYPES.keys())}")
                raise ValueError(f"Invalid model type: {model_type}")
        
            self.current_model_type = model_type
        
            # Get default model for this type if not specified
            if model_name is None:
                model_name = MODEL_TYPES[model_type]["default_model"]
                if model_name is None:
                    logger.warning(f"No default model available for type {model_type}")
                    self.current_model_name = None
                    self.current_model_info = None
                    self.labels = None
                    return
        
            self.current_model_name = model_name
            self.current_model_info = None
            self.labels = None
        
            # Load the initial model information if available
            if self.current_model_name:
                self._load_model(self.current_model_name, self.current_model_type)
        
            # Load labels if applicable
            label_type = MODEL_TYPES[model_type]["label_type"]
            if label_type:
                self._load_labels(label_type)
    
        # Lock for thread safety during model loading
        self.model_lock = threading.Lock()
        self.is_model_loading = False
    
        # TensorRT model cache
        self.trt_models = {}
    
        # Check for TensorRT availability
        self.has_tensorrt = self._check_tensorrt()
    
        # Check for ONNX availability
        self.has_onnx = self._check_onnx()
    
    def _check_tensorrt(self):
        """
        Check if TensorRT is available.
        
        Returns:
            bool: True if TensorRT is available, False otherwise
        """
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit
            logger.info("TensorRT is available")
            return True
        except ImportError:
            logger.warning("TensorRT is not available. Some functionality may be limited.")
            return False
    
    def _check_onnx(self):
        """
        Check if ONNX is available.
        
        Returns:
            bool: True if ONNX is available, False otherwise
        """
        try:
            import onnx
            import onnxruntime
            logger.info("ONNX is available")
            return True
        except ImportError:
            logger.warning("ONNX is not available. Some functionality may be limited.")
            return False
    
    def _load_model(self, model_name, model_type):
        """
        Load a model by name and type.
        This method only prepares the model information, the actual loading
        is done by the inference modules.
        
        Args:
            model_name (str): Name of the model to load
            model_type (str): Type of the model to load
        
        Returns:
            bool: True if successful, False otherwise
        """
        with self.model_lock:
            self.is_model_loading = True
            
            try:
                # Check if model type is valid
                if model_type not in MODEL_TYPES:
                    logger.error(f"Model type '{model_type}' not supported. Available types: {', '.join(MODEL_TYPES.keys())}")
                    self.is_model_loading = False
                    return False
                
                # Check if model is available for this type
                type_info = MODEL_TYPES[model_type]
                if model_name not in type_info["available_models"] and model_name not in AVAILABLE_MODELS:
                    logger.error(f"Model '{model_name}' not supported for type '{model_type}'. "
                                f"Available models: {', '.join(type_info['available_models'])}")
                    self.is_model_loading = False
                    return False
                
                # Ensure model file is available
                model_path = self._ensure_model_available(model_name, model_type)
                if not model_path:
                    logger.error(f"Failed to get model path for {model_name}")
                    self.is_model_loading = False
                    return False
                
                # Determine paths for different formats
                model_dir = os.path.join(self.models_dir, model_type)
                pytorch_path = model_path
                onnx_path = os.path.join(model_dir, f"{model_name}.onnx")
                trt_path = os.path.join(model_dir, f"{model_name}.engine")
                
                # Create model info dictionary
                self.current_model_info = {
                    'name': model_name,
                    'type': model_type,
                    'path': model_path,
                    'pytorch_path': pytorch_path,
                    'onnx_path': onnx_path,
                    'trt_path': trt_path,
                    'has_pytorch': os.path.exists(pytorch_path),
                    'has_onnx': os.path.exists(onnx_path),
                    'has_trt': os.path.exists(trt_path),
                }
                
                self.current_model_name = model_name
                self.current_model_type = model_type
                
                logger.info(f"Model {model_name} of type {model_type} info prepared")
                return True
            
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                return False
            
            finally:
                self.is_model_loading = False
    
    def _load_labels(self, label_type):
        """
        Load labels for a specific type.
        
        Args:
            label_type (str): Type of labels to load
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Skip if label type is None
            if label_type is None:
                logger.info("No labels required for this model type")
                self.labels = None
                return True
                
            # Ensure labels file is available
            labels_path = self._ensure_labels_available(label_type)
            
            # Skip if labels path is None
            if labels_path is None:
                logger.warning(f"No labels available for type: {label_type}")
                self.labels = None
                return False
                
            # Load the labels
            with open(labels_path, 'r') as f:
                self.labels = json.load(f)
            
            logger.info(f"Labels loaded successfully: {len(self.labels) if isinstance(self.labels, dict) else 'unknown'} entries")
            return True
        
        except Exception as e:
            logger.error(f"Error loading labels: {e}")
            self.labels = None
            return False
    
    def _ensure_model_available(self, model_name, model_type):
        """
        Check if the model is available locally, and if not, download it.
        
        Args:
            model_name (str): Name of the model to check/download
            model_type (str): Type of the model
        
        Returns:
            str: Path to the model file or None if not available
        """
        try:
            # Create model type directory if it doesn't exist
            model_dir = os.path.join(self.models_dir, model_type)
            os.makedirs(model_dir, exist_ok=True)
            
            # Check if model exists in the type-specific directory
            model_path = os.path.join(model_dir, f"{model_name}.pth")
            
            if not os.path.exists(model_path):
                logger.info(f"Model {model_name} for type {model_type} not found locally. Downloading...")
                
                if model_name in AVAILABLE_MODELS:
                    downloaded_path = download_model(model_name, self.models_dir, model_type)
                    if downloaded_path:
                        logger.info(f"Model {model_name} downloaded to {downloaded_path}")
                        return downloaded_path
                    else:
                        logger.error(f"Failed to download model {model_name}")
                        return None
                else:
                    logger.error(f"Model {model_name} not available for download")
                    return None
            else:
                logger.info(f"Model {model_name} for type {model_type} found at {model_path}")
                return model_path
            
        except Exception as e:
            logger.error(f"Error ensuring model availability: {e}")
            return None
    
    def _ensure_labels_available(self, label_type):
        """
        Check if the labels are available locally, and if not, download them.
        
        Args:
            label_type (str): Type of labels to check/download
        
        Returns:
            str: Path to the labels file or None if not available
        """
        try:
            # Create labels directory if it doesn't exist
            os.makedirs(self.labels_dir, exist_ok=True)
            
            # Check if labels exist
            labels_path = os.path.join(self.labels_dir, f"{label_type}.json")
            
            if not os.path.exists(labels_path):
                logger.info(f"Labels for {label_type} not found locally. Downloading...")
                
                if label_type in IMAGENET_LABELS_URLS:
                    downloaded_path = download_imagenet_labels(label_type, self.labels_dir)
                    if downloaded_path:
                        logger.info(f"Labels for {label_type} downloaded to {downloaded_path}")
                        return downloaded_path
                    else:
                        logger.error(f"Failed to download labels for {label_type}")
                        return None
                else:
                    logger.error(f"Labels for {label_type} not available for download")
                    return None
            else:
                logger.info(f"Labels for {label_type} found at {labels_path}")
                return labels_path
                
        except Exception as e:
            logger.error(f"Error ensuring labels availability: {e}")
            return None
    
    def change_model(self, model_name, model_type=None):
        """
        Change the current model.
        
        Args:
            model_name (str): Name of the model to change to
            model_type (str): Type of the model (if None, uses current type)
        
        Returns:
            tuple: (success, error_message)
        """
        # If model type not specified, use current type
        if model_type is None:
            model_type = self.current_model_type
        
        # Check if model is already being loaded
        if self.is_model_loading:
            return False, "Another model is currently being loaded. Please try again later."
        
        # Check if model type is valid
        if model_type not in MODEL_TYPES:
            return False, f"Model type '{model_type}' not supported. Available types: {', '.join(MODEL_TYPES.keys())}"
        
        # Check if model name is valid for this type
        type_info = MODEL_TYPES[model_type]
        if (model_name not in type_info["available_models"] and 
            model_name not in AVAILABLE_MODELS):
            return False, f"Model '{model_name}' not supported for type '{model_type}'. Available models: {', '.join(type_info['available_models'])}"
        
        # Check if model is already loaded
        if model_name == self.current_model_name and model_type == self.current_model_type:
            return True, None
        
        # Load the new model
        success = self._load_model(model_name, model_type)
        
        # If model type changed, load appropriate labels
        if success and model_type != self.current_model_type:
            label_type = MODEL_TYPES[model_type]["label_type"]
            if label_type:
                self._load_labels(label_type)
        
        if success:
            return True, None
        else:
            return False, f"Failed to load model {model_name} of type {model_type}"
    
    def get_model_info(self):
        """
        Get the current model information.
        
        Returns:
            dict: The current model information
        """
        return self.current_model_info
    
    def get_labels(self):
        """
        Get the loaded labels.
        
        Returns:
            dict: The loaded labels
        """
        return self.labels
    
    def get_available_model_types(self):
        """
        Get all available model types.
        
        Returns:
            list: List of available model types
        """
        return list(MODEL_TYPES.keys())
    
    def get_available_models(self, model_type=None):
        """
        Get all available models for a specific type.
        
        Args:
            model_type (str): Type of models to get (if None, uses current type)
        
        Returns:
            list: List of available models
        """
        if model_type is None:
            model_type = self.current_model_type
        
        if model_type not in MODEL_TYPES:
            logger.error(f"Invalid model type: {model_type}")
            return []
        
        return MODEL_TYPES[model_type]["available_models"]
    
    def register_trt_model(self, model_name, trt_model):
        """
        Register a TensorRT model in the cache.
        
        Args:
            model_name (str): Name of the model
            trt_model: TensorRT model instance
        """
        cache_key = f"{self.current_model_type}_{model_name}"
        self.trt_models[cache_key] = trt_model
        logger.info(f"TensorRT model {model_name} of type {self.current_model_type} registered in cache")
    
    def get_trt_model(self, model_name, model_type=None):
        """
        Get a TensorRT model from the cache.
        
        Args:
            model_name (str): Name of the model
            model_type (str): Type of the model (if None, uses current type)
        
        Returns:
            The TensorRT model instance or None if not found
        """
        if model_type is None:
            model_type = self.current_model_type
            
        cache_key = f"{model_type}_{model_name}"
        return self.trt_models.get(cache_key)
    
    def load_engine(self, engine_path):
        """
        Load a TensorRT engine file.
        
        Args:
            engine_path (str): Path to the engine file
        
        Returns:
            TensorRT engine or None if loading failed
        """
        if not self.has_tensorrt:
            logger.error("TensorRT is not available")
            return None
        
        # Check if engine_path is None
        if engine_path is None:
            logger.error("Engine path is None")
            return None
            
        # Verify engine_path is a string, not a dict
        if not isinstance(engine_path, (str, bytes, os.PathLike)):
            logger.error(f"Invalid engine path type: {type(engine_path)}. Must be string, bytes, or os.PathLike.")
            return None
            
        # Check if file exists
        if not os.path.exists(engine_path):
            logger.error(f"Engine file not found: {engine_path}")
            return None
            
        logger.info(f"Loading TensorRT engine: {engine_path}")
        
        try:
            import tensorrt as trt
            
            # Create logger
            trt_logger = trt.Logger(trt.Logger.INFO)
            
            with open(engine_path, "rb") as f, trt.Runtime(trt_logger) as runtime:
                engine = runtime.deserialize_cuda_engine(f.read())
                
            if not engine:
                logger.error("Failed to load engine")
                return None
                
            logger.info("Engine loaded successfully")
            return engine
        except Exception as e:
            logger.error(f"Error loading engine: {e}")
            return None
    
    def pytorch_to_onnx(self, model_name=None, model_type=None, input_shape=None):
        """
        Convert a PyTorch model to ONNX format.
        
        Args:
            model_name (str): Name of the model (if None, uses current model)
            model_type (str): Type of the model (if None, uses current type)
            input_shape (tuple): Input shape for the model (if None, uses default for model type)
        
        Returns:
            str: Path to the ONNX model or None if conversion failed
        """
        if not self.has_onnx:
            logger.error("ONNX is not available")
            return None
        
        # Use current model and type if not specified
        if model_name is None:
            model_name = self.current_model_name
        if model_type is None:
            model_type = self.current_model_type
            
        # Set default input shape based on model type if not provided
        if input_shape is None:
            if model_type == "classification":
                input_shape = (1, 3, 224, 224)
            elif model_type == "detection":
                input_shape = (1, 3, 300, 300)
            elif model_type == "segmentation":
                input_shape = (1, 3, 513, 513)
            elif model_type == "pose":
                input_shape = (1, 3, 368, 368)
            elif model_type == "action":
                input_shape = (1, 3, 16, 224, 224)  # (batch, channels, frames, height, width)
            else:
                logger.error(f"No default input shape for model type: {model_type}")
                return None
        
        try:
            # Get the model directory for this type
            model_dir = os.path.join(self.models_dir, model_type)
            os.makedirs(model_dir, exist_ok=True)
            
            # Get the PyTorch model path
            pytorch_path = os.path.join(model_dir, f"{model_name}.pth")
            if not os.path.exists(pytorch_path):
                logger.error(f"PyTorch model not found: {pytorch_path}")
                return None
            
            # Get the ONNX model path
            onnx_path = os.path.join(model_dir, f"{model_name}.onnx")
            
            # Check if ONNX model already exists
            if os.path.exists(onnx_path):
                logger.info(f"ONNX model already exists: {onnx_path}")
                return onnx_path
            
            # Load the PyTorch model
            if model_name in AVAILABLE_MODELS:
                model_fn = AVAILABLE_MODELS[model_name]
                model = model_fn(pretrained=False)
                model.load_state_dict(torch.load(pytorch_path, map_location=torch.device('cpu')))
                model.eval()
                
                # Create a dummy input
                dummy_input = torch.randn(input_shape)
                
                # Export to ONNX
                logger.info(f"Exporting PyTorch model to ONNX: {onnx_path}")
                torch.onnx.export(
                    model,
                    dummy_input,
                    onnx_path,
                    export_params=True,
                    opset_version=12,
                    do_constant_folding=True,
                    input_names=['input'],
                    output_names=['output'],
                    dynamic_axes={
                        'input': {0: 'batch_size'},
                        'output': {0: 'batch_size'}
                    }
                )
                
                # Verify the ONNX model
                import onnx
                onnx_model = onnx.load(onnx_path)
                onnx.checker.check_model(onnx_model)
                
                logger.info(f"PyTorch model successfully converted to ONNX: {onnx_path}")
                
                # Update the current model info if this is the current model
                if self.current_model_name == model_name and self.current_model_type == model_type:
                    self.current_model_info['has_onnx'] = True
                    self.current_model_info['onnx_path'] = onnx_path
                
                return onnx_path
            else:
                logger.error(f"Model {model_name} not available in AVAILABLE_MODELS")
                return None
        
        except Exception as e:
            logger.error(f"Error converting PyTorch model to ONNX: {e}")
            return None
    
    def onnx_to_tensorrt(self, model_name=None, model_type=None, precision='fp32'):
        """
        Convert an ONNX model to TensorRT format.
        
        Args:
            model_name (str): Name of the model (if None, uses current model)
            model_type (str): Type of the model (if None, uses current type)
            precision (str): Precision to use for the TensorRT model ('fp32', 'fp16', or 'int8')
        
        Returns:
            str: Path to the TensorRT engine or None if conversion failed
        """
        if not self.has_tensorrt:
            logger.error("TensorRT is not available")
            return None
        
        # Use current model and type if not specified
        if model_name is None:
            model_name = self.current_model_name
        if model_type is None:
            model_type = self.current_model_type
        
        try:
            import tensorrt as trt
            
            # Get the model directory for this type
            model_dir = os.path.join(self.models_dir, model_type)
            os.makedirs(model_dir, exist_ok=True)
            
            # Get the ONNX model path
            onnx_path = os.path.join(model_dir, f"{model_name}.onnx")
            if not os.path.exists(onnx_path):
                # Try to convert from PyTorch to ONNX first
                onnx_path = self.pytorch_to_onnx(model_name, model_type)
                if not onnx_path:
                    logger.error(f"ONNX model not found and conversion failed")
                    return None
            
            # Get the TensorRT engine path
            engine_path = os.path.join(model_dir, f"{model_name}.engine")
            
            # Check if TensorRT engine already exists
            if os.path.exists(engine_path):
                logger.info(f"TensorRT engine already exists: {engine_path}")
                return engine_path
            
            # Create TensorRT builder and config
            logger.info(f"Creating TensorRT engine from ONNX model: {onnx_path}")
            
            trt_logger = trt.Logger(trt.Logger.INFO)
            builder = trt.Builder(trt_logger)
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            parser = trt.OnnxParser(network, trt_logger)
            
            # Parse the ONNX model
            with open(onnx_path, 'rb') as model_file:
                if not parser.parse(model_file.read()):
                    for error in range(parser.num_errors):
                        logger.error(f"ONNX parsing error: {parser.get_error(error)}")
                    return None
            
            # Create config
            config = builder.create_builder_config()
            config.max_workspace_size = 1 << 30  # 1GB
            
            # Set precision
            if precision == 'fp16' and builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                logger.info("Using FP16 precision")
            elif precision == 'int8' and builder.platform_has_fast_int8:
                config.set_flag(trt.BuilderFlag.INT8)
                logger.info("Using INT8 precision")
            else:
                logger.info("Using FP32 precision")
            
            # Build the engine
            engine = builder.build_engine(network, config)
            if not engine:
                logger.error("Failed to build TensorRT engine")
                return None
            
            # Save the engine
            with open(engine_path, 'wb') as f:
                f.write(engine.serialize())
            
            logger.info(f"TensorRT engine successfully created: {engine_path}")
            
            # Update the current model info if this is the current model
            if self.current_model_name == model_name and self.current_model_type == model_type:
                self.current_model_info['has_trt'] = True
                self.current_model_info['trt_path'] = engine_path
            
            return engine_path
        
        except Exception as e:
            logger.error(f"Error converting ONNX model to TensorRT: {e}")
            return None
    
    def ensure_model_formats(self, model_name=None, model_type=None, formats=None):
        """
        Ensure that the model is available in the specified formats.
        
        Args:
            model_name (str): Name of the model (if None, uses current model)
            model_type (str): Type of the model (if None, uses current type)
            formats (list): List of formats to ensure ('pytorch', 'onnx', 'tensorrt')
                           If None, ensure all available formats
        
        Returns:
            dict: Dictionary of model paths for each format
        """
        # Use current model and type if not specified
        if model_name is None:
            model_name = self.current_model_name
        if model_type is None:
            model_type = self.current_model_type
        
        if formats is None:
            formats = ['pytorch']
            if self.has_onnx:
                formats.append('onnx')
            if self.has_tensorrt:
                formats.append('tensorrt')
        
        result = {}
        
        # Get the model directory for this type
        model_dir = os.path.join(self.models_dir, model_type)
        os.makedirs(model_dir, exist_ok=True)
        
        # Ensure the PyTorch model is available
        if 'pytorch' in formats:
            pytorch_path = self._ensure_model_available(model_name, model_type)
            if pytorch_path:
                result['pytorch'] = pytorch_path
        
        # Ensure the ONNX model is available
        if 'onnx' in formats and self.has_onnx:
            onnx_path = os.path.join(model_dir, f"{model_name}.onnx")
            if not os.path.exists(onnx_path):
                onnx_path = self.pytorch_to_onnx(model_name, model_type)
            if onnx_path:
                result['onnx'] = onnx_path
        
        # Ensure the TensorRT engine is available
        if 'tensorrt' in formats and self.has_tensorrt:
            engine_path = os.path.join(model_dir, f"{model_name}.engine")
            if not os.path.exists(engine_path):
                engine_path = self.onnx_to_tensorrt(model_name, model_type)
            if engine_path:
                result['tensorrt'] = engine_path
        
        return result
    
    def test_trt_engine(self, model_name=None, model_type=None):
        """
        Test if a TensorRT engine can be loaded and executed.
        
        Args:
            model_name (str): Name of the model (if None, uses current model)
            model_type (str): Type of the model (if None, uses current type)
        
        Returns:
            bool: True if the engine is valid, False otherwise
        """
        if not self.has_tensorrt:
            logger.error("TensorRT is not available")
            return False
        
        # Use current model and type if not specified
        if model_name is None:
            model_name = self.current_model_name
        if model_type is None:
            model_type = self.current_model_type
        
        try:
            import tensorrt as trt
            
            # Get the model directory for this type
            model_dir = os.path.join(self.models_dir, model_type)
            
            # Get the TensorRT engine path
            engine_path = os.path.join(model_dir, f"{model_name}.engine")
            if not os.path.exists(engine_path):
                logger.error(f"TensorRT engine not found: {engine_path}")
                return False
            
            # Load the engine
            engine = self.load_engine(engine_path)
            if not engine:
                return False
            
            # Create a context (this is an important test)
            try:
                logger.info("Creating execution context...")
                context = engine.create_execution_context()
                logger.info("Context created successfully")
                
                # Clean up
                del context
                del engine
                
                return True
            except Exception as e:
                logger.error(f"Error creating context: {e}")
                return False
        
        except Exception as e:
            logger.error(f"Error testing TensorRT engine: {e}")
            return False
            
    def add_model_type(self, model_type, default_model=None, available_models=None, label_type=None):
        """
        Add a new model type dynamically.
        
        Args:
            model_type (str): Type of model to add
            default_model (str): Default model for this type
            available_models (list): List of available models for this type
            label_type (str): Type of labels for this type
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not available_models:
            available_models = []
            
        try:
            if model_type in MODEL_TYPES:
                logger.warning(f"Model type {model_type} already exists. Updating...")
            
            MODEL_TYPES[model_type] = {
                "default_model": default_model,
                "available_models": available_models,
                "label_type": label_type
            }
            
            logger.info(f"Model type {model_type} added successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error adding model type: {e}")
            return False
            
    def add_model_to_type(self, model_type, model_name):
        """
        Add a model to an existing model type.
        
        Args:
            model_type (str): Type of model
            model_name (str): Name of model to add
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if model_type not in MODEL_TYPES:
                logger.error(f"Model type {model_type} does not exist")
                return False
                
            if model_name in MODEL_TYPES[model_type]["available_models"]:
                logger.warning(f"Model {model_name} already exists for type {model_type}")
                return True
                
            MODEL_TYPES[model_type]["available_models"].append(model_name)
            logger.info(f"Model {model_name} added to type {model_type}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding model to type: {e}")
            return False
            
    def is_model_available(self, model_name, model_type):
        """
        Check if a model is available for a specific type.
        
        Args:
            model_name (str): Name of the model
            model_type (str): Type of the model
        
        Returns:
            bool: True if available, False otherwise
        """
        if model_type not in MODEL_TYPES:
            return False
            
        return model_name in MODEL_TYPES[model_type]["available_models"] or model_name in AVAILABLE_MODELS