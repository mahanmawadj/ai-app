import cv2
import logging
import uuid
import os
import asyncio
import json
import time
import threading
import traceback

from aiohttp import web
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder, MediaRelay
from av import VideoFrame


relay = MediaRelay()
logger = logging.getLogger("pc")
pcs = set()

ROOT = os.path.dirname(__file__)

class VideoTransformTrack(MediaStreamTrack):
    """
    A video stream track that transforms frames from an another track.
    """

    kind = "video"

    def __init__(self, track, stream_manager):
        super().__init__()
        self.frame_count = 0
        self.frame_time = time.time()
        self.fps = 0
        self.track = track
        self.stream_manager = stream_manager

    async def recv(self):
        frame = await self.track.recv()
    
        # Convert frame to numpy array for OpenCV processing
        img = frame.to_ndarray(format="bgr24")
    
        # Calculate FPS
        current_time = time.time()
        if current_time - self.frame_time >= 1.0:
            self.fps = self.frame_count
            self.frame_count = 0
            self.frame_time = current_time
        self.frame_count += 1
    
        # Process frame with active models
        processed_img = self.stream_manager.process_frame(img)
        
        # Add FPS text to the image
        cv2.putText(processed_img, f"FPS: {self.fps}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
        # Rebuild a VideoFrame with the processed image
        new_frame = VideoFrame.from_ndarray(processed_img, format="bgr24")
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base
        return new_frame

class StreamManager:
    """Manages WebRTC streams and connections with lazy loading of models"""
    
    def __init__(self, model_handler):
        """
        Initialize stream manager with model handler for lazy loading
        
        Args:
            model_handler: ModelHandler instance for managing models
        """
        self.model_handler = model_handler
        self.inference_models = {}  # Will be populated lazily
        self.model_states = {}  # Track which models are enabled/disabled
        self.model_locks = {}  # Individual locks for each model type
        self.loading_models = set()  # Track models currently being loaded
        self.peer_connections = set()
        self.input_tracks = {}
        
        # Initialize locks for each model type
        for model_type in ["classification", "detection", "pose", "action", "segmentation"]:
            self.model_locks[model_type] = threading.Lock()
            self.model_states[model_type] = False

    def _load_inference_module(self, model_type):
        """
        Lazy load an inference module when needed
        
        Args:
            model_type (str): Type of model to load
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Check if already loaded
        if model_type in self.inference_models:
            return True
            
        # Check if currently loading
        if model_type in self.loading_models:
            logger.info(f"Model {model_type} is already being loaded")
            return False
            
        try:
            self.loading_models.add(model_type)
            logger.info(f"Lazy loading {model_type} inference module...")
            
            # Import the appropriate module dynamically
            if model_type == "classification":
                from inference.classifier import Classifier
                module_class = Classifier
            elif model_type == "detection":
                from inference.detector import ObjectDetector
                module_class = ObjectDetector
            elif model_type == "pose":
                from inference.pose_estimator import PoseEstimator
                module_class = PoseEstimator
            elif model_type == "action":
                from inference.action_recognizer import ActionRecognizer
                module_class = ActionRecognizer
            elif model_type == "segmentation":
                from inference.segmenter import Segmenter
                module_class = Segmenter
            else:
                logger.error(f"Unknown model type: {model_type}")
                return False
            
            # Get model info and labels from model handler
            # First ensure the model handler has the right model type loaded
            if self.model_handler.current_model_type != model_type:
                # Get the default model for this type
                from models.model_handler import MODEL_TYPES
                if model_type in MODEL_TYPES:
                    default_model = MODEL_TYPES[model_type]["default_model"]
                    if default_model:
                        success, error = self.model_handler.change_model(default_model, model_type)
                        if not success:
                            logger.error(f"Failed to load default model for {model_type}: {error}")
                            return False
                    else:
                        logger.error(f"No default model available for {model_type}")
                        return False
                else:
                    logger.error(f"Unknown model type: {model_type}")
                    return False
            
            model_info = self.model_handler.get_model_info()
            labels = self.model_handler.get_labels()
            
            # Create the inference module instance
            inference_module = module_class(model_info, labels)
            
            # Store the module
            self.inference_models[model_type] = inference_module
            
            logger.info(f"Successfully loaded {model_type} inference module")
            return True
            
        except Exception as e:
            logger.error(f"Error loading {model_type} inference module: {e}")
            logger.error(traceback.format_exc())
            return False
        finally:
            self.loading_models.discard(model_type)

    def set_model_state(self, model_type, enabled):
        """
        Enable or disable a model, loading it if necessary
        
        Args:
            model_type (str): Type of model
            enabled (bool): Whether to enable or disable
        """
        with self.model_locks[model_type]:
            # If enabling and model not loaded, load it now
            if enabled and model_type not in self.inference_models:
                if not self._load_inference_module(model_type):
                    logger.error(f"Failed to load {model_type} model")
                    return
            
            # Update state
            self.model_states[model_type] = enabled
            logger.info(f"Model {model_type} {'enabled' if enabled else 'disabled'}")

    def get_model_state(self, model_type):
        """
        Get the current state of a model
        
        Args:
            model_type (str): Type of model
            
        Returns:
            bool: Whether the model is enabled
        """
        return self.model_states.get(model_type, False)

    def process_frame(self, frame):
        """Process a frame with active inference models"""
        processed_frame = frame.copy()
        
        # Apply each active model to the frame
        for model_type, enabled in self.model_states.items():
            if not enabled:
                continue
                
            # Check if model is loaded
            if model_type not in self.inference_models:
                continue
                
            with self.model_locks[model_type]:
                try:
                    inference_model = self.inference_models[model_type]
                    
                    # Run inference
                    results = inference_model.infer(processed_frame)
                    
                    # Draw results on frame
                    if results is not None:
                        processed_frame = inference_model.draw_results(processed_frame, results)
                        
                except Exception as e:
                    logger.error(f"Error processing frame with {model_type}: {e}")
        
        return processed_frame

    def change_model(self, model_name, model_type):
        """
        Change the model for a specific type
        
        Args:
            model_name (str): Name of the new model
            model_type (str): Type of the model
            
        Returns:
            tuple: (success, error_message)
        """
        # Ensure we have a lock for this model type
        if model_type not in self.model_locks:
            self.model_locks[model_type] = threading.Lock()
            
        with self.model_locks[model_type]:
            try:
                # Change the model in the model handler
                success, error = self.model_handler.change_model(model_name, model_type)
                if not success:
                    return False, error
                
                # Get model info
                model_info = self.model_handler.get_model_info()
                
                # Check what formats are available and build a clean model_info
                has_engine = model_info.get('has_trt', False) and os.path.exists(model_info.get('trt_path', ''))
                has_onnx = model_info.get('has_onnx', False) and os.path.exists(model_info.get('onnx_path', ''))
                has_pytorch = model_info.get('has_pytorch', False) and os.path.exists(model_info.get('pytorch_path', ''))
                
                # Create a clean model_info with only existing paths
                clean_model_info = model_info.copy()
                
                # Remove non-existent paths so base.py uses the right fallback
                if not has_engine:
                    clean_model_info.pop('trt_path', None)
                    clean_model_info['has_trt'] = False
                
                if not has_onnx:
                    clean_model_info.pop('onnx_path', None)
                    clean_model_info['has_onnx'] = False
                
                # Determine what we have and what we need
                if has_engine:
                    logger.info(f"TensorRT engine found for {model_name}")
                elif has_onnx:
                    logger.info(f"ONNX model found for {model_name}. TensorRT will convert it automatically.")
                    # Make sure the ONNX path is available for base.py to use
                    clean_model_info['path'] = clean_model_info['onnx_path']
                elif has_pytorch:
                    logger.info(f"Only PyTorch model found for {model_name}.")
                    
                    # Check if ONNX is available for conversion
                    if not self.model_handler.has_onnx:
                        error_msg = (
                            "Model conversion requires ONNX. Please install it with:\n"
                            "pip install onnx onnxruntime\n\n"
                            "Alternatively, you can:\n"
                            "1. Download a pre-converted .onnx or .engine file\n"
                            "2. Convert the model on another machine and copy the files"
                        )
                        logger.error(error_msg)
                        return False, error_msg
                    
                    # Try to convert PyTorch to ONNX
                    logger.info(f"Converting {model_name} from PyTorch to ONNX...")
                    onnx_path = self.model_handler.pytorch_to_onnx(model_name, model_type)
                    if not onnx_path:
                        return False, "Failed to convert model to ONNX format"
                    
                    # Reload model info after conversion
                    self.model_handler._load_model(model_name, model_type)
                    model_info = self.model_handler.get_model_info()
                    # Set the path to the ONNX file
                    model_info['path'] = model_info['onnx_path']
                else:
                    return False, f"No model files found for {model_name}"
                
                # If this model type is loaded, update it
                if model_type in self.inference_models:
                    logger.info(f"Updating {model_type} inference module with new model: {model_name}")
                    
                    # Get labels
                    labels = self.model_handler.get_labels()
                    
                    # Update the inference module
                    # The inference module will handle ONNX to TensorRT conversion if needed
                    success = self.inference_models[model_type].update_model(model_info, labels)
                    
                    if not success:
                        return False, f"Failed to update {model_type} inference module"
                
                return True, None
                
            except Exception as e:
                logger.error(f"Error changing model: {e}")
                logger.error(traceback.format_exc())
                return False, str(e)

    async def handle_offer(self, request):
        """Handle WebRTC offer from client"""
        params = await request.json()
        offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

        pc = RTCPeerConnection()
        pc_id = "PeerConnection(%s)" % uuid.uuid4()
        pcs.add(pc)

        def log_info(msg, *args):
            logger.info(pc_id + " " + msg, *args)

        log_info("Created for %s", request.remote)

        @pc.on("datachannel")
        def on_datachannel(channel):
            @channel.on("message")
            def on_message(message):
                if isinstance(message, str) and message.startswith("ping"):
                    channel.send("pong" + message[4:])

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            log_info("Connection state is %s", pc.connectionState)
            if pc.connectionState == "failed":
                await pc.close()
                pcs.discard(pc)

        @pc.on("track")
        def on_track(track):
            log_info("Track %s received", track.kind)

            if track.kind == "video":
                pc.addTrack(
                    VideoTransformTrack(
                        relay.subscribe(track), self
                    )
                )

            @track.on("ended")
            async def on_ended():
                log_info("Track %s ended", track.kind)

        # handle offer
        await pc.setRemoteDescription(offer)

        # send answer
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
            ),
        )

async def on_shutdown(app):
    # close peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()