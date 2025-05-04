"""
WebRTC streaming implementation for Windows with RTX GPU.
This replaces the Jetson-specific streaming functionality from the original implementation.
"""

import asyncio
import threading
import time
import cv2
import numpy as np
import json
import ssl
import logging
import os
from typing import Dict, List, Set, Optional

# WebRTC imports
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceServer
from aiortc.contrib.media import MediaStreamTrack, MediaBlackhole
from aiortc.mediastreams import MediaStreamError
from av import VideoFrame

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("stream")

# Global variables
active_connections = set()


class VideoStreamTrack(MediaStreamTrack):
    """
    A MediaStreamTrack that outputs video frames from a camera or
    processed frames from an inference pipeline.
    """
    kind = "video"

    def __init__(self, camera, inference_processor=None):
        """Initialize with camera and optional inference processor"""
        super().__init__()
        self.camera = camera
        self.inference_processor = inference_processor
        self.frame_count = 0
        self.frame_time = time.time()
        self.fps = 0
        self.running = True
    
    async def recv(self):
        """Get the next frame"""
        if not self.running:
            raise MediaStreamError("Stream is not running")
        
        # Calculate FPS
        current_time = time.time()
        if current_time - self.frame_time >= 1.0:
            self.fps = self.frame_count
            self.frame_count = 0
            self.frame_time = current_time
        self.frame_count += 1
        
        # Read frame from camera
        ret, frame = self.camera.read()
        if not ret:
            # If camera read fails, return a blank frame
            height, width = 720, 1280
            frame = np.zeros((height, width, 3), np.uint8)
        
        # Process frame with inference if available
        if self.inference_processor is not None:
            frame = self.inference_processor(frame)
        
        # Add FPS counter to frame
        cv2.putText(frame, f"FPS: {self.fps}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Convert to VideoFrame for WebRTC
        video_frame = VideoFrame.from_ndarray(frame, format="bgr24")
        video_frame.pts = self.frame_count * 1000  # millisecond timestamp
        
        return video_frame


class StreamManager:
    """Manages WebRTC streams and connections"""
    
    def __init__(self, inference_models=None):
        """Initialize stream manager"""
        self.camera = None
        self.camera_id = 0
        self.inference_models = inference_models if inference_models else {}
        self.active_models = {}
        self.peer_connections = set()
        self.input_tracks = {}
        self.lock = threading.Lock()
    
    def init_camera(self, camera_id=0, width=1280, height=720):
        """Initialize camera"""
        logger.info(f"Initializing camera {camera_id} with resolution {width}x{height}")
        try:
            self.camera_id = camera_id
            self.camera = cv2.VideoCapture(camera_id)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            
            # Check if camera opened successfully
            if not self.camera.isOpened():
                logger.error(f"Failed to open camera {camera_id}")
                return False
            
            return True
        except Exception as e:
            logger.error(f"Error initializing camera: {e}")
            return False
    
    def release_camera(self):
        """Release camera resources"""
        if self.camera is not None:
            self.camera.release()
            self.camera = None
    
    def process_frame(self, frame):
        """Process a frame with active inference models"""
        processed_frame = frame.copy()
        
        # Apply each active model to the frame
        with self.lock:
            for model_name, model in self.active_models.items():
                # Skip if model is not enabled
                if not model.get('enabled', False):
                    continue
                
                # Get the model and process frame
                inference_model = model.get('model')
                if inference_model:
                    # Run inference
                    results = inference_model.infer(processed_frame)
                    
                    # Draw results on frame
                    processed_frame = inference_model.draw_results(processed_frame, results)
        
        return processed_frame
    
    async def handle_offer(self, offer, ice_servers=None):
        """Handle WebRTC offer from client"""
        if ice_servers is None:
            ice_servers = []
        
        # Create peer connection
        pc = RTCPeerConnection(configuration={"iceServers": ice_servers})
        self.peer_connections.add(pc)
        
        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            logger.info(f"Connection state: {pc.connectionState}")
            if pc.connectionState == "failed" or pc.connectionState == "closed":
                await self.close_peer_connection(pc)
        
        @pc.on("track")
        def on_track(track):
            logger.info(f"Received track: {track.kind}")
            if track.kind == "video":
                # Store incoming video track for future use (e.g., from client's webcam)
                self.input_tracks[pc] = track
                
                # Add to media sink to consume the track
                pc.addTrack(MediaBlackhole())
        
        # Ensure camera is initialized
        if self.camera is None or not self.camera.isOpened():
            if not self.init_camera(self.camera_id):
                logger.error("Failed to initialize camera")
                return None
        
        # Create video track with inference processing
        video_track = VideoStreamTrack(
            self.camera,
            inference_processor=self.process_frame
        )
        
        # Add track to peer connection
        pc.addTrack(video_track)
        
        # Set remote description
        await pc.setRemoteDescription(RTCSessionDescription(
            sdp=offer.get("sdp"),
            type=offer.get("type")
        ))
        
        # Create answer
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        
        return {
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type
        }
    
    async def close_peer_connection(self, pc):
        """Close a peer connection"""
        logger.info("Closing peer connection")
        
        # Remove from tracked connections
        if pc in self.peer_connections:
            self.peer_connections.remove(pc)
        
        # Remove any input tracks
        if pc in self.input_tracks:
            del self.input_tracks[pc]
        
        # Close the connection
        await pc.close()
    
    async def close_all_connections(self):
        """Close all peer connections"""
        logger.info("Closing all peer connections")
        
        # Close all connections
        for pc in list(self.peer_connections):
            await self.close_peer_connection(pc)
        
        # Release camera
        self.release_camera()
    
    def set_model_state(self, model_name, enabled):
        """Enable or disable an inference model"""
        with self.lock:
            if model_name in self.inference_models:
                if model_name not in self.active_models and enabled:
                    # Add model to active models
                    self.active_models[model_name] = {
                        'model': self.inference_models[model_name],
                        'enabled': enabled
                    }
                elif model_name in self.active_models:
                    # Update model state
                    self.active_models[model_name]['enabled'] = enabled
            else:
                logger.warning(f"Model {model_name} not found")
    
    def get_model_state(self, model_name):
        """Get the state of an inference model"""
        with self.lock:
            if model_name in self.active_models:
                return self.active_models[model_name]['enabled']
            return False


# Create SSL context for HTTPS
def create_ssl_context():
    """Create SSL context for HTTPS/WSS"""
    # Check for existing certificate
    if os.path.exists("cert.pem") and os.path.exists("key.pem"):
        # Use existing certificate
        ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        ssl_context.load_cert_chain("cert.pem", "key.pem")
        return ssl_context
    
    # Create self-signed certificate
    from OpenSSL import crypto
    
    # Create key
    key = crypto.PKey()
    key.generate_key(crypto.TYPE_RSA, 2048)
    
    # Create certificate
    cert = crypto.X509()
    cert.get_subject().CN = "localhost"
    cert.set_serial_number(1000)
    cert.gmtime_adj_notBefore(0)
    cert.gmtime_adj_notAfter(10*365*24*60*60)  # 10 years
    cert.set_issuer(cert.get_subject())
    cert.set_pubkey(key)
    cert.sign(key, "sha256")
    
    # Save certificate
    with open("cert.pem", "wb") as f:
        f.write(crypto.dump_certificate(crypto.FILETYPE_PEM, cert))
    with open("key.pem", "wb") as f:
        f.write(crypto.dump_privatekey(crypto.FILETYPE_PEM, key))
    
    # Create SSL context
    ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    ssl_context.load_cert_chain("cert.pem", "key.pem")
    
    return ssl_context