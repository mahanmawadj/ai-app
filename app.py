#!/usr/bin/env python3
"""
Flask web application for deep learning inference using TensorRT on Windows with RTX GPU.
This replaces the Jetson-specific implementation from the original repo.
"""

import os
import sys
import argparse
import threading
import logging
import json
import time
import asyncio
from pathlib import Path

# Flask and web server imports
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import ssl

# Local module imports
from inference import ObjectDetector, Classifier, PoseEstimator, ActionRecognizer, Segmenter
from stream import StreamManager, create_ssl_context

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

# Create Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variables
stream_manager = None
args = None


def load_models(args):
    """
    Load inference models based on command line arguments
    """
    models = {}
    
    # Load object detection model if specified
    if args.detection:
        model_path = get_model_path(args.detection, 'detection')
        if model_path:
            logger.info(f"Loading detection model: {model_path}")
            models['detection'] = ObjectDetector(model_path)
    
    # Load classification model if specified
    if args.classification:
        model_path = get_model_path(args.classification, 'classification')
        if model_path:
            logger.info(f"Loading classification model: {model_path}")
            models['classification'] = Classifier(model_path)
    
    # Load pose estimation model if specified
    if args.pose:
        model_path = get_model_path(args.pose, 'pose')
        if model_path:
            logger.info(f"Loading pose model: {model_path}")
            models['pose'] = PoseEstimator(model_path)
    
    # Load action recognition model if specified
    if args.action:
        model_path = get_model_path(args.action, 'action')
        if model_path:
            logger.info(f"Loading action model: {model_path}")
            models['action'] = ActionRecognizer(model_path)
    
    # Load segmentation model if specified
    if args.segmentation:
        model_path = get_model_path(args.segmentation, 'segmentation')
        if model_path:
            logger.info(f"Loading segmentation model: {model_path}")
            models['segmentation'] = Segmenter(model_path)
    
    return models


def get_model_path(model_name, model_type):
    """
    Get the path to a model file
    """
    # Check if the model name is a direct path to a file
    if os.path.isfile(model_name):
        return model_name
    
    # Look in models directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, 'models', model_type)
    
    # Create the directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Check for .engine file
    engine_path = os.path.join(model_dir, f"{model_name}.engine")
    if os.path.isfile(engine_path):
        return engine_path
    
    # Check for .onnx file
    onnx_path = os.path.join(model_dir, f"{model_name}.onnx")
    if os.path.isfile(onnx_path):
        return onnx_path
    
    logger.error(f"Model file not found: {model_name}")
    return None


def property_endpoint(prop, default_value=None):
    """
    Decorator for property endpoints that handle both GET and PUT
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            if request.method == 'GET':
                return jsonify({prop: func(*args, **kwargs)})
            elif request.method == 'PUT':
                data = request.json
                if prop in data:
                    return jsonify({prop: func(*args, **kwargs, value=data[prop])})
                return jsonify({"error": f"Missing property: {prop}"}), 400
            return jsonify({"error": "Method not allowed"}), 405
        
        # Add route with both methods
        wrapper.__name__ = func.__name__
        return app.route(f"/api/{prop}", methods=['GET', 'PUT'])(wrapper)
    
    return decorator


# Define routes
@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')


@app.route('/js/<path:path>')
def send_js(path):
    """Serve JavaScript files"""
    return send_from_directory('static/js', path)


@app.route('/css/<path:path>')
def send_css(path):
    """Serve CSS files"""
    return send_from_directory('static/css', path)


@app.route('/api/webrtc_offer', methods=['POST'])
async def webrtc_offer():
    """Handle WebRTC offer from client"""
    global stream_manager
    
    # Get offer from request
    offer = request.json
    
    # Handle offer
    answer = await stream_manager.handle_offer(offer)
    
    if answer:
        return jsonify(answer)
    return jsonify({"error": "Failed to create answer"}), 500


@property_endpoint('detection_enabled', False)
def detection_enabled(value=None):
    """Get or set detection model state"""
    global stream_manager
    
    if value is not None:
        stream_manager.set_model_state('detection', value)
    
    return stream_manager.get_model_state('detection')


@property_endpoint('classification_enabled', False)
def classification_enabled(value=None):
    """Get or set classification model state"""
    global stream_manager
    
    if value is not None:
        stream_manager.set_model_state('classification', value)
    
    return stream_manager.get_model_state('classification')


@property_endpoint('pose_enabled', False)
def pose_enabled(value=None):
    """Get or set pose model state"""
    global stream_manager
    
    if value is not None:
        stream_manager.set_model_state('pose', value)
    
    return stream_manager.get_model_state('pose')


@property_endpoint('action_enabled', False)
def action_enabled(value=None):
    """Get or set action model state"""
    global stream_manager
    
    if value is not None:
        stream_manager.set_model_state('action', value)
    
    return stream_manager.get_model_state('action')


@property_endpoint('segmentation_enabled', False)
def segmentation_enabled(value=None):
    """Get or set segmentation model state"""
    global stream_manager
    
    if value is not None:
        stream_manager.set_model_state('segmentation', value)
    
    return stream_manager.get_model_state('segmentation')


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='TensorRT Flask Application for Windows')
    
    # WebRTC options
    parser.add_argument('--port', type=int, default=8050, help='Port to run the server on')
    parser.add_argument('--input', type=str, default='0', help='Camera device ID or video file path')
    
    # Model options
    parser.add_argument('--detection', type=str, help='Detection model to use (e.g., ssd-mobilenet-v2)')
    parser.add_argument('--classification', type=str, help='Classification model to use (e.g., resnet18)')
    parser.add_argument('--pose', type=str, help='Pose estimation model to use (e.g., resnet18-body)')
    parser.add_argument('--action', type=str, help='Action recognition model to use (e.g., resnet18-kinetics)')
    parser.add_argument('--segmentation', type=str, help='Segmentation model to use (e.g., fcn-resnet18)')
    
    return parser.parse_args()


async def shutdown_server():
    """Shut down the server and clean up resources"""
    global stream_manager
    
    # Close all WebRTC connections
    await stream_manager.close_all_connections()
    
    # Exit the application
    os._exit(0)


def init_stream_manager(args):
    """Initialize the stream manager with models and camera"""
    global stream_manager
    
    # Load inference models
    models = load_models(args)
    
    # Create stream manager with models
    stream_manager = StreamManager(models)
    
    # Initialize camera
    camera_id = args.input
    try:
        # Try to convert to integer (for webcam index)
        camera_id = int(camera_id)
    except ValueError:
        # Keep as string (for video file path)
        pass
    
    # Initialize camera with specified ID
    stream_manager.init_camera(camera_id)
    
    # Enable models by default if specified on command line
    if args.detection and 'detection' in models:
        stream_manager.set_model_state('detection', True)
    
    if args.classification and 'classification' in models:
        stream_manager.set_model_state('classification', True)
    
    if args.pose and 'pose' in models:
        stream_manager.set_model_state('pose', True)
    
    if args.action and 'action' in models:
        stream_manager.set_model_state('action', True)
    
    if args.segmentation and 'segmentation' in models:
        stream_manager.set_model_state('segmentation', True)


def run_flask_app(args):
    """Run the Flask application with SSL"""
    # Create SSL context
    ssl_context = create_ssl_context()
    
    # Run Flask app
    app.run(
        host='0.0.0.0',
        port=args.port,
        ssl_context=ssl_context,
        debug=False,
        threaded=True
    )


def main():
    """Main entry point"""
    global args
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Initialize stream manager
    init_stream_manager(args)
    
    # Print instructions
    local_url = f"https://localhost:{args.port}"
    print(f"\nRunning Flask application with TensorRT inference")
    print(f"Navigate to: {local_url}")
    print("Note: Since we're using a self-signed certificate, you'll need to accept the security warning in your browser.")
    
    # Start Flask application
    run_flask_app(args)


if __name__ == "__main__":
    main()