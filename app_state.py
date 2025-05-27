"""
Global application state management
"""
import logging
from models.model_handler import ModelHandler
from stream import StreamManager

logger = logging.getLogger(__name__)

# Global application state
model_handler = None
stream_manager = None

def initialize_app(args):
    """
    Initialize the application with given arguments
    
    Args:
        args: Command line arguments
        
    Returns:
        bool: True if successful, False otherwise
    """
    global model_handler, stream_manager
    
    try:
        # Initialize model handler without loading any specific model
        # Models will be loaded lazily when enabled through the UI
        model_handler = ModelHandler(
            model_type=None,  # No default model type
            model_name=None,  # No default model
            models_dir=args.models_dir,
            labels_dir=args.labels_dir
        )
        
        # Initialize stream manager with model handler
        stream_manager = StreamManager(model_handler)
        
        logger.info("Application initialized successfully with lazy loading enabled")
        return True
    
    except Exception as e:
        logger.error(f"Error initializing app: {e}")
        return False

def get_model_handler():
    """Get the global model handler instance"""
    return model_handler

def get_stream_manager():
    """Get the global stream manager instance"""
    return stream_manager