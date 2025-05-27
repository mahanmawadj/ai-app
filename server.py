import argparse
import logging
import os
import ssl
import sys
from aiohttp import web

# Import route handlers
from routes import setup_routes
from stream import on_shutdown
from test import check_numpy, check_opencv, check_cuda, check_tensorrt
from config import Config

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

ROOT = os.path.dirname(__file__)

def doChecks():
    """Run all checks"""
    print("=" * 60)
    print("TensorRT Environment Test")
    print("=" * 60)
    
    # Basic info
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    
    # Check required modules
    check_numpy()
    check_opencv()
    cuda_ok = check_cuda()
    tensorrt_ok = check_tensorrt()
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary:")
    if cuda_ok and tensorrt_ok:
        print("✅ All critical components are installed and working!")
    else:
        print("❌ There are issues with your installation:")
        if not cuda_ok:
            print("  - CUDA/PyCUDA is not properly installed or configured")
        if not tensorrt_ok:
            print("  - TensorRT is not properly installed or configured")
        print("\nPlease fix these issues before running the TensorRT application.")
        sys.exit(1)
    print("=" * 60)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="AI Computer Vision App with TensorRT")
    
    parser.add_argument("--cert-file", default=os.path.join(ROOT, "cert.pem"), 
                        help="SSL certificate file (for HTTPS)")
    
    parser.add_argument("--key-file", default=os.path.join(ROOT, "key.pem"), 
                        help="SSL key file (for HTTPS)")
    
    parser.add_argument("--host", default="localhost", 
                        help="Host for HTTP server (default: localhost)")
    
    parser.add_argument("--port", type=int, default=8080, 
                        help="Port for HTTP server (default: 8080)")
    
    parser.add_argument("--models-dir", type=str, default=Config.DEFAULT_MODELS_DIR, 
                        help=f"Directory to store models (default: {Config.DEFAULT_MODELS_DIR})")
    
    parser.add_argument("--labels-dir", type=str, default=Config.DEFAULT_LABELS_DIR, 
                        help=f"Directory to store label files (default: {Config.DEFAULT_LABELS_DIR})")
    
    parser.add_argument("--verbose", "-v", action="count")
    
    return parser.parse_args()

def main():
    print("Starting application...")
    args = parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # Run environment checks
    doChecks()
    
    # Initialize the application
    from app_state import initialize_app
    if not initialize_app(args):
        logger.error("Failed to initialize app")
        sys.exit(1)
    
    # Set up SSL if certificates are provided
    if args.cert_file and os.path.exists(args.cert_file):
        ssl_context = ssl.SSLContext()
        ssl_context.load_cert_chain(args.cert_file, args.key_file)
    else:
        ssl_context = None

    # Create web app
    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    
    # Set up all routes
    setup_routes(app)

    # Start the web app
    web.run_app(
        app, 
        access_log=None, 
        host=args.host, 
        port=args.port, 
        ssl_context=ssl_context
    )

if __name__ == "__main__":
    main()