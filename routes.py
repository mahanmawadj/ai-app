"""
Route definitions for the web application
"""
import os
from aiohttp import web

# Import handlers
from handlers.static_handlers import index, javascript, css
from handlers.model_handlers import (
    get_current_model, get_available_models, handle_model_change,
    toggle_model, get_model_states
)
from handlers.stream_handlers import webrtcoffer

ROOT = os.path.dirname(__file__)
TEMPLATES = os.path.join(ROOT, 'templates')
JS = os.path.join(ROOT, 'static', 'js')
CSS = os.path.join(ROOT, 'static', 'css')

def setup_routes(app):
    """
    Set up all routes for the application
    
    Args:
        app: The aiohttp application instance
    """
    # Static content routes
    app.router.add_get("/", index)
    app.router.add_get("/client.js", javascript)
    app.router.add_get("/styles.css", css)
    
    # WebRTC routes
    app.router.add_post("/offer", webrtcoffer)
    
    # Model API routes
    app.router.add_get("/api/model", get_current_model)
    app.router.add_get("/api/models", get_available_models)
    app.router.add_post("/api/model/change", handle_model_change)
    app.router.add_put("/api/{model_type}", toggle_model)  # This captures the full "classification_enabled"
    app.router.add_get("/api/{model_type}", get_model_states)
    
    # You can add more route groups here as needed