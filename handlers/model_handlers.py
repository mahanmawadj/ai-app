"""
Handlers for model-related API endpoints
"""
import json
import logging
from aiohttp import web
from app_state import get_model_handler, get_stream_manager

logger = logging.getLogger(__name__)

async def get_current_model(request):
    """Return the currently loaded model information"""
    model_handler = get_model_handler()
    
    model_info = model_handler.get_model_info() if model_handler else None
    
    content = json.dumps({
        'model_name': model_handler.current_model_name if model_handler else None,
        'model_type': model_handler.current_model_type if model_handler else None,
        'is_loading': model_handler.is_model_loading if model_handler else False,
        'has_pytorch': model_info.get('has_pytorch', False) if model_info else False,
        'has_onnx': model_info.get('has_onnx', False) if model_info else False,
        'has_trt': model_info.get('has_trt', False) if model_info else False
    })

    return web.Response(content_type="application/json", text=content)

async def get_available_models(request):
    """Return all available models and types"""
    model_handler = get_model_handler()
    
    if not model_handler:
        return web.Response(
            content_type="application/json",
            text=json.dumps({'error': 'Model handler not initialized'}),
            status=500
        )
    
    available_types = model_handler.get_available_model_types()
    available_models = {}
    
    for model_type in available_types:
        available_models[model_type] = model_handler.get_available_models(model_type)
    
    content = json.dumps({
        'model_types': available_types,
        'models': available_models
    })
    
    return web.Response(content_type="application/json", text=content)

async def change_model(request):
    """Change the current model based on the request"""
    stream_manager = get_stream_manager()
    
    if not stream_manager:
        return web.Response(
            content_type="application/json",
            text=json.dumps({'success': False, 'error': 'Stream manager not initialized'}),
            status=500
        )
    
    try:
        # Parse the request data
        data = await request.json()
        new_model_name = data.get('model_name')
        new_model_type = data.get('model_type', 'classification')  # Default to classification
        
        # Change the model through stream manager
        success, error = stream_manager.change_model(new_model_name, new_model_type)
        
        if success:
            content = json.dumps({
                'success': True,
                'model_name': new_model_name,
                'model_type': new_model_type
            })
        else:
             content = json.dumps({
                'success': False,
                'error': error
            })
    except Exception as e:
        logger.error(f"Error changing model: {e}")
        content = json.dumps({
            'success': False,
            'error': str(e)
        })
    return web.Response(content_type="application/json", text=content)

async def toggle_model(request):
    """Enable or disable a specific model type based on the endpoint"""
    stream_manager = get_stream_manager()
    
    if not stream_manager:
        return web.Response(
            content_type="application/json",
            text=json.dumps({'success': False, 'error': 'Stream manager not initialized'}),
            status=500
        )
    
    try:
        # Extract model type from the URL
        model_type_enabled = request.match_info.get('model_type', '')
        model_type = model_type_enabled.replace('_enabled', '')
        
        # Parse the request data
        data = await request.json()
        enabled = data.get(model_type_enabled, False)
        
        # Validate model type
        if model_type not in ["detection", "classification", "pose", "action", "segmentation"]:
            content = json.dumps({
                'success': False,
                'error': f"Invalid model type: {model_type}"
            })
            return web.Response(content_type="application/json", text=content, status=400)
        
        # Set model state (will lazy load if needed)
        stream_manager.set_model_state(model_type, enabled)
        
        content = json.dumps({
            'success': True,
            model_type_enabled: enabled
        })
    except Exception as e:
        logger.error(f"Error toggling model: {e}")
        content = json.dumps({
            'success': False,
            'error': str(e)
        })
    return web.Response(content_type="application/json", text=content)

async def get_model_states(request):
    """Get the current state of a specific model"""
    stream_manager = get_stream_manager()
    
    if not stream_manager:
        return web.Response(
            content_type="application/json",
            text=json.dumps({'error': 'Stream manager not initialized'}),
            status=500
        )
    
    try:
        # Extract model type from the URL
        model_type_enabled = request.match_info.get('model_type', '')
        model_type = model_type_enabled.replace('_enabled', '')
        
        # Get the state
        enabled = stream_manager.get_model_state(model_type)
        
        content = json.dumps({
            model_type_enabled: enabled
        })
        
    except Exception as e:
        logger.error(f"Error getting model state: {e}")
        content = json.dumps({
            'error': str(e)
        })
        
    return web.Response(content_type="application/json", text=content)