"""
Handlers for WebRTC streaming
"""
from app_state import get_stream_manager

async def webrtcoffer(request):
    """Handle WebRTC offer from client"""
    stream_manager = get_stream_manager()
    
    if not stream_manager:
        return web.Response(
            content_type="application/json",
            text=json.dumps({'error': 'Stream manager not initialized'}),
            status=500
        )
    
    return await stream_manager.handle_offer(request)