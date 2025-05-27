"""
Handlers for static files (HTML, JS, CSS)
"""
import os
from aiohttp import web

ROOT = os.path.dirname(os.path.dirname(__file__))
TEMPLATES = os.path.join(ROOT, 'templates')
JS = os.path.join(ROOT, 'static', 'js')
CSS = os.path.join(ROOT, 'static', 'css')

async def index(request):
    """Serve the main HTML page"""
    content = open(os.path.join(TEMPLATES, "index.html"), "r").read()
    return web.Response(content_type="text/html", text=content)

async def javascript(request):
    """Serve the client JavaScript file"""
    content = open(os.path.join(JS, "client.js"), "r").read()
    return web.Response(content_type="application/javascript", text=content)

async def css(request):
    """Serve the CSS stylesheet"""
    content = open(os.path.join(CSS, "styles.css"), "r").read()
    return web.Response(content_type="text/css", text=content)