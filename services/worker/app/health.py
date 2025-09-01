"""Health check HTTP server for the worker."""
import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict

from app.config import settings
from app.storage import verify_storage_health

logger = logging.getLogger(__name__)


class HealthStatus:
    """Global health status tracker."""
    
    def __init__(self):
        self.is_healthy = False
        self.last_check = None
        self.worker_running = False
        self.services = {}
    
    def update_worker_status(self, running: bool):
        """Update worker running status."""
        self.worker_running = running
    
    def get_status(self) -> Dict[str, Any]:
        """Get current health status."""
        try:
            # Check storage health
            storage_ok = verify_storage_health()
            
            self.services = {
                "worker": "ok" if self.worker_running else "down",
                "chroma": "ok" if storage_ok else "error",
                "rabbitmq": "ok",
                "config": "ok"
            }
            
            # Overall health: all services must be OK
            self.is_healthy = all(status == "ok" for status in self.services.values())
            
            return {
                "status": "ok" if self.is_healthy else "error",
                "worker_mode": settings.analysis_mode,
                "services": self.services,
                "version": "1.0.0"
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "services": {"worker": "error"}
            }


# Global health status instance
health_status = HealthStatus()


class HealthHandler(BaseHTTPRequestHandler):
    """HTTP handler for health checks."""
    
    def do_GET(self):
        """Handle GET requests to /health."""
        if self.path == '/health':
            try:
                status_data = health_status.get_status()
                status_code = 200 if status_data["status"] == "ok" else 503
                
                self.send_response(status_code)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                
                response = json.dumps(status_data, indent=2)
                self.wfile.write(response.encode('utf-8'))
                
            except Exception as e:
                logger.error(f"Health endpoint error: {e}")
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                
                error_response = json.dumps({
                    "status": "error",
                    "error": str(e)
                })
                self.wfile.write(error_response.encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        """Suppress default HTTP logging."""
        pass


def start_health_server(port: int = 8000):
    """Start health check HTTP server in a background thread."""
    def run_server():
        try:
            server = HTTPServer(('0.0.0.0', port), HealthHandler)
            logger.info(f"Health server running on port {port}")
            server.serve_forever()
        except Exception as e:
            logger.error(f"Health server failed: {e}")
    
    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()
    return thread
