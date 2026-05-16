"""GPUMind REST API — Control and monitor daemon via HTTP."""

import json
import asyncio
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .core import GPUMind

import logging
log = logging.getLogger("gpumind.api")


class GPUMindHandler(BaseHTTPRequestHandler):
    """HTTP handler for GPUMind API."""
    
    daemon: "GPUMind" = None  # set by APIServer
    
    def do_GET(self):
        if self.path == "/status" or self.path == "/":
            self._json_response(200, self.daemon.get_status())
        elif self.path == "/metrics":
            self._json_response(200, self.daemon.monitor.export_prometheus())
        elif self.path.startswith("/gpu/"):
            try:
                gpu_id = int(self.path.split("/")[2])
                state = self.daemon.states.get(gpu_id)
                if state:
                    self._json_response(200, {
                        "id": state.gpu_id,
                        "mode": state.mode.value,
                        "hashrate": state.mining_hashrate,
                        "temperature": state.temperature,
                        "power": state.power_watts,
                        "utilization": state.utilization,
                    })
                else:
                    self._json_response(404, {"error": "GPU not found"})
            except (ValueError, IndexError):
                self._json_response(400, {"error": "Invalid GPU ID"})
        else:
            self._json_response(404, {"error": "Not found"})
    
    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length else b""
        
        try:
            data = json.loads(body) if body else {}
        except json.JSONDecodeError:
            self._json_response(400, {"error": "Invalid JSON"})
            return
        
        if self.path == "/infer":
            # Queue inference request
            prompt = data.get("prompt", "")
            if not prompt:
                self._json_response(400, {"error": "Missing prompt"})
                return
            
            # Add to queue
            try:
                self.daemon._inference_queue.put_nowait({
                    "prompt": prompt,
                    "max_tokens": data.get("max_tokens", 512),
                })
                self._json_response(202, {
                    "status": "queued",
                    "position": self.daemon._inference_queue.qsize()
                })
            except Exception as e:
                self._json_response(503, {"error": str(e)})
        
        elif self.path == "/mode":
            # Force mode switch
            from .core import Mode
            gpu_id = data.get("gpu_id", 0)
            mode_str = data.get("mode", "")
            state = self.daemon.states.get(gpu_id)
            
            if not state:
                self._json_response(404, {"error": "GPU not found"})
                return
            
            try:
                target = Mode(mode_str)
                asyncio.ensure_future(self.daemon._switch_mode(state, target))
                self._json_response(200, {"status": "switching", "mode": mode_str})
            except ValueError:
                self._json_response(400, {"error": f"Invalid mode: {mode_str}"})
        
        else:
            self._json_response(404, {"error": "Not found"})
    
    def _json_response(self, status: int, data: dict):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2).encode())
    
    def log_message(self, format, *args):
        """Suppress default HTTP logging."""
        pass


class APIServer:
    """Async wrapper for HTTP API server."""
    
    def __init__(self, daemon: "GPUMind", host: str = "0.0.0.0", port: int = 8500):
        self.daemon = daemon
        self.host = host
        self.port = port
    
    async def start(self):
        """Start API server in background."""
        GPUMindHandler.daemon = self.daemon
        
        server = HTTPServer((self.host, self.port), GPUMindHandler)
        log.info(f"API server listening on {self.host}:{self.port}")
        
        # Run in executor (blocking HTTP server)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, server.serve_forever)
