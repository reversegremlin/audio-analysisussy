#!/usr/bin/env python3
"""
Simple development server for Kaleidoscope Studio frontend.
Serves static files and provides API endpoints for audio analysis.
"""

import json
import mimetypes
import os
import sys
import tempfile
import threading
import uuid
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Track render tasks
render_tasks = {}


class KaleidoscopeHandler(SimpleHTTPRequestHandler):
    """HTTP handler for the Kaleidoscope Studio."""

    def __init__(self, *args, **kwargs):
        # Set directory to frontend folder
        self.directory = str(Path(__file__).parent)
        super().__init__(*args, directory=self.directory, **kwargs)

    def do_POST(self):
        """Handle POST requests for API endpoints."""
        parsed = urlparse(self.path)

        if parsed.path == "/api/analyze":
            self.handle_analyze()
        elif parsed.path == "/api/render":
            self.handle_render()
        else:
            self.send_error(404, "Not found")

    def do_GET(self):
        """Handle GET requests."""
        parsed = urlparse(self.path)

        if parsed.path.startswith("/api/render/status/"):
            task_id = parsed.path.split("/")[-1]
            self.handle_render_status(task_id)
        elif parsed.path.startswith("/api/render/download/"):
            task_id = parsed.path.split("/")[-1]
            self.handle_render_download(task_id)
        else:
            # Serve static files
            super().do_GET()

    def handle_analyze(self):
        """Analyze audio file and return manifest."""
        try:
            content_length = int(self.headers["Content-Length"])
            post_data = self.rfile.read(content_length)

            # Parse multipart form data (simplified)
            # In production, use proper multipart parsing
            boundary = self.headers["Content-Type"].split("boundary=")[1]

            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                # Extract file content (simplified - in production use proper parsing)
                temp_path = f.name
                # For now, we'll use the data directly

            # Import and run analysis
            from audio_analysisussy import AudioPipeline

            pipeline = AudioPipeline(target_fps=60)
            result = pipeline.process(temp_path)

            # Clean up
            os.unlink(temp_path)

            # Send response
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(result["manifest"]).encode())

        except Exception as e:
            self.send_error(500, str(e))

    def handle_render(self):
        """Start video rendering task."""
        try:
            # Create task ID
            task_id = str(uuid.uuid4())

            render_tasks[task_id] = {
                "progress": 0,
                "message": "Starting render...",
                "complete": False,
                "error": None,
                "output_path": None,
            }

            # In production, this would start a background render task
            # For now, just return the task ID

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"task_id": task_id}).encode())

        except Exception as e:
            self.send_error(500, str(e))

    def handle_render_status(self, task_id):
        """Get render task status."""
        if task_id not in render_tasks:
            self.send_error(404, "Task not found")
            return

        task = render_tasks[task_id]

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(task).encode())

    def handle_render_download(self, task_id):
        """Download rendered video."""
        if task_id not in render_tasks:
            self.send_error(404, "Task not found")
            return

        task = render_tasks[task_id]
        if not task["complete"] or not task["output_path"]:
            self.send_error(400, "Render not complete")
            return

        # Send file
        output_path = Path(task["output_path"])
        if not output_path.exists():
            self.send_error(404, "File not found")
            return

        self.send_response(200)
        self.send_header("Content-Type", "video/mp4")
        self.send_header(
            "Content-Disposition", f'attachment; filename="{output_path.name}"'
        )
        self.send_header("Content-Length", str(output_path.stat().st_size))
        self.end_headers()

        with open(output_path, "rb") as f:
            self.wfile.write(f.read())

    def log_message(self, format, *args):
        """Custom log format."""
        print(f"[Kaleidoscope] {args[0]}")


def run_server(port=8080):
    """Run the development server."""
    server_address = ("", port)
    httpd = HTTPServer(server_address, KaleidoscopeHandler)

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   🎨 Kaleidoscope Studio                                     ║
║                                                              ║
║   Server running at: http://localhost:{port}                  ║
║                                                              ║
║   Open in browser to start creating visualizations!          ║
║   Press Ctrl+C to stop                                       ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        httpd.shutdown()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Kaleidoscope Studio Server")
    parser.add_argument("-p", "--port", type=int, default=8080, help="Port to run on")
    args = parser.parse_args()

    run_server(args.port)
