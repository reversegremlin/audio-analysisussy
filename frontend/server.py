#!/usr/bin/env python3
"""
Simple development server for Chromascope Studio frontend.
Serves static files and provides API endpoints for audio analysis and video rendering.
"""

import cgi
import json
import mimetypes
import os
import sys
import tempfile
import threading
import uuid
from http.server import HTTPServer, SimpleHTTPRequestHandler
from io import BytesIO
from pathlib import Path
from urllib.parse import parse_qs, urlparse

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Track render tasks
render_tasks = {}
render_tasks_lock = threading.Lock()


class KaleidoscopeHandler(SimpleHTTPRequestHandler):
    """HTTP handler for the Chromascope Studio."""

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
            from chromascope import AudioPipeline

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
            # Parse multipart form data
            content_type = self.headers.get("Content-Type", "")
            if "multipart/form-data" not in content_type:
                self.send_error(400, "Expected multipart/form-data")
                return

            # Parse the form data
            content_length = int(self.headers["Content-Length"])
            body = self.rfile.read(content_length)

            # Extract boundary
            boundary = content_type.split("boundary=")[1].strip()
            if boundary.startswith('"') and boundary.endswith('"'):
                boundary = boundary[1:-1]

            # Parse multipart data
            environ = {
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": content_type,
                "CONTENT_LENGTH": content_length,
            }
            fs = cgi.FieldStorage(
                fp=BytesIO(body),
                headers=self.headers,
                environ=environ,
            )

            # Get audio file
            audio_item = fs["audio"]
            if not audio_item.file:
                self.send_error(400, "No audio file provided")
                return

            # Get config
            config_str = fs.getvalue("config", "{}")
            config = json.loads(config_str)

            # Save audio to temp file
            audio_suffix = Path(audio_item.filename).suffix or ".mp3"
            audio_temp = tempfile.NamedTemporaryFile(
                suffix=audio_suffix, delete=False
            )
            audio_temp.write(audio_item.file.read())
            audio_temp.close()

            # Create task ID
            task_id = str(uuid.uuid4())

            with render_tasks_lock:
                render_tasks[task_id] = {
                    "progress": 0,
                    "message": "Starting render...",
                    "complete": False,
                    "error": None,
                    "output_path": None,
                }

            # Start background render thread
            thread = threading.Thread(
                target=self._render_video_task,
                args=(task_id, audio_temp.name, config),
                daemon=True,
            )
            thread.start()

            # Return task ID immediately
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"task_id": task_id}).encode())

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.send_error(500, str(e))

    def _render_video_task(self, task_id, audio_path, config):
        """Background task to render video."""
        output_path = None
        try:
            from chromascope.render_video import render_video

            # Create output file
            output_fd, output_path = tempfile.mkstemp(suffix=".mp4")
            os.close(output_fd)

            def progress_callback(pct, msg):
                with render_tasks_lock:
                    if task_id in render_tasks:
                        render_tasks[task_id]["progress"] = pct
                        render_tasks[task_id]["message"] = msg

            # Get render parameters from config
            width = config.get("exportWidth", 1920)
            height = config.get("exportHeight", 1080)
            fps = config.get("exportFps", 60)
            mirrors = config.get("mirrors", 8)
            trail = config.get("trailAlpha", 40)
            style = config.get("style", "geometric")

            # Map resolution strings to dimensions
            if isinstance(width, str):
                res_map = {"720p": (1280, 720), "1080p": (1920, 1080), "4k": (3840, 2160)}
                width, height = res_map.get(width, (1920, 1080))

            render_video(
                audio_path=Path(audio_path),
                output_path=Path(output_path),
                width=width,
                height=height,
                fps=fps,
                num_mirrors=mirrors,
                trail_alpha=trail,
                progress_callback=progress_callback,
                style=style,
            )

            with render_tasks_lock:
                render_tasks[task_id]["complete"] = True
                render_tasks[task_id]["output_path"] = output_path
                render_tasks[task_id]["progress"] = 100
                render_tasks[task_id]["message"] = "Complete!"

        except Exception as e:
            import traceback
            traceback.print_exc()
            with render_tasks_lock:
                render_tasks[task_id]["error"] = str(e)
                render_tasks[task_id]["message"] = f"Error: {e}"
            # Clean up output file on error
            if output_path and os.path.exists(output_path):
                os.unlink(output_path)
        finally:
            # Clean up audio temp file
            if os.path.exists(audio_path):
                os.unlink(audio_path)

    def handle_render_status(self, task_id):
        """Get render task status."""
        with render_tasks_lock:
            if task_id not in render_tasks:
                self.send_error(404, "Task not found")
                return
            task = dict(render_tasks[task_id])

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
║   🎨 Chromascope Studio                                     ║
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

    parser = argparse.ArgumentParser(description="Chromascope Studio Server")
    parser.add_argument("-p", "--port", type=int, default=8080, help="Port to run on")
    args = parser.parse_args()

    run_server(args.port)
