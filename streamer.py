"""
streamer.py
Synchronized Low-Latency MJPEG Web Streamer.

Optimizations for Visual SLAM:
- Zero-copy frame reference swapping (`push_frame` only copies when an active
  browser client is connected, saving ~3.7 MB/frame of RAM bandwidth at 30 FPS).
- Uses ThreadingHTTPServer so concurrent browser clients never block the main SLAM loop.
"""

import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2


_frame = None
_seq = 0
_active_clients = 0
_lock = threading.Lock()
_cond = threading.Condition(_lock)


class _DaemonThreadingHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def has_clients():
    """Return True if at least one browser client is actively viewing the MJPEG stream."""
    return _active_clients > 0


def push_frame(frame):
    """
    Publish the latest composited frame.
    Avoids expensive full-frame memory copies when no browser client is connected.
    """
    global _frame, _seq
    with _cond:
        _frame = frame.copy() if _active_clients > 0 else frame
        _seq += 1
        if _active_clients > 0:
            _cond.notify_all()


def get_frame_state():
    """Return (frame, seq) under lock."""
    with _lock:
        return _frame, _seq


class _Handler(BaseHTTPRequestHandler):
    def do_HEAD(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()

    def do_GET(self):
        global _active_clients
        if self.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(b"""
            <html>
            <head><title>YOLO Mask + Feature Tracking</title></head>
            <body style='background:#0f172a;color:#f8fafc;text-align:center;
                         font-family:sans-serif;margin:0;padding:20px;'>
                <h2 style='color:#38bdf8;margin-bottom:6px;'>
                    YOLO Mask Tracking + Visual SLAM (Ultra-Low Latency)
                </h2>
                <div style='display:inline-block;border:2px solid #334155;
                            border-radius:10px;overflow:hidden;
                            box-shadow:0 8px 30px rgba(0,0,0,0.6);background:#000;'>
                    <img src='/video_feed' style='max-width:100%;height:auto;display:block;'>
                </div>
            </body>
            </html>
            """)
        elif self.path == "/video_feed":
            self.send_response(200)
            self.send_header("Content-type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
            self.end_headers()
            last_sent = -1
            with _lock:
                _active_clients += 1
            try:
                while True:
                    with _cond:
                        _cond.wait_for(lambda: _seq != last_sent, timeout=0.1)
                        if _frame is None or _seq == last_sent:
                            continue
                        curr = _frame
                        last_sent = _seq

                    ok, jpg = cv2.imencode(".jpg", curr, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                    if not ok:
                        continue
                    data = jpg.tobytes()
                    self.wfile.write(
                        b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                        + str(len(data)).encode() + b"\r\n\r\n" + data + b"\r\n"
                    )
            except Exception:
                pass
            finally:
                with _lock:
                    _active_clients = max(0, _active_clients - 1)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        return


def get_local_ip():
    """Detect LAN IP address for external device access."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


def start(port):
    """Launch the multi-threaded MJPEG server on a daemon thread."""
    server = _DaemonThreadingHTTPServer(("0.0.0.0", port), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    local_ip = get_local_ip()
    print("=" * 62)
    print("  [WEB STREAM READY]")
    print(f"  Local URL   : http://localhost:{port}/")
    print(f"  Network URL : http://{local_ip}:{port}/")
    print("=" * 62)
    return server
