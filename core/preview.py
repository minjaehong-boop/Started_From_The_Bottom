import os, time, threading, cv2
from http.server import BaseHTTPRequestHandler, HTTPServer
import numpy as np

def _has_display() -> bool:
    if os.name == "posix":
        return bool(os.environ.get("DISPLAY"))
    return True

class _MJPEGHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != '/':
            self.send_error(404); return
        self.send_response(200)
        self.send_header('Age', 0)
        self.send_header('Cache-Control', 'no-cache, private')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
        self.end_headers()
        try:
            while True:
                frame = self.server.get_jpeg()  # type: ignore
                if frame is None:
                    time.sleep(0.01); continue
                self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except Exception:
            pass

class LivePreview:
    def __init__(self, window_name: str = "preview", port: int = 8090):
        self.window_name = window_name
        self.use_window  = _has_display()
        self.closed      = False
        self._last_jpeg = None
        self._last_lock = threading.Lock()
        self._server    = None
        self._thread    = None
        self._port      = port

        if self.use_window:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        else:
            class _Srv(HTTPServer):
                def get_jpeg(s):
                    with self._last_lock:
                        return self._last_jpeg
            self._server = _Srv(('0.0.0.0', self._port), _MJPEGHandler)
            self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
            self._thread.start()
            print(f"[preview] MJPEG server: http://localhost:{self._port}")

    def show(self, frame_bgr: np.ndarray, waitkey_ms: int = 1):
        if self.closed:
            return
        if self.use_window:
            cv2.imshow(self.window_name, frame_bgr)
            if cv2.waitKey(waitkey_ms) & 0xFF == 27:
                self.close()
        else:
            ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ok:
                with self._last_lock:
                    self._last_jpeg = buf.tobytes()

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.use_window:
            try: cv2.destroyWindow(self.window_name)
            except: pass
        else:
            try:
                if self._server: self._server.shutdown()
            except: pass
